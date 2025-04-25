# src/gemma3_1b_simple.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
import logging
import sentencepiece as spm
import os
import sys
import argparse

# Setup basic logging (only config)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Tokenizer Implementation --- Start ---
class SimpleTokenizer:
    """A simple wrapper for the SentencePiece tokenizer."""

    def __init__(self, model_path: str):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Tokenizer model file not found at: {model_path}")

        try:
            self.processor = spm.SentencePieceProcessor(model_file=model_path)
            self._bos_id = self.processor.bos_id()
            self._eos_id = self.processor.eos_id()
            self._pad_id = self.processor.pad_id()
            # No logging inside __init__ for reusability
            # if self._pad_id == -1:
            #     logging.warning("SentencePiece model does not define a pad_id...")
            # logging.info(f"Tokenizer loaded. BOS ID: {self._bos_id}, ...")
        except Exception as e:
            # Still log critical load errors
            logging.error(f"Failed to load SentencePiece model from {model_path}: {e}")
            raise

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def pad_id(self) -> int:
        return self._pad_id

    def encode(self, text: str) -> List[int]:
        return self.processor.encode(text, add_bos=False, add_eos=False)

    def decode(self, ids: List[int]) -> str:
        return self.processor.decode(ids)


# --- Tokenizer Implementation --- End ---


# --- Helper functions ---
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_transposed = x.transpose(1, 2)
    x_stacked = torch.stack(torch.chunk(x_transposed.float(), 2, dim=-1), dim=-1)
    x_complex = torch.view_as_complex(x_stacked)
    freqs_cis_broadcast = freqs_cis.unsqueeze(0).unsqueeze(2)
    x_rotated_complex = x_complex * freqs_cis_broadcast
    x_rotated_real = torch.view_as_real(x_rotated_complex)
    x_out_real = torch.cat(torch.chunk(x_rotated_real, 2, dim=-1), dim=-2).squeeze(-1)
    x_out = x_out_real.transpose(1, 2)
    return x_out.type_as(x)


# --- Model Configuration ---
@dataclass
class SimpleGemmaConfig:
    hidden_size: int = 1152
    num_hidden_layers: int = 26
    num_attention_heads: int = 4
    num_key_value_heads: int = 1
    head_dim: int = 256
    intermediate_size: int = 6912
    vocab_size: int = 262144
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    use_qk_norm: bool = True
    use_post_ffw_norm: bool = True  # Corresponds to post_feedforward_layernorm
    use_pre_ffw_norm: bool = True  # Corresponds to pre_feedforward_layernorm
    rope_theta_local: float = 10000.0
    rope_theta_global: float = 1000000.0
    attn_types: List[str] = field(
        default_factory=lambda: ["local", "local", "local", "local", "local", "global"]
        * 4
        + ["local", "local"]
    )
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if len(self.attn_types) != self.num_hidden_layers:
            raise ValueError(
                f"Length of attn_types ({len(self.attn_types)}) must match num_hidden_layers ({self.num_hidden_layers})"
            )


# --- Model Layers ---
class SimpleRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()) * (1 + self.weight.float())
        return output.type_as(x)


class SimpleGemmaMLP(nn.Module):
    def __init__(self, config: SimpleGemmaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class SimpleGemmaAttention(nn.Module):
    def __init__(self, config: SimpleGemmaConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )
        if config.use_qk_norm:
            self.query_norm = SimpleRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_norm = SimpleRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.query_norm = nn.Identity()
            self.key_norm = nn.Identity()
        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        # Split Q, K, V
        q_start = 0
        q_end = self.num_heads * self.head_dim
        k_start = q_end
        k_end = q_end + self.num_kv_heads * self.head_dim
        v_start = k_end
        v_end = k_end + self.num_kv_heads * self.head_dim
        query_states = qkv[:, :, q_start:q_end]
        key_states = qkv[:, :, k_start:k_end]
        value_states = qkv[:, :, v_start:v_end]

        # Reshape
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # QK Norm
        query_states = self.query_norm(query_states)
        key_states = self.key_norm(key_states)

        # Apply RoPE
        query_states = apply_rotary_emb(query_states, freqs_cis=freqs_cis)
        key_states = apply_rotary_emb(key_states, freqs_cis=freqs_cis)

        # Repeat K, V heads for MQA/GQA
        if self.num_kv_groups > 1:
            key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)

        # Scale Q
        query_states = query_states * self.scaling

        # Attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        # Apply mask
        if attention_mask is not None:
            # Mask size check removed for library usage
            # if attention_mask.size() != (bsz, 1, q_len, q_len):
            #      logging.warning(...)
            attn_weights = attn_weights + attention_mask.float()

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            query_states
        )

        # Output
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output


class SimpleGemmaDecoderLayer(nn.Module):
    def __init__(self, config: SimpleGemmaConfig):
        super().__init__()
        self.self_attn = SimpleGemmaAttention(config)
        self.mlp = SimpleGemmaMLP(config)
        self.input_layernorm = SimpleRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = SimpleRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # Norms around MLP block, names match checkpoint keys
        self.pre_feedforward_layernorm = SimpleRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = SimpleRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention Block
        residual = hidden_states
        normed_hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(normed_hidden_states, attention_mask, freqs_cis)
        normed_attn_output = self.post_attention_layernorm(attn_output)
        hidden_states = residual + normed_attn_output

        # MLP Block
        residual = hidden_states
        normed_hidden_states_for_mlp = self.pre_feedforward_layernorm(hidden_states)
        mlp_output = self.mlp(normed_hidden_states_for_mlp)
        normed_mlp_output = self.post_feedforward_layernorm(mlp_output)
        hidden_states = residual + normed_mlp_output

        return hidden_states


# --- Full Model ---
class SimpleGemmaModel(nn.Module):
    def __init__(self, config: SimpleGemmaConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SimpleGemmaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = SimpleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Precompute RoPE frequencies
        local_freqs_cis = precompute_freqs_cis(
            config.head_dim,
            config.max_position_embeddings * 2,
            theta=config.rope_theta_local,
        )
        global_freqs_cis = precompute_freqs_cis(
            config.head_dim,
            config.max_position_embeddings * 2,
            theta=config.rope_theta_global,
        )
        self.register_buffer("local_freqs_cis", local_freqs_cis, persistent=False)
        self.register_buffer("global_freqs_cis", global_freqs_cis, persistent=False)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        # Ensure buffers are on the correct device (might be moved by .to(device))
        local_freqs_cis = self.local_freqs_cis.to(device)[:seq_len]
        global_freqs_cis = self.global_freqs_cis.to(device)[:seq_len]

        for idx, layer in enumerate(self.layers):
            attn_type = self.config.attn_types[idx]
            current_freqs_cis = (
                local_freqs_cis if attn_type == "local" else global_freqs_cis
            )
            hidden_states = layer(hidden_states, attention_mask, current_freqs_cis)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class SimpleGemmaForCausalLM(nn.Module):
    def __init__(self, config: SimpleGemmaConfig):
        super().__init__()
        self.config = config
        self.embedder = nn.Embedding(config.vocab_size, config.hidden_size)
        self.model = SimpleGemmaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedder.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for training or simple inference.

        Handles creation of attention mask internally.
        """
        bsz, seq_len = input_ids.shape

        # 1. Create Causal Attention Mask (if not provided)
        if attention_mask is None:
            attention_mask = torch.ones(
                (bsz, seq_len), dtype=torch.long, device=input_ids.device
            )

        # Expand mask to 4D: (bsz, 1, q_len, kv_len) and incorporate causal nature
        causal_mask_val = torch.finfo(torch.float32).min
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), causal_mask_val, device=input_ids.device),
            diagonal=1,
        )
        padding_mask_expanded = attention_mask[:, None, None, :].to(causal_mask.dtype)
        causal_mask = causal_mask.unsqueeze(0)
        final_attention_mask = torch.where(
            padding_mask_expanded == 0, causal_mask_val, causal_mask
        )
        final_attention_mask = final_attention_mask.to(next(self.parameters()).dtype)

        # 2. Get Embeddings & Normalize
        hidden_states = self.embedder(input_ids)
        hidden_states = hidden_states * (self.config.hidden_size**0.5)

        # 3. Pass through Transformer blocks
        hidden_states = self.model(hidden_states, final_attention_mask)

        # 4. Compute Logits
        logits = self.lm_head(hidden_states)
        logits = logits.float()  # Ensure float32 for loss/sampling

        # 5. Calculate Loss (optional)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # Default ignore_index=-100
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return logits, loss


# --- Weight Loading Function ---
# Return type changed to include loading status
LoadResult = Tuple[
    Optional[List[str]], Optional[List[str]]
]  # (missing_keys, unexpected_keys)


def load_pretrained(
    config: SimpleGemmaConfig, ckpt_path: str, device: Optional[str] = None
) -> Tuple[SimpleGemmaForCausalLM, LoadResult]:
    # No internal logging here for reusability
    # logging.info(f"Starting weight loading from: {ckpt_path}")

    model = SimpleGemmaForCausalLM(config)
    model_state_dict = model.state_dict()

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            original_state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict):
            original_state_dict = checkpoint
        else:
            raise TypeError("Loaded checkpoint is not a dictionary.")
        # Remove temporary inspection logging
        # logging.info("--- Inspecting Checkpoint Keys...")

    except FileNotFoundError:
        logging.error(
            f"Checkpoint file not found at {ckpt_path}"
        )  # Keep critical error logs
        raise
    except Exception as e:
        logging.error(f"Error loading checkpoint file: {e}")  # Keep critical error logs
        raise

    mapped_state_dict = {}
    skipped_keys_mapping = []
    used_original_keys = set()

    for key_ours in model_state_dict.keys():
        potential_orig_key = None
        if key_ours == "lm_head.weight":
            if config.tie_word_embeddings:
                # logging.debug(f"Skipping tied lm_head.weight: {key_ours}")
                continue
            else:
                potential_orig_key = "lm_head.weight"
        elif key_ours == "embedder.weight":
            potential_orig_key_options = ["embedder.weight", "model.embedder.weight"]
            for name in potential_orig_key_options:
                if name in original_state_dict:
                    potential_orig_key = name
                    break
        elif key_ours == "model.norm.weight":
            potential_orig_key_options = ["model.norm.weight", "norm.weight"]
            for name in potential_orig_key_options:
                if name in original_state_dict:
                    potential_orig_key = name
                    break
        elif key_ours.startswith("model.layers."):
            potential_orig_key = key_ours

        if potential_orig_key is None:
            potential_orig_key = key_ours
            # logging.debug(f"Applying fallback mapping for key: {key_ours} -> {potential_orig_key}")

        if potential_orig_key in original_state_dict:
            if (
                original_state_dict[potential_orig_key].shape
                != model_state_dict[key_ours].shape
            ):
                # Collect skipped keys instead of logging warning
                skipped_keys_mapping.append(
                    key_ours + f" (shape mismatch, tried {potential_orig_key})"
                )
                continue
            mapped_state_dict[key_ours] = original_state_dict[potential_orig_key]
            used_original_keys.add(potential_orig_key)
            # logging.debug(f"Mapped: {key_ours} <-- {potential_orig_key}")
        else:
            # Collect skipped keys instead of logging warning
            skipped_keys_mapping.append(
                key_ours + f" (no mapping found, tried '{potential_orig_key}')"
            )

    load_result_info: LoadResult = (None, None)
    try:
        load_result = model.load_state_dict(mapped_state_dict, strict=False)
        # Collect results instead of logging info/errors
        # logging.info(f"State dict loaded (strict=False). Result: {load_result}")
        missing_keys = load_result.missing_keys
        unexpected_keys = load_result.unexpected_keys

        # Don't treat tied lm_head as missing if embedder was loaded
        if (
            config.tie_word_embeddings
            and "lm_head.weight" in missing_keys
            and "embedder.weight" in mapped_state_dict
        ):
            missing_keys.remove("lm_head.weight")

        load_result_info = (missing_keys or None, unexpected_keys or None)

    except Exception as e:
        logging.error(
            f"Error during model.load_state_dict: {e}"
        )  # Keep critical error logs
        raise

    if device:
        model.to(device)
        # logging.info(f"Model moved to device: {device}")

    # Return model and loading results
    return model, load_result_info


# --- Example Usage / Main Block ---
if __name__ == "__main__":
    # --- Argument Parsing --- Start ---
    parser = argparse.ArgumentParser(
        description="Load a Gemma 3 1B model and run simple inference."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the Gemma model checkpoint (.ckpt) file.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to the SentencePiece tokenizer model (.model) file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="For each of the 50 states, give me an approximate number of people who live there.",
        help="The prompt to use for generation.",
    )
    # Add an argument for device selection
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (e.g., 'cuda', 'cpu').",
    )

    args = parser.parse_args()
    # --- Argument Parsing --- End ---

    # Use parsed arguments
    MODEL_PATH = args.model_path
    TOKENIZER_PATH = args.tokenizer_path
    DEVICE = args.device
    prompt = args.prompt

    logging.info(f"Using device: {DEVICE}")

    # --- Load Config ---
    config = SimpleGemmaConfig()
    logging.info(f"Model Configuration: {config}")

    # --- Load Tokenizer ---
    logging.info(f"Loading tokenizer from: {TOKENIZER_PATH}")
    try:
        tokenizer = SimpleTokenizer(TOKENIZER_PATH)
        # Log tokenizer details here
        logging.info(
            f"Tokenizer loaded. BOS ID: {tokenizer.bos_id}, EOS ID: {tokenizer.eos_id}, PAD ID: {tokenizer.pad_id}"
        )
        if tokenizer.pad_id == -1:
            logging.warning(
                "Tokenizer model does not define a pad_id. Padding during batching might require special handling."
            )
        eos_id = tokenizer.eos_id
        bos_id = tokenizer.bos_id
    except FileNotFoundError:
        logging.error(
            f"Tokenizer file not found at {TOKENIZER_PATH}. Use --tokenizer-path to specify the correct file. Exiting."
        )
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading tokenizer: {e}. Exiting.")
        sys.exit(1)

    # --- Load Pretrained Weights ---
    logging.info(f"Loading pretrained model weights from: {MODEL_PATH}")
    try:
        model, load_info = load_pretrained(config, MODEL_PATH, device=DEVICE)
        missing_keys, unexpected_keys = load_info

        logging.info(f"Model loaded onto device: {DEVICE}")
        if missing_keys:
            logging.warning(
                f"State dict loading finished with MISSING keys: {missing_keys}"
            )
        if unexpected_keys:
            logging.warning(
                f"State dict loading finished with UNEXPECTED keys: {unexpected_keys}"
            )
        if not missing_keys and not unexpected_keys:
            logging.info(
                "State dict loaded successfully with no missing or unexpected keys."
            )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total Parameters: {total_params / 1e9:.3f} B")
        logging.info(f"Trainable Parameters: {trainable_params / 1e9:.3f} B")

    except FileNotFoundError:
        logging.error(
            f"Model checkpoint file not found at {MODEL_PATH}. Use --model-path to specify the correct file. Exiting."
        )
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to load pretrained model: {e}", exc_info=True)
        sys.exit(1)

    # --- Simple Inference Test ---
    logging.info("--- Starting Simple Inference Test ---")
    model.eval()  # Set model to evaluation mode

    max_gen_len = 100
    temperature = 0.9  # Slightly higher temperature for potentially more diverse output
    top_k = 50  # Widen top-k

    logging.info(f"Prompt: {prompt}")
    logging.info(f"Max generation length: {max_gen_len}")
    logging.info(f"Temperature: {temperature}")
    logging.info(f"Top-k: {top_k}")

    try:
        # Tokenize prompt
        prompt_tokens_list = tokenizer.encode(prompt)
        if bos_id != -1:
            prompt_tokens = [bos_id] + prompt_tokens_list
        else:
            prompt_tokens = prompt_tokens_list
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=DEVICE)

        generated_tokens = list(prompt_tokens)
        total_gen_len = 0

        with torch.no_grad():
            for i in range(max_gen_len):
                current_input_ids = torch.tensor(
                    [generated_tokens], dtype=torch.long, device=DEVICE
                )
                # current_seq_len = current_input_ids.shape[1] # Not strictly needed if mask generated internally

                # No need to recreate full mask each time, but simpler for now
                logits, _ = model(
                    input_ids=current_input_ids
                )  # Attention mask created internally

                # Get logits for the very last token
                last_token_logits = logits[0, -1, :]

                # Sampling
                if temperature > 0:
                    last_token_logits = last_token_logits / temperature
                    if top_k > 0:
                        v, _ = torch.topk(last_token_logits, top_k)
                        last_token_logits[last_token_logits < v[-1]] = -float("Inf")
                    probs = F.softmax(last_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).item()
                else:  # Greedy
                    next_token_id = torch.argmax(last_token_logits, dim=-1).item()

                # Stop if EOS is generated
                if next_token_id == eos_id:
                    logging.info(
                        f"EOS token generated at step {i + 1}. Stopping generation."
                    )
                    break

                generated_tokens.append(next_token_id)
                total_gen_len += 1

        # Decode the full sequence
        generated_text = tokenizer.decode(generated_tokens)

        logging.info("--- Inference Finished ---")
        print("\n--- Generated Text (Prompt + Completion) ---")
        print(generated_text)
        print("---------------------------------------------")

    except Exception as e:
        logging.error(f"Failed during inference test: {e}", exc_info=True)

    logging.info("Script finished.")
