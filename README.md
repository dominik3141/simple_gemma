# Gemma 3 Pure PyTorch in a Single File

This is a simple implementation of Gemma 3 in a single file.

## Why?

While [Google's reference implementation](https://github.com/google-deepmind/gemma) of Gemma 3 is a lot (!) more efficient with regard to inference, it's an absolute pain to calculate gradients using their library from PyTorch (all their training code is in JAX).

This simple file makes it easy to experiment with different architectures.  
Ever woke up at night wondering what would happen if we made all attention layers local? Neither have I, but in case that ever happens, I just have to modify a single file to find out.

## Dependencies

Only `torch` and `sentencepiece`.

## Usage

Currently, the config is specific to the 1b instruct version, but this can easily be adjusted.

### Inference Example

1. **Download Model and Tokenizer:** Obtain the Gemma model checkpoint (`.ckpt`) and the SentencePiece tokenizer (`.model`) file for the desired variant (e.g., from Hugging Face Hub).

2. **Run the script:**

```bash
python gemma3_1b_simple.py \
  --model-path /path/to/your/model.ckpt \
  --tokenizer-path /path/to/your/tokenizer.model \
  --prompt "For each of the 50 states, give me an approximate number of people who live there."
```