# Portable Brain Directory (`brain/`)

Drop any quantized `.gguf` LLM weights here (for example, `Gemma-4-E2B-IT-AB.gguf`).

## Rules
- `nc-core` auto-scans `./brain` for `*.gguf` files at startup and supports hot-swapping via the Brain Picker.
- **Never commit `.gguf` or binary model weights to Git** (enforced by `.gitignore`).
- Keep filenames simple (no spaces preferred, e.g. `Qwen2.5-Coder-3B-Q4_K_M.gguf`).
