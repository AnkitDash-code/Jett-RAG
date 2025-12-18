# Offline RAG Project (KoboldCpp + GGUF)

This repo is set up to run a local LLM via **KoboldCpp** using a **GGUF** model file.

## What you need to download

1. **KoboldCpp for Windows** (`koboldcpp.exe`)

   - Download from: https://github.com/LostRuins/koboldcpp/releases/tag/v1.103

2. **Mistral 7B Instruct v0.2 (GGUF)**
   - Model hub page: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF

---

## Step 1 — Download `koboldcpp.exe`

1. Open the KoboldCpp release page:
   - https://github.com/LostRuins/koboldcpp/releases/tag/v1.103
2. In the **Assets** section, download the Windows executable (commonly named something like `koboldcpp.exe` or similar).
3. Place `koboldcpp.exe` in the LLM-Backend repo root.

Recommended layout for this repo:

- Put it in a new folder at the repo root, for example:
  - `LLM-Backend/koboldcpp.exe`

(You can also place it elsewhere; just remember the path you chose.)

---

## Step 2 — Download a GGUF model file

1. Open the model page:
   - https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
2. Choose **one** `.gguf` file to download.

Notes on choosing a file:

- Smaller quantizations (e.g., `Q4_...`) generally use less RAM/VRAM and run faster.
- Larger quantizations (e.g., `Q6`, `Q8`) are higher quality but need more memory.

Recommended layout for this repo:

- Put the `.gguf` file in:
  - `./LLM-Backend/mistral-7b-instruct-v0.2.Q4_K_M.gguf`

This repo already expects/contains a file named:

- `LLM-Backend/mistral-7b-instruct-v0.2.Q4_K_M.gguf`

If you download a different GGUF filename, either:

- rename it to match what your scripts/config expect, or
- update your config/scripts to point at the new filename.

---

## Quick sanity check (optional)

From the repo root, confirm you have:

- `koboldcpp/koboldcpp.exe`
- `backend_api/<your-model>.gguf`

---

## Troubleshooting

- If Windows SmartScreen blocks `koboldcpp.exe`, you may need to click **More info → Run anyway**.
- If the model fails to load, you likely need a smaller quantization (less memory use) or to close other memory-heavy apps.

---

## Links

- KoboldCpp release (v1.103): https://github.com/LostRuins/koboldcpp/releases/tag/v1.103
- Mistral 7B Instruct v0.2 GGUF (TheBloke): https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
