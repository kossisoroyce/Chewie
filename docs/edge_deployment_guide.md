# 📱 Edge Deployment Guide for AfriCHW-Medical

This guide explains how to deploy the fine-tuned **Llama 3.2 3B** model on edge devices (Android, iOS, Raspberry Pi) using the GGUF format.

## 1. Model Optimization (GGUF)

First, convert your fine-tuned model to GGUF using the `notebooks/convert_to_gguf_colab.ipynb` notebook.
- **Recommended Format:** `Q4_K_M` (4-bit quantization)
- **Size:** ~2.0 GB
- **RAM Required:** ~2.5 GB

## 2. Android Deployment 🤖

### Option A: MLC LLM (Recommended for App Integration)
1. Download **MLC Chat** from the Google Play Store.
2. The Llama 3.2 3B model is supported natively.
3. To load your *custom* model, you need to compile it for MLC (requires more advanced setup) or use the "Custom Model" feature if available.

### Option B: UserLAnd / Termux (For CHW Techs)
Run full `llama.cpp` on the phone.

1. **Install Termux** from F-Droid (Play Store version is outdated).
2. **Install Dependencies:**
   ```bash
   pkg install clang wget git cmake
   ```
3. **Clone & Build:**
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   make
   ```
4. **Download Model:**
   Transfer your `.gguf` file to the phone (e.g., Downloads folder).
5. **Run:**
   ```bash
   ./llama-cli -m /sdcard/Download/africhw-llama-3.2-3b-Q4_K_M.gguf -p "You are a CHW assistant. How do I treat malaria?" -n 256
   ```

### Option C: Private LLM Apps (Easiest)
Apps like **Layla** or **Maid** (paid/free variants) allow loading local GGUF files.
1. Copy `.gguf` file to phone storage.
2. Open app -> Load Custom Model -> Select file.
3. Set prompt template to **Llama 3**.

## 3. iOS Deployment 🍎

1. **Download "MLC Chat"** or **"PocketPal"** from App Store.
2. PocketPal supports importing GGUF files from iCloud/Files app.
3. Import your `africhw-llama-3.2-3b-Q4_K_M.gguf`.

## 4. Raspberry Pi (Offline Clinic Server) 🥧

Ideal for rural clinics with no internet.

**Hardware:** Raspberry Pi 4 (8GB) or Pi 5 (8GB).

1. **Install OS:** Raspberry Pi OS (64-bit).
2. **Build llama.cpp:**
   ```bash
   sudo apt update && sudo apt install git g++ make
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   make
   ```
3. **Run Server Mode:**
   This creates a local API that other devices in the clinic can connect to via Wi-Fi.
   ```bash
   ./llama-server -m africhw-model.gguf --host 0.0.0.0 --port 8080
   ```
4. **Connect:**
   Clinic tablets can access the AI via `http://<raspberry-pi-ip>:8080`.

## 5. Laptop / Desktop 💻

**LM Studio** (Windows/Mac/Linux)
1. Download [LM Studio](https://lmstudio.ai).
2. Drag and drop your `.gguf` file into the search bar.
3. Select the model.
4. Choose "Llama 3" preset.
5. Chat!

## 6. Performance Benchmarks (Estimated)

| Device | Speed (Tokens/sec) | Usability |
|--------|--------------------|-----------|
| Snapdragon 8 Gen 2 (Android) | ~15-20 t/s | ⚡ Fast |
| Mid-range Android (Pixel 6) | ~8-10 t/s | ✅ Usable |
| Raspberry Pi 5 | ~4-6 t/s | 🆗 Readable speed |
| Raspberry Pi 4 | ~1-2 t/s | 🐢 Slow but works |

---
**Note:** For CHWs in the field, **Android (via Termux or Private LLM App)** is the most robust offline solution.
