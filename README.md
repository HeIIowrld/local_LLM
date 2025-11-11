# LLMs on Intel NPU with OpenVINO-genai

This project demonstrates running NPU-optimized LLM on an **Intel Ultra Series 1 & 2 NPU** using **OpenVINO GenAI**.  

---

## ✨ Features
- Runs NPU-optimized LLM locally on Intel NPU
- Uses **OpenVINO GenAI**
- Works **offline** (no GPU required)
- Built with **Hugging Face Transformers + Optimum-Intel**

---

## 📦 Installation

Clone this repository and install the dependencies:

```bash
python -m venv llm
#for powershell terminal security authorization issue
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
llm\Scripts\activate
pip install -r requirements.txt
```

---

## ⚡ Model Download

Download the model locally by the following command

```bash
huggingface-cli download OpenVINO/Qwen3-8B-int4-cw-ov --local-dir ./Qwen3-8B
huggingface-cli download OpenVINO/DeepSeek-R1-Distill-Qwen-7B-int4-cw-ov --local-dir ./DeepSeek-R1-Distill-Qwen-7B
huggingface-cli download OpenVINO/gpt-j-6b-int4-cw-ov --local-dir ./gpt-j-6b
huggingface-cli download OpenVINO/Phi-3.5-mini-instruct-int4-cw-ov --local-dir ./Phi-3.5-mini-instruct-int4

- This will create a folder for each with the OpenVINO-optimized model.

---

## ▶️ Usage

- Run inference with: specifying model's name

```bash
python run.py
```
