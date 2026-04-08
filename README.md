# 🌪️ Vortex-LLM: Geometric Volume Regularization for LLMs

> *"This architecture cannot be trained efficiently on a traditional PCIe GPU cluster. It relies on non-linear continuous algebraic topologies and Unified Memory Zero-Copy architecture. Trained and verified entirely on a MacBook Air M4."*

## 💡 The Philosophy (The Manifesto)
Modern LLMs suffer from "Oversmoothing" — as depth increases, feature vectors collapse into identical representations. Traditional scaling laws solve this by brute-forcing with $100M+ NVIDIA clusters. 

**Vortex-LLM takes a mathematical shortcut.** By extracting the three core physical fields (Input, Attention, MLP) and calculating their 3D Geometric Volume (Determinant) via a $3 \times 3$ matrix, we introduce a **Volume Expansion Penalty** into the loss function. We force the LLM to physically "stretch" its thought space, accelerating the "Grokking" phenomenon on complex logic tasks.

## 🚀 Key Features
- **Zero-Copy Math Injection:** Built natively on Apple `MLX`. Extracts internal tensors without PCIe bottleneck delays.
- **Rule of Sarrus Custom Operator:** Bypasses framework API limitations with a high-concurrency tensor-sliced determinant calculator.
- **Right-Brain Activation:** The custom loss function penalizes linear thinking and rewards high-dimensional feature divergence.

## ⚙️ Quick Start (M-Series Mac Required)

### 1. Setup
Install dependencies:
```bash
pip install mlx mlx-lm huggingface_hub
```
Make sure to login to HuggingFace via CLI first, and prepare a QQQ.txt with Q&A pairs.

### 2. Run the Vortex Fine-Tuner
```bash
python train_vortex.py