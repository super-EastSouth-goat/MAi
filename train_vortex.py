import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
import random
import math
import pandas as pd
from huggingface_hub import login
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
login("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") # 替换成你的 Hugging Face 访问令牌

# ========================================================
# 1. 后勤准备：加载数据并进行 70/15/15 切分
# ========================================================
def load_and_split_data(path="dataset"):
    print("📂 正在读取你的数据集...")
    test_df = pd.read_parquet("dataset/test/test-00000-of-00001.parquet")
    if "text" in test_df.columns:
        test_lines = test_df["text"].tolist()
    else:
        test_lines = ("问题: " + test_df["question"].astype(str) + "\n答案: " + test_df["answer"].astype(str)).tolist()
        
    train_df = pd.read_parquet("dataset/train/train-00000-of-00001.parquet")
    if "text" in train_df.columns:
        train_lines = train_df["text"].tolist()
    else:
        train_lines = ("问题: " + train_df["question"].astype(str) + "\n答案: " + train_df["answer"].astype(str)).tolist()
    
    train_data =train_lines
    test_data =test_lines
    
    print(f"📊 数据就绪: 训练集 {len(train_data)} | 测试集 {len(test_data)}")
    return train_data, test_data

# ========================================================
# 2. 核心数学：计算 3D 几何体积 (替代 MoE 专家)
# ========================================================
def compute_volume_loss(hidden_in, hidden_attn, hidden_mlp):
    """
    用这一层的输入、Attention输出、MLP输出构建 3x3 矩阵，求体积
    """
    # 提取最后 3 个 Token 的特征作为 Q, K, v 的代表向量
    # 在 4096 维空间里，我们取它们的点积投影
    q_mock = hidden_in[:, -1, :]   # 当前词
    k_mock = hidden_in[:, -2, :]   # 上一个词
    v_mock = hidden_in[:, -3, :]   # 上上个词
    
    # 构建 3x3 矩阵 (在 Batch 维度上进行)
    # 用三大物理场与特征向量做点积，瞬间降维
    row1 = mx.stack([mx.sum(hidden_in[:, -1, :] * q_mock, axis=-1), 
                     mx.sum(hidden_in[:, -1, :] * k_mock, axis=-1), 
                     mx.sum(hidden_in[:, -1, :] * v_mock, axis=-1)], axis=-1)
                     
    row2 = mx.stack([mx.sum(hidden_attn[:, -1, :] * q_mock, axis=-1), 
                     mx.sum(hidden_attn[:, -1, :] * k_mock, axis=-1), 
                     mx.sum(hidden_attn[:, -1, :] * v_mock, axis=-1)], axis=-1)
                     
    row3 = mx.stack([mx.sum(hidden_mlp[:, -1, :] * q_mock, axis=-1), 
                     mx.sum(hidden_mlp[:, -1, :] * k_mock, axis=-1), 
                     mx.sum(hidden_mlp[:, -1, :] * v_mock, axis=-1)], axis=-1)
    
    M = mx.stack([row1, row2, row3], axis=-1) # shape: (batch_size, 3, 3)
    
# 假设 M 是形状为 (batch, 3, 3) 的张量
# M[:, 0, 0] 就是 a, M[:, 0, 1] 就是 b... 以此类推
    a, b, c = M[:, 0, 0], M[:, 0, 1], M[:, 0, 2]
    d, e, f = M[:, 1, 0], M[:, 1, 1], M[:, 1, 2]
    g, h, i = M[:, 2, 0], M[:, 2, 1], M[:, 2, 2]

    
    det_M = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
    volume = mx.abs(det_M*10000000.0) + 1e-6
    
    return volume.mean()

# ========================================================
# 3. 拦截与重塑：带体积惩罚的自定义 Loss
# ========================================================
def vortex_loss_fn(model, input_ids, targets, volume_weight=0.01): # 把权重稍微调大一点点
    logits = model(input_ids)
    ce_loss = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)).mean()
    
    # ⭐️ 修复线性相关性 ⭐️
    hidden_states = model.model.embed_tokens(input_ids) 
    
    # 利用非线性函数，强行把特征推向正交维度，撑开平行六面体！
    hidden_attn = mx.maximum(hidden_states, 0) # ReLU
    hidden_mlp = mx.sin(hidden_states)         # Sin波
    
    volume = compute_volume_loss(hidden_states, hidden_attn, hidden_mlp)
    
    # 依然是死记硬背误差 减去 思维体积
    total_loss = ce_loss - (volume_weight * mx.log(volume))
    
    return total_loss, ce_loss, volume
    # 正常的前向传播
    logits = model(input_ids)
    
    # 标准的死记硬背误差 (Cross Entropy)
    ce_loss = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)).mean()
    
    # 为了演示，我们直接抓取最终层前的一些激活状态模拟三大物理场
    # (在极其深度的定制中，我们会从 Hook 里抓取特定层的状态)
    hidden_states = model.model.embed_tokens(input_ids) # 模拟输入场
    hidden_attn = hidden_states * 1.1 # 模拟注意力的拉伸
    hidden_mlp = hidden_states * 0.9  # 模拟 MLP 的压缩
    
    volume = compute_volume_loss(hidden_states, hidden_attn, hidden_mlp)
    
    # 蝴蝶效应：用做对题的 Loss，减去我们渴望撑开的“体积”
    # 注意用 mx.log 防止体积过大导致梯度爆炸
    total_loss = ce_loss - (volume_weight * mx.log(volume))
    
    return total_loss, ce_loss, volume

# ========================================================
# 4. 主控台：训练循环入口
# ========================================================
def main():
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    train_data, test_data = load_and_split_data("dataset")

    def run_training_experiment(run_name, vw_value, epochs=50):
        print(f"\n🚀 启动 M4 统一内存... 载入 Gemma-2-2B ({run_name})")
        model, tokenizer = load("google/gemma-2-2b-it")
        
        # 1. 全部冻结
        model.freeze()
        
        # 2. 智能遍历解冻：只要名字里带有 'embed' 或 'head' 或者是最后一层，统统解冻！
        for name, module in model.named_modules():
            if "embed" in name or "head" in name or "layers.25" in name: # 假设26层
                module.unfreeze()
        
        optimizer = optim.AdamW(learning_rate=2e-4)
        
        # 仅修改传入的 volume_weight 权重 
        def custom_loss_fn(m, x, y):
            return vortex_loss_fn(m, x, y, volume_weight=vw_value)
            
        step_fn = nn.value_and_grad(model, custom_loss_fn)
        
        print(f"\n⚔️ {run_name} 训练开始 (volume_weight={vw_value})")
        print("-" * 60)
        
        history = {"ce_loss": [], "volume": []}
        
        for epoch in range(epochs):
            sample_text = random.choice(train_data)
            tokens = tokenizer.encode(sample_text)
            if len(tokens) < 5: continue
            
            input_ids = mx.array([tokens[:-1]])
            targets = mx.array([tokens[1:]])
            
            (total_loss, ce_loss, volume), grads = step_fn(model, input_ids, targets)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            ce_val = ce_loss.item()
            vol_val = volume.item()
            history["ce_loss"].append(ce_val)
            history["volume"].append(vol_val)
            
            print(f"Epoch {epoch+1:02d} | CE Loss: {ce_val:.4f} | 🌪️ 思维体积: {vol_val:.6f} | 最终 Loss: {total_loss.item():.4f}")
            
        return history

    history_vortex = run_training_experiment("Vortex (实验组)", vw_value=0.01, epochs=50)
    history_baseline = run_training_experiment("Baseline (对照组)", vw_value=0.0, epochs=50)

    # ========================================================
    # 5. 终极数据可视化 
    # ========================================================
    print("\n📊 正在生成图表 A (Grokking 心电图)...")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    steps_vortex = list(range(1, len(history_vortex["ce_loss"]) + 1))
    
    color1 = 'tab:red'
    ax1.set_xlabel('Training Steps (训练步数)')
    ax1.set_ylabel('CE Loss (死记硬背的误差)', color=color1)
    ax1.plot(steps_vortex, history_vortex["ce_loss"], color=color1, label='CE Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()  
    color2 = 'tab:blue'
    ax2.set_ylabel('Volume (思维体积)', color=color2)  
    ax2.plot(steps_vortex, history_vortex["volume"], color=color2, label='Volume', linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('图表 A: Grokking (顿悟) 心电图 - Vortex Model')
    fig.tight_layout()  
    plt.savefig('chart_A_grokking.png', dpi=300)
    print("✅ 图表 A 保存为 'chart_A_grokking.png'")

    print("📊 正在生成图表 B (Ablation 对比实验图)...")
    plt.figure(figsize=(10, 6))
    
    steps_baseline = list(range(1, len(history_baseline["ce_loss"]) + 1))
    plt.plot(steps_vortex, history_vortex["ce_loss"], color='tab:red', label='Vortex Model (volume_weight=0.01)', linewidth=2)
    plt.plot(steps_baseline, history_baseline["ce_loss"], color='tab:gray', label='Baseline Model (volume_weight=0)', linewidth=2, linestyle='-.')
    
    plt.xlabel('Training Steps (训练步数)')
    plt.ylabel('CE Loss')
    plt.title('图表 B: 对比实验图 (Ablation Study)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('chart_B_ablation.png', dpi=300)
    print("✅ 图表 B 保存为 'chart_B_ablation.png'")

if __name__ == "__main__":
    main()