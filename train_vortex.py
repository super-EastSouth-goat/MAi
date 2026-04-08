import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
import random
import math

from huggingface_hub import login


login("YOUR_HUGGINGFACE_TOKEN") # 替换成你的 Hugging Face 访问令牌

# ========================================================
# 1. 后勤准备：加载数据并进行 70/15/15 切分
# ========================================================
def load_and_split_data(filepath="QQQ.txt"):
    print("📂 正在读取你的 100 道脑筋急转弯题...")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            # 假设你的 txt 每行是一道题（格式如：问题 | 答案）
            lines = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print("⚠️ 找不到 QQQ.txt！我生成了 10 条虚拟数据用于测试打样...")
        lines = [f"测试问题 {i} —— 这是一个充满旋度的测试答案 {i}" for i in range(100)]
        
    random.shuffle(lines)
    
    # 按照 70/15/15 切分
    train_data = lines[:70]
    val_data = lines[70:85]
    test_data = lines[85:]
    
    print(f"📊 数据就绪: 训练集 {len(train_data)} | 验证集 {len(val_data)} | 测试集 {len(test_data)}")
    return train_data, val_data, test_data

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
    print("🚀 启动 M4 统一内存... 载入 Gemma-2-2B")
    model, tokenizer = load("google/gemma-2-2b-it")
    
    # 1. 全部冻结
    model.freeze()
    
    # 2. 智能遍历解冻：只要名字里带有 'embed' 或 'head' 或者是最后一层，统统解冻！
    for name, module in model.named_modules():
        if "embed" in name or "head" in name or "layers.25" in name: # 假设26层
            module.unfreeze()
    
    train_data, val_data, test_data = load_and_split_data("QQQ.txt")
    
    # 设置学习率和优化器 (AdamW 是业界标配)
    optimizer = optim.AdamW(learning_rate=2e-4)
    
    # MLX 的核心：用 value_and_grad 自动计算你的复杂涡旋公式的微分！
    step_fn = nn.value_and_grad(model, vortex_loss_fn)
    
    epochs = 10 # 跑 10 轮
    
    print("\n⚔️ 几何体积重塑训练开始！(观察 Volume 值的飙升)")
    print("-" * 60)
    
    for epoch in range(epochs):
        # 从训练集随便抽一道题（这里简化为 Batch Size 1）
        sample_text = random.choice(train_data)
        
        # 将文字转为张量格式
        tokens = tokenizer.encode(sample_text)
        if len(tokens) < 5: continue
        
        # 错位预测：输入前 N-1 个词，预测后 N-1 个词
        input_ids = mx.array([tokens[:-1]])
        targets = mx.array([tokens[1:]])
        
        # 1. 计算带有涡旋的 Loss，并自动求出梯度
        (total_loss, ce_loss, volume), grads = step_fn(model, input_ids, targets)
        
        # 2. 将梯度应用到模型权重上 (更新那 1% 解冻的参数)
        optimizer.update(model, grads)
        
        # 3. 强制 M4 的 GPU 评估计算图（MLX 是惰性计算）
        mx.eval(model.parameters(), optimizer.state)
        
        # 把 .2f 改成 .6f 
        print(f"Epoch {epoch+1:02d} | CE Loss: {ce_loss.item():.4f} | 🌪️ 思维体积: {volume.item():.6f} | 最终 Loss: {total_loss.item():.4f}")

if __name__ == "__main__":
    main()