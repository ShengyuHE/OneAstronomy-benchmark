# Light Curve Codec Inference 使用说明

## 概述

这是一个用于 AstroM3 Phase 2 光变曲线编码器与量化器的推理模块，支持：
- **编码**: 将光变曲线压缩为离散 tokens（用于存储/传输）
- **解码**: 从 tokens 重建光变曲线
- **端到端重建**: 编码 + 解码完整流程

## 模型架构

```
输入光变曲线 (B, T=200)
       │
       ▼
┌──────────────────────┐
│ Chronos-2-small      │ ← 预训练时间序列 backbone
│ (frozen, 512D)       │
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ grad_mlp Projection  │ ← 梯度感知 MLP (512D → 5D)
│                      │
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ FSQ Quantizer        │ ← 有限标量量化器
│ levels=[4,4,5,5,5]   │   vocab = 4×4×5×5×5 = 1600
└──────────────────────┘
       │
       ▼
离散 tokens (B, 80)
       │
       ▼
┌──────────────────────┐
│ Transformer Decoder  │ ← 3 层 Transformer (512D hidden)
│ (8 heads, 2048 FF)   │
└──────────────────────┘
       │
       ▼
重建光变曲线 (B, 200)
```

### 关键参数

| 组件 | 参数 | 值 |
|------|------|-----|
| Backbone | Chronos 模型 | chronos-2-small |
| Backbone 输出维度 | 512D | - |
| Projection | grad_mlp | 输入 512D → 输出 5D |
| Quantizer | FSQ levels | [4, 4, 5, 5, 5] |
| Vocabulary Size | 码表大小 | 1600 |
| Decoder | Transformer | 3 层 × 8 头 |
| Decoder Hidden | - | 512 |
| Patch Size | - | 16 |
| 序列长度 | fixed_length | 200 |

## 快速开始

### 1. 从 Checkpoint 加载

```python
from claudecode.light_curve_codec_inference import LightCurveCodecInference

# 从 Phase 2 checkpoint 加载
codec = LightCurveCodecInference.from_checkpoint(
    checkpoint_path="outputs/training_astrom3_p2_fsq_1600/checkpoints/best_model.pt",
    config_path="configs/phase2_reset/p2_reset_astrom3_fsq_1600.json",  # 可选
    device="cuda"  # 或 "cpu"
)
```

### 2. 编码：光变曲线 → Tokens

```python
import torch

# 准备输入数据 (单个样本)
flux = torch.randn(200)  # 流量值
delta_t = torch.cumsum(torch.ones(200), dim=0)  # 累积时间
delta_t_interval = torch.ones(200)  # 时间间隔

# 编码为 tokens
encoding = codec.encode(flux, delta_t, delta_t_interval)

print(f"Tokens 形状：{encoding.tokens.shape}")  # (80,) - 80 个 patches
print(f"Latent 形状：{encoding.latent.shape}")  # (80, 5) - 80 个 patches, 5D latent
print(f"词表大小：{codec.vocab_size}")  # 1600
```

**批量编码**:

```python
# 批量处理 (4 个样本)
flux_batch = torch.randn(4, 200)
delta_t_batch = torch.arange(200).float().unsqueeze(0).expand(4, -1)
delta_t_interval_batch = torch.ones(4, 200)

encoding_batch = codec.encode(flux_batch, delta_t_batch, delta_t_interval_batch)
print(f"批量 Tokens: {encoding_batch.tokens.shape}")  # (4, 80)
```

### 3. 解码：Tokens → 重建光变曲线

```python
# 从 tokens 解码
output = codec.decode(encoding.tokens, target_length=200)

print(f"重建流量：{output.flux.shape}")  # (200,)
print(f"重建时间：{output.delta_t.shape}")  # (200,)
```

### 4. 端到端重建

```python
# 完整流程：编码 + 解码
reconstruction = codec.reconstruct(flux, delta_t, delta_t_interval)

# 同时获取 tokens
reconstruction, tokens = codec.reconstruct(flux, delta_t, delta_t_interval, return_tokens=True)
```

## 完整示例

```python
import torch
from claudecode.light_curve_codec_inference import LightCurveCodecInference

def main():
    # 1. 加载模型
    codec = LightCurveCodecInference.from_checkpoint(
        checkpoint_path="outputs/training_astrom3_p2_fsq_1600/checkpoints/best_model.pt",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"模型加载完成，词表大小：{codec.vocab_size}")

    # 2. 准备测试数据
    batch_size = 4
    seq_length = 200

    flux = torch.randn(batch_size, seq_length)
    delta_t = torch.arange(seq_length).float().unsqueeze(0).expand(batch_size, -1)
    delta_t_interval = torch.ones(batch_size, seq_length)

    # 3. 编码
    encoding = codec.encode(flux, delta_t, delta_t_interval)
    print(f"编码完成，Tokens 形状：{encoding.tokens.shape}")
    print(f"码本使用率：{codec.get_codebook_usage(encoding.tokens):.2%}")

    # 4. 解码
    output = codec.decode(encoding.tokens, target_length=seq_length)
    print(f"解码完成，重建形状：{output.flux.shape}")

    # 5. 端到端重建评估
    reconstruction = codec.reconstruct(flux, delta_t, delta_t_interval)

    # 计算重建误差 (norm 空间)
    flux_mae = (reconstruction.flux - flux).abs().mean()
    print(f"Flux MAE (norm 空间): {flux_mae.item():.6f}")

    # 6. 保存配置
    codec.save_config("codec_config.json")

if __name__ == "__main__":
    main()
```

## 输出说明

### EncodingOutput (编码输出)

| 属性 | 形状 | 说明 |
|------|------|------|
| `tokens` | (B, N) | 离散 token 索引，N=num_patches |
| `latent` | (B, N, D) | 量化前的连续 latent (D=5) |
| `codes` | (B, N) | FSQ 编码索引（可用于存储） |

### LightCurveOutput (解码输出)

| 属性 | 形状 | 说明 |
|------|------|------|
| `flux` | (B, T) | 重建流量值（norm 空间） |
| `delta_t` | (B, T) | 重建累积时间 |
| `delta_t_interval` | (B, T) | 重建时间间隔 |
| `flux_norm` | (B, T) | 同 flux (norm 空间) |
| `delta_t_norm` | (B, T) | 同 delta_t (norm 空间) |
| `delta_t_interval_norm` | (B, T) | 同 delta_t_interval |

## 高级用法

### 1. Tokens 存储与传输

```python
# 编码并保存 tokens 为 numpy 数组
encoding = codec.encode(flux, delta_t, delta_t_interval)
tokens_np = encoding.tokens.cpu().numpy()
import numpy as np
np.save("tokens.npy", tokens_np)

# 加载并重建
tokens_loaded = torch.from_numpy(np.load("tokens.npy"))
output = codec.decode(tokens_loaded, target_length=200)
```

### 2. 码本使用率分析

```python
# 分析测试集上的码本使用率
all_tokens = []
for batch in test_loader:
    enc = codec.encode(batch["flux"], batch["delta_t"], batch["delta_t_interval"])
    all_tokens.append(enc.tokens)

all_tokens = torch.cat(all_tokens, dim=0)
usage = codec.get_codebook_usage(all_tokens)
print(f"整体码本使用率：{usage:.2%}")  # 实验值约 8.7%
```

### 3. 模型配置导出

```python
# 导出配置
config = codec.to_dict()
print(f"嵌入维度：{config['embed_dim']}")
print(f"词表大小：{config['vocab_size']}")
print(f"FSQ levels: {config['levels']}")

# 保存为 JSON
codec.save_config("my_codec_config.json")
```

## 与其他脚本的集成

### 与 eval_light_curve_codec_astrom3_p2.py 对比

当前推理类的功能等价于评估脚本，但提供了更简洁的 API：

| 评估脚本 | 推理类 |
|----------|--------|
| 手动构建 encoder/decoder | `LightCurveCodecInference.from_checkpoint()` |
| 手动处理 state_dict | 自动加载 |
| 需要 config 文件 | 可选 config（checkpoint 内嵌） |
| 复杂的数据加载 | 简洁的 `encode()` 接口 |

### 在 4M 模型中使用

```python
# 作为 4M 模型的一个 modality codec
from aion.codecs.manager import ModalityCodecManager

codec = LightCurveCodecInference.from_checkpoint(
    "outputs/training_astrom3_p2_fsq_1600/checkpoints/best_model.pt"
)

# 注册到管理器
manager.register_codec("light_curve", codec)

# 使用 manager 统一调用
tokens = manager.encode("light_curve", flux, delta_t, delta_t_interval)
```

## 实验数据参考

基于 AstroM3 Phase 2 实验记录（`docs/experiment_log.md`）:

| 指标 | Phase 2 (FSQ 1600) |
|------|-------------------|
| Best Epoch | 19 |
| Best Val Loss | 0.4753 |
| Flux MAE Raw | 1.2063 |
| Flux RMSE Raw | 3.3216 |
| Flux R² Normal Raw | 0.9861 |
| Flux Cross Correlation | 0.5193 |
| Codebook Usage | 8.74% (~140/1600 codes) |
| 测试集样本数 | 2225 |

## 故障排查

### 问题：Checkpoint 加载失败

```python
# 确保路径正确
ckpt_path = Path("outputs/training_astrom3_p2_fsq_1600/checkpoints/best_model.pt")
if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint 不存在：{ckpt_path}")
```

### 问题：输出形状不匹配

```python
# 检查输入形状
print(f"Flux 形状：{flux.shape}")  # 应为 (B, T) 或 (T,)

# 检查 target_length
output = codec.decode(tokens, target_length=200)  # 明确指定
```

### 问题：码本使用率过低

这是 Phase 2 训练的已知问题（见 `docs/experiment_log.md`），commitment_loss_weight=0 导致。
后续实验可尝试：
1. 减小 vocab（levels=[4,4,5,5] → 400 codes）
2. 增加 commitment_loss_weight
3. 使用 LFQ 量化器

## 文件结构

```
claudecode/
├── light_curve_codec_inference.py  # 推理类
└── README.md                        # 本文档

AION/
├── codecs/
│   ├── light_curve_p2_reset_encoder.py  # P2-Reset 编码器
│   ├── light_curve_v5_1_decoder.py      # V5.1 解码器
│   └── quantizers/
│       └── __init__.py                  # FSQ 量化器
├── configs/phase2_reset/
│   └── p2_reset_astrom3_fsq_1600.json   # Phase 2 配置
└── outputs/
    └── training_astrom3_p2_fsq_1600/
        └── checkpoints/
            └── best_model.pt            # 训练好的 checkpoint
```

## 下一步

基于当前实验结果，后续可探索的方向：

1. **Exp-004**: 缩减 vocab 到 400 (levels=[4,4,5,5])，提高码本利用率
2. **Exp-005**: 扩大 embed_dim 到 8D，增加容量
3. **Exp-006**: 使用 LFQ 量化器，真正的 entropy loss 驱动均匀使用码本

详见 `docs/experiment_log.md` "待处理实验" 部分。
