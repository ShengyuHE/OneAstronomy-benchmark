# SpectrumCodec.encode() 中间结果维度分析

> 基于 `aion/codecs/spectrum.py` 源码分析，以 `batch_size=32` 为例

## 输入数据维度

| 变量 | 维度 | 说明 |
|------|------|------|
| `flux` | `[32, ~7781]` | 光谱流量（波长≤10000Å过滤后） |
| `ivar` | `[32, ~7781]` | 逆方差权重 |
| `mask` | `[32, ~7781]` | 像素掩码 |
| `wavelength` | `[32, ~7781]` | 波长网格 |

---

## `_encode()` 方法中间结果维度

| 行号 | 变量 | 维度 | 说明 |
|------|------|------|------|
| 58-61 | `flux`, `ivar`, `mask`, `wavelength` | `[32, ~7781]` | 输入光谱数据（波长≤10000Å过滤后） |
| 65 | `spectrum` | `[32, ~7781]` | NaN处理后的flux |
| 71 | `istd` | `[32, ~7781]` | `sqrt(ivar)` |
| 74-76 | `normalization` | `[32]` | 归一化因子（沿波长维度求和） |
| 81 | `normalization` | `[32]` | log10压缩后 |
| 84 | `normalization` | `[32]` | 量化后（仍为标量） |
| 87 | `n` | `[32, 1]` | 归一化系数 |
| 88-89 | `spectrum`, `istd` | `[32, ~7781]` | 归一化+缩放后 |
| 92-93 | `spectrum`, `istd` | `[32, 8704]` | 投影到潜在网格（`num_pixels=8704`） |
| 96 | `x` | `[32, 2, 8704]` | stack后：flux和istd两个通道 |
| 97 | `h` | `[32, 512, 272]` | ConvNext编码器输出（`latent_channels=512`） |
| 98 | `h` | `[32, 512, 272]` | LayerNorm后 |
| 99 | `h` | `[32, 10, 272]` | `quant_conv`后（`embedding_dim=10`） |

---

## `encode()` 方法中间结果维度

| 行号 | 变量 | 维度 | 说明 |
|------|------|------|------|
| 111 | `embedding` | `[32, 10, 272]` | `_encode`返回 |
| 111 | `normalization` | `[32]` | `_encode`返回 |
| 114 | `embedding` | `[32, 272]` | LFQ量化后（离散token索引） |
| 117 | `normalization` | `[32]` | 标量量化后（离散token索引） |
| 120 | `embedding` | `[32, 273]` | 拼接后：1个norm_token + 272个spectrum_tokens |

---

## 维度变化流程图

```
输入光谱 [32, ~7781]
    │
    ▼ latent_grid.to_latent()
潜在网格 [32, 8704]
    │
    ▼ torch.stack([spectrum, istd], dim=1)
双通道 [32, 2, 8704]
    │
    ▼ ConvNextEncoder1d (下采样32x)
编码特征 [32, 512, 272]
    │
    ▼ quant_conv (512→10通道)
嵌入向量 [32, 10, 272]
    │
    ▼ LFQ量化
离散tokens [32, 272]
    │
    ▼ 拼接normalization token
最终输出 [32, 273]
```

---

## ConvNextEncoder1d 下采样计算

编码器通过4个stage逐步下采样：

| Stage | 输入维度 | 输出维度 | 下采样率 |
|-------|----------|----------|----------|
| 1 | `[32, 2, 8704]` | `[32, 96, 4352]` | 2x |
| 2 | `[32, 96, 4352]` | `[32, 192, 2176]` | 2x |
| 3 | `[32, 192, 2176]` | `[32, 384, 1088]` | 2x |
| 4 | `[32, 384, 1088]` | `[32, 512, 272]` | 4x |

**总下采样率**: 2×2×2×4 = **32x**

**Token数量计算**: `8704 / 32 = 272` 个token位置

---

## 关键配置参数

```python
SpectrumCodec(
    encoder_depths=(3, 3, 9, 3),      # 每个stage的block数量
    encoder_dims=(96, 192, 384, 768), # 每个stage的通道数
    latent_channels=512,              # 潜在空间通道数
    embedding_dim=10,                 # 量化前嵌入维度
    num_pixels=8704,                  # 潜在网格像素数
    codebook_size=1024,               # 码本大小
    dim=10,                           # LFQ维度
)
```

---

## 输出格式

最终输出字典：

```python
{
    "tok_spectrum_desi": torch.Tensor  # shape: [batch, 273]
}
```

- **273个token** = 1个归一化token + 272个光谱tokens
- 每个token来自 `codebook_size=1024` 的离散码本
