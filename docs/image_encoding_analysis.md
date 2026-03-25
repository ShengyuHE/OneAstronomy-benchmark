# AION 图像模态编码操作详解

## 概述

本文档详细解释 `encode_val.py` 第95行 `codec_manager.encode(image)` 对图像模态执行的具体操作。

```python
im_encoded = codec_manager.encode(image)  # 第95行
```

---

## 整体调用链

```
CodecManager.encode() → ImageCodec.encode() → ImageCodec._encode() → Quantizer.encode()
```

---

## 详细步骤

### 1. CodecManager.encode()

**文件位置**: [manager.py:73-98](file:///mnt/oneocean_200TBnas_feitian/oneastronomy/zwg/AION/aion/codecs/manager.py#L73-L98)

```python
def encode(self, *modalities: Modality) -> dict[str, torch.Tensor]:
    # 1. 根据 modality 类型查找对应的 codec
    # LegacySurveyImage → ImageCodec (来自 MODALITY_CODEC_MAPPING)
    codec = self._load_codec(type(modality))  # 从 HuggingFace 加载预训练权重
    
    # 2. 调用 codec 的 encode 方法
    tokenized = codec.encode(modality)
    
    # 3. 返回 {token_key: tokens} 字典
    # 对于 LegacySurveyImage，token_key = "tok_image"
```

### 2. ImageCodec.encode()

**文件位置**: [base.py:57-77](file:///mnt/oneocean_200TBnas_feitian/oneastronomy/zwg/AION/aion/codecs/base.py#L57-L77)

```python
def encode(self, x: ModalityType) -> Float[Tensor, "b n_tokens"]:
    # 1. 调用 _encode 获取连续潜在表示
    embedding = self._encode(x)
    
    # 2. 量化为离散 token
    return self.quantizer.encode(embedding)
```

### 3. ImageCodec._encode()

**文件位置**: [image.py:97-118](file:///mnt/oneocean_200TBnas_feitian/oneastronomy/zwg/AION/aion/codecs/image.py#L97-L118)

这是核心编码逻辑，对图像执行以下操作：

| 步骤 | 操作 | 说明 |
|------|------|------|
| **预处理** | `center_crop(flux_tensor)` | 中心裁剪至 **96×96** 像素 |
| | `clamp(processed_flux, bands)` | 像素值裁剪（去除异常值） |
| | `rescaler.forward()` | 根据调查类型（DES/HSC）重缩放 |
| | `_range_compress()` | **arcsinh 范围压缩**：`arcsinh(x/0.01) * 0.01 * 10` |
| **通道处理** | `image_padder.forward()` | 填充至统一通道数（9波段） |
| | `subsample_in()` | 多调查投影（54维） |
| **编码器** | `encoder(processed_flux)` | **MagVitAE 编码器**（卷积下采样） |
| **投影** | `pre_quant_proj(h)` | 投影至量化维度（5维） |
| **展平** | `h.reshape(b, c, -1)` | 展平空间维度 → `(batch, 5, 24×24)` |

### 4. FiniteScalarQuantizer.encode()

**文件位置**: [quantizers/__init__.py:131-147](file:///mnt/oneocean_200TBnas_feitian/oneastronomy/zwg/AION/aion/codecs/quantizers/__init__.py#L131-L147)

将连续潜在表示量化为离散整数 token：

```python
# 将 5 维连续向量映射为离散整数索引
# 使用有限标量量化（FSQ）
return (zhat * self._basis).sum(axis=-1).to(torch.int32)
```

---

## 数据流总结

```
输入: LegacySurveyImage
      flux: (batch, 4, 96, 96)  # DES-G/R/I/Z 四波段
      bands: ["DES-G", "DES-R", "DES-I", "DES-Z"]
          ↓
预处理: 裁剪 → 裁剪 → 重缩放 → arcsinh压缩 → 通道填充 → 线性投影
          ↓
编码器: MagVitAE (卷积下采样 × 2)
          ↓
潜在表示: (batch, 512, 24, 24)
          ↓
投影: pre_quant_proj (512 → 5)
          ↓
量化前: (batch, 5, 24, 24) → (batch, 5, 576)
          ↓
量化: FiniteScalarQuantizer (FSQ)
          ↓
输出: tokens (batch, 576)  # 24×24 = 576 个离散 token
```

---

## 关键配置参数

**文件位置**: [modality_info.py:254-272](file:///mnt/oneocean_200TBnas_feitian/oneastronomy/zwg/AION/aion/fourm/modality_info.py#L254-L272)

- **input_size**: 96 (图像尺寸)
- **patch_size**: 4 (patch 大小)
- **vocab_size**: 4375 (词表大小)
- **token 数量**: (96/4)² = 576 个 token

---

## 维度变化详解

### MagVitAE 编码器输出

根据 [magvit.py:122-208](file:///mnt/oneocean_200TBnas_feitian/oneastronomy/zwg/AION/aion/codecs/modules/magvit.py#L122-L208) 和 [image.py:192-198](file:///mnt/oneocean_200TBnas_feitian/oneastronomy/zwg/AION/aion/codecs/image.py#L192-L198) 的配置：

| 阶段 | 操作 | 维度变化 |
|------|------|----------|
| 输入 | - | `(batch, 54, 96, 96)` |
| conv_in | 54 → 128 | `(batch, 128, 96, 96)` |
| 下采样 1 | 128 → 256, 空间 ÷2 | `(batch, 256, 48, 48)` |
| 下采样 2 | 256 → 512, 空间 ÷2 | `(batch, 512, 24, 24)` |
| **encoder 输出** | - | **`(batch, 512, 24, 24)`** |

### 关键维度变化位置

**通道维度 512 → 5** 发生在 [image.py:114](file:///mnt/oneocean_200TBnas_feitian/oneastronomy/zwg/AION/aion/codecs/image.py#L114)：

```python
h = self.pre_quant_proj(h)  # 第114行
```

这里 `pre_quant_proj` 是一个 1×1 卷积：

```python
self.pre_quant_proj = torch.nn.Conv2d(
    hidden_dims, embedding_dim, kernel_size=1, stride=1, padding=0
)  # 512 → 5
```

### 完整数据流

```
encoder 输出:     (batch, 512, 24, 24)
                        ↓
pre_quant_proj:   (batch, 5, 24, 24)    ← 第114行
                        ↓
reshape:          (batch, 5, 576)       ← 第117行
```

### 注意事项

- **空间维度**：始终是 **24×24**（96÷2÷2 = 24）
- **通道维度**：从 **512 → 5**，发生在第 114 行的 `pre_quant_proj`

如果看到 12×12 的空间维度，可能是因为使用了不同的 `n_compressions` 参数（比如 3 次下采样：96→48→24→12）。

---

## 相关文件索引

| 文件 | 说明 |
|------|------|
| [encode_val.py](file:///mnt/oneocean_200TBnas_feitian/oneastronomy/zwg/AION/tools/encode_val.py) | 验证脚本入口 |
| [manager.py](file:///mnt/oneocean_200TBnas_feitian/oneastronomy/zwg/AION/aion/codecs/manager.py) | CodecManager 实现 |
| [image.py](file:///mnt/oneocean_200TBnas_feitian/oneastronomy/zwg/AION/aion/codecs/image.py) | ImageCodec 实现 |
| [base.py](file:///mnt/oneocean_200TBnas_feitian/oneastronomy/zwg/AION/aion/codecs/base.py) | Codec 基类 |
| [magvit.py](file:///mnt/oneocean_200TBnas_feitian/oneastronomy/zwg/AION/aion/codecs/modules/magvit.py) | MagVitAE 编码器 |
| [config.py](file:///mnt/oneocean_200TBnas_feitian/oneastronomy/zwg/AION/aion/codecs/config.py) | 模态-编码器映射配置 |
