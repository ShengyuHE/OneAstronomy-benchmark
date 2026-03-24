```{raw} html
<div class="hero-section">
  <div class="hero-background"></div>
  <h1 class="hero-title">AION-1</h1>
  <p class="hero-subtitle">AstronomIcal Omnimodal Network</p>
  <p class="hero-description">Large-Scale Multimodal Foundation Model for Astronomy</p>
  <div class="hero-buttons">
    <a href="#quick-start" class="btn-primary">Get Started →</a>
    <!-- <a href="https://arxiv.org/abs/2406.00000" class="btn-secondary">Read the Paper</a> -->
    <a href="https://colab.research.google.com/github/polymathic-ai/AION/blob/main/notebooks/Tutorial.ipynb" class="btn-secondary">Run on Colab</a>
  </div>
</div>
```

# AION-1 Documentation

## 🌟 Why AION-1?

Trained on over 200 million astronomical objects, AION-1 (AstronomIcal Omnimodal Network) is the first Foundation Model capable of unifying multiband imaging, spectroscopy, and photometry from major ground- and space-based observatories into a single framework.

Compared to traditional machine learning approaches in Astronomy, AION-1 stands out on several points:
- **Enabling Flexible Data Fusion**: Scientists can use any combination of available observations without redesigning their analysis pipeline
- **Enabling Easy Adaptation to Downstream Tasks**: Scientists can adapt AION-1 to new tasks in a matter of minutes and reach SOTA performance
- **Excelling in Low-Data Regimes**: AION-1 achieves competitive results with orders of magnitude less labeled data than supervised approaches
- **Providing Universal Representations**: The learned embeddings capture physically meaningful structure useful across diverse downstream tasks

## 🚀 Quick Start

Assuming you have PyTorch installed, you can install AION trivially with:
```bash
pip install polymathic-aion
```

Then you can load the pretrained model and start analyzing astronomical data:
```python
import torch
from aion import AION
from aion.codecs import CodecManager
from aion.modalities import LegacySurveyImage

# Load model and codec manager
model = AION.from_pretrained('aion-base').to('cuda')  # or 'aion-large', 'aion-xlarge'
codec_manager = CodecManager(device='cuda')

# Prepare your astronomical data (example: Legacy Survey image)
image = LegacySurveyImage(
    flux=your_image_tensor,  # Shape: [batch, 4, height, width] for g,r,i,z bands
    bands=['DES-G', 'DES-R', 'DES-I', 'DES-Z']
)

# Encode data to tokens
tokens = codec_manager.encode(image)

# Option 1: Extract embeddings for downstream tasks
embeddings = model.encode(tokens, num_encoder_tokens=600)

# Option 2: Generate predictions (e.g., redshift)
from aion.modalities import Z
preds = model(
    codec_manager.encode(image),
    target_modality=Z,
)
```

## 📚 Documentation

```{eval-rst}
.. grid:: 1 1 1 2
   :gutter: 3

   .. grid-item-card:: API Reference
      :link: api.html
      :class-card: doc-card

      Complete API documentation with all classes and methods
```

```{toctree}
:hidden:
:maxdepth: 2

api
```
