API Reference
=============

This page provides comprehensive API documentation for all AION components, automatically generated from the source code.

.. currentmodule:: aion

Main Model
----------

.. automodule:: aion.model
   :members:
   :undoc-members:
   :show-inheritance:

Modalities
----------

The modality system defines data structures for all 39 astronomical data types supported by AION.

Base Classes
~~~~~~~~~~~~

.. automodule:: aion.modalities
   :members: Modality, Image, Spectrum, Scalar
   :undoc-members:
   :show-inheritance:

Image Modalities
~~~~~~~~~~~~~~~~

.. automodule:: aion.modalities
   :members: LegacySurveyImage, HSCImage
   :undoc-members:
   :show-inheritance:

Spectrum Modalities
~~~~~~~~~~~~~~~~~~~

.. automodule:: aion.modalities
   :members: DESISpectrum, SDSSSpectrum
   :undoc-members:
   :show-inheritance:

Catalog Modalities
~~~~~~~~~~~~~~~~~~

.. automodule:: aion.modalities
   :members: LegacySurveyCatalog, LegacySurveySegmentationMap
   :undoc-members:
   :show-inheritance:

Scalar Modalities
~~~~~~~~~~~~~~~~~

Legacy Survey Scalars
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: aion.modalities
   :members: LegacySurveyFluxG, LegacySurveyFluxR, LegacySurveyFluxI, LegacySurveyFluxZ, LegacySurveyFluxW1, LegacySurveyFluxW2, LegacySurveyFluxW3, LegacySurveyFluxW4, LegacySurveyShapeR, LegacySurveyShapeE1, LegacySurveyShapeE2, LegacySurveyEBV
   :undoc-members:
   :show-inheritance:

HSC Scalars
~~~~~~~~~~~

.. automodule:: aion.modalities
   :members: HSCAG, HSCAR, HSCAI, HSCAZ, HSCAY, HSCMagG, HSCMagR, HSCMagI, HSCMagZ, HSCMagY, HSCShape11, HSCShape22, HSCShape12
   :undoc-members:
   :show-inheritance:

Gaia Scalars
~~~~~~~~~~~~

.. automodule:: aion.modalities
   :members: GaiaFluxG, GaiaFluxBp, GaiaFluxRp, GaiaParallax, GaiaXpBp, GaiaXpRp
   :undoc-members:
   :show-inheritance:

Coordinate Scalars
~~~~~~~~~~~~~~~~~~

.. automodule:: aion.modalities
   :members: Ra, Dec, Z
   :undoc-members:
   :show-inheritance:

Utility Types
~~~~~~~~~~~~~

.. automodule:: aion.modalities
   :members: ScalarModalities, ModalityType
   :undoc-members:
   :show-inheritance:

Codec System
------------

The codec system handles tokenization of different modality types.

Core Codec Classes
~~~~~~~~~~~~~~~~~~

.. automodule:: aion.codecs.manager
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.codecs.base
   :members:
   :undoc-members:
   :show-inheritance:

Codec Implementations
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: aion.codecs.image
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.codecs.spectrum
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.codecs.catalog
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.codecs.scalar_field
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.codecs.scalar
   :members:
   :undoc-members:
   :show-inheritance:

Quantizers
~~~~~~~~~~

.. automodule:: aion.codecs.quantizers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.codecs.quantizers.scalar
   :members:
   :undoc-members:
   :show-inheritance:

4M Transformer
--------------

Core transformer architecture and components.

Main Transformer
~~~~~~~~~~~~~~~~

.. automodule:: aion.fourm.fm
   :members:
   :undoc-members:
   :show-inheritance:

Embedding Layers
~~~~~~~~~~~~~~~~

.. automodule:: aion.fourm.encoder_embeddings
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.fourm.decoder_embeddings
   :members:
   :undoc-members:
   :show-inheritance:

Transformer Components
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: aion.fourm.fm_utils
   :members:
   :undoc-members:
   :show-inheritance:

Generation
~~~~~~~~~~

.. automodule:: aion.fourm.generate
   :members:
   :undoc-members:
   :show-inheritance:

LoRA Support
~~~~~~~~~~~~

.. automodule:: aion.fourm.lora_utils
   :members:
   :undoc-members:
   :show-inheritance:

Modality Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: aion.fourm.modality_info
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.fourm.modality_transforms
   :members:
   :undoc-members:
   :show-inheritance:

Codec Modules
-------------

Specialized neural network modules used in codecs.

Architecture Components
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: aion.codecs.modules.magvit
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.codecs.modules.convnext
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.codecs.modules.convblocks
   :members:
   :undoc-members:
   :show-inheritance:

Specialized Modules
~~~~~~~~~~~~~~~~~~~

.. automodule:: aion.codecs.modules.spectrum
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.codecs.modules.ema
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.codecs.modules.subsampler
   :members:
   :undoc-members:
   :show-inheritance:

Configuration and Utilities
----------------------------

.. automodule:: aion.codecs.config
   :members:
   :undoc-members:
   :show-inheritance:
