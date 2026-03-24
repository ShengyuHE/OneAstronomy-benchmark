from contextlib import contextmanager
from threading import local
from typing import Optional

from huggingface_hub import hub_mixin

from aion.codecs.base import Codec
from aion.modalities import Modality


ORIGINAL_CONFIG_NAME = hub_mixin.constants.CONFIG_NAME
ORIGINAL_PYTORCH_WEIGHTS_NAME = hub_mixin.constants.PYTORCH_WEIGHTS_NAME
ORIGINAL_SAFETENSORS_SINGLE_FILE = hub_mixin.constants.SAFETENSORS_SINGLE_FILE

# Thread-local storage for codec context
_thread_local = local()


@contextmanager
def _codec_path_context(modality: type[Modality]):
    """Thread-safe context manager for temporarily overriding HuggingFace constants.

    Args:
        modality: The modality type to create paths for

    Yields:
        None
    """
    # Store original values
    original_config = hub_mixin.constants.CONFIG_NAME
    original_weights = hub_mixin.constants.PYTORCH_WEIGHTS_NAME
    original_safetensors = hub_mixin.constants.SAFETENSORS_SINGLE_FILE

    try:
        # Set codec-specific paths
        hub_mixin.constants.CONFIG_NAME = (
            f"codecs/{modality.name}/{ORIGINAL_CONFIG_NAME}"
        )
        hub_mixin.constants.PYTORCH_WEIGHTS_NAME = (
            f"codecs/{modality.name}/{ORIGINAL_PYTORCH_WEIGHTS_NAME}"
        )
        hub_mixin.constants.SAFETENSORS_SINGLE_FILE = (
            f"codecs/{modality.name}/{ORIGINAL_SAFETENSORS_SINGLE_FILE}"
        )
        yield
    finally:
        # Always restore original values
        hub_mixin.constants.CONFIG_NAME = original_config
        hub_mixin.constants.PYTORCH_WEIGHTS_NAME = original_weights
        hub_mixin.constants.SAFETENSORS_SINGLE_FILE = original_safetensors


def _validate_modality(modality: type[Modality]) -> None:
    """Validate that the modality is properly configured.

    Args:
        modality: The modality type to validate

    Raises:
        ValueError: If the modality is invalid
    """
    if not isinstance(modality, type):
        raise ValueError(f"Expected modality to be a type, got {type(modality)}")

    if not issubclass(modality, Modality):
        raise ValueError(f"Modality {modality} must be a subclass of Modality")

    if not hasattr(modality, "name") or not isinstance(modality.name, str):
        raise ValueError(
            f"Modality {modality} must have a 'name' class attribute of type str"
        )

    if not modality.name.strip():
        raise ValueError(f"Modality {modality} name cannot be empty")


class CodecPytorchHubMixin(hub_mixin.PyTorchModelHubMixin):
    """Mixin for PyTorch models that correspond to codecs.
    Codec don't have their own model repo.
    Instead they lie in the transformer model repo as subfolders.
    """

    @staticmethod
    def _validate_codec_modality(codec: type[Codec], modality: type[Modality]):
        """Validate that a codec class is compatible with a modality.

        Args:
            codec: The codec class to validate
            modality: The modality type to validate against

        Raises:
            TypeError: If the codec is not a valid codec class or is incompatible with the modality
            ValueError: If the modality has no corresponding codec configuration
        """
        # Import MODALITY_CODEC_MAPPING here to avoid circular import
        from aion.codecs.config import MODALITY_CODEC_MAPPING

        if not issubclass(codec, Codec):
            raise TypeError("Only codecs can be loaded using this method.")
        if modality not in MODALITY_CODEC_MAPPING:
            raise ValueError(f"Modality {modality} has no corresponding codec.")
        elif MODALITY_CODEC_MAPPING[modality] != codec:
            raise TypeError(
                f"Modality {modality} is associated with {MODALITY_CODEC_MAPPING[modality]} codec but {codec} requested."
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        modality: type[Modality],
        *model_args,
        **kwargs,
    ):
        """Load a codec model from a pretrained model repository.

        Args:
            pretrained_model_name_or_path (str): The name or path of the pretrained
                model repository.
            modality (type[Modality]): The modality type for this codec.
            *model_args: Additional positional arguments to pass to the model
                constructor.
            **kwargs: Additional keyword arguments to pass to the model
                constructor.

        Returns:
            The loaded codec model.

        Raises:
            ValueError: If the class is not a codec subclass or modality is invalid.
        """
        # Validate codec-modality compatibility
        cls._validate_codec_modality(cls, modality)

        # Validate modality
        _validate_modality(modality)

        # Use thread-safe context manager to override paths
        with _codec_path_context(modality):
            model = super().from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )

        # Store modality reference on the model instance for later use
        model._modality = modality
        return model

    def save_pretrained(
        self, save_directory, modality: Optional[type[Modality]] = None, *args, **kwargs
    ):
        """Save the codec model to a pretrained model repository.

        Args:
            save_directory (str): The directory to save the model to.
            modality (Optional[type[Modality]]): The modality type for this codec.
                If not provided, will use the modality stored during from_pretrained.
            *args: Additional positional arguments to pass to the save method.
            **kwargs: Additional keyword arguments to pass to the save method.

        Raises:
            ValueError: If the instance is not a codec or modality cannot be determined.
        """
        if not issubclass(self.__class__, Codec):
            raise ValueError("Only codec instances can be saved using this method.")

        # Determine modality to use
        if modality is not None:
            _validate_modality(modality)
            target_modality = modality
        elif hasattr(self, "_modality"):
            target_modality = self._modality
        else:
            raise ValueError(
                "No modality specified. Either provide modality parameter or "
                "load the codec using from_pretrained() which stores the modality."
            )

        # Construct the path to the codec subfolder
        codec_path = f"{save_directory}/codecs/{target_modality.name}"
        super().save_pretrained(codec_path, *args, **kwargs)
