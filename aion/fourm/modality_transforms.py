# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np
import torch


class AbstractTransform(ABC):
    @abstractmethod
    def load(self, sample):
        pass

    @abstractmethod
    def preprocess(self, sample):
        pass

    @abstractmethod
    def image_augment(
        self,
        v,
        crop_coords: Tuple,
        flip: bool,
        orig_size: Tuple,
        target_size: Tuple,
        rand_aug_idx: Optional[int],
        resample_mode: str = None,
    ):
        pass

    @abstractmethod
    def postprocess(self, v):
        pass


class TokTransform(AbstractTransform):
    def __init__(self):
        pass

    def load(self, path):
        sample = np.load(path).astype(int)
        return sample

    def preprocess(self, sample):
        return sample

    def image_augment(
        self,
        v,
        crop_coords: Tuple,
        flip: bool,
        orig_size: Tuple,
        target_size: Tuple,
        rand_aug_idx: Optional[int],
        resample_mode: str = None,
    ):
        if rand_aug_idx is None:
            raise ValueError(
                "Crop settings / augmentation index are missing but a pre-tokenized modality is being used"
            )
        v = torch.tensor(v[rand_aug_idx])
        return v

    def postprocess(self, sample):
        return sample
