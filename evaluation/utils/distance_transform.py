# This file contains code from the SW-FastEdit repository:
# https://github.com/Zrrr1997/SW-FastEdit
#
# The original implementation accompanies the following research paper:
#
# M. Hadlich, Z. Marinov, M. Kim, E. Nasca, J. Kleesiek, and R. Stiefelhagen,
# "Sliding Window Fastedit: A Framework for Lesion Annotation in Whole-Body PET Images,"
# 2024 IEEE International Symposium on Biomedical Imaging (ISBI), Athens, Greece, 2024, 
# pp. 1-5, doi: 10.1109/ISBI56570.2024.10635459.
#
# Keywords: Training; Image segmentation; Solid modeling; Annotations; Memory management;
# Whole-body PET; Manuals; Interactive Segmentation; PET; Sliding Window; 
# Lung Cancer; Melanoma; Lymphoma

from __future__ import annotations

import logging
from typing import List, Tuple

import cupy as cp
import numpy as np
import torch

# Details here: https://docs.rapids.ai/api/cucim/nightly/api/#cucim.core.operations.morphology.distance_transform_edt
from cucim.core.operations.morphology import distance_transform_edt as distance_transform_edt_cupy
# from numpy.typing import ArrayLike
# from scipy.ndimage import distance_transform_edt

np.seterr(all="raise")

logger = logging.getLogger("sw_fastedit")

"""
CUDA enabled distance transforms using cupy
"""

def get_random_choice_from_tensor(
    t: torch.Tensor | cp.ndarray,
    *,
    # device: torch.device,
    max_threshold: int = None,
    size=1,
) -> Tuple[List[int], int] | None:
    device = t.device
    cp.random.seed(42)
    
    with cp.cuda.Device(device.index):
        if not isinstance(t, cp.ndarray):
            t_cp = cp.asarray(t)
        else:
            t_cp = t

        if cp.sum(t_cp) <= 0:
            # No valid distance has been found. Dont raise, just empty return
            return None, None

        # Probability transform
        if max_threshold is None:
            # divide by the maximum number of elements in a volume, otherwise we will get overflows..
            max_threshold = int(cp.floor(cp.log(cp.finfo(cp.float32).max))) / (800 * 800 * 800)

        # Clip the distance transform to avoid overflows and negative probabilities
        clipped_distance = t_cp.clip(min=0, max=max_threshold)

        flattened_t_cp = clipped_distance.flatten()

        probability = cp.exp(flattened_t_cp) - 1.0
        idx = cp.where(flattened_t_cp > 0)[0]
        probabilities = probability[idx] / cp.sum(probability[idx])
        assert idx.shape == probabilities.shape
        assert cp.all(cp.greater_equal(probabilities, 0))

        # Choosing an element based on the probabilities
        seed = cp.random.choice(a=idx, size=size, p=probabilities)
        dst = flattened_t_cp[seed.item()]

        # Get the elements index
        g = cp.asarray(cp.unravel_index(seed, t_cp.shape)).transpose().tolist()[0]
        index = g
        # g[0] = dst.item()
    assert len(g) == len(t_cp.shape), f"g has wrong dimensions! {len(g)} != {len(t_cp.shape)}"
    return index, dst.item()
