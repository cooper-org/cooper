from collections.abc import Iterable
import logging

logger = logging.getLogger(__name__)

import torch


def compare_values(val1, val2):

    if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
        # convert both to the same CUDA device
        if str(val1.device) != "cuda:0":
            val1 = val1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(val2.device) != "cuda:0":
            val2 = val2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        return torch.allclose(val1, val2)

    if isinstance(val1, dict) and isinstance(val2, dict):
        if val1.keys() != val2.keys():
            return False
        else:
            return all([compare_values(val1[k], val2[k]) for k in val1.keys()])

    if isinstance(val1, Iterable) and isinstance(val2, Iterable):
        if len(val1) != len(val2):
            return False
        return all([compare_values(ii, jj) for ii, jj in zip(val1, val2)])

    else:
        return val1 == val2


def validate_state_dicts(model_state_dict_1, model_state_dict_2):
    # Edited from: https://gist.github.com/rohan-varma/a0a75e9a0fbe9ccc7420b04bff4a7212

    if model_state_dict_1 is None and model_state_dict_2 is None:
        return True

    if len(model_state_dict_1) != len(model_state_dict_2):
        logger.info(
            f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
        )
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_2.items()
        }

    for ((k_1, val1), (k_2, val2)) in zip(
        model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            logger.info(f"Key mismatch: {k_1} vs {k_2}")
            return False

        if not compare_values(val1, val2):
            logger.info(f"Attribute mismatch: {val1} vs {val2}")
            return False

    return True
