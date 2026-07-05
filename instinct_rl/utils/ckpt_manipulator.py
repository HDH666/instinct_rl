"""
# A python module that manipulates torch checkpoint file in a hacky way.
Each function should be used with caution and should be used only when thoughtfully considered.
---
Args:
    source_state_dict: the state_dict loaded using torch.load
    algo_state_dict: the algorithm state_dict summarized from algorithm as an example
---
Returns:
    new_state_dict: the state_dict that has been manipulated or directly saved as a checkpoint file.
"""

from collections import OrderedDict
from typing import Literal

import regex as re
import torch


def strict_him_source_checkpoint(source_state_dict: dict, algo_state_dict: dict):
    """Adapt a source HIM checkpoint only when model weights match exactly.

    This is intended for checkpoints saved by ``him_parkour``'s
    ``HIMOnPolicyRunner``. It allows top-level metadata to pass through, but
    model weights must match the target HIM model by key and tensor shape.
    """
    _require_top_level_key(source_state_dict, "model_state_dict")
    _require_top_level_key(source_state_dict, "iter")
    _require_top_level_key(source_state_dict, "infos")
    _require_top_level_key(algo_state_dict, "model_state_dict")

    if "optimizer_state_dict" in algo_state_dict:
        _require_optimizer_state_dict(source_state_dict, "optimizer_state_dict")
    if "estimator_optimizer_state_dict" in algo_state_dict:
        _require_optimizer_state_dict(source_state_dict, "estimator_optimizer_state_dict")

    source_model_state_dict = source_state_dict["model_state_dict"]
    target_model_state_dict = algo_state_dict["model_state_dict"]
    _validate_matching_model_state_dict(source_model_state_dict, target_model_state_dict)

    adapted_state_dict = OrderedDict()
    for key, value in source_state_dict.items():
        if key == "model_state_dict":
            adapted_state_dict[key] = OrderedDict(
                (model_key, source_model_state_dict[model_key]) for model_key in target_model_state_dict.keys()
            )
        else:
            adapted_state_dict[key] = value
    return adapted_state_dict


def _require_top_level_key(state_dict: dict, key: str):
    if key not in state_dict:
        raise ValueError(f"Source HIM checkpoint is missing required top-level key: {key}")


def _require_optimizer_state_dict(state_dict: dict, key: str):
    _require_top_level_key(state_dict, key)
    optimizer_state_dict = state_dict[key]
    if not isinstance(optimizer_state_dict, dict):
        raise ValueError(f"Source HIM checkpoint {key} must be a dict.")
    missing_keys = {"state", "param_groups"} - set(optimizer_state_dict.keys())
    if missing_keys:
        raise ValueError(f"Source HIM checkpoint {key} is missing optimizer metadata keys: {sorted(missing_keys)}")


def _validate_matching_model_state_dict(source_model_state_dict: dict, target_model_state_dict: dict):
    source_keys = set(source_model_state_dict.keys())
    target_keys = set(target_model_state_dict.keys())
    missing_keys = sorted(target_keys - source_keys)
    extra_keys = sorted(source_keys - target_keys)
    if missing_keys or extra_keys:
        message_parts = ["Source HIM checkpoint model_state_dict does not match target HIM model."]
        if missing_keys:
            message_parts.append(f"missing model weights: {missing_keys}")
        if extra_keys:
            message_parts.append(f"extra model weights: {extra_keys}")
        raise ValueError(" ".join(message_parts))

    shape_mismatches = []
    for key in target_model_state_dict.keys():
        source_tensor = source_model_state_dict[key]
        target_tensor = target_model_state_dict[key]
        if not isinstance(source_tensor, torch.Tensor) or not isinstance(target_tensor, torch.Tensor):
            raise ValueError(f"Source HIM checkpoint model weight {key} must be a torch.Tensor.")
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            shape_mismatches.append(f"{key}: source{tuple(source_tensor.shape)} != target{tuple(target_tensor.shape)}")
    if shape_mismatches:
        raise ValueError("Source HIM checkpoint model_state_dict has shape mismatches: " + "; ".join(shape_mismatches))


def replace_encoder0(source_state_dict, algo_state_dict):
    print("\033[1;36m Replacing encoder.0 weights with untrained weights and avoid critic_encoder.0 \033[0m")
    new_model_state_dict = OrderedDict()
    for key in algo_state_dict["model_state_dict"].keys():
        if "critic_encoders.0" in key:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
        elif "encoders.0" in key:
            print(
                "key:", key, "shape:", algo_state_dict["model_state_dict"][key].shape, "using untrained module weights."
            )
            new_model_state_dict[key] = algo_state_dict["model_state_dict"][key]
        else:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
    new_state_dict = dict(
        model_state_dict=new_model_state_dict,
        # No optimizer_state_dict
        iter=source_state_dict["iter"],
        infos=source_state_dict["infos"],
    )
    return new_state_dict


def append_GRU_weights(source_state_dict, algo_state_dict):
    print("\033[1;36m Appending GRU weights to fit the new model \033[0m")
    print("\033[1;36m Operating on both actor and critic \033[0m")
    new_model_state_dict = OrderedDict()
    for key in algo_state_dict["model_state_dict"].keys():
        if ("memory_a" in key or "memory_c" in key) and "rnn" in key and "weight_ih" in key:
            print(
                "key:",
                key,
                "shape:",
                source_state_dict["model_state_dict"][key].shape,
                "is updated to shape:",
                algo_state_dict["model_state_dict"][key].shape,
            )
            new_model_state_dict[key] = algo_state_dict["model_state_dict"][key]
            new_model_state_dict[key][:, : source_state_dict["model_state_dict"][key].shape[1]] = source_state_dict[
                "model_state_dict"
            ][key]
            new_model_state_dict[key][:, source_state_dict["model_state_dict"][key].shape[1] :] /= 10
        else:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
    new_state_dict = dict(
        model_state_dict=new_model_state_dict,
        # No optimizer_state_dict
        iter=source_state_dict["iter"],
        infos=source_state_dict["infos"],
    )
    return new_state_dict


def append_GRU_weights_newStd(source_state_dict, algo_state_dict):
    return_ = append_GRU_weights(source_state_dict, algo_state_dict)
    print(
        "\033[1;36m Setting the std of the new actor to {} \033[0m".format(
            algo_state_dict["model_state_dict"]["std"].mean().cpu().item()
        )
    )
    return_["model_state_dict"]["std"][:] = algo_state_dict["model_state_dict"]["std"][:]
    return return_


def reinitialize_actor_critic_backbone(source_state_dict, algo_state_dict):
    print("\033[1;36m Reinitializing the actor and critic backbone \033[0m")
    new_model_state_dict = OrderedDict()
    for key in algo_state_dict["model_state_dict"].keys():
        if (
            "actor." in key
            or "critic." in key
            or "critics." in key
            or "memory_a" in key
            or "memory_c" in key
            or "std" in key
        ):
            if key not in source_state_dict["model_state_dict"]:
                print(
                    "key:",
                    key,
                    "shape:",
                    algo_state_dict["model_state_dict"][key].shape,
                    "using untrained module weights.",
                )
            else:
                print(
                    "key:",
                    key,
                    "shape:",
                    source_state_dict["model_state_dict"][key].shape,
                    "is updated to shape:",
                    algo_state_dict["model_state_dict"][key].shape,
                )
            new_model_state_dict[key] = algo_state_dict["model_state_dict"][key]
        else:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
    new_state_dict = dict(
        model_state_dict=new_model_state_dict,
        # No optimizer_state_dict
        iter=source_state_dict["iter"],
        infos=source_state_dict["infos"],
    )
    return new_state_dict


def ignore_missing_key(source_state_dict, algo_state_dict):
    """Ignore the missing critic mlp weights and use the initialized ones."""
    print("\033[1;36m Ignoring missing key and using the initialized weights \033[0m")
    new_model_state_dict = OrderedDict()
    missing_keys = []
    for key in algo_state_dict["model_state_dict"].keys():
        if key in source_state_dict["model_state_dict"]:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
        else:
            new_model_state_dict[key] = algo_state_dict["model_state_dict"][key]
            missing_keys.append(key)
    new_state_dict = dict(
        model_state_dict=new_model_state_dict,
        # No optimizer_state_dict
        iter=source_state_dict["iter"],
        infos=source_state_dict["infos"],
    )
    print("\033[1;36m Missing keys: \033[0m", missing_keys)
    return new_state_dict


def fit_smaller_weight(
    source_state_dict: dict,
    algo_state_dict: dict,
    weight_name_regex: str = ".*",
    weight_match_mode: Literal["start", "end"] = "start",
):
    """To fix the weight matrix in algo_state_dict which is smaller than the one in source_state_dict,
    we will copy the part of the weight matrix from source_state_dict to algo_state_dict.
    ## Args:
        weight_name_regex: str
            The regex to match the weight name in algo_state_dict.
        weight_match_mode: Literal["start", "end"]
            If "start", weight_algo = weight_source[:weight_algo.shape[0], :weight_algo.shape[1]]
            If "end", weight_algo = weight_source[-weight_algo.shape[0]:, -weight_algo.shape[1]:]
    """
    print("\033[1;36m Fitting smaller weight matrix, matching \033[0m")
    new_model_state_dict = OrderedDict()
    for key in algo_state_dict["model_state_dict"].keys():
        if re.match(weight_name_regex, key):
            weight_algo = algo_state_dict["model_state_dict"][key]
            weight_source = source_state_dict["model_state_dict"][key]
            if weight_match_mode == "start":
                new_model_state_dict[key] = weight_source[: weight_algo.shape[0], : weight_algo.shape[1]]
            elif weight_match_mode == "end":
                new_model_state_dict[key] = weight_source[-weight_algo.shape[0] :, -weight_algo.shape[1] :]
            else:
                raise ValueError(f"Invalid weight_match_mode: {weight_match_mode}. Must be one of ['start', 'end'].")
        else:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
    new_state_dict = dict(
        model_state_dict=new_model_state_dict,
    )
    for k in source_state_dict.keys():
        if k not in new_state_dict and not k.startswith("optimizer_state_dict"):
            new_state_dict[k] = source_state_dict[k]
    return new_state_dict


def newStd(
    source_state_dict: dict,
    algo_state_dict: dict,
):
    """Replicate everything except for policy std"""
    print(
        "\033[1;36m Setting the std of the new actor to {} \033[0m".format(
            algo_state_dict["model_state_dict"]["std"].mean().cpu().item()
        )
    )
    new_state_dict = OrderedDict()
    for state_dict_key in source_state_dict.keys():
        if state_dict_key == "model_state_dict":
            new_state_dict[state_dict_key] = OrderedDict()
            for model_state_dict_key in source_state_dict[state_dict_key].keys():
                if "std" == model_state_dict_key:
                    new_state_dict[state_dict_key][model_state_dict_key] = algo_state_dict["model_state_dict"][
                        model_state_dict_key
                    ]
                else:
                    new_state_dict[state_dict_key][model_state_dict_key] = source_state_dict[state_dict_key][
                        model_state_dict_key
                    ]
        else:
            new_state_dict[state_dict_key] = source_state_dict[state_dict_key]
    return new_state_dict
