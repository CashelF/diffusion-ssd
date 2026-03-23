import hashlib
import importlib.util
import json
import os
import sys

import transformers
from transformers import AutoConfig, AutoModelForMaskedLM


def _load_config_json(model_path: str) -> dict:
    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _load_custom_class(model_path: str, auto_map_key: str):
    config_json = _load_config_json(model_path)
    auto_map = config_json.get("auto_map", {})
    target = auto_map.get(auto_map_key)
    if target is None:
        raise ValueError(
            f"config.json at {model_path} does not define auto_map[{auto_map_key!r}]"
        )

    module_name, class_name = target.rsplit(".", 1)
    module_path = os.path.join(model_path, f"{module_name}.py")
    if not os.path.isfile(module_path):
        raise FileNotFoundError(
            f"Custom HF module file not found for {auto_map_key}: {module_path}"
        )

    _ensure_transformers_submodules(
        [
            "cache_utils",
            "modeling_outputs",
            "processing_utils",
            "modeling_attn_mask_utils",
            "utils",
        ]
    )

    digest = hashlib.sha1(f"{model_path}:{module_name}".encode("utf-8")).hexdigest()
    import_name = f"ssd_hf_remote_{digest}_{module_name}"
    if import_name in sys.modules:
        module = sys.modules[import_name]
    else:
        spec = importlib.util.spec_from_file_location(import_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to import custom HF module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[import_name] = module
        spec.loader.exec_module(module)

    return getattr(module, class_name)


def _ensure_transformers_submodules(submodules: list[str]) -> None:
    transformers_root = os.path.dirname(transformers.__file__)

    for submodule in submodules:
        full_name = f"transformers.{submodule}"
        if full_name in sys.modules:
            continue

        base_path = os.path.join(transformers_root, *submodule.split("."))
        if os.path.isdir(base_path):
            module_path = os.path.join(base_path, "__init__.py")
            search_locations = [base_path]
        else:
            module_path = f"{base_path}.py"
            search_locations = None

        if not os.path.exists(module_path):
            continue

        spec = importlib.util.spec_from_file_location(
            full_name,
            module_path,
            submodule_search_locations=search_locations,
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)


def load_config(model_path: str, *, trust_remote_code: bool = False):
    try:
        return AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
    except ImportError:
        if not trust_remote_code or not os.path.isdir(model_path):
            raise
        config_class = _load_custom_class(model_path, "AutoConfig")
        return config_class.from_pretrained(model_path)


def load_masked_lm(model_path: str, *, torch_dtype=None):
    try:
        return AutoModelForMaskedLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    except ImportError:
        if not os.path.isdir(model_path):
            raise
        config = load_config(model_path, trust_remote_code=True)
        model_class = _load_custom_class(model_path, "AutoModelForMaskedLM")
        return model_class.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch_dtype,
        )
