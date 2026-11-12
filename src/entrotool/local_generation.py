from __future__ import annotations

from threading import Lock
from typing import Any, Mapping


_INSTANCES: dict[tuple[Any, ...], "LocalTransformersLLM"] = {}
_INSTANCES_LOCK = Lock()


class LocalTransformersLLM:
    """Thread-safe local text generation adapter for Transformers models."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        self.spec = dict(spec)
        self._load_lock = Lock()
        self._generate_lock = Lock()
        self._processor: Any = None
        self._model: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._processor is None or self._model is None:
            with self._load_lock:
                if self._processor is None or self._model is None:
                    import torch
                    from transformers import AutoModelForMultimodalLM, AutoProcessor

                    model_path = str(self.spec["model_path"])
                    self._processor = AutoProcessor.from_pretrained(
                        model_path,
                        local_files_only=True,
                    )
                    dtype_name = str(self.spec.get("torch_dtype", "bfloat16"))
                    dtype = getattr(torch, dtype_name, torch.bfloat16)
                    self._model = AutoModelForMultimodalLM.from_pretrained(
                        model_path,
                        dtype=dtype,
                        device_map=self.spec.get("device_map", "auto"),
                        local_files_only=True,
                    )
        return self._processor, self._model

    def complete(self, prompt: str, spec: Mapping[str, Any] | None = None) -> str:
        processor, model = self._load()
        generation = {**self.spec, **dict(spec or {})}
        messages = [{"role": "user", "content": prompt}]
        with self._generate_lock:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=bool(generation.get("enable_thinking", False)),
            )
            device = getattr(model, "device", None)
            if device is not None:
                inputs = inputs.to(device)
            input_length = inputs["input_ids"].shape[-1]
            output = model.generate(
                **inputs,
                max_new_tokens=int(
                    generation.get(
                        "max_new_tokens", generation.get("max_tokens", 2048)
                    )
                ),
                do_sample=bool(generation.get("do_sample", False)),
                temperature=float(generation.get("temperature", 0.0))
                if generation.get("do_sample", False)
                else None,
            )
            text = processor.decode(
                output[0][input_length:], skip_special_tokens=True
            )
        return str(text).strip()

    def close(self) -> None:
        pass


def create_local_llm(spec: Mapping[str, Any]) -> LocalTransformersLLM:
    key = (
        str(spec["model_path"]),
        str(spec.get("device_map", "auto")),
        str(spec.get("torch_dtype", "bfloat16")),
    )
    with _INSTANCES_LOCK:
        if key not in _INSTANCES:
            _INSTANCES[key] = LocalTransformersLLM(spec)
        return _INSTANCES[key]
