from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class OnnxInference:
    def __init__(self, model_path: str | Path, backend: str = "opencv"):
        self.backend = backend
        if backend == "opencv":
            self._net = cv2.dnn.readNetFromONNX(str(model_path))
            self._session = None
            self._input_name = None
            return
        if backend != "cuda":
            raise ValueError(f"Unsupported inference backend: {backend}")
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "CUDA inference requires the optional 'onnxruntime-gpu' package."
            ) from error
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "ONNX Runtime CUDAExecutionProvider is unavailable on this machine."
            )
        self._net = None
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CUDAExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def forward(self, tensor: np.ndarray) -> np.ndarray:
        if self._net is not None:
            self._net.setInput(tensor)
            return np.asarray(self._net.forward())
        return np.asarray(
            self._session.run(None, {self._input_name: tensor})[0]
        )
