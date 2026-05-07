"""Model serialization, loading, and ONNX export utilities.

Enables:
- Saving/loading PyTorch models (checkpoint + state dict)
- ONNX export for cross-platform inference
- Model versioning and metadata
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class ModelRegistry:
    """Manage model versions and metadata."""

    def __init__(self, registry_path: Path = Path("model_weights")):
        """Initialize registry.

        Args:
            registry_path: Root directory for model storage.
        """
        self.registry_path = registry_path
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / "manifest.json"

        self.manifest: Dict[str, Any] = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        """Load manifest from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {}

    def _save_manifest(self) -> None:
        """Save manifest to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def register(
        self,
        model_name: str,
        model: nn.Module,
        version: str,
        metadata: Dict[str, Any],
    ) -> Path:
        """Register and save a model.

        Args:
            model_name: Model identifier (e.g., "lstm_rul").
            model: PyTorch model.
            version: Version string (e.g., "1.0.0").
            metadata: Dict with config, metrics, etc.

        Returns:
            Path to saved model.
        """
        model_dir = self.registry_path / f"{model_name}_v{version}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save state dict
        model_path = model_dir / "model.pt"
        torch.save(model.state_dict(), model_path)

        # Save metadata
        meta_path = model_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Update manifest
        self.manifest[f"{model_name}:{version}"] = {
            "path": str(model_path),
            "metadata": meta_path.name,
            "created": str(Path.cwd()),
        }
        self._save_manifest()

        return model_path

    def load(
        self,
        model_name: str,
        version: str,
        model_class: type,
        device: str = "cpu",
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Load a model and its metadata.

        Args:
            model_name: Model identifier.
            version: Version string.
            model_class: Model class to instantiate.
            device: "cpu" or "cuda".

        Returns:
            (model, metadata)
        """
        model_dir = self.registry_path / f"{model_name}_v{version}"

        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_dir}")

        # Load metadata
        meta_path = model_dir / "metadata.json"
        with open(meta_path) as f:
            metadata = json.load(f)

        # Instantiate model
        config = metadata.get("config", {})
        model = model_class(**config)

        # Load weights
        model_path = model_dir / "model.pt"
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        return model, metadata


def export_to_onnx(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    output_path: Path,
    model_name: str = "model",
    opset_version: int = 12,
    dynamic_axes: Optional[Dict[str, Dict]] = None,
) -> None:
    """Export PyTorch model to ONNX format.

    **Benefits of ONNX**:
    - Cross-platform inference (CPU, GPU, mobile, browser)
    - Optimized runtime (ONNX Runtime, TensorRT)
    - Interoperability (PyTorch ↔ TensorFlow ↔ ONNX)

    Args:
        model: PyTorch model (must be in eval mode).
        input_shape: Shape of dummy input (e.g., (1, 30, 14)).
        output_path: Path to save ONNX model.
        model_name: Name for ONNX model.
        opset_version: ONNX opset version (default 12).
        dynamic_axes: Dict specifying dynamic dimensions
                     (e.g., {"input": {0: "batch"}, "output": {0: "batch"}}).

    Example:
        >>> export_to_onnx(
        ...     model=transformer,
        ...     input_shape=(1, 30, 14),
        ...     output_path=Path("transformer_rul.onnx"),
        ...     dynamic_axes={
        ...         "input": {0: "batch"},
        ...         "output": {0: "batch"}
        ...     }
        ... )
    """
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, *input_shape[1:])

    # Export
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes or {"input": {0: "batch"}, "output": {0: "batch"}},
        verbose=False,
    )

    print(f"✅ Model exported to {output_path}")


def load_onnx_model(model_path: Path):
    """Load ONNX model for inference.

    Requires: onnxruntime (`pip install onnxruntime`)

    Args:
        model_path: Path to ONNX model.

    Returns:
        ONNX Runtime inference session.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime not installed. Install via: pip install onnxruntime")

    session = ort.InferenceSession(str(model_path))
    return session


def predict_with_onnx(
    session: Any,
    X: np.ndarray,
) -> np.ndarray:
    """Run inference using ONNX Runtime.

    Args:
        session: ONNX Runtime session.
        X: Input array of shape (batch, seq_len, n_features).

    Returns:
        Predictions of shape (batch,).
    """
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    predictions = session.run(
        [output_name],
        {input_name: X.astype(np.float32)},
    )

    return predictions[0].flatten()


class ONNXPredictor:
    """Wrapper for convenient ONNX inference."""

    def __init__(self, model_path: Path):
        """Initialize predictor.

        Args:
            model_path: Path to ONNX model.
        """
        self.session = load_onnx_model(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions.

        Args:
            X: Input of shape (batch, seq_len, n_features).

        Returns:
            Predictions of shape (batch,).
        """
        return predict_with_onnx(self.session, X)

    def predict_single(self, x: np.ndarray) -> float:
        """Predict for single sample.

        Args:
            x: Single sample of shape (seq_len, n_features).

        Returns:
            Single RUL prediction.
        """
        x_batch = np.expand_dims(x, axis=0)
        pred = self.predict(x_batch)
        return float(pred[0])


def benchmark_onnx_vs_pytorch(
    pytorch_model: nn.Module,
    onnx_path: Path,
    X_test: np.ndarray,
    num_iterations: int = 100,
) -> Dict[str, Any]:
    """Compare inference speed: PyTorch vs ONNX.

    Args:
        pytorch_model: PyTorch model (must be on CPU).
        onnx_path: Path to ONNX model.
        X_test: Test data.
        num_iterations: Number of iterations for averaging.

    Returns:
        Dict with timing and accuracy comparison.
    """
    import time

    pytorch_model.eval()
    onnx_session = load_onnx_model(onnx_path)

    # PyTorch inference
    pytorch_times = []
    pytorch_model = pytorch_model.to("cpu")

    for _ in range(num_iterations):
        X_batch = torch.tensor(X_test[:10]).float()  # First 10 samples
        start = time.time()
        with torch.no_grad():
            pytorch_preds = pytorch_model(X_batch).numpy()
        pytorch_times.append(time.time() - start)

    # ONNX inference
    onnx_times = []
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    for _ in range(num_iterations):
        start = time.time()
        onnx_preds = onnx_session.run(
            [output_name],
            {input_name: X_test[:10].astype(np.float32)},
        )[0]
        onnx_times.append(time.time() - start)

    # Compare predictions
    max_diff = np.max(np.abs(pytorch_preds - onnx_preds.flatten()))

    return {
        "pytorch_mean_ms": np.mean(pytorch_times) * 1000,
        "pytorch_std_ms": np.std(pytorch_times) * 1000,
        "onnx_mean_ms": np.mean(onnx_times) * 1000,
        "onnx_std_ms": np.std(onnx_times) * 1000,
        "speedup": np.mean(pytorch_times) / np.mean(onnx_times),
        "max_prediction_diff": float(max_diff),
    }


# ── Export Functions ──────────────────────────────────────────────────────

def export_models_to_onnx(
    models: Dict[str, nn.Module],
    input_shape: Tuple[int, ...],
    output_dir: Path = Path("model_weights"),
) -> None:
    """Export all models to ONNX.

    Args:
        models: Dict of {model_name: model}.
        input_shape: Input shape (e.g., (1, 30, 14)).
        output_dir: Output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        model.eval()
        onnx_path = output_dir / f"{model_name}.onnx"

        print(f"🔄 Exporting {model_name}...")
        export_to_onnx(
            model=model,
            input_shape=input_shape,
            output_path=onnx_path,
            model_name=model_name,
        )

    print(f"✅ All models exported to {output_dir}")
