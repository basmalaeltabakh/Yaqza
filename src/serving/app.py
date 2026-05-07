"""FastAPI serving app for RUL predictions using ONNX models.

Provides REST endpoints for:
- Single sequence prediction
- Batch prediction
- Model health checks
- Metadata endpoints
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Try to import ONNX utilities (graceful degradation if ONNX not available)
try:
    from src.serving.model_utils import ONNXPredictor
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


# ── Request/Response Models ────────────────────────────────────────────────

class RULPredictionRequest(BaseModel):
    """Request for RUL prediction.
    
    Example:
        {
            "sequence": [[100.2, 52.1, ...], [100.5, 52.3, ...], ...],
            "model": "transformer"
        }
    """
    sequence: List[List[float]] = Field(
        ...,
        description="Sensor readings (seq_len × n_features)",
    )
    model: str = Field(default="transformer", description="Model to use")


class RULPredictionResponse(BaseModel):
    """Response with RUL prediction."""
    rul_cycles: float = Field(..., description="Predicted Remaining Useful Life")
    confidence: Optional[float] = Field(None, description="Prediction confidence (0-1)")
    model: str = Field(..., description="Model used")


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    onnx_available: bool
    models_loaded: List[str]


class ModelMetadata(BaseModel):
    """Model metadata."""
    model_name: str
    version: str
    input_shape: List[int]
    output_shape: List[int]
    framework: str = "ONNX"


# ── App Creation ───────────────────────────────────────────────────────────

def create_app(model_dir: Optional[Path] = None) -> FastAPI:
    """Create FastAPI app with RUL prediction endpoints.

    Args:
        model_dir: Directory containing ONNX models (defaults to model_weights/).

    Returns:
        FastAPI application.
    """
    app = FastAPI(
        title="Yaqza RUL Prediction API",
        description="Predict Remaining Useful Life of industrial equipment",
        version="1.0.0",
    )

    if model_dir is None:
        model_dir = Path(__file__).parent.parent.parent / "model_weights"

    # Load models (only if ONNX available)
    loaded_models = {}

    if HAS_ONNX:
        transformer_path = model_dir / "transformer.onnx"
        lstm_path = model_dir / "lstm.onnx"

        if transformer_path.exists():
            try:
                loaded_models["transformer"] = ONNXPredictor(transformer_path)
            except Exception as e:
                print(f"⚠️  Failed to load transformer: {e}")

        if lstm_path.exists():
            try:
                loaded_models["lstm"] = ONNXPredictor(lstm_path)
            except Exception as e:
                print(f"⚠️  Failed to load LSTM: {e}")

    # ── Endpoints ──────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthCheckResponse)
    def health_check():
        """Health check endpoint."""
        return HealthCheckResponse(
            status="healthy",
            onnx_available=HAS_ONNX,
            models_loaded=list(loaded_models.keys()),
        )

    @app.post("/predict", response_model=RULPredictionResponse)
    def predict(request: RULPredictionRequest):
        """Predict RUL from a sensor sequence.

        Args:
            request: Prediction request with sequence and model name.

        Returns:
            RUL prediction.

        Raises:
            HTTPException: If model not found or prediction fails.
        """
        if not HAS_ONNX:
            raise HTTPException(
                status_code=503,
                detail="ONNX Runtime not available. Install onnxruntime.",
            )

        if request.model not in loaded_models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model}' not found. Available: {list(loaded_models.keys())}",
            )

        try:
            # Convert to numpy array
            X = np.array(request.sequence, dtype=np.float32)

            # Ensure correct shape (1, seq_len, n_features)
            if X.ndim == 2:
                X = np.expand_dims(X, axis=0)
            elif X.ndim != 3 or X.shape[0] != 1:
                raise ValueError(f"Expected shape (1, seq_len, n_features), got {X.shape}")

            # Predict
            predictor = loaded_models[request.model]
            rul_pred = predictor.predict(X)[0]

            return RULPredictionResponse(
                rul_cycles=float(rul_pred),
                confidence=None,  # Could add uncertainty estimation here
                model=request.model,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    @app.post("/predict_batch")
    def predict_batch(requests: List[RULPredictionRequest]):
        """Batch prediction endpoint.

        Args:
            requests: List of prediction requests.

        Returns:
            List of predictions.
        """
        results = []
        for req in requests:
            try:
                result = predict(req)
                results.append(result)
            except HTTPException as e:
                results.append({"error": e.detail})

        return results

    @app.get("/models/metadata/{model_name}", response_model=ModelMetadata)
    def get_model_metadata(model_name: str):
        """Get metadata for a model.

        Args:
            model_name: Name of the model.

        Returns:
            Model metadata.
        """
        if model_name not in loaded_models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        # Placeholder metadata (could be extended with actual ONNX introspection)
        return ModelMetadata(
            model_name=model_name,
            version="1.0.0",
            input_shape=[1, 30, 14],
            output_shape=[1],
            framework="ONNX",
        )

    @app.get("/models")
    def list_models():
        """List available models."""
        return {"models": list(loaded_models.keys())}

    return app


# ── Create App Instance ────────────────────────────────────────────────────

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
    # Run the app with uvicorn for local development
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
