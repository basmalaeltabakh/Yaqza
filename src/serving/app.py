"""FastAPI serving app (placeholder).

Provides a minimal REST endpoint to exercise the repository's serving path.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List


class PredictRequest(BaseModel):
    features: List[float] = []


def create_app() -> FastAPI:
    app = FastAPI(title="Yaqza Serving", version="0.1.0")
    app.name = "yaqza-serving"  # for compatibility with existing tests

    @app.post("/predict")
    def predict(req: PredictRequest):
        # Simple placeholder prediction: sum of input features
        value = sum(req.features) if req.features else 0.0
        return {"prediction": value, "features": req.features}

    return app


app = create_app()

if __name__ == "__main__":
    # Run the app with uvicorn for local development
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
