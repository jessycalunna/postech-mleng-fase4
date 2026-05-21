"""
API FastAPI para servir o modelo LSTM treinado no Databricks.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import numpy as np
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

# Configurações
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
MODEL_PATH    = ARTIFACTS_DIR / "lstm_model.keras"
SCALER_PATH   = ARTIFACTS_DIR / "scaler.pkl"
META_PATH     = ARTIFACTS_DIR / "metadata.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("lstm-api")

# Métricas Prometheus
REQ_COUNT = Counter(
    "http_requests_total", "Total de requisições HTTP", ["method", "endpoint", "status"]
)
REQ_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Latência das requisições HTTP",
    ["endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PRED_LATENCY = Histogram(
    "model_inference_duration_seconds",
    "Tempo gasto apenas na inferência do modelo",
)
PRED_COUNT = Counter("model_predictions_total", "Total de predições feitas")
PRED_ERRORS = Counter("model_prediction_errors_total", "Total de erros de predição")
CPU_GAUGE = Gauge("process_cpu_percent", "Uso de CPU do processo (%)")
MEM_GAUGE = Gauge("process_memory_mb", "Memória residente do processo (MB)")

# Estado do Modelo
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo, scaler e metadados."""
    log.info("Carregando artefatos de %s", ARTIFACTS_DIR.resolve())
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Modelo não encontrado em {MODEL_PATH}. "
        )
    state["model"] = load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        state["scaler"] = pickle.load(f)
    with open(META_PATH) as f:
        state["metadata"] = json.load(f)
    state["window_size"] = int(state["metadata"]["window_size"])
    log.info(
        "Modelo carregado | ticker=%s | window=%s",
        state["metadata"]["ticker"], state["window_size"],
    )
    yield
    log.info("Encerrando API")


app = FastAPI(
    title="LSTM Pretidor da PETR4",
    description="API para previsão do preço de fechamento de ações com LSTM",
    version="1.0.0",
    lifespan=lifespan,
)


# Métricas Middleware
@app.middleware("http")
async def monitor(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    endpoint = request.url.path
    REQ_LATENCY.labels(endpoint=endpoint).observe(elapsed)
    REQ_COUNT.labels(
        method=request.method, endpoint=endpoint, status=response.status_code
    ).inc()
    CPU_GAUGE.set(psutil.cpu_percent(interval=None))
    MEM_GAUGE.set(psutil.Process().memory_info().rss / 1024 / 1024)
    response.headers["X-Response-Time-ms"] = f"{elapsed * 1000:.2f}"
    return response


# Schemas
class PredictRequest(BaseModel):
    prices: List[float] = Field(
        ...,
        description=(
            "Lista de preços de fechamento históricos. "
        ),
        examples=[[35.1, 35.4, 35.8, 36.0, 35.7]],
    )


class PredictResponse(BaseModel):
    predicted_price: float
    window_size_used: int
    inference_time_ms: float


# Endpoints
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/metadata")
def metadata():
    return state["metadata"]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    window = state["window_size"]
    if len(req.prices) < window:
        PRED_ERRORS.inc()
        raise HTTPException(
            status_code=400,
            detail=f"Forneça ao menos {window} preços (recebido: {len(req.prices)}).",
        )

    try:
        ult = np.array(req.prices[-window:], dtype=np.float32).reshape(-1, 1)
        scaled = state["scaler"].transform(ult)
        x = scaled.reshape(1, window, 1)

        t0 = time.perf_counter()
        y_scaled = state["model"].predict(x, verbose=0)
        infer_ms = (time.perf_counter() - t0) * 1000
        PRED_LATENCY.observe(infer_ms / 1000.0)

        y = float(state["scaler"].inverse_transform(y_scaled)[0, 0])
        PRED_COUNT.inc()
        log.info("Predição: %.4f (entrada com %d preços)", y, len(req.prices))
        return PredictResponse(
            predicted_price=y, window_size_used=window, inference_time_ms=infer_ms
        )
    except HTTPException:
        raise
    except Exception as e:
        PRED_ERRORS.inc()
        log.exception("Erro na predição")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/metrics")
def metrics():
    """Endpoint Prometheus-compatible. Configure scrape_interval no Prometheus."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
