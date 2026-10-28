from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from flask import Response
from pathlib import Path
import logging
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g

app = Flask(__name__)
HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests received",
    ["method", "path", "status"],
)

HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total prediction attempts",
    ["result"],
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the prediction model loaded successfully",
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s "
           "request_id=%(request_id)s %(message)s",
)

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True

for handler in logging.getLogger().handlers:
    handler.addFilter(RequestIdFilter())

logger = logging.getLogger(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
]


@app.before_request
def log_request_start():
    g.request_id = request.headers.get(
        "X-Request-ID", str(uuid.uuid4())
    )
    g.start_time = time.perf_counter()

    logger.info(
        "request_started method=%s path=%s",
        request.method,
        request.path,
    )


@app.after_request
def log_request_end(response):
    elapsed_ms = (
        (time.perf_counter() - g.start_time) * 1000
        if hasattr(g, "start_time") else 0
    )
        if request.path != "/metrics":
        HTTP_REQUESTS.labels(
            method=request.method,
            path=request.path,
            status=str(response.status_code),
        ).inc()

        HTTP_LATENCY.labels(
            method=request.method,
            path=request.path,
        ).observe(elapsed_ms / 1000.0)

    response.headers["X-Request-ID"] = getattr(
        g, "request_id", "-"
    )

    logger.info(
        "request_completed method=%s path=%s status=%s duration_ms=%.2f",
        request.method,
        request.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    logger.exception("unexpected_application_error")
    return jsonify({
        "error": "Internal server error",
        "request_id": getattr(g, "request_id", "-"),
    }), 500


def load_model():
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "student_result_model.pkl was not found. "
                "Run the training pipeline first."
            )

        model = joblib.load(MODEL_PATH)
        MODEL_LOADED.set(1)
        return model

    except Exception:
        MODEL_LOADED.set(0)
        logger.exception("event=model_load_failed model_file=%s", MODEL_PATH)
        raise


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction",
        "version": "2.1"
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        logger.warning("validation_error reason=missing_json_body")
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "validation_error reason=missing_fields fields=%s",
            missing_fields,
        )
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields,
        }), 400

    try:
        sample = pd.DataFrame([{
            feature: float(data[feature])
            for feature in FEATURES
        }])
    except (TypeError, ValueError):
        logger.warning("validation_error reason=non_numeric_features")
        return jsonify({
            "error": "Feature values must be numeric"
        }), 400

    try:
        model = load_model()
        prediction_code = int(model.predict(sample)[0])
        PREDICTIONS.labels(result="success").inc()

    except Exception:
        PREDICTIONS.labels(result="error").inc()
        logger.exception("event=prediction_failed")
        raise
    prediction = "PASS" if prediction_code == 1 else "FAIL"

    logger.info(
        "prediction_completed prediction=%s prediction_code=%s",
        prediction,
        prediction_code,
    )

    return jsonify({
        "prediction": prediction,
        "prediction_code": prediction_code,
    })

@app.get("/metrics")
def metrics():
    return Response(
        response=generate_latest(),
        mimetype=CONTENT_TYPE_LATEST,
    )
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
