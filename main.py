from DTO import PredictionRequest, PredictionOutput
from pydantic import ValidationError
from flask import Flask, jsonify, request
import datetime
import numpy as np
import pandas as pd
import logging
import pickle
import json

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("predictions.log"), logging.StreamHandler()]
    )

log = logging.getLogger(__name__)

path_model = "model/delivery.pkl"
CALIBRATION_PATH = "model/calibration.json"

with open(path_model, "rb") as f:
    model = pickle.load(f)
    log.info("Modello caricato correttamente da %s", path_model)

#carico il file di calibrazione per la stima dell'intervallo di confidenza
with open(CALIBRATION_PATH, encoding="utf-8") as f:
    calibration = json.load(f)

app = Flask(__name__)


def build_prediction_output(raw_prediction):
    prediction = float(raw_prediction)

    lower = max(
        0.0,
        prediction + calibration["residual_q_lower"],
    )
    upper = max(
        lower,
        prediction + calibration["residual_q_upper"],
    )

    margin = calibration["absolute_residual_q95"]

    reliability_score = 1.0 / (
        1.0 + margin / max(abs(prediction), 1e-9)
    )

    return PredictionOutput(
        estimated_delivery_time=round(prediction, 2),
        unit=calibration["output_unit"],
        reliability_score=round(reliability_score, 3),
        confidence_interval=(
            round(lower, 2),
            round(upper, 2),
        ),
        calibration_source=calibration["source"],
    )


@app.errorhandler(ValidationError)
def handle_validation_error(e):
    return jsonify({"status": "error", "errors": e.errors()}), 422

@app.route("/", methods=["GET"])
def home():
    log.info("App in esecuzione. Stato operativo: OK")
    return jsonify({"status": "OK", "message": "App in esecuzione.", "timestamp": datetime.datetime.now().isoformat()})

@app.route("/health", methods=["GET"])
def health():
    if model is None:
        log.error("Modello non caricato correttamente. Stato operativo: ERROR")
        return jsonify({"status": "ERROR", "message": "Modello non caricato correttamente.", "timestamp": datetime.datetime.now().isoformat()})
    return jsonify({"status": "OK", "timestamp": datetime.datetime.now().isoformat()})

MODEL_VERSION = "1.0.0"
MODEL_TRAINED_ON = "2026-01-15"

@app.route("/model/info", methods=["GET"])
def model_info():
    return jsonify({
    "model_version": MODEL_VERSION,
    "trained_on": MODEL_TRAINED_ON,
    "model_features": model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else [],
    "sklearn_version": "1.6.1",
    "model_type": model.__class__.__name__,
    # Estrae il nome assegnato e il nome della classe dell'oggetto per ogni passaggio
    "model_steps": [
        {"step_name": name, "class_name": step.__class__.__name__} 
        for name, step in model.steps
    ] if hasattr(model, 'steps') else []
})

@app.route("/predict", methods=["POST"])
def predict():
    params = request.get_json()
    if params is None:
        log.error("Nessun parametro ricevuto nella richiesta.")
        return jsonify({"status": "error",
            "error": "Richiesta JSON mancante o malformata",
            "timestamp": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()}), 400
    if not isinstance(params, dict):
        return jsonify({
            "status": "error",
            "error": "Il body deve essere un oggetto JSON",
            "timestamp": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
        }), 422
    
    log.info("Parametri ricevuti: %s", params)

    try:
        request_param = PredictionRequest.model_validate(params)
    except ValidationError as e:
        log.warning("Errore di validazione: %s", e)

        return jsonify({
            "status": "error",
            "error": "validation_error",
            "details": e.errors(
                include_context=False,
                include_url=False
            ),
            "timestamp": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
        }), 422
        
        
    current_record = pd.DataFrame([{
        "pickup_location": request_param.pickup_location,
        "delivery_location": request_param.delivery_location,
        "weight": request_param.weight,
        "service_type": request_param.service_type
    }])

    try:
        
        prediction = model.predict(current_record)[0]
        output = build_prediction_output(prediction)
        log.info("Predizione effettuata per il record: %s", current_record)
        
    except Exception as e:
        log.error("Errore durante la predizione per il record %s: %s", current_record, str(e))
        return jsonify({
            "status": "error",
            "error": "prediction_error",
            "message": "Errore interno durante la predizione",
            "timestamp": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
        }), 500
    return jsonify({
        "prediction": output.model_dump(),
        "status": "success",
        "timestamp": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
    }), 200
   

MAX_BATCH_SIZE = 10
@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json()  # lista di dict
    if data is None:
        return jsonify({
            "status": "error",
            "error": "Richiesta JSON mancante o malformata",
            "timestamp": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
        }), 400

    if not isinstance(data, list):
        return jsonify({
            "status": "error",
            "error": "Il body deve essere un array JSON",
            "timestamp": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
        }), 422

    if len(data) == 0:
        return jsonify({
            "status": "error",
            "error": "Il batch non può essere vuoto",
            "timestamp": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
        }), 422

    if len(data) > MAX_BATCH_SIZE:
        return jsonify({
            "status": "error",
            "error": "batch_too_large",
            "max_batch_size": MAX_BATCH_SIZE,
            "timestamp": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
        }), 413
        
    predictions = []
    for index, item_param in enumerate(data): 
        if not isinstance(item_param, dict):
            predictions.append({
                "index": index,
                "status": "error",
                "error": "Ogni elemento deve essere un oggetto JSON"
            })
            continue
        try:
            request_param = PredictionRequest.model_validate(item_param)
        except ValidationError as e:
            log.warning( "Errore di validazione nel record %s: %s", index, e)

            predictions.append({
                "index": index,
                "status": "error",
                "error": "validation_error",
                "details": e.errors(
                    include_context=False,
                    include_url=False
                )
            })
            continue   
            
        current_record = pd.DataFrame([{
            "pickup_location": request_param.pickup_location,
            "delivery_location": request_param.delivery_location,
            "weight": request_param.weight,
            "service_type": request_param.service_type
        }])

        try:
            prediction = model.predict(current_record)[0]
            output = build_prediction_output(prediction)
            predictions.append({
                "prediction": output.model_dump(),
                "status": "success"
            })
            log.info("Predizione effettuata per il record: %s", item_param)
        except Exception:
            log.exception("Errore durante la predizione per il record %s: %s", index)
            predictions.append({
                "index": index,
                "status": "error",
                "error": "prediction_error",
                "message": "Errore interno durante la predizione"
            })
    if all(
        item["status"] == "success"
        for item in predictions
    ):
        overall_status = "success"
    elif any(
        item["status"] == "success"
        for item in predictions
    ):
        overall_status = "partial_success"
    else:
        overall_status = "error"

    return jsonify({
        "predictions": predictions,
        "status": overall_status,
        "timestamp": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
    }), 200
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)