from DTO import PredictionRequest, PredictionOutput
from flask import Flask, jsonify, request
import datetime
import numpy as np
import pandas as pd
import logging
import pickle

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("predictions.log"), logging.StreamHandler()]
    )

log = logging.getLogger(__name__)

path_model = "model/delivery.pkl"

with open(path_model, "rb") as f:
    model = pickle.load(f)
    log.info("Modello caricato correttamente da %s", path_model)

app = Flask(__name__)

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
        return jsonify({"error": "Richiesta JSON mancante o malformata"}), 400
    log.info("Parametri ricevuti: %s", params)

    request_param = PredictionRequest(
        pickuplocation=params.get("pickup_location"),
        deliverylocation=params.get("delivery_location"),
        weight=params.get("weight"),
        servicetype=params.get("service_type"),
    )

    current_record = pd.DataFrame([{
        "pickup_location": request_param.pickuplocation,
        "delivery_location": request_param.deliverylocation,
        "weight": request_param.weight,
        "service_type": request_param.servicetype
    }])

    try:
        
        prediction = model.predict(current_record)[0]
        log.info("Predizione effettuata per il record: %s", current_record)
        status = "success"
    except Exception as e:
        log.error("Errore durante la predizione per il record %s: %s", current_record, str(e))
        status = "error"
        return jsonify({"error": str(e), "status": status, "timestamp": datetime.datetime.now().isoformat()})

    output = PredictionOutput(estimated_delivery_time=prediction)
    return jsonify({"prediction": output.model_dump(), "status": status, "timestamp": datetime.datetime.now().isoformat()}) 

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json()  # lista di dict
    predictions = []
    for item_param in data:
        request_param = PredictionRequest(pickuplocation=item_param.get("pickup_location"),
                deliverylocation=item_param.get("delivery_location"),
                weight=item_param.get("weight"),
                servicetype=item_param.get("service_type"),)
        
        current_record = pd.DataFrame([{
            "pickup_location": request_param.pickuplocation,
            "delivery_location": request_param.deliverylocation,
            "weight": request_param.weight,
            "service_type": request_param.servicetype
        }])
        try:
            
            prediction = model.predict(current_record)[0]
            predictions.append({
                "prediction": PredictionOutput(estimated_delivery_time=float(prediction)).model_dump(),
                "status": "success"
            })
            log.info("Predizione effettuata per il record: %s", item_param)
            status = "success"
        except Exception as e:
            log.error("Errore durante la predizione per il record %s: %s", item_param, str(e))
            status = "error"
            predictions.append({"error": str(e)})

    return jsonify({"predictions": predictions, "status": status, "timestamp": datetime.datetime.now().isoformat()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
    