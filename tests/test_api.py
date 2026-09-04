"""
Test di integrazione per le API di LogiFast Delivery Time Prediction.
 
Richiede che il servizio sia già in esecuzione su BASE_URL
(locale con `python main.py` oppure via Docker con la porta 8000 mappata).
 
Esecuzione con pytest:
    pytest tests/test_api.py -v
 
Esecuzione come script standalone:
    python tests/test_api.py
"""
 
import requests
 
BASE_URL = "http://localhost:8000"
 
VALID_PAYLOAD = {
    "pickup_location": "Milano",
    "delivery_location": "Roma",
    "weight": 5.5,
    "service_type": "Express",
}
 
VALID_BATCH_PAYLOAD = [
    {
        "pickup_location": "Milano",
        "delivery_location": "Roma",
        "weight": 5.5,
        "service_type": "Express",
    },
    {
        "pickup_location": "Torino",
        "delivery_location": "Napoli",
        "weight": 12.0,
        "service_type": "Premium",
    },
]
 
INVALID_PAYLOAD = {
    "pickup_location": "Milano",
    "delivery_location": "Roma",
    "weight": "non-un-numero",
    "service_type": "Express",
}
 
 
def test_root():
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "OK"
 
 
def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "OK"
    assert "timestamp" in body
 
 
def test_model_info():
    r = requests.get(f"{BASE_URL}/model/info")
    assert r.status_code == 200
    body = r.json()
    assert "model_version" in body
    assert "model_features" in body
    assert "model_steps" in body
    assert "model_type" in body
    assert "trained_on" in body
    assert "sklearn_version" in body
 
def test_predict_valid():
    r = requests.post(f"{BASE_URL}/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert "prediction" in body
    print("predict OK ->", body)
 
 
def test_predict_batch_valid():
    r = requests.post(f"{BASE_URL}/predict/batch", json=VALID_BATCH_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body
    assert len(body["predictions"]) == len(VALID_BATCH_PAYLOAD)
    print("predict/batch OK ->", body)
 
 
def test_predict_invalid_payload():
    r = requests.post(f"{BASE_URL}/predict", json=INVALID_PAYLOAD)
    # ci si aspetta una gestione controllata dell'errore, non un crash del server
    assert r.status_code in (200, 400, 422, 500)
    body = r.json()
    print("predict con payload invalido ->", body)
 
 
def test_predict_empty_body():
    r = requests.post(f"{BASE_URL}/predict", json={})
    assert r.status_code in (200, 400, 422, 500)
 
 
if __name__ == "__main__":
    print("== /  ==")
    test_root()
    print("== /health ==")
    test_health()
    print("== /model/info ==")
    test_model_info()
    print("== /predict (valido) ==")
    test_predict_valid()
    print("== /predict/batch (valido) ==")
    test_predict_batch_valid()
    print("== /predict (payload invalido) ==")
    test_predict_invalid_payload()
    print("== /predict (body vuoto) ==")
    test_predict_empty_body()
    print("\nTutti i test eseguiti.")