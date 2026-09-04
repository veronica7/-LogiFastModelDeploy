# LogiFast Delivery Time Prediction API

API REST in Flask per la stima del tempo di consegna (time-to-delivery) degli ordini di LogiFast Solutions, dal ritiro alla consegna finale.

## Indice

- [Descrizione](#descrizione)
- [Struttura del progetto](#struttura-del-progetto)
- [Avvio in locale](#avvio-in-locale)
- [Avvio con Docker](#avvio-con-docker)
- [Documentazione API](#documentazione-api)
- [Test di integrazione](#test-di-integrazione)
- [Note su versionamento e monitoraggio](#note-su-versionamento-e-monitoraggio)

## Descrizione

Il servizio espone un modello di machine learning (`model/delivery.pkl`) che predice il tempo di consegna di un ordine a partire da:

- luogo di ritiro
- luogo di consegna
- data/ora di ritiro
- peso del pacco
- tipo di servizio (`Express` / `Premium`)

## Struttura del progetto

```
.
├── main.py                # Applicazione Flask
├── DTO.py                 # Modelli Pydantic (request/response)
├── model/
│   └── delivery.pkl       # Modello serializzato
├── tests/
│   └── test_api.py        # Test di integrazione
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Avvio in locale

Requisiti: Python 3.11+

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
python main.py
```

Il servizio sarà disponibile su `http://localhost:8000`.

## Avvio con Docker

```bash
docker build -t logifast-delivery-api .
docker run -p 8000:8000 logifast-delivery-api
```

Verifica che il container sia in esecuzione e con la porta mappata correttamente:

```bash
docker ps
```

Nella colonna `PORTS` deve comparire `0.0.0.0:8000->8000/tcp`.

## Documentazione API

### `GET /health`

Stato operativo del servizio.

**Risposta 200:**
```json
{
  "status": "OK",
  "timestamp": "2026-09-04T10:00:00.000000"
}
```

### `GET /model/info`

Versione e metadati del modello caricato.

**Risposta 200:**
```json
{
  "model_version": "1.0.0",
  "trained_on": "2026-01-15",
  "sklearn_version": "1.6.1",
  "features": ["pickup_location", "delivery_location", "pickup_datetime", "weight", "service_type"]
}
```

### `POST /predict`

Predizione per un singolo ordine.

**Request body:**
```json
{
  "pickuplocation": "Milano",
  "deliverylocation": "Roma",
  "pickupdatetime": "2026-09-03T10:00:00",
  "weight": 5.5,
  "servicetype": "Express"
}
```

**Risposta 200:**
```json
{
  "prediction": {
    "predicted_probability": 132.5
  },
  "status": "success",
  "timestamp": "2026-09-04T10:00:00.000000"
}
```

### `POST /predict/batch`

Predizione per una lista di ordini.

**Request body:**
```json
[
  {
    "pickuplocation": "Milano",
    "deliverylocation": "Roma",
    "pickupdatetime": "2026-09-03T10:00:00",
    "weight": 5.5,
    "servicetype": "Express"
  },
  {
    "pickuplocation": "Torino",
    "deliverylocation": "Napoli",
    "pickupdatetime": "2026-09-04T14:30:00",
    "weight": 12.0,
    "servicetype": "Premium"
  }
]
```

**Risposta 200:**
```json
{
  "predictions": [
    { "prediction": { "predicted_probability": 132.5 }, "status": "success" },
    { "prediction": { "predicted_probability": 210.0 }, "status": "success" }
  ],
  "timestamp": "2026-09-04T10:00:00.000000"
}
```

## Test di integrazione

I test si trovano in `tests/test_api.py` e verificano che gli endpoint rispondano correttamente. Richiedono che il servizio sia già attivo (locale o via Docker) su `http://localhost:8000`.

Installazione dipendenza di test:

```bash
pip install pytest requests
```

Esecuzione:

```bash
pytest tests/test_api.py -v
```

Oppure, senza pytest, come script standalone:

```bash
python tests/test_api.py
```

## Note su versionamento e monitoraggio

- La versione del modello è esposta tramite `GET /model/info` ed è tracciata manualmente nella costante `MODEL_VERSION` in `main.py`.
- Le richieste e le predizioni vengono loggate su file (`predictions.log`) oltre che su console, per supportare attività di monitoraggio e audit.
- Per la strategia completa di versionamento, automazione, monitoraggio e governance, vedere il documento `docs/mlops-strategy.md` (o equivalente) allegato alla consegna del progetto.