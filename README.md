# LogiFast Delivery Time Prediction API

API REST Flask per esporre un modello di Machine Learning che stima il tempo di consegna di una spedizione LogiFast.

## Funzionalità

Il servizio carica l’artefatto `model/delivery.pkl` all’avvio ed espone predizioni singole e batch. La validazione applicativa accetta soltanto le venti città e i due tipi di servizio conosciuti dal modello. Gli errori del client sono restituiti in JSON con codici HTTP coerenti. Le risposte di predizione includono il valore stimato, l’unità, uno score di affidabilità euristico e un intervallo di predizione derivato dai residui di calibrazione.

`pickup_datetime` non viene accettata dall’API perché non compare tra le feature effettivamente utilizzate dalla pipeline fitted. Modificare la data o l’ora non cambia la predizione dell’artefatto corrente.

## Struttura del progetto

```text
.
├── main.py
├── DTO.py
├── model/
│   ├── delivery.pkl
│   └── calibration.json
├── data/
|   ├── calibration_residuals.csv
│   └── synthetic_delivery_dataset.csv
├── notebook/
│   └── sensitivity_metrics.py
├── tests/
│   └── test_api.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

Il notebook rende riproducibili l’ispezione della pipeline, l’analisi di sensibilità, il calcolo delle metriche e la generazione dei parametri di calibrazione. Il dataset è sintetico e non rappresenta dati reali di training o test.

## Contratto del modello

| Campo | Tipo | Vincoli |
|---|---:|---|
| `pickup_location` | stringa | Una delle venti città supportate |
| `delivery_location` | stringa | Una delle venti città supportate |
| `weight` | numero | Valore finito e strettamente maggiore di zero |
| `service_type` | stringa | `Express` oppure `Premium` |

Le città supportate sono: `Ancona`, `Bari`, `Bologna`, `Cagliari`, `Catania`, `Firenze`, `Genova`, `Lecce`, `Milano`, `Napoli`, `Palermo`, `Perugia`, `Pescara`, `Reggio Calabria`, `Roma`, `Salerno`, `Sassari`, `Torino`, `Trapani` e `Verona`.

## Avvio in locale

È richiesto Python 3.11 o una versione compatibile con le dipendenze dichiarate.

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python main.py
```

Il servizio viene esposto su `http://localhost:8000`.

## Avvio con Docker

```bash
docker build -t logifast-delivery-api:1.1.0 .
docker run --rm -p 8000:8000 logifast-delivery-api:1.1.0
```

Verifica rapida:

```bash
curl --fail http://localhost:8000/health
```

## API

### `GET /`

Restituisce lo stato generale dell’applicazione.

### `GET /health`

**Risposta `200 OK`:**

```json
{
  "status": "OK",
  "timestamp": "2026-09-11T14:30:00+00:00"
}
```

### `GET /model/info`

Espone metadati verificabili su API, modello, dati di calibrazione e dipendenze.

```json
{
    "model_features": [
        "pickup_location",
        "delivery_location",
        "weight",
        "service_type"
    ],
    "model_steps": [
        {
            "class_name": "ColumnTransformer",
            "step_name": "preprocess"
        },
        {
            "class_name": "LinearRegression",
            "step_name": "regressor"
        }
    ],
    "model_type": "Pipeline",
    "model_version": "1.0.0",
    "sklearn_version": "1.6.1",
    "trained_on": "2026-01-15"
}
```

La data di training non viene dichiarata se non è disponibile da una fonte verificabile. Non deve essere ricavata o inventata a partire dal file Pickle.

### `POST /predict`

**Request body:**

```json
{
  "pickup_location": "Milano",
  "delivery_location": "Roma",
  "weight": 5.5,
  "service_type": "Express"
}
```

Esempio con `curl`:

```bash
curl --fail-with-body \
  -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_location": "Milano",
    "delivery_location": "Roma",
    "weight": 5.5,
    "service_type": "Express"
  }'
```

**Risposta `200 OK`:**

```json
{
  "prediction": {
    "estimated_delivery_time": 43.05,
    "unit": "minutes",
    "reliability_score": 0.48,
    "confidence_interval": [
            137.51,
            1106.03
        ],
  },
  "status": "success",
  "timestamp": "2026-09-11T14:30:00+00:00"
}
```

Il valore mostrato è soltanto un esempio. Lo score e l’intervallo devono essere calcolati in esecuzione a partire dai parametri salvati in `model/calibration.json`.

### Interpretazione dell’affidabilità

Il notebook calcola i residui su un dataset di calibrazione espresso nella stessa unità dell’output del modello. La versione dimostrativa utilizza il 95° percentile dell’errore assoluto, indicato con `absolute_residual_q95`.

```text
lower = max(0.0,prediction + calibration["residual_q_lower"],)
upper = max(lower,prediction + calibration["residual_q_upper"],)
margin = calibration["absolute_residual_q95"]
reliability_score = 1.0 / (1.0 + margin / max(abs(prediction), 1e-9))
```

Lo score è compreso tra zero e uno. Un intervallo ampio rispetto alla predizione produce uno score più basso. Questa è una misura euristica di supporto, non una probabilità di correttezza. Poiché la calibrazione corrente usa un target sintetico. Lo score non può essere presentato come affidabilità reale del modello finché non sono disponibili tempi di consegna osservati.

### `POST /predict/batch`

Il body deve essere un array JSON. Ogni elemento viene validato e processato indipendentemente, in modo che un record errato non faccia perdere gli altri risultati.

```json
[
  {
    "pickup_location": "Milano",
    "delivery_location": "Roma",
    "weight": 5.5,
    "service_type": "Express"
  },
  {
    "pickup_location": "Torino",
    "delivery_location": "Napoli",
    "weight": 12.0,
    "service_type": "Premium"
  }
]
```

**Risposta `200 OK`:**

```json
{
    "predictions": [
        {
            "prediction": {
                "confidence_interval": [
                    136.5,
                    1105.02
                ],
                "estimated_delivery_time": 43.05,
                "reliability_score": 0.042,
                "unit": "minutes"
            },
            "status": "success"
        },
        {
            "prediction": {
                "confidence_interval": [
                    143.67,
                    1112.19
                ],
                "estimated_delivery_time": 50.22,
                "reliability_score": 0.048,
                "unit": "minutes"
            },
            "status": "success"
        }
    ],
    "status": "success",
    "timestamp": "2026-09-11T16:24:27.786949+00:00"
}
```

Il server deve rifiutare un batch vuoto, un body che non sia una lista e un numero di elementi superiore al limite configurato.

## Errori JSON

Un payload non valido restituisce sempre JSON, mai una pagina HTML.

**Esempio `422 Unprocessable Entity`:**

```json
{
    "details": [
        {
            "input": "Atlantide",
            "loc": [
                "pickup_location"
            ],
            "msg": "Value error, Città non riconosciuta: Atlantide",
            "type": "value_error"
        }
    ],
    "error": "validation_error",
    "status": "error",
    "timestamp": "2026-09-11T16:25:27.994520+00:00"
}
```

| Caso | Codice HTTP |
|---|---:|
| JSON mancante o malformato | `400` |
| Campi mancanti, tipi errati o valori fuori dominio | `422` |
| Metodo non consentito | `405` |
| Errore interno non previsto | `500` |

## Esplorazione e validazione

Il notebook `notebook/sensitivity_metrics.py` documenta:

1. struttura della pipeline e feature effettivamente usate;
2. categorie accettate dal `OneHotEncoder`;
3. coefficiente leggermente negativo associato a `weight`;
4. irrilevanza di `pickup_datetime` per l’artefatto corrente;
5. metriche dimostrative sul dataset sintetico;
6. distribuzione dei residui e parametri usati dall’API per l’intervallo.

Esecuzione:

```bash
python notebook/sensitivity_metrics.py
```

Le metriche sul dataset sintetico non misurano la performance reale. Il dataset non proviene dalla distribuzione di training e non fornisce una ground truth osservata.

## Test

I test prevedono l'avvio del server manualmente (python main.py  in un terminale). In un nuovo terminale si esegue:

```bash
pytest tests/test_api.py
```

La suite copre endpoint validi, JSON malformato, body vuoto, campi mancanti, peso negativo, città e servizio sconosciuti, batch troppo grande e batch con record misti. 

## Limiti noti

L’artefatto non include il dataset originale, il target osservato, l’unità del target o metadati completi di training. Il dataset presente nella repository è sintetico. Le sue metriche e gli intervalli che ne derivano hanno valore dimostrativo. L’unità `hours` usata in questa anteprima deve essere confermata con il fornitore del modello o con la traccia originale prima della pubblicazione definitiva.

Il modello ignora `pickup_datetime`. Il coefficiente del peso è leggermente negativo. Entrambi i comportamenti sono stati verificati e devono essere considerati prima di un utilizzo operativo.

