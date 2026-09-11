"""
LogiFast — Esplorazione e validazione del modello `delivery.pkl`
 
Questo notebook rende **verificabile e riproducibile** l"Analisi di
sensibilità" e le "Metriche di valutazione"). 

"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
MODEL_PATH = "./model/delivery.pkl"
DATA_PATH = "./data/synthetic_delivery_dataset.csv"
 
pd.set_option("display.width", 120)
plt.rcParams["figure.figsize"] = (7, 4)


## 1. Caricamento dell'artefatto e ispezione della pipeline
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
 
print("Tipo oggetto:", type(model).__name__)
print("Step della pipeline:")
for name, step in model.steps:
    print(f"  - {name}: {type(step).__name__}")
 
print("\\nColonne attese in input (feature_names_in_):")
print(list(model.feature_names_in_))

"""
Le colonne riportate da `feature_names_in_` sono quelle **effettivamente
utilizzate dal preprocessing fitted**. Come si vede, `pickup_datetime` non
compare: anche se l'API la accetta/accettava nello schema applicativo, la
pipeline non la vede mai..
"""
 
print("Preprocessing e regressore:")
preprocess = model.named_steps["preprocess"]
regressor = model.named_steps["regressor"]
 
print("Preprocessing:", type(preprocess).__name__)
for name, transformer, cols in preprocess.transformers_:
    print(f"  - transformer '{name}' ({type(transformer).__name__}) su colonne: {cols}")
 
print("Modello finale:", type(regressor).__name__)
 
feature_names_out = preprocess.get_feature_names_out()
print(f"Numero di feature dopo preprocessing: {len(feature_names_out)}")

summary = pd.DataFrame({
    "Voce": [
        "Tipo",
        "Preprocessing",
        "Encoding",
        "Modello finale",
        "Categoriche",
        "Numerica",
        "Variabile nello schema ma non usata",
        "Feature dopo preprocessing",
    ],
    "Valore riscontrato": [
        type(model).__name__ + " (scikit-learn)",
        type(preprocess).__name__,
        type(preprocess.named_transformers_["cat"]).__name__,
        type(regressor).__name__,
        ", ".join([c for n, t, cols in preprocess.transformers_ if n == "cat" for c in cols]),
        ", ".join([c for n, t, cols in preprocess.transformers_ if n == "num" for c in cols]),
        "pickup_datetime",
        len(feature_names_out),
    ],
})
summary

# 2. Coefficienti della regressione: il finding su `weight`
coefs = pd.Series(regressor.coef_, index=feature_names_out)
weight_coef = coefs[[c for c in coefs.index if c.endswith("weight")][0]]
 
print(f"Intercetta: {regressor.intercept_:.4f}")
print(f"Coefficiente di 'weight': {weight_coef:.6f}")
print()
print(f"Interpretazione: a parità di rotta e servizio, ogni kg aggiuntivo sposta la predizione di circa {weight_coef:.4f} minuti.")
print("Il segno è negativo: il modello NON riflette l'intuizione secondo cui un pacco più pesante richieda più tempo. Comportamento da segnalare al team Data Science, come indicato in relazione.")

## 3. Analisi di sensibilità

def build_record(pickup, delivery, weight, service, pickup_datetime=None):
    row = {
        "pickup_location": pickup,
        "delivery_location": delivery,
        "weight": weight,
        "service_type": service,
    }
    if pickup_datetime is not None:
        row["pickup_datetime"] = pickup_datetime
    return pd.DataFrame([row])

# 3.1 Variazione del peso, Roma -> Milano, Express
weights = [0.5, 1, 2, 5, 10, 15, 20, 25, 30]
rows = pd.concat(
    [build_record("Roma", "Milano", w, "Express") for w in weights],
    ignore_index=True,
)
rows["predicted_minutes"] = model.predict(rows)
rows

plt.plot(rows["weight"], rows["predicted_minutes"], marker="o")
plt.xlabel("Peso (kg)")
plt.ylabel("Tempo di consegna predetto (min)")
plt.title("Sensibilità al peso — Roma → Milano, Express")
plt.grid(alpha=0.3)
plt.show()

# 3.2 Confronto Express vs Premium, stessa tratta e stesso peso
express = build_record("Roma", "Milano", 5.0, "Express")
premium = build_record("Roma", "Milano", 5.0, "Premium")
 
pred_express = model.predict(express)[0]
pred_premium = model.predict(premium)[0]
 
print(f"Express: {pred_express:.2f} min")
print(f"Premium: {pred_premium:.2f} min")
print(f"Differenza (Premium - Express): {pred_premium - pred_express:.2f} min")

# 3.3 Variazione di pickup_datetime a parità di tutto il resto.
# La pipeline seleziona le colonne per nome (vedi transformers_ sopra), quindi passare una colonna extra non utilizzata non genera un errore:
# viene semplicemente ignorata. Dimostriamo che sullo stesso artefatto, la variabile non contribuisce alla predizione.

early_morning = build_record("Roma", "Milano", 5.0, "Express", "2026-01-01T07:00:00")
late_evening = build_record("Roma", "Milano", 5.0, "Express", "2026-07-15T20:00:00")
 
pred_early = model.predict(early_morning)[0]
pred_late = model.predict(late_evening)[0]
 
print(f"pickup_datetime = 2026-01-01T07:00:00 -> {pred_early:.6f} min")
print(f"pickup_datetime = 2026-07-15T20:00:00 -> {pred_late:.6f} min")
print(f"Differenza: {pred_late - pred_early:.10f} min")
assert pred_early == pred_late, "Le predizioni differiscono: verificare la pipeline"
print("Confermato: la predizione e' identica al variare di pickup_datetime.")

## 4. Metriche di valutazione sul dataset sintetico
 
df = pd.read_csv(DATA_PATH)
print(df.shape)
df.head()

feature_cols = ["pickup_location", "delivery_location", "weight", "service_type"]
recomputed = model.predict(df[feature_cols])
 
max_abs_diff = np.max(np.abs(recomputed - df["model_prediction_min"].to_numpy()))
print(f"Differenza massima tra predizione ricalcolata e colonna nel CSV: {max_abs_diff:.6f}")
# Tolleranza 0.01: la colonna nel CSV risulta arrotondata a 2 decimali in fase
# di generazione del dataset sintetico, non e' un disallineamento della pipeline.
assert max_abs_diff < 1e-2, "Le predizioni ricalcolate non combaciano col CSV"
print("Confermato (entro l'arrotondamento a 2 decimali del CSV):"
      " model_prediction_min e' riproducibile da model.predict().")

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}
 
y_true = df["synthetic_delivery_time_min"]
y_pred = df["model_prediction_min"]
 
metrics = regression_metrics(y_true, y_pred)
metrics_df = pd.DataFrame(
    {
        "Metrica": ["MAE (min)", "RMSE (min)", "MAPE (%)", "R2"],
        "Valore sintetico": [
            round(metrics["MAE"], 2),
            round(metrics["RMSE"], 2),
            round(metrics["MAPE"], 2),
            round(metrics["R2"], 3),
        ],
    }
)
metrics_df

residuals = y_true - y_pred
 
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
axes[0].scatter(y_pred, y_true, alpha=0.3, s=10)
lims = [min(y_pred.min(), y_true.min()), max(y_pred.max(), y_true.max())]
axes[0].plot(lims, lims, color="red", linestyle="--", label="y = x (ideale)")
axes[0].set_xlabel("Predizione (min)")
axes[0].set_ylabel("Target sintetico (min)")
axes[0].set_title("Predetto vs. target sintetico")
axes[0].legend()
 
axes[1].hist(residuals, bins=40)
axes[1].set_xlabel("Residuo (target - predizione, min)")
axes[1].set_ylabel("Frequenza")
axes[1].set_title("Distribuzione dei residui")
 
plt.tight_layout()
plt.show()
 
print(f"Deviazione standard dei residui: {residuals.std():.2f} min")

## 5. Verifica dell'intervallo di confidenza usato da `/predict`
 
# L'API espone uno `score di affidabilità` costruito come intervallo `predizione ± 1.96 * std(residui)`, 
# dove lo std dei residui è quello appena calcolato su questo stesso dataset. Verifichiamo qui il valore usato in
# `DTO.py` / `main.py`, così che resti tracciabile da dove viene.

residual_std = float(residuals.std())
z = 1.96  # ~95% in un'approssimazione normale dei residui
 
example_pred = pred_express  # Roma -> Milano, Express, 5 kg (vedi sezione 3)
lower = example_pred - z * residual_std
upper = example_pred + z * residual_std
 
print(f"Deviazione standard dei residui (dataset sintetico): {residual_std:.4f} min")
print(f"Esempio (Roma->Milano, Express, 5kg): predizione={example_pred:.2f} min, "
      f"intervallo 95% approssimato=[{lower:.2f}, {upper:.2f}] min")