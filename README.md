# -LogiFastModelDeploy
Sviluppo e deploy di un modello di Machine Learning per la previsione dei tempi di consegna. Il progetto espone il modello tramite API REST Flask e integra principi MLOps per versionamento, testing, monitoraggio, gestione delle performance e strategie di retraining e rollback.

# 🚚 Deploy di un modello predittivo per la stima dei tempi di consegna

## 📋 Descrizione

**LogiFast Solutions** è un'azienda che gestisce consegne urbane e interurbane per e-commerce e rivenditori.

Il progetto implementa una soluzione di **Machine Learning per la previsione del tempo di consegna di un ordine**, dal momento della presa in carico fino alla consegna finale.

Il modello predittivo viene esposto tramite una **API REST sviluppata con Flask**, consentendo a sistemi esterni, operatori e applicazioni aziendali di ottenere previsioni in tempo reale.

Il progetto integra inoltre principi fondamentali di **MLOps**, con particolare attenzione a:

- versionamento di modello e dataset;
- tracciamento dei metadati di training;
- testing e validazione;
- monitoraggio delle predizioni;
- rilevazione del data/concept drift;
- alerting;
- strategie di rollback;
- pianificazione del retraining.

L'obiettivo è simulare un flusso realistico di **Machine Learning in produzione**, mantenendo l'implementazione focalizzata sugli aspetti richiesti dal progetto senza introdurre un'infrastruttura MLOps completa.

---

## 🎯 Obiettivi

Gli obiettivi principali del progetto sono:

1. Analizzare e validare il modello predittivo fornito.
2. Definire una metodologia di valutazione delle performance.
3. Implementare un'API REST con Flask.
4. Esporre endpoint per predizioni singole e batch.
5. Implementare endpoint di health check e informazioni sul modello.
6. Definire una strategia di versionamento per modello e dataset.
7. Progettare un workflow MLOps per training, testing e deployment.
8. Definire un sistema di monitoraggio delle performance e del drift.
9. Definire strategie di rollback e retraining.
10. Documentare architettura, API, metriche e modalità di utilizzo.

---

## 🧠 Modello di Machine Learning

Il modello predittivo è fornito in formato **Pickle (`.pkl`)**.

Il file può essere scaricato dal repository del progetto e caricato tramite Python:

```python
import pickle

with open("delivery.pkl", "rb") as f:
    model = pickle.load(f)
