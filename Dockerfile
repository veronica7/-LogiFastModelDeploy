# Immagine base leggera con Python
FROM python:3.11-slim

# Directory di lavoro nel container
WORKDIR /app

# Copia solo requirements per sfruttare la cache dei layer Docker
COPY requirements.txt .

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto del codice, inclusi main.py, DTO.py e la cartella model/
COPY . .

# Espone la porta usata da Flask
EXPOSE 8000

# Variabili d'ambiente utili in produzione
ENV FLASK_ENV=production \
    PYTHONUNBUFFERED=1

# Avvio dell'applicazione
CMD ["python", "main.py"]