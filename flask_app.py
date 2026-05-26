import os
from flask import Flask, render_template, request
import requests

app = Flask(__name__)

# URL de l'API FastAPI - Utilise une variable d'environnement sur Render ou localhost par défaut
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")

def check_backend():
    """Vérifie si le backend FastAPI est en ligne."""
    try:
        health_url = API_URL.replace("/predict", "/health")
        response = requests.get(health_url, timeout=2)
        return response.status_code == 200
    except:
        return False

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    backend_status = check_backend()
    
    if not backend_status:
        error = "⚠️ Le Serveur Backend (FastAPI) est éteint. Lancez 'python api.py' dans un autre terminal."

    if request.method == "POST" and backend_status:
        try:
            # Mapping des réponses OUI/NON et Localisation
            is_forest = request.form.get("is_forest") == "yes"
            is_water = request.form.get("is_water") == "yes"
            location = request.form.get("location_preset")

            # Valeurs par défaut intelligentes
            forest_val = 0.85 if is_forest else 0.15
            water_val = 0.90 if is_water else 0.20
            
            # Coordonnées moyennes par région
            coords = {
                "ituri": (1.15, 29.5),
                "nkivu_north": (0.45, 29.4),
                "nkivu_south": (-1.65, 29.2),
                "other": (0.0, 28.0)
            }
            lat, lon = coords.get(location, (0.0, 28.0))

            payload = {
                "total_cases": float(request.form["total_cases"]),
                "total_deaths": float(request.form["total_deaths"]),
                "population_density": float(request.form["population_density"]),
                "forest_coverage_percent": forest_val,
                "water_proximity_index": water_val,
                "latitude": lat,
                "longitude": lon
            }
            
            # Appel à l'API FastAPI
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                prediction = response.json()
                prediction["zone"] = request.form["zone_name"]
                prediction["province"] = request.form["province"]
            else:
                error = f"Erreur API : {response.text}"
        except Exception as e:
            error = f"Erreur de connexion : {str(e)}"
            
    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
