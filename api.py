from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import os
import sys

# Ajout du dossier parent pour l'import de src.models
sys.path.append(os.getcwd())
from src.models import EcoEbolaClustering

app = FastAPI(title="Ebola Prediction API", version="1.0")

# Chargement global du modèle
MODEL_PATH = 'models/ebola_clustering_final.pkl'
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Modèle introuvable. Exécutez 'python main.py' d'abord.")

model = EcoEbolaClustering.load_model(MODEL_PATH)

class HealthZoneData(BaseModel):
    total_cases: float
    total_deaths: float
    population_density: float
    forest_coverage_percent: float
    water_proximity_index: float
    latitude: float
    longitude: float

@app.post("/predict")
async def predict(data: HealthZoneData):
    try:
        # Conversion Pydantic -> DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # Normalisation
        scaled_data = model.scaler.transform(input_df)
        
        # Prédiction du cluster
        cluster_id = int(model.kmeans.predict(scaled_data)[0])
        
        # Calcul du Risque de Contamination (Score pondéré 0-100)
        # On base le risque sur la densité, les cas et la proximité des risques
        base_risk = (data.total_cases * 0.4) + (data.population_density / 50) + (data.forest_coverage_percent * 20)
        risk_percentage = min(98.5, max(5.0, base_risk)) # Cap entre 5% et 98.5%
        
        row = data.dict()
        
        # Structure de réponse professionnelle pour centre de recherche/traitement
        result = {
            "cluster_id": cluster_id,
            "risk_percentage": round(risk_percentage, 1),
            "scientific_rationale": "",
            "protocols": {
                "barrier_measures": [],
                "hygiene_measures": [],
                "medical_surveillance": []
            },
            "color_code": ""
        }

        if row['population_density'] > 500 and row['total_cases'] > 10:
            result["profile_label"] = "Épicentre Urbain à Haute Transmission"
            result["scientific_rationale"] = "La corrélation entre forte densité démographique et charge virale active indique une propagation interhumaine exponentielle."
            result["color_code"] = "red"
            result["protocols"]["barrier_measures"] = ["Port du masque FFP2 obligatoire en public", "Interdiction des rassemblements > 10 personnes", "Distanciation sociale stricte (2m)"]
            result["protocols"]["hygiene_measures"] = ["Désinfection systématique des lieux publics", "Installation de stations de lavage PCI (Prévention et Contrôle des Infections)", "Gestion sécurisée des déchets biomédicaux"]
            result["protocols"]["medical_surveillance"] = ["Vaccination réactive (Ring Vaccination)", "Recherche active des contacts (Contact Tracing 24h/24)", "Triage avancé aux entrées de la zone"]
            
        elif row['forest_coverage_percent'] > 0.6:
            result["profile_label"] = "Zone de Réservoir Sylvatique (Risque Zoonotique)"
            result["scientific_rationale"] = "L'indice de couverture forestière suggère une interface Homme-Faune critique favorisant le spillover viral."
            result["color_code"] = "green"
            result["protocols"]["barrier_measures"] = ["Éviter tout contact avec la faune sauvage", "Protection cutanée complète lors des activités agricoles", "Limiter les incursions en forêt profonde"]
            result["protocols"]["hygiene_measures"] = ["Interdiction de manipulation de viande de brousse", "Traitement thermique rigoureux des aliments", "Lavage des mains après manipulation d'outils forestiers"]
            result["protocols"]["medical_surveillance"] = ["Monitoring des mortalités animales suspectes", "Surveillance biologique des chasseurs", "Stockage pré-positionné de kits de protection (EPI)"]

        elif row['water_proximity_index'] > 0.7:
            result["profile_label"] = "Interface Lacustre et Flux Migratoires"
            result["scientific_rationale"] = "La proximité hydrologique couple le risque sanitaire aux flux économiques et migratoires transfrontaliers."
            result["color_code"] = "blue"
            result["protocols"]["barrier_measures"] = ["Contrôle thermique aux débarcadères", "Masque obligatoire dans les embarcations", "Enregistrement des voyageurs"]
            result["protocols"]["hygiene_measures"] = ["Chloration systématique de l'eau aux ports", "Désinfection des embarcations de commerce", "Campagnes d'hygiène sur les marchés de poissons"]
            result["protocols"]["medical_surveillance"] = ["Postes de santé mobiles aux frontières d'eau", "Alerte précoce sur les flux en provenance de zones actives", "Collaboration sanitaire inter-provinces"]

        else:
            result["profile_label"] = "Zone de Vigilance / Profil Mixte"
            result["scientific_rationale"] = "Profil équilibré nécessitant un maintien des standards de prévention de base."
            result["color_code"] = "orange"
            result["protocols"]["barrier_measures"] = ["Respect des gestes barrières standard", "Utilisation de masques dans les centres de santé"]
            result["protocols"]["hygiene_measures"] = ["Maintien des points d'eau potable", "Sensibilisation à l'hygiène de base"]
            result["protocols"]["medical_surveillance"] = ["Veille épidémiologique hebdomadaire", "Reporting des cas suspects via système d'alerte national"]

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # On force le port 8000 pour la cohérence avec le frontend
    print("Démarrage du Serveur de Prédiction (Backend)...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
