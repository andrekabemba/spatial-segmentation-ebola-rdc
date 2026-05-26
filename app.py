import streamlit as st
import pandas as pd
import pickle
import os
import sys

# Ajout du dossier parent pour l'import de src.models
sys.path.append(os.getcwd())
from src.models import EcoEbolaClustering

def load_trained_model():
    model_path = 'models/ebola_clustering_final.pkl'
    if not os.path.exists(model_path):
        st.error("Modèle introuvable ! Veuillez exécuter 'python main.py' d'abord.")
        return None
    return EcoEbolaClustering.load_model(model_path)

def main():
    st.set_page_config(page_title="Ebola Risk Predictor - RDC", layout="wide")
    
    st.title("🏥 Ebola Spatial Segmentation - Predictor")
    st.markdown("""
    Cette interface permet de prédire le **profil de vulnérabilité** d'une zone de santé (Ituri, Nord-Kivu ou autre province) 
    en utilisant les modèles d'apprentissage automatique entraînés sur les facteurs éco-épidémiologiques.
    """)

    model = load_trained_model()
    if not model:
        return

    # Barre latérale pour les entrées utilisateur
    st.sidebar.header("Paramètres de la Zone de Santé")
    
    province = st.sidebar.text_input("Province", "Ex: Sud-Kivu")
    zone_name = st.sidebar.text_input("Nom de la Zone de Santé", "Ex: Bukavu")
    
    st.sidebar.divider()
    
    total_cases = st.sidebar.number_input("Total des Cas (Historique/Prévu)", min_value=0, value=5)
    total_deaths = st.sidebar.number_input("Total des Décès", min_value=0, value=2)
    pop_density = st.sidebar.number_input("Densité de Population (hab/km²)", min_value=0, value=150)
    forest_cov = st.sidebar.slider("Couverture Forestière (%)", 0, 100, 45) / 100.0
    water_prox = st.sidebar.slider("Indice de Proximité Eau (0-1)", 0.0, 1.0, 0.5)
    lat = st.sidebar.number_input("Latitude", value=-2.5)
    lon = st.sidebar.number_input("Longitude", value=28.8)

    if st.sidebar.button("Prédire le Profil de Risque"):
        # Préparation des données pour la prédiction
        input_data = pd.DataFrame([{
            'total_cases': total_cases,
            'total_deaths': total_deaths,
            'population_density': pop_density,
            'forest_coverage_percent': forest_cov,
            'water_proximity_index': water_prox,
            'latitude': lat,
            'longitude': lon
        }])

        # Normalisation et Prédiction
        scaled_input = model.scaler.transform(input_data)
        cluster_id = model.kmeans.predict(scaled_input)[0]
        
        # Récupération du label (via une simulation rapide du mécanisme de labellisation)
        # On utilise un DataFrame temporaire pour réutiliser la logique label_clusters
        temp_df = input_data.copy()
        temp_df['cluster_kmeans'] = cluster_id
        
        # Pour que la labellisation fonctionne, il nous faut les profils du modèle original
        # Ici on utilise une version simplifiée de la logique de labellisation pour l'UI
        risk_profile = ""
        if pop_density > 500 and total_cases > 10:
            risk_profile = "🔴 Épicentre Urbain (Risque de propagation massive)"
        elif forest_cov > 0.6:
            risk_profile = "🟢 Réservoir Sylvatique (Risque d'émergence initiale)"
        elif water_prox > 0.7:
            risk_profile = "🔵 Interface Lacustre (Risque lié aux flux migratoires)"
        else:
            risk_profile = "🟡 Zone de Profil Mixte / Modéré"

        # Affichage des résultats
        st.subheader(f"Résultats pour la Zone : {zone_name} ({province})")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Cluster ID", cluster_id)
            st.info(f"**Profil Identifié :** {risk_profile}")
        
        with col2:
            st.write("**Récapitulatif des indicateurs :**")
            st.write(input_data)

        st.success("Analyse terminée. Ces résultats doivent être validés par des experts de santé publique.")

    # Affichage des données d'entraînement (Optionnel)
    if st.checkbox("Voir les zones de santé de référence (Données d'entraînement)"):
        if os.path.exists('data/segmentation_resultats.csv'):
            ref_df = pd.read_csv('data/segmentation_resultats.csv', sep=';')
            st.dataframe(ref_df)

if __name__ == "__main__":
    main()
