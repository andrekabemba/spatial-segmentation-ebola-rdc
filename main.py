import os
import pandas as pd
from src.models import EcoEbolaClustering

def run_pipeline():
    print("--- Démarrage du Pipeline Spatial Segmentation Ebola ---")
    
    # 1. Chargement des données
    data_path = 'data/donnees_ebola_enrichies.csv'
    if not os.path.exists(data_path):
        print(f"Erreur : Le fichier {data_path} est introuvable.")
        return

    df = pd.read_csv(data_path, sep=';')
    
    # 2. Prétraitement / Agrégation
    print("Agrégation des données par zone de santé...")
    agg_rules = {
        'province': 'first',
        'total_cases': 'max',
        'total_deaths': 'max',
        'population_density': 'first',
        'forest_coverage_percent': 'first',
        'water_proximity_index': 'first',
        'latitude': 'first',
        'longitude': 'first'
    }
    df_zone = df.groupby('health_zone').agg(agg_rules).reset_index()
    
    # Sauvegarde des données préparées
    df_zone.to_csv('data/donnees_zones_preparees.csv', index=False, sep=';')
    print(f"Données agrégées : {df_zone.shape[0]} zones identifiées.")

    # 3. Clustering
    print("Application des algorithmes de clustering (K-Means & CHA)...")
    pipeline = EcoEbolaClustering(n_clusters=3)
    
    df_results = pipeline.apply_kmeans(df_zone)
    df_results = pipeline.apply_hierarchical(df_results)
    
    # 4. Profilage et Labellisation
    print("Labellisation des clusters...")
    df_results = pipeline.label_clusters(df_results, 'cluster_kmeans')
    
    profils = pipeline.get_cluster_profiles(df_results, 'cluster_kmeans')
    print("\n--- Profils des Clusters (Moyennes) ---")
    print(profils.round(2))
    
    print("\n--- Exemple de Segmentation ---")
    print(df_results[['health_zone', 'cluster_kmeans', 'cluster_kmeans_label']].head(10))
    
    # 5. Sauvegarde du modèle final
    os.makedirs('models', exist_ok=True)
    pipeline.save_model('models/ebola_clustering_final.pkl')
    
    # Sauvegarde des résultats
    df_results.to_csv('data/segmentation_resultats.csv', index=False, sep=';')
    print("\n--- Pipeline terminé avec succès ! ---")
    print("Modèle sauvegardé dans 'models/ebola_clustering_final.pkl'")
    print("Résultats sauvegardés dans 'data/segmentation_resultats.csv'")

if __name__ == "__main__":
    run_pipeline()
