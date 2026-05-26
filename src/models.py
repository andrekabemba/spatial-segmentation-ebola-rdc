import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import pickle
import os

class EcoEbolaClustering:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.hierarchical = AgglomerativeClustering(n_clusters=self.n_clusters)
        self.features = [
            'total_cases', 'total_deaths', 'population_density', 
            'forest_coverage_percent', 'water_proximity_index', 
            'latitude', 'longitude'
        ]

    def fit_prepare_data(self, df):
        """Prend un DataFrame et retourne les données normalisées."""
        data_to_scale = df[self.features]
        scaled_data = self.scaler.fit_transform(data_to_scale)
        return scaled_data

    def apply_kmeans(self, df):
        """Applique K-Means et ajoute les labels au DataFrame."""
        scaled_data = self.fit_prepare_data(df)
        df['cluster_kmeans'] = self.kmeans.fit_predict(scaled_data)
        return df

    def apply_hierarchical(self, df):
        """Applique le Clustering Hiérarchique et ajoute les labels au DataFrame."""
        scaled_data = self.fit_prepare_data(df)
        df['cluster_hierarchical'] = self.hierarchical.fit_predict(scaled_data)
        return df

    def get_cluster_profiles(self, df, cluster_column):
        """Calcule les profils moyens pour chaque cluster."""
        return df.groupby(cluster_column)[self.features].mean()

    def label_clusters(self, df, cluster_column):
        """Attribue des noms descriptifs aux clusters basés sur leurs profils."""
        profiles = self.get_cluster_profiles(df, cluster_column)

        labels = {}
        for cluster_id in profiles.index:
            row = profiles.loc[cluster_id]

            # Logique de classification simple basée sur la description du projet
            if row['population_density'] > 500 and row['total_cases'] > 10:
                labels[cluster_id] = "Épicentre Urbain (Risque de propagation)"
            elif row['forest_coverage_percent'] > 0.6:
                labels[cluster_id] = "Réservoir Sylvatique (Risque d'émergence)"
            elif row['water_proximity_index'] > 0.7:
                labels[cluster_id] = "Interface Lacustre (Flux migratoires)"
            else:
                labels[cluster_id] = f"Zone de Profil Mixte (Cluster {cluster_id})"

        df[f'{cluster_column}_label'] = df[cluster_column].map(labels)
        return df

    def save_model(self, path='models/ebola_clustering_final.pkl'):

        """Sauvegarde le scaler et les modèles dans un fichier pickle."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'scaler': self.scaler,
            'kmeans': self.kmeans,
            'hierarchical': self.hierarchical,
            'features': self.features
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Modèles sauvegardés dans {path}")

    @staticmethod
    def load_model(path='models/ebola_clustering_model.pkl'):
        """Charge le scaler et les modèles à partir d'un fichier pickle."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = EcoEbolaClustering()
        instance.scaler = model_data['scaler']
        instance.kmeans = model_data['kmeans']
        instance.hierarchical = model_data['hierarchical']
        instance.features = model_data['features']
        return instance
