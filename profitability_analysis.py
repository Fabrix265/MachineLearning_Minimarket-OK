#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Evaluación de Rentabilidad
Análisis de clustering para productos basado en patrones de venta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class ProfitabilityAnalysis:
    def __init__(self, df):
        self.df = df.copy()
        self.df_pivot = None
        self.clusters = None
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=4, random_state=42)
        
    def prepare_data(self):
        """Prepara los datos para el análisis de clustering"""
        print("Preparando datos para análisis de rentabilidad...")
        
        # Convertir fecha
        self.df['Fecha'] = pd.to_datetime(self.df['Fecha'])
        self.df['Mes_Period'] = self.df['Fecha'].dt.to_period('M')
        
        # Agrupar por producto y mes
        df_agg = self.df.groupby(['Producto', 'Mes_Period'])['Cantidad'].sum().reset_index()
        
        # Pivoteo: productos en filas, meses en columnas
        self.df_pivot = df_agg.pivot(index='Producto', columns='Mes_Period', values='Cantidad').fillna(0)
        
        print(f" Datos preparados: {len(self.df_pivot)} productos, {len(self.df_pivot.columns)} períodos")
        
    def perform_clustering(self):
        """Ejecuta el análisis de clustering"""
        print("Ejecutando análisis de clustering...")
        
        # Escalado de datos
        X_scaled = self.scaler.fit_transform(self.df_pivot)
        
        # Aplicar KMeans
        self.clusters = self.kmeans.fit_predict(X_scaled)
        self.df_pivot['cluster'] = self.clusters
        
        # Generar rentabilidad simulada (en un caso real usarías datos reales)
        np.random.seed(42)
        rentabilidad = pd.DataFrame({
            'Producto': self.df_pivot.index,
            'Rentabilidad': np.random.uniform(50, 3000, len(self.df_pivot))
        }).set_index('Producto')
        
        # Unir rentabilidad con clusters
        self.df_pivot = self.df_pivot.join(rentabilidad)
        
        print(" Clustering completado")
        
    def show_cluster_analysis(self):
        """Muestra el análisis de clusters"""
        print("\n" + "="*60)
        print("           ANÁLISIS DE RENTABILIDAD POR CLUSTERS")
        print("="*60)
        
        # Resumen de rentabilidad por cluster
        rent_summary = self.df_pivot.groupby('cluster')['Rentabilidad'].describe()
        
        print("\nRESUMEN DE RENTABILIDAD POR CLUSTER:")
        print("-" * 50)
        for cluster in sorted(self.df_pivot['cluster'].unique()):
            cluster_data = rent_summary.loc[cluster]
            num_productos = len(self.df_pivot[self.df_pivot['cluster'] == cluster])
            print(f"\nCluster {cluster} ({num_productos} productos):")
            print(f"  Rentabilidad promedio: S/ {cluster_data['mean']:.2f}")
            print(f"  Rentabilidad mínima:   S/ {cluster_data['min']:.2f}")
            print(f"  Rentabilidad máxima:   S/ {cluster_data['max']:.2f}")
            print(f"  Desviación estándar:   S/ {cluster_data['std']:.2f}")
        
        # Interpretación de clusters
        self.interpret_clusters()
        
    def interpret_clusters(self):
        """Proporciona interpretación de los clusters"""
        print("\n" + "="*60)
        print("           INTERPRETACIÓN DE CLUSTERS")
        print("="*60)
        
        cluster_interpretations = {
            0: "Productos de Alta Rotación - Rentabilidad Media",
            1: "Productos Premium - Alta Rentabilidad",
            2: "Productos de Baja Rotación - Rentabilidad Variable",
            3: "Productos Estacionales - Rentabilidad Fluctuante"
        }
        
        for cluster in sorted(self.df_pivot['cluster'].unique()):
            productos_cluster = self.df_pivot[self.df_pivot['cluster'] == cluster]
            rent_promedio = productos_cluster['Rentabilidad'].mean()
            
            print(f"\n CLUSTER {cluster}: {cluster_interpretations.get(cluster, 'Cluster Especial')}")
            print(f"   Rentabilidad promedio: S/ {rent_promedio:.2f}")
            print(f"   Productos ({len(productos_cluster)}):")
            
            # Mostrar top 5 productos del cluster
            top_productos = productos_cluster.nlargest(5, 'Rentabilidad')
            for i, (producto, data) in enumerate(top_productos.iterrows(), 1):
                print(f"   {i}. {producto} (S/ {data['Rentabilidad']:.2f})")
            
            if len(productos_cluster) > 5:
                print(f"   ... y {len(productos_cluster) - 5} productos más")
                
    def show_all_products_by_cluster(self):
        """Muestra todos los productos agrupados por cluster"""
        print("\n" + "="*60)
        print("         PRODUCTOS AGRUPADOS POR CLUSTER")
        print("="*60)
        
        for cluster in sorted(self.df_pivot['cluster'].unique()):
            productos_cluster = self.df_pivot[self.df_pivot['cluster'] == cluster].index.tolist()
            print(f"\n CLUSTER {cluster} - {len(productos_cluster)} productos:")
            print("-" * 40)
            
            # Ordenar productos alfabéticamente
            productos_cluster.sort()
            
            # Mostrar en columnas para mejor visualización
            for i, producto in enumerate(productos_cluster):
                if i % 3 == 0 and i > 0:
                    print()
                print(f"{producto:<25}", end=" ")
            print()  # Nueva línea al final
            
    def plot_clusters(self):
        """Genera visualización de clusters usando PCA"""
        print("\nGenerando visualización de clusters...")
        
        # Datos sin la columna cluster y rentabilidad para PCA
        cluster_data = self.df_pivot.drop(['cluster', 'Rentabilidad'], axis=1)
        X_scaled = self.scaler.fit_transform(cluster_data)
        
        # PCA para visualización 2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Crear gráfico
        plt.figure(figsize=(12, 8))
        colors = plt.cm.get_cmap('tab10', len(set(self.clusters)))
        
        for i, cluster in enumerate(sorted(set(self.clusters))):
            mask = self.df_pivot['cluster'] == cluster
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       s=80, alpha=0.7, color=colors(i), 
                       label=f'Cluster {cluster}', edgecolors='black', linewidth=0.5)
        
        plt.title('Análisis de Clusters - Productos por Patrón de Ventas\n(Visualización PCA 2D)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]:.1%} varianza)', fontsize=12)
        plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]:.1%} varianza)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Información adicional sobre PCA
        print(f"\n Varianza explicada por componentes PCA:")
        print(f"   Componente 1: {pca.explained_variance_ratio_[0]:.1%}")
        print(f"   Componente 2: {pca.explained_variance_ratio_[1]:.1%}")
        print(f"   Total explicada: {sum(pca.explained_variance_ratio_):.1%}")
        
    def generate_recommendations(self):
        """Genera recomendaciones basadas en el análisis"""
        print("\n" + "="*60)
        print("              RECOMENDACIONES ESTRATÉGICAS")
        print("="*60)
        
        recommendations = {
            0: [
                "• Mantener stock constante de estos productos",
                "• Considerar promociones para aumentar márgenes",
                "• Evaluar optimización de costos de inventario"
            ],
            1: [
                "• Enfocar esfuerzos de marketing en estos productos",
                "• Mantener calidad premium del servicio",
                "• Considerar expansión de líneas similares"
            ],
            2: [
                "• Revisar estrategia de precios",
                "• Evaluar descontinuación de productos de muy baja rotación",
                "• Implementar promociones específicas"
            ],
            3: [
                "• Planificar stock según estacionalidad",
                "• Desarrollar campañas de marketing estacionales",
                "• Considerar productos complementarios"
            ]
        }
        
        for cluster in sorted(self.df_pivot['cluster'].unique()):
            num_productos = len(self.df_pivot[self.df_pivot['cluster'] == cluster])
            rent_promedio = self.df_pivot[self.df_pivot['cluster'] == cluster]['Rentabilidad'].mean()
            
            print(f"\n CLUSTER {cluster} ({num_productos} productos - Rentabilidad: S/ {rent_promedio:.2f}):")
            
            if cluster in recommendations:
                for rec in recommendations[cluster]:
                    print(f"   {rec}")
            else:
                print("   • Requiere análisis específico adicional")
    
    def run_analysis(self):
        """Ejecuta el análisis completo de rentabilidad"""
        try:
            self.prepare_data()
            self.perform_clustering()
            self.show_cluster_analysis()
            self.show_all_products_by_cluster()
            self.plot_clusters()
            self.generate_recommendations()
            
            # Preguntar si quiere ver detalles adicionales
            print("\n" + "="*60)
            while True:
                print("\nOpciones adicionales:")
                print("1. Ver productos específicos de un cluster")
                print("2. Comparar rentabilidad entre clusters")
                print("3. Volver al menú principal")
                
                opcion = input("\nSeleccione una opción (1-3): ").strip()
                
                if opcion == '1':
                    self.show_cluster_details()
                elif opcion == '2':
                    self.compare_clusters()
                elif opcion == '3':
                    break
                else:
                    print("Opción no válida.")
                    
        except Exception as e:
            print(f"Error en análisis de rentabilidad: {e}")
            
    def show_cluster_details(self):
        """Muestra detalles específicos de un cluster"""
        try:
            cluster_num = int(input(f"Ingrese número de cluster (0-{max(self.clusters)}): "))
            if cluster_num not in self.df_pivot['cluster'].values:
                print("Cluster no válido.")
                return
                
            cluster_products = self.df_pivot[self.df_pivot['cluster'] == cluster_num]
            
            print(f"\n DETALLES DEL CLUSTER {cluster_num}:")
            print("-" * 40)
            
            for producto, data in cluster_products.iterrows():
                print(f"• {producto}: S/ {data['Rentabilidad']:.2f}")
                
        except ValueError:
            print("Por favor ingrese un número válido.")
            
    def compare_clusters(self):
        """Compara métricas entre clusters"""
        print("\n COMPARACIÓN ENTRE CLUSTERS:")
        print("-" * 50)
        
        comparison = self.df_pivot.groupby('cluster').agg({
            'Rentabilidad': ['count', 'mean', 'std', 'min', 'max']
        }).round(2)
        
        comparison.columns = ['Cantidad', 'Promedio', 'Desv_Std', 'Mínimo', 'Máximo']
        print(comparison)


if __name__ == "__main__":
    # Código de prueba
    print("Módulo de Evaluación de Rentabilidad - Modo de prueba")