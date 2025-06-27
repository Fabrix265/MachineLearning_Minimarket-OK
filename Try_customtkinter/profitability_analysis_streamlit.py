#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Evaluación de Rentabilidad - Versión Streamlit
Análisis de clustering para productos basado en patrones de venta
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class ProfitabilityAnalysisStreamlit:
    def __init__(self, df):
        self.df = df.copy()
        self.df_pivot = None
        self.clusters = None
        self.scaler = StandardScaler()
        self.kmeans = None
        self.pca = PCA(n_components=2)
        
    def prepare_data(self):
        """Prepara los datos para el análisis de clustering"""
        # Convertir fecha si no está convertida
        if not pd.api.types.is_datetime64_any_dtype(self.df['Fecha']):
            self.df['Fecha'] = pd.to_datetime(self.df['Fecha'])

        self.df['Mes_Period'] = self.df['Fecha'].dt.to_period('M')

        # Agrupar por producto y mes
        df_agg = self.df.groupby(['Producto', 'Mes_Period'])['Cantidad'].sum().reset_index()

        # Pivoteo: productos en filas, meses en columnas
        self.df_pivot = df_agg.pivot(index='Producto', columns='Mes_Period', values='Cantidad').fillna(0)

        # Convertir nombres de columnas a string para evitar errores con scikit-learn
        self.df_pivot.columns = self.df_pivot.columns.astype(str)

        return True
    
    def perform_clustering(self, n_clusters=4):
        """Ejecuta el análisis de clustering"""
        if self.df_pivot is None:
            self.prepare_data()

        # Escalado de datos
        X_scaled = self.scaler.fit_transform(self.df_pivot)

        # Aplicar KMeans
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = self.kmeans.fit_predict(X_scaled)
        self.df_pivot['cluster'] = self.clusters

        # Generar rentabilidad simulada basada en patrones reales
        np.random.seed(42)

        # Calcular métricas base para cada producto
        producto_stats = self.df.groupby('Producto').agg({
            'Cantidad': ['sum', 'mean', 'std'],
            'PrecioTotal': ['sum', 'mean'],
            'PrecioUnitario': 'mean'
        }).fillna(0)

        # Aplanar columnas
        producto_stats.columns = ['_'.join(col) for col in producto_stats.columns]

        # Simular rentabilidad más realista basada en ventas
        rentabilidad_base = producto_stats['PrecioTotal_sum'] * np.random.uniform(0.1, 0.4, len(producto_stats))
        rentabilidad_variacion = np.random.uniform(0.8, 1.2, len(producto_stats))

        rentabilidad = pd.DataFrame({
            'Producto': producto_stats.index,
            'Rentabilidad': rentabilidad_base * rentabilidad_variacion,
            'Ventas_Totales': producto_stats['PrecioTotal_sum'],
            'Cantidad_Total': producto_stats['Cantidad_sum'],
            'Precio_Promedio': producto_stats['PrecioUnitario_mean']
        }).set_index('Producto')

        # Unir rentabilidad con clusters
        self.df_pivot = self.df_pivot.join(rentabilidad)

        # ✅ Guardar en session_state
        st.session_state.df_pivot = self.df_pivot
        st.session_state.clustering_done = True

        return True

    def find_optimal_clusters(self, max_clusters=10):
        """Encuentra el número óptimo de clusters usando el método del codo"""
        if self.df_pivot is None:
            self.prepare_data()
        
        X_scaled = self.scaler.fit_transform(self.df_pivot.drop(['cluster'], axis=1, errors='ignore'))
        
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
        return K_range, inertias, silhouette_scores
    
    def plot_cluster_optimization(self):
        """Muestra gráficos para determinar el número óptimo de clusters"""
        K_range, inertias, silhouette_scores = self.find_optimal_clusters()
        
        # Crear subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Método del Codo', 'Puntuación Silhouette'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Gráfico del codo
        fig.add_trace(
            go.Scatter(x=list(K_range), y=inertias, mode='lines+markers',
                      name='Inercia', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Gráfico silhouette
        fig.add_trace(
            go.Scatter(x=list(K_range), y=silhouette_scores, mode='lines+markers',
                      name='Silhouette Score', line=dict(color='red', width=2)),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Número de Clusters", row=1, col=1)
        fig.update_xaxes(title_text="Número de Clusters", row=1, col=2)
        fig.update_yaxes(title_text="Inercia", row=1, col=1)
        fig.update_yaxes(title_text="Puntuación Silhouette", row=1, col=2)
        
        fig.update_layout(
            title_text="Optimización del Número de Clusters",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def plot_clusters_pca(self):
        """Genera visualización de clusters usando PCA"""
        if 'cluster' not in self.df_pivot.columns:
            return None
        
        # Datos para PCA (sin cluster y métricas de rentabilidad)
        cluster_data = self.df_pivot.drop(['cluster', 'Rentabilidad', 'Ventas_Totales', 
                                          'Cantidad_Total', 'Precio_Promedio'], axis=1)
        X_scaled = self.scaler.fit_transform(cluster_data)
        
        # Aplicar PCA
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Crear DataFrame para el gráfico
        pca_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': self.df_pivot['cluster'].astype(str),
            'Producto': self.df_pivot.index,
            'Rentabilidad': self.df_pivot['Rentabilidad']
        })
        
        # Crear gráfico interactivo
        fig = px.scatter(
            pca_df, x='PC1', y='PC2', color='Cluster',
            hover_data=['Producto', 'Rentabilidad'],
            title=f'Análisis de Clusters - Visualización PCA<br>'
                  f'<sub>Varianza explicada: PC1={self.pca.explained_variance_ratio_[0]:.1%}, '
                  f'PC2={self.pca.explained_variance_ratio_[1]:.1%}</sub>',
            labels={
                'PC1': f'Componente Principal 1 ({self.pca.explained_variance_ratio_[0]:.1%})',
                'PC2': f'Componente Principal 2 ({self.pca.explained_variance_ratio_[1]:.1%})'
            }
        )
        
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
        fig.update_layout(height=500)
        
        return fig
    
    def plot_cluster_profitability(self):
        """Gráfico de rentabilidad por cluster"""
        if 'Rentabilidad' not in self.df_pivot.columns:
            return None
        
        # Preparar datos
        cluster_data = []
        for cluster in sorted(self.df_pivot['cluster'].unique()):
            cluster_products = self.df_pivot[self.df_pivot['cluster'] == cluster]
            for _, product in cluster_products.iterrows():
                cluster_data.append({
                    'Cluster': f'Cluster {cluster}',
                    'Producto': product.name,
                    'Rentabilidad': product['Rentabilidad']
                })
        
        cluster_df = pd.DataFrame(cluster_data)
        
        # Crear box plot
        fig = px.box(
            cluster_df, x='Cluster', y='Rentabilidad',
            title='Distribución de Rentabilidad por Cluster',
            points="outliers"
        )
        
        fig.update_layout(height=400)
        return fig
    
    def get_cluster_summary(self):
        """Genera resumen estadístico de clusters"""
        if 'Rentabilidad' not in self.df_pivot.columns:
            return None
        
        summary_stats = []
        for cluster in sorted(self.df_pivot['cluster'].unique()):
            cluster_data = self.df_pivot[self.df_pivot['cluster'] == cluster]
            
            stats = {
                'Cluster': cluster,
                'Productos': len(cluster_data),
                'Rentabilidad_Promedio': cluster_data['Rentabilidad'].mean(),
                'Rentabilidad_Mediana': cluster_data['Rentabilidad'].median(),
                'Rentabilidad_Std': cluster_data['Rentabilidad'].std(),
                'Rentabilidad_Min': cluster_data['Rentabilidad'].min(),
                'Rentabilidad_Max': cluster_data['Rentabilidad'].max(),
                'Ventas_Promedio': cluster_data['Ventas_Totales'].mean(),
                'Cantidad_Promedio': cluster_data['Cantidad_Total'].mean()
            }
            summary_stats.append(stats)
        
        return pd.DataFrame(summary_stats)
    
    def get_cluster_interpretations(self):
        """Proporciona interpretaciones y recomendaciones por cluster"""
        interpretations = {}
        
        if 'Rentabilidad' not in self.df_pivot.columns:
            return interpretations
        
        summary = self.get_cluster_summary()
        
        for _, row in summary.iterrows():
            cluster = int(row['Cluster'])
            rent_promedio = row['Rentabilidad_Promedio']
            ventas_promedio = row['Ventas_Promedio']
            
            # Clasificar cluster basado en rentabilidad y ventas
            if rent_promedio > summary['Rentabilidad_Promedio'].median() and ventas_promedio > summary['Ventas_Promedio'].median():
                categoria = "⭐ Productos Estrella"
                descripcion = "Alta rentabilidad y altas ventas"
                recomendaciones = [
                    "Mantener estrategia actual",
                    "Considerar aumentar inventario",
                    "Enfocar marketing en productos similares"
                ]
            elif rent_promedio > summary['Rentabilidad_Promedio'].median():
                categoria = "💎 Productos Premium"
                descripcion = "Alta rentabilidad, ventas moderadas"
                recomendaciones = [
                    "Impulsar campañas de marketing",
                    "Mantener calidad premium",
                    "Evaluar estrategias de up-selling"
                ]
            elif ventas_promedio > summary['Ventas_Promedio'].median():
                categoria = "🔄 Productos de Volumen"
                descripcion = "Altas ventas, rentabilidad moderada"
                recomendaciones = [
                    "Optimizar costos operativos",
                    "Considerar promociones por volumen",
                    "Evaluar mejoras en márgenes"
                ]
            else:
                categoria = "⚠️ Productos de Revisión"
                descripcion = "Rentabilidad y ventas por debajo del promedio"
                recomendaciones = [
                    "Revisar estrategia de precios",
                    "Evaluar descontinuación",
                    "Implementar promociones específicas"
                ]
            
            interpretations[cluster] = {
                'categoria': categoria,
                'descripcion': descripcion,
                'recomendaciones': recomendaciones,
                'metricas': {
                    'productos': int(row['Productos']),
                    'rentabilidad_promedio': rent_promedio,
                    'ventas_promedio': ventas_promedio
                }
            }
        
        return interpretations
    
    def show_interface(self):
        """Interfaz principal de Streamlit para el análisis de rentabilidad"""
        st.markdown("### 🎯 Análisis de Rentabilidad por Clustering")
        # Recuperar df_pivot si ya fue generado
        if st.session_state.get('df_pivot') is not None:
            self.df_pivot = st.session_state.df_pivot

        # Inicializar estado
        if 'profit_analysis_done' not in st.session_state:
            st.session_state.profit_analysis_done = False

        # Sidebar para configuración
        with st.sidebar:
            st.markdown("#### ⚙️ Configuración del Análisis")
            n_clusters = st.slider("Número de clusters", min_value=2, max_value=8, value=4)

            if st.button("🔄 Ejecutar Análisis", type="primary"):
                with st.spinner("Procesando datos y ejecutando clustering..."):
                    success = self.perform_clustering(n_clusters)
                    if success:
                        st.session_state.profit_analysis_done = True
                        st.success("✅ Análisis completado")

            if st.button("🔁 Reiniciar análisis"):
                st.session_state.profit_analysis_done = False

        # Mostrar análisis solo si ya se ejecutó
        if not st.session_state.profit_analysis_done:
            st.info("👆 Configure los parámetros en la barra lateral y ejecute el análisis para comenzar.")
            
            # Mostrar vista previa de datos
            with st.expander("📊 Vista Previa de Datos"):
                st.write("**Estructura de datos disponible:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Productos", self.df['Producto'].nunique())
                    st.metric("Total Registros", len(self.df))
                with col2:
                    st.metric("Rango de Fechas", 
                            f"{self.df['Fecha'].min().strftime('%Y-%m')} a {self.df['Fecha'].max().strftime('%Y-%m')}")
                    st.metric("Categorías", self.df['Categoría'].nunique())
            return

        # Mostrar resultados del análisis
        st.markdown("#### 📈 Resultados del Análisis")

        # Tabs para organizar contenido
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Resumen Ejecutivo", 
            "🎯 Análisis por Clusters", 
            "📈 Visualizaciones", 
            "🔍 Exploración Detallada"
        ])

        with tab1:
            self.show_executive_summary()

        with tab2:
            self.show_cluster_analysis()

        with tab3:
            self.show_visualizations()

        with tab4:
            self.show_detailed_exploration()

    
    def show_executive_summary(self):
        """Muestra resumen ejecutivo"""
        summary = self.get_cluster_summary()
        
        st.markdown("#### 🎯 Resumen Ejecutivo")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Productos Analizados",
                f"{len(self.df_pivot):,}"
            )
        
        with col2:
            st.metric(
                "Clusters Identificados",
                f"{len(summary)}"
            )
        
        with col3:
            st.metric(
                "Rentabilidad Promedio",
                f"S/ {summary['Rentabilidad_Promedio'].mean():,.0f}"
            )
        
        with col4:
            total_rentabilidad = summary['Rentabilidad_Promedio'].sum() * summary['Productos']
            st.metric(
                "Rentabilidad Total Estimada",
                f"S/ {total_rentabilidad.sum():,.0f}"
            )
        
        # Tabla resumen
        st.markdown("#### 📋 Resumen por Cluster")
        
        # Formatear tabla para mejor visualización
        display_summary = summary.copy()
        display_summary['Rentabilidad_Promedio'] = display_summary['Rentabilidad_Promedio'].apply(lambda x: f"S/ {x:,.0f}")
        display_summary['Ventas_Promedio'] = display_summary['Ventas_Promedio'].apply(lambda x: f"S/ {x:,.0f}")
        display_summary['Rentabilidad_Std'] = display_summary['Rentabilidad_Std'].apply(lambda x: f"S/ {x:,.0f}")
        
        # Renombrar columnas
        display_summary.columns = [
            'Cluster', 'Productos', 'Rent. Promedio', 'Rent. Mediana',
            'Desv. Estándar', 'Rent. Mínima', 'Rent. Máxima',
            'Ventas Promedio', 'Cantidad Promedio'
        ]
        
        st.dataframe(display_summary, use_container_width=True)
        
        # Insights principales
        st.markdown("#### 💡 Insights Principales")
        interpretations = self.get_cluster_interpretations()
        
        for cluster, data in interpretations.items():
            with st.expander(f"Cluster {cluster}: {data['categoria']}"):
                st.write(f"**{data['descripcion']}**")
                st.write(f"📊 {data['metricas']['productos']} productos | "
                        f"Rentabilidad promedio: S/ {data['metricas']['rentabilidad_promedio']:,.0f}")
    
    def show_cluster_analysis(self):
        """Muestra análisis detallado por clusters"""
        interpretations = self.get_cluster_interpretations()
        
        for cluster in sorted(interpretations.keys()):
            data = interpretations[cluster]
            
            st.markdown(f"### {data['categoria']} (Cluster {cluster})")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**{data['descripcion']}**")
                
                st.markdown("**🎯 Recomendaciones Estratégicas:**")
                for rec in data['recomendaciones']:
                    st.write(f"• {rec}")
            
            with col2:
                st.metric("Productos", data['metricas']['productos'])
                st.metric("Rentabilidad Prom.", f"S/ {data['metricas']['rentabilidad_promedio']:,.0f}")
                st.metric("Ventas Prom.", f"S/ {data['metricas']['ventas_promedio']:,.0f}")
            
            # Mostrar productos del cluster
            with st.expander(f"Ver productos del Cluster {cluster}"):
                cluster_products = self.df_pivot[self.df_pivot['cluster'] == cluster]
                product_display = cluster_products[['Rentabilidad', 'Ventas_Totales', 'Cantidad_Total']].copy()
                product_display.columns = ['Rentabilidad', 'Ventas Totales', 'Cantidad Total']
                product_display = product_display.round(2)
                st.dataframe(product_display, use_container_width=True)
            
            st.divider()
    
    def show_visualizations(self):
        """Muestra las visualizaciones"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🎯 Optimización de Clusters")
            fig_opt = self.plot_cluster_optimization()
            st.plotly_chart(fig_opt, use_container_width=True)
        
        with col2:
            st.markdown("##### 💰 Rentabilidad por Cluster")
            fig_profit = self.plot_cluster_profitability()
            if fig_profit:
                st.plotly_chart(fig_profit, use_container_width=True)
        
        st.markdown("##### 🗺️ Mapa de Clusters (Análisis PCA)")
        fig_pca = self.plot_clusters_pca()
        if fig_pca:
            st.plotly_chart(fig_pca, use_container_width=True)
        
        # Información adicional sobre PCA
        if hasattr(self, 'pca'):
            st.info(f"ℹ️ **Información PCA:** Los dos primeros componentes principales explican "
                   f"{sum(self.pca.explained_variance_ratio_):.1%} de la varianza total en los datos.")
    
    def show_detailed_exploration(self):
        """Interfaz para exploración detallada"""
        st.markdown("#### 🔍 Exploración Detallada")
        
        # Selector de cluster
        clusters_disponibles = sorted(self.df_pivot['cluster'].unique())
        selected_cluster = st.selectbox(
            "Seleccione un cluster para análisis detallado:",
            clusters_disponibles,
            format_func=lambda x: f"Cluster {x}"
        )
        
        if selected_cluster is not None:
            cluster_data = self.df_pivot[self.df_pivot['cluster'] == selected_cluster]
            
            st.markdown(f"##### Análisis del Cluster {selected_cluster}")
            
            # Métricas del cluster
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Productos", len(cluster_data))
            
            with col2:
                st.metric("Rentabilidad Prom.", f"S/ {cluster_data['Rentabilidad'].mean():,.0f}")
            
            with col3:
                st.metric("Ventas Totales", f"S/ {cluster_data['Ventas_Totales'].sum():,.0f}")
            
            with col4:
                st.metric("Unidades Totales", f"{cluster_data['Cantidad_Total'].sum():,.0f}")
            
            # Top productos del cluster
            st.markdown("##### 🏆 Top 10 Productos por Rentabilidad")
            top_products = cluster_data.nlargest(10, 'Rentabilidad')[
                ['Rentabilidad', 'Ventas_Totales', 'Cantidad_Total', 'Precio_Promedio']
            ].round(2)
            top_products.columns = ['Rentabilidad', 'Ventas Totales', 'Cantidad Total', 'Precio Promedio']
            st.dataframe(top_products, use_container_width=True)
            
            # Gráfico de distribución
            st.markdown("##### 📊 Distribución de Rentabilidad en el Cluster")
            fig_hist = px.histogram(
                cluster_data, x='Rentabilidad',
                title=f'Distribución de Rentabilidad - Cluster {selected_cluster}',
                nbins=20
            )
            st.plotly_chart(fig_hist, use_container_width=True)