#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Predicción de Ventas Futuras - Streamlit
Análisis de tendencias y predicción usando regresión lineal
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta


class SalesPredictionStreamlit:
    def __init__(self, df):
        self.df = df.copy()
        self.prepare_data()
        
    def prepare_data(self):
        """Prepara los datos para el análisis"""
        # Codificación de variables categóricas
        self.df["ProductoCod"] = self.df["Producto"].astype("category").cat.codes
        self.df["CategoriaCod"] = self.df["Categoría"].astype("category").cat.codes
        
    def evaluate_models(self):
        """Evalúa diferentes modelos de predicción"""
        with st.spinner("Evaluando modelos de machine learning..."):
            X = self.df[["ProductoCod", "CategoriaCod", "Mes", "PrecioUnitario", "Año"]]
            y = self.df["PrecioTotal"]
            
            # División de datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Modelos a evaluar
            modelos = {
                "Regresión Lineal": LinearRegression(),
                "Árbol de Decisión": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            }
            
            resultados = []
            
            for nombre, modelo in modelos.items():
                modelo.fit(X_train, y_train)
                pred = modelo.predict(X_test)
                
                mae = mean_absolute_error(y_test, pred)
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                r2 = r2_score(y_test, pred)
                
                resultados.append({
                    "Modelo": nombre,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R²": r2
                })
            
            return pd.DataFrame(resultados)
    
    def show_monthly_sales_by_year(self):
        """Muestra gráficos de ventas mensuales por año"""
        # Agrupar por año y mes
        ventas_mensuales = self.df.groupby(["Año", "Mes"])["PrecioTotal"].sum().reset_index()
        ventas_mensuales["Fecha"] = pd.to_datetime(
            ventas_mensuales["Año"].astype(str) + "-" + 
            ventas_mensuales["Mes"].astype(str) + "-01"
        )
        
        años = sorted(ventas_mensuales["Año"].unique())
        
        # Crear gráfico interactivo
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, año in enumerate(años):
            datos_año = ventas_mensuales[ventas_mensuales["Año"] == año]
            
            fig.add_trace(go.Scatter(
                x=datos_año["Mes"],
                y=datos_año["PrecioTotal"],
                mode='lines+markers',
                name=f'Año {año}',
                line=dict(width=3, color=colors[i % len(colors)]),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="Ventas Mensuales por Año",
            xaxis_title="Mes",
            yaxis_title="Ventas Totales (S/)",
            hovermode='x unified',
            height=500,
            xaxis=dict(tickmode='linear', tick0=1, dtick=1)
        )
        
        return fig
    
    def show_annual_sales(self):
        """Muestra gráfico de ventas anuales"""
        ventas_anuales = self.df.groupby("Año")["PrecioTotal"].sum().reset_index()
        
        fig = px.bar(
            ventas_anuales,
            x="Año",
            y="PrecioTotal",
            title="Ventas Totales por Año",
            text="PrecioTotal",
            color="PrecioTotal",
            color_continuous_scale="Blues"
        )
        
        fig.update_traces(
            texttemplate='S/ %{text:,.0f}',
            textposition='outside'
        )
        
        fig.update_layout(
            height=500,
            xaxis_title="Año",
            yaxis_title="Ventas Totales (S/)",
            showlegend=False
        )
        
        return fig
    
    def predict_future_sales(self, months_ahead=12):
        """Predice ventas futuras usando regresión lineal"""
        # Filtrar datos desde abril 2024 en adelante
        ventas_filtradas = self.df[(self.df["Año"] == 2024) & (self.df["Mes"] >= 4)]
        
        if ventas_filtradas.empty:
            st.error("No hay datos suficientes desde abril 2024 para la predicción.")
            return None, None
        
        # Agrupar ventas totales por mes
        ventas_mensuales = ventas_filtradas.groupby(["Año", "Mes"])["PrecioTotal"].sum().reset_index()
        ventas_mensuales = ventas_mensuales.sort_values(by=["Año", "Mes"]).reset_index(drop=True)
        ventas_mensuales["t"] = np.arange(1, len(ventas_mensuales) + 1)
        
        # Modelo de regresión lineal
        X = ventas_mensuales[["t"]]
        y = ventas_mensuales["PrecioTotal"]
        
        modelo = LinearRegression()
        modelo.fit(X, y)
        
        # Predecir ventas futuras
        t_futuro = np.arange(len(ventas_mensuales) + 1, 
                           len(ventas_mensuales) + months_ahead + 1).reshape(-1, 1)
        ventas_predichas = modelo.predict(t_futuro)
        
        # Crear fechas futuras
        ultimo_mes = ventas_mensuales.loc[len(ventas_mensuales)-1, "Mes"]
        ultimo_año = ventas_mensuales.loc[len(ventas_mensuales)-1, "Año"]
        
        fechas_futuras = []
        mes = ultimo_mes
        año = ultimo_año
        for _ in range(months_ahead):
            mes += 1
            if mes > 12:
                mes = 1
                año += 1
            fechas_futuras.append(pd.Timestamp(year=año, month=mes, day=1))
        
        # DataFrame con predicciones
        df_prediccion = pd.DataFrame({
            "Fecha": fechas_futuras,
            "VentasPredichas": ventas_predichas
        })
        
        # Crear gráfico de predicción
        fechas_historicas = pd.to_datetime(
            ventas_mensuales["Año"].astype(str) + "-" + 
            ventas_mensuales["Mes"].astype(str) + "-01"
        )
        
        fig = go.Figure()
        
        # Ventas históricas
        fig.add_trace(go.Scatter(
            x=fechas_historicas,
            y=ventas_mensuales["PrecioTotal"],
            mode='lines+markers',
            name='Ventas Históricas',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        # Predicciones
        fig.add_trace(go.Scatter(
            x=df_prediccion["Fecha"],
            y=df_prediccion["VentasPredichas"],
            mode='lines+markers',
            name='Ventas Predichas',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=8, symbol='square')
        ))
        
        fig.update_layout(
            title="Predicción de Ventas Futuras (Regresión Lineal)",
            xaxis_title="Fecha",
            yaxis_title="Ventas Totales (S/)",
            hovermode='x unified',
            height=600
        )
        
        return fig, df_prediccion
    
    def show_interface(self):
        """Muestra la interfaz completa del módulo"""
        st.markdown("### 🔍 Análisis y Predicción de Ventas")
        
        # Tabs para organizar el contenido
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Análisis Histórico", 
            "🤖 Evaluación de Modelos", 
            "🔮 Predicción Futura",
            "📋 Resumen Ejecutivo"
        ])
        
        with tab1:
            st.markdown("#### 📈 Tendencias Históricas")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    self.show_monthly_sales_by_year(), 
                    use_container_width=True
                )
            
            with col2:
                st.plotly_chart(
                    self.show_annual_sales(), 
                    use_container_width=True
                )
            
            # Estadísticas adicionales
            st.markdown("#### 📊 Estadísticas Descriptivas")
            ventas_por_mes = self.df.groupby(["Año", "Mes"])["PrecioTotal"].sum()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Promedio Mensual", f"S/ {ventas_por_mes.mean():,.2f}")
            with col2:
                st.metric("Mediana Mensual", f"S/ {ventas_por_mes.median():,.2f}")
            with col3:
                st.metric("Máximo Mensual", f"S/ {ventas_por_mes.max():,.2f}")
            with col4:
                st.metric("Mínimo Mensual", f"S/ {ventas_por_mes.min():,.2f}")
        
        with tab2:
            st.markdown("#### 🤖 Comparación de Modelos de Machine Learning")
            
            with st.spinner("Entrenando y evaluando modelos..."):
                resultados_df = self.evaluate_models()
            
            # Mostrar tabla de resultados
            st.dataframe(
                resultados_df.style.format({
                    'MAE': '{:.2f}',
                    'RMSE': '{:.2f}',
                    'R²': '{:.3f}'
                }).highlight_max(subset=['R²'], color='lightgreen')
                .highlight_min(subset=['MAE', 'RMSE'], color='lightgreen'),
                use_container_width=True
            )
            
            # Explicación de métricas
            with st.expander("ℹ️ Explicación de Métricas"):
                st.markdown("""
                - **MAE (Error Absoluto Medio)**: Promedio de errores absolutos. Menor es mejor.
                - **RMSE (Raíz del Error Cuadrático Medio)**: Raíz cuadrada del promedio de errores al cuadrado. Menor es mejor.
                - **R² (Coeficiente de Determinación)**: Proporción de varianza explicada por el modelo. Más cercano a 1 es mejor.
                """)
            
            # Gráfico de comparación
            fig_comparacion = px.bar(
                resultados_df.melt(id_vars=['Modelo'], 
                                 value_vars=['MAE', 'RMSE'],
                                 var_name='Métrica', 
                                 value_name='Valor'),
                x='Modelo', y='Valor', color='Métrica',
                title='Comparación de Errores por Modelo',
                barmode='group'
            )
            st.plotly_chart(fig_comparacion, use_container_width=True)
        
        with tab3:
            st.markdown("#### 🔮 Predicción de Ventas Futuras")
            
            # Configuración de predicción
            col1, col2 = st.columns([1, 3])
            
            with col1:
                months_ahead = st.slider(
                    "Meses a predecir:",
                    min_value=1,
                    max_value=24,
                    value=12,
                    step=1
                )
                
                if st.button("🚀 Generar Predicción", type="primary"):
                    st.session_state.prediction_generated = True
                    st.session_state.months_ahead = months_ahead
            
            # Generar predicción si se ha solicitado
            if st.session_state.get('prediction_generated', False):
                with st.spinner(f"Generando predicción para {st.session_state.months_ahead} meses..."):
                    fig_pred, df_pred = self.predict_future_sales(st.session_state.months_ahead)
                
                if fig_pred is not None:
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Mostrar tabla de predicciones
                    st.markdown("#### 📋 Predicciones Detalladas")
                    
                    # Formatear las predicciones
                    df_pred_formatted = df_pred.copy()
                    df_pred_formatted['Mes/Año'] = df_pred_formatted['Fecha'].dt.strftime('%B %Y')
                    df_pred_formatted['Ventas Predichas'] = df_pred_formatted['VentasPredichas'].apply(
                        lambda x: f"S/ {x:,.2f}"
                    )
                    
                    st.dataframe(
                        df_pred_formatted[['Mes/Año', 'Ventas Predichas']],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Métricas de predicción
                    total_predicho = df_pred['VentasPredichas'].sum()
                    promedio_mensual = df_pred['VentasPredichas'].mean()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Predicho", f"S/ {total_predicho:,.2f}")
                    with col2:
                        st.metric("Promedio Mensual", f"S/ {promedio_mensual:,.2f}")
                    with col3:
                        # Comparar con promedio histórico
                        hist_avg = self.df.groupby(["Año", "Mes"])["PrecioTotal"].sum().mean()
                        crecimiento = ((promedio_mensual - hist_avg) / hist_avg) * 100
                        st.metric("Crecimiento vs Histórico", f"{crecimiento:+.1f}%")
        
        with tab4:
            st.markdown("#### 📋 Resumen Ejecutivo")
            
            # Insights clave
            st.markdown("##### 🎯 Insights Clave")
            
            # Análisis de tendencia
            ventas_mensuales = self.df.groupby(["Año", "Mes"])["PrecioTotal"].sum().reset_index()
            ventas_mensuales = ventas_mensuales.sort_values(by=["Año", "Mes"])
            
            # Calcular tendencia general
            X = np.arange(len(ventas_mensuales)).reshape(-1, 1)
            y = ventas_mensuales["PrecioTotal"].values
            modelo_tendencia = LinearRegression().fit(X, y)
            pendiente = modelo_tendencia.coef_[0]
            
            if pendiente > 0:
                tendencia = "📈 Creciente"
                color_tendencia = "green"
            else:
                tendencia = "📉 Decreciente" 
                color_tendencia = "red"
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Tendencia General:**")
                st.markdown(f"<span style='color: {color_tendencia}'>{tendencia}</span>", 
                           unsafe_allow_html=True)
                st.markdown(f"Cambio mensual promedio: S/ {pendiente:,.2f}")
                
                # Mejor y peor mes
                mejor_mes = ventas_mensuales.loc[ventas_mensuales["PrecioTotal"].idxmax()]
                peor_mes = ventas_mensuales.loc[ventas_mensuales["PrecioTotal"].idxmin()]
                
                st.markdown(f"**Mejor mes:** {mejor_mes['Mes']:02d}/{mejor_mes['Año']} - S/ {mejor_mes['PrecioTotal']:,.2f}")
                st.markdown(f"**Peor mes:** {peor_mes['Mes']:02d}/{peor_mes['Año']} - S/ {peor_mes['PrecioTotal']:,.2f}")
            
            with col2:
                # Estacionalidad
                estacionalidad = self.df.groupby("Mes")["PrecioTotal"].sum().sort_values(ascending=False)
                
                st.markdown("**Estacionalidad (Top 3 meses):**")
                meses_nombres = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
                               5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
                               9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
                
                for i, (mes, ventas) in enumerate(estacionalidad.head(3).items()):
                    st.markdown(f"{i+1}. {meses_nombres[mes]}: S/ {ventas:,.2f}")
            
            # Recomendaciones
            st.markdown("##### 💡 Recomendaciones")
            recomendaciones = [
                "Aprovechar los meses de mayor demanda para impulsar las ventas",
                "Implementar estrategias de marketing en los meses de menor rendimiento",
                "Mantener un inventario adecuado basado en las predicciones",
                "Monitorear regularmente el rendimiento vs las predicciones"
            ]
            
            for rec in recomendaciones:
                st.markdown(f"• {rec}")


# Función auxiliar para integrar con el main
def show_sales_prediction_interface(df):
    """Función para mostrar la interfaz desde el main"""
    predictor = SalesPredictionStreamlit(df)
    predictor.show_interface()