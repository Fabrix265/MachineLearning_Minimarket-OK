#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Predicción de Stock para Streamlit
Incluye predicción para productos existentes y nuevos
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import sklearn
from packaging import version


class StockPredictionStreamlit:
    def __init__(self, df):
        self.df = df.copy()
        self.df_grouped_existing = None
        self.df_grouped_new = None
        self.encoder_existing = None
        self.model_existing = None
        self.le_cat = None
        self.model_new = None
        
        # Métricas de los modelos
        self.mae_existing = 0
        self.rmse_existing = 0
        self.r2_existing = 0
        self.mae_new = 0
        self.rmse_new = 0
        self.r2_new = 0
        
        self.prepare_data()
        
    def prepare_data(self):
        """Prepara los datos para ambos tipos de predicción"""
        try:
            # Para productos existentes
            self.prepare_existing_products_data()
            # Para productos nuevos
            self.prepare_new_products_data()
        except Exception as e:
            st.error(f"Error al preparar datos: {e}")
            
    def prepare_existing_products_data(self):
        """Prepara datos para predicción de productos existentes"""
        # Agrupar datos por Producto y Mes
        self.df_grouped_existing = self.df.groupby(['Producto', 'Mes']).agg({
            'Cantidad': 'sum',
            'PrecioUnitario': 'mean',
            'PrecioTotal': 'sum',
            'Categoría': 'first',
            'MétodoPago': 'first'
        }).reset_index()
        
        # Transformar la variable objetivo
        self.df_grouped_existing['CantidadLog'] = np.log1p(self.df_grouped_existing['Cantidad'])
        
        # Preparar features
        features = self.df_grouped_existing[['Producto', 'Mes', 'Categoría', 'MétodoPago']].copy()
        features['PrecioUnitario'] = self.df_grouped_existing['PrecioUnitario']
        features['PrecioTotal'] = self.df_grouped_existing['PrecioTotal']
        target = self.df_grouped_existing['CantidadLog']
        
        # Codificar variables categóricas
        if version.parse(sklearn.__version__) >= version.parse("1.2"):
            self.encoder_existing = OneHotEncoder(sparse_output=False)
        else:
            self.encoder_existing = OneHotEncoder(sparse=False)
            
        encoded = self.encoder_existing.fit_transform(features[['Producto', 'Categoría', 'MétodoPago']])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder_existing.get_feature_names_out())
        
        X = pd.concat([encoded_df, features[['Mes', 'PrecioUnitario', 'PrecioTotal']].reset_index(drop=True)], axis=1)
        y = target.values
        
        # Entrenar modelo
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model_existing = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_existing.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred_log = self.model_existing.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_test_exp = np.expm1(y_test)
        
        self.mae_existing = mean_absolute_error(y_test_exp, y_pred)
        self.rmse_existing = np.sqrt(mean_squared_error(y_test_exp, y_pred))
        self.r2_existing = r2_score(y_test_exp, y_pred)
        
    def prepare_new_products_data(self):
        """Prepara datos para predicción de productos nuevos"""
        # Preparar datos temporales
        df_temp = self.df.copy()
        df_temp['Fecha'] = pd.to_datetime(df_temp['Fecha'])
        df_temp['Año'] = df_temp['Fecha'].dt.year
        df_temp['Mes_num'] = df_temp['Fecha'].dt.month
        df_temp['Trimestre'] = df_temp['Fecha'].dt.quarter
        
        # Codificar categoría
        self.le_cat = LabelEncoder()
        df_temp['Categoria_encoded'] = self.le_cat.fit_transform(df_temp['Categoría'])
        
        # Agrupar por mes, producto, etc.
        self.df_grouped_new = df_temp.groupby(['Producto', 'Categoria_encoded', 'Año', 'Mes_num', 'Trimestre']).agg({
            'Cantidad': 'sum',
            'PrecioUnitario': 'mean'
        }).reset_index()
        
        # Feature engineering
        cat_price_avg = self.df_grouped_new.groupby('Categoria_encoded')['PrecioUnitario'].transform('mean')
        self.df_grouped_new['Precio_relativo'] = self.df_grouped_new['PrecioUnitario'] / cat_price_avg
        
        # Variables para el modelo
        features = ['Categoria_encoded', 'PrecioUnitario', 'Precio_relativo', 'Mes_num', 'Trimestre']
        X = self.df_grouped_new[features]
        y = self.df_grouped_new['Cantidad']
        
        # División y entrenamiento
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model_new = RandomForestRegressor(n_estimators=300, max_depth=25, min_samples_split=4, random_state=42)
        self.model_new.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = self.model_new.predict(X_test)
        self.mae_new = mean_absolute_error(y_test, y_pred)
        self.rmse_new = mean_squared_error(y_test, y_pred) ** 0.5
        self.r2_new = r2_score(y_test, y_pred)
        
    def show_existing_product_interface(self):
        """Interfaz para predicción de productos existentes"""
        st.subheader("📦 Predicción de Stock - Producto Existente")
        
        with st.container():
            # Información del modelo
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"{self.mae_existing:.2f}")
            with col2:
                st.metric("RMSE", f"{self.rmse_existing:.2f}")
            with col3:
                st.metric("R²", f"{self.r2_existing:.3f}")
        
        st.markdown("---")
        
        # Formulario de predicción
        with st.form("existing_product_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                productos_disponibles = sorted(self.df_grouped_existing['Producto'].unique())
                producto_seleccionado = st.selectbox(
                    "Seleccione el producto:",
                    productos_disponibles,
                    help="Productos disponibles en el histórico de ventas"
                )
                
            with col2:
                mes_seleccionado = st.selectbox(
                    "Seleccione el mes:",
                    list(range(1, 13)),
                    format_func=lambda x: [
                        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
                    ][x-1],
                    help="Mes para el cual desea predecir el stock"
                )
            
            submitted = st.form_submit_button("🔮 Predecir Stock", type="primary")
            
        if submitted:
            try:
                # Obtener datos del producto
                fila = self.df_grouped_existing[
                    self.df_grouped_existing['Producto'] == producto_seleccionado
                ].iloc[0]
                
                # Preparar datos para predicción
                prod_df = pd.DataFrame({
                    'Producto': [producto_seleccionado],
                    'Categoría': [fila['Categoría']],
                    'MétodoPago': [fila['MétodoPago']],
                    'Mes': [mes_seleccionado],
                    'PrecioUnitario': [fila['PrecioUnitario']],
                    'PrecioTotal': [fila['PrecioTotal']],
                })
                
                encoded_input = self.encoder_existing.transform(
                    prod_df[['Producto', 'Categoría', 'MétodoPago']]
                )
                encoded_input_df = pd.DataFrame(
                    encoded_input, 
                    columns=self.encoder_existing.get_feature_names_out()
                )
                input_data = pd.concat([
                    encoded_input_df, 
                    prod_df[['Mes', 'PrecioUnitario', 'PrecioTotal']].reset_index(drop=True)
                ], axis=1)
                
                # Realizar predicción
                pred_log = self.model_existing.predict(input_data)[0]
                pred = np.expm1(pred_log)
                stock_recomendado = max(round(pred), 0)
                
                # Mostrar resultado
                st.success("✅ Predicción completada exitosamente")
                
                # Resultado principal
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 2rem;
                        border-radius: 15px;
                        text-align: center;
                        color: white;
                        margin: 1rem 0;
                    ">
                        <h2 style="margin: 0; font-size: 1.5rem;">📦 STOCK RECOMENDADO</h2>
                        <h1 style="margin: 0.5rem 0; font-size: 3rem;">{stock_recomendado}</h1>
                        <p style="margin: 0; font-size: 1.2rem;">unidades</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Información detallada
                with st.expander("📊 Información Detallada", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Producto:**", producto_seleccionado)
                        st.write("**Categoría:**", fila['Categoría'])
                        st.write("**Mes:**", [
                            "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                            "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
                        ][mes_seleccionado-1])
                        
                    with col2:
                        st.write("**Precio Unitario:**", f"S/ {fila['PrecioUnitario']:.2f}")
                        st.write("**Método de Pago:**", fila['MétodoPago'])
                        st.write("**Predicción Raw:**", f"{pred:.2f}")
                
                # Mostrar histórico del producto
                self.show_product_history(producto_seleccionado)
                
            except Exception as e:
                st.error(f"❌ Error en la predicción: {e}")
    
    def show_new_product_interface(self):
        """Interfaz para predicción de productos nuevos"""
        st.subheader("🆕 Predicción de Stock - Producto Nuevo")
        
        with st.container():
            # Información del modelo
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"{self.mae_new:.2f}")
            with col2:
                st.metric("RMSE", f"{self.rmse_new:.2f}")
            with col3:
                st.metric("R²", f"{self.r2_new:.3f}")
        
        st.markdown("---")
        
        # Formulario de predicción
        with st.form("new_product_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                nombre_producto = st.text_input(
                    "Nombre del nuevo producto:",
                    placeholder="Ej: Galletas Chocolate Premium",
                    help="Ingrese el nombre del producto nuevo"
                )
                
                categorias_disponibles = sorted(self.le_cat.classes_)
                categoria_seleccionada = st.selectbox(
                    "Categoría del producto:",
                    categorias_disponibles,
                    help="Seleccione la categoría que mejor describa el producto"
                )
                
            with col2:
                precio_unitario = st.number_input(
                    "Precio unitario (S/):",
                    min_value=0.01,
                    max_value=1000.0,
                    value=5.0,
                    step=0.1,
                    help="Precio de venta unitario del producto"
                )
                
                mes_seleccionado = st.selectbox(
                    "Mes de lanzamiento:",
                    list(range(1, 13)),
                    format_func=lambda x: [
                        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
                    ][x-1],
                    help="Mes en que se lanzará el producto"
                )
            
            submitted = st.form_submit_button("🔮 Predecir Stock", type="primary")
            
        if submitted:
            if not nombre_producto.strip():
                st.error("❌ Por favor ingrese un nombre para el producto")
                return
                
            try:
                # Calcular trimestre automáticamente
                trimestre_calculado = ((mes_seleccionado - 1) // 3) + 1
                
                # Realizar predicción
                categoria_encoded = self.le_cat.transform([categoria_seleccionada])[0]
                cat_price_prom = self.df_grouped_new[
                    self.df_grouped_new['Categoria_encoded'] == categoria_encoded
                ]['PrecioUnitario'].mean()
                precio_relativo = precio_unitario / cat_price_prom if cat_price_prom > 0 else 1.0
                
                X_new = pd.DataFrame({
                    'Categoria_encoded': [categoria_encoded],
                    'PrecioUnitario': [precio_unitario],
                    'Precio_relativo': [precio_relativo],
                    'Mes_num': [mes_seleccionado],
                    'Trimestre': [trimestre_calculado]
                })
                
                pred = self.model_new.predict(X_new)[0]
                stock_recomendado = max(round(pred), 0)
                
                # Mostrar resultado
                st.success("✅ Predicción completada exitosamente")
                
                # Resultado principal
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                        padding: 2rem;
                        border-radius: 15px;
                        text-align: center;
                        color: white;
                        margin: 1rem 0;
                    ">
                        <h2 style="margin: 0; font-size: 1.5rem;">🆕 STOCK INICIAL RECOMENDADO</h2>
                        <h1 style="margin: 0.5rem 0; font-size: 3rem;">{stock_recomendado}</h1>
                        <p style="margin: 0; font-size: 1.2rem;">unidades</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Información detallada
                with st.expander("📊 Información Detallada", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Producto:**", nombre_producto)
                        st.write("**Categoría:**", categoria_seleccionada)
                        st.write("**Precio Unitario:**", f"S/ {precio_unitario:.2f}")
                        
                    with col2:
                        st.write("**Mes de Lanzamiento:**", [
                            "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                            "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
                        ][mes_seleccionado-1])
                        st.write("**Trimestre:**", trimestre_calculado)
                        st.write("**Precio Relativo:**", f"{precio_relativo:.2f}")
                
                # Mostrar comparación con categoría
                self.show_category_comparison(categoria_seleccionada, precio_unitario)
                
            except Exception as e:
                st.error(f"❌ Error en la predicción: {e}")
    
    def show_product_history(self, producto):
        """Muestra el histórico de ventas del producto"""
        try:
            hist_data = self.df[self.df['Producto'] == producto].copy()
            if len(hist_data) == 0:
                return
                
            # Agrupar por mes
            hist_monthly = hist_data.groupby('Mes').agg({
                'Cantidad': 'sum',
                'PrecioTotal': 'sum'
            }).reset_index()
            
            # Crear gráfico
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_monthly['Mes'],
                y=hist_monthly['Cantidad'],
                mode='lines+markers',
                name='Ventas Históricas',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title=f"📈 Histórico de Ventas - {producto}",
                xaxis_title="Mes",
                yaxis_title="Cantidad Vendida",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"No se pudo mostrar el histórico: {e}")
    
    def show_category_comparison(self, categoria, precio):
        """Muestra comparación con otros productos de la categoría"""
        try:
            # Obtener datos de la categoría
            cat_data = self.df[self.df['Categoría'] == categoria].copy()
            if len(cat_data) == 0:
                return
                
            # Estadísticas de la categoría
            cat_stats = cat_data.groupby('Producto').agg({
                'Cantidad': 'sum',
                'PrecioUnitario': 'mean'
            }).reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**📊 Estadísticas Categoría '{categoria}':**")
                st.write(f"- Productos en categoría: {len(cat_stats)}")
                st.write(f"- Precio promedio: S/ {cat_stats['PrecioUnitario'].mean():.2f}")
                st.write(f"- Ventas promedio/producto: {cat_stats['Cantidad'].mean():.0f} unidades")
                
            with col2:
                # Gráfico de dispersión precio vs ventas
                fig = px.scatter(
                    cat_stats, 
                    x='PrecioUnitario', 
                    y='Cantidad',
                    title=f"Precio vs Ventas - {categoria}",
                    labels={'PrecioUnitario': 'Precio Unitario (S/)', 'Cantidad': 'Ventas Totales'}
                )
                
                # Agregar punto del nuevo producto
                fig.add_trace(go.Scatter(
                    x=[precio],
                    y=[0],  # No tenemos ventas aún
                    mode='markers',
                    marker=dict(color='red', size=12, symbol='star'),
                    name='Nuevo Producto'
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.warning(f"No se pudo mostrar la comparación: {e}")
    
    def show_model_performance(self):
        """Muestra gráficos de rendimiento de los modelos"""
        st.subheader("📊 Rendimiento de los Modelos")
        
        tab1, tab2 = st.tabs(["Productos Existentes", "Productos Nuevos"])
        
        with tab1:
            self.show_existing_model_performance()
            
        with tab2:
            self.show_new_model_performance()
    
    def show_existing_model_performance(self):
        """Rendimiento del modelo de productos existentes"""
        try:
            # Preparar datos para evaluación
            features = self.df_grouped_existing[['Producto', 'Mes', 'Categoría', 'MétodoPago']].copy()
            features['PrecioUnitario'] = self.df_grouped_existing['PrecioUnitario']
            features['PrecioTotal'] = self.df_grouped_existing['PrecioTotal']
            target = self.df_grouped_existing['CantidadLog']
            
            encoded = self.encoder_existing.transform(features[['Producto', 'Categoría', 'MétodoPago']])
            encoded_df = pd.DataFrame(encoded, columns=self.encoder_existing.get_feature_names_out())
            X = pd.concat([encoded_df, features[['Mes', 'PrecioUnitario', 'PrecioTotal']].reset_index(drop=True)], axis=1)
            
            X_train, X_test, y_train, y_test = train_test_split(X, target.values, test_size=0.2, random_state=42)
            y_pred_log = self.model_existing.predict(X_test)
            y_pred = np.expm1(y_pred_log)
            y_test_exp = np.expm1(y_test)
            
            # Gráfico de predicción vs real
            fig = px.scatter(
                x=y_test_exp, 
                y=y_pred,
                title="Predicción vs Valores Reales - Productos Existentes",
                labels={'x': 'Cantidad Real', 'y': 'Cantidad Predicha'}
            )
            
            # Línea de predicción perfecta
            min_val, max_val = min(y_test_exp.min(), y_pred.min()), max(y_test_exp.max(), y_pred.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Predicción Perfecta',
                line=dict(color='red', dash='dash')
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Métricas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"{self.mae_existing:.2f}")
            with col2:
                st.metric("RMSE", f"{self.rmse_existing:.2f}")
            with col3:
                st.metric("R²", f"{self.r2_existing:.3f}")
                
        except Exception as e:
            st.error(f"Error al mostrar rendimiento: {e}")
    
    def show_new_model_performance(self):
        """Rendimiento del modelo de productos nuevos"""
        try:
            # Preparar datos para evaluación
            features_new = ['Categoria_encoded', 'PrecioUnitario', 'Precio_relativo', 'Mes_num', 'Trimestre']
            X_new = self.df_grouped_new[features_new]
            y_new = self.df_grouped_new['Cantidad']
            
            X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
            y_pred_new = self.model_new.predict(X_test_new)
            
            # Gráfico de predicción vs real
            fig = px.scatter(
                x=y_test_new, 
                y=y_pred_new,
                title="Predicción vs Valores Reales - Productos Nuevos",
                labels={'x': 'Cantidad Real', 'y': 'Cantidad Predicha'}
            )
            
            # Línea de predicción perfecta
            min_val, max_val = min(y_test_new.min(), y_pred_new.min()), max(y_test_new.max(), y_pred_new.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Predicción Perfecta',
                line=dict(color='red', dash='dash')
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Métricas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"{self.mae_new:.2f}")
            with col2:
                st.metric("RMSE", f"{self.rmse_new:.2f}")
            with col3:
                st.metric("R²", f"{self.r2_new:.3f}")
                
        except Exception as e:
            st.error(f"Error al mostrar rendimiento: {e}")
    
    def show_interface(self):
        """Interfaz principal del módulo"""
        # Selector de tipo de predicción
        tipo_prediccion = st.radio(
            "Seleccione el tipo de predicción:",
            ["📦 Producto Existente", "🆕 Producto Nuevo", "📊 Rendimiento de Modelos"],
            horizontal=True
        )
        
        st.markdown("---")
        
        if tipo_prediccion == "📦 Producto Existente":
            self.show_existing_product_interface()
        elif tipo_prediccion == "🆕 Producto Nuevo":
            self.show_new_product_interface()
        elif tipo_prediccion == "📊 Rendimiento de Modelos":
            self.show_model_performance()