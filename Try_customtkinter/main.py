#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Gestión de Minimarket - Streamlit
Aplicación web con interfaz gráfica
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Importar módulos existentes (comentados para evitar errores de importación)
# Descomenta estas líneas cuando tengas los módulos
try:
    from data_processor import DataProcessor
    from product_report_streamlit import ProductReportStreamlit
    from sales_prediction_streamlit import SalesPredictionStreamlit
    from stock_prediction_streamlit import StockPredictionStreamlit
    from profitability_analysis_streamlit import ProfitabilityAnalysisStreamlit
    from basket_analysis_streamlit import BasketAnalysisStreamlit
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importando módulos: {e}")
    MODULES_AVAILABLE = False

#Clase principal
class MinimarketStreamlitApp:
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()

    def setup_page_config(self):
        """Configuración inicial de la página"""
        st.set_page_config(
            page_title="Sistema Minimarket ML",
            page_icon="💼",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #c0392b 0%, #e74c3c 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .main-header h1 {
            color: white;
            text-align: center;
            margin: 0;
            font-size: 2.5rem;
        }
        .sidebar-header {
            background: linear-gradient(180deg, #c0392b 0%, #e74c3c 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .sidebar-header h2 {
            color: white;
            text-align: center;
            margin: 0;
        }
        </style>
        """, unsafe_allow_html=True)

    def initialize_session_state(self):
        """Inicializa el estado de la sesión"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'data_processor' not in st.session_state:
            st.session_state.data_processor = None
        if 'stock_predictor' not in st.session_state:
            st.session_state.stock_predictor = None

    def load_data(self):
        """Carga los datos del CSV y prepara los módulos"""
        if not MODULES_AVAILABLE:
            st.error("❌ Los módulos requeridos no están disponibles. Verifica las importaciones.")
            return
        
        try:
            with st.spinner('Cargando y procesando datos...'):
                data_processor = DataProcessor()
                df = data_processor.load_and_preprocess_data('ventas_minimarket.csv')

                st.session_state.df = df
                st.session_state.data_processor = data_processor
                st.session_state.data_loaded = True
                st.session_state.stock_predictor = StockPredictionStreamlit(df)

            st.success('✅ Datos cargados y modelos de stock preparados correctamente!')
        except FileNotFoundError:
            st.error("❌ No se encontró el archivo 'ventas_minimarket.csv'. Asegúrate de que esté en la misma carpeta que tu aplicación.")
        except Exception as e:
            st.error(f'❌ Error al cargar datos o preparar modelos: {e}')
            st.info("Asegúrate de que 'ventas_minimarket.csv' existe y tiene las columnas requeridas (Producto, Mes, Cantidad, PrecioUnitario, PrecioTotal, Categoría, MétodoPago, Fecha).")

    def show_data_summary(self):
        if not st.session_state.data_loaded:
            return

        df = st.session_state.df
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Registros", f"{len(df):,}")
        col2.metric("Productos Únicos", f"{df['Producto'].nunique():,}")
        col3.metric("Ventas Totales", f"S/. {df['PrecioTotal'].sum():,.2f}")
        col4.metric("Unidades Vendidas", f"{df['Cantidad'].sum():,}")

        with st.expander("📊 Información Detallada del Dataset"):
            col1, col2 = st.columns(2)
            col1.write("**Período de datos:**")
            col1.write(f"Desde: {df['Fecha'].min().strftime('%d/%m/%Y')}")
            col1.write(f"Hasta: {df['Fecha'].max().strftime('%d/%m/%Y')}")
            col1.write(f"**Categorías:** {df['Categoría'].nunique()}")

            top_productos = df.groupby('Producto')['Cantidad'].sum().sort_values(ascending=False).head(5)
            col2.write("**Top 5 Productos más vendidos:**")
            for i, (producto, cantidad) in enumerate(top_productos.items(), 1):
                col2.write(f"{i}. {producto}: {cantidad:,} unidades")

    def show_sidebar_menu(self):
        with st.sidebar:
            st.markdown("""
            <div class="sidebar-header">
                <h2>💼 MENÚ PRINCIPAL</h2>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.data_loaded:
                st.success("✅ Datos cargados")
            else:
                st.error("❌ Datos no cargados")
            
            if not MODULES_AVAILABLE:
                st.warning("⚠️ Módulos no disponibles")
            
            st.markdown("---")

            menu_options = {
                "🏠 Inicio": "inicio",
                "📊 Reporte de Producto": "reporte_producto",
                "📈 Predicción de Ventas": "prediccion_ventas",
                "📦 Predicción de Stock - Existente": "stock_existente",
                "🆕 Predicción de Stock - Nuevo": "stock_nuevo",
                "📊 Rendimiento Modelos Stock": "rendimiento_stock",
                "💰 Evaluación de Rentabilidad": "rentabilidad",
                "🛍️ Análisis de Cesta": "cesta_compra"
            }

            selected_option = st.radio("Seleccione una opción:", list(menu_options.keys()), key="menu_selection")
            return menu_options[selected_option]

    def show_header(self):
        st.markdown("""
        <div class="main-header">
            <h1>💼 Sistema de Gestión Minimarket ML</h1>
        </div>
        """, unsafe_allow_html=True)

    def show_inicio(self):
        st.markdown("## 🏠 Bienvenido al Sistema de Gestión")
        st.info("Este sistema te ayuda a optimizar la gestión de tu minimarket utilizando técnicas de Machine Learning para predicción y análisis.")

        if not MODULES_AVAILABLE:
            st.error("❌ Los módulos requeridos no están disponibles. Por favor, verifica que todos los archivos Python estén en la misma carpeta.")
            st.info("Módulos requeridos: data_processor.py, product_report_streamlit.py, sales_prediction_streamlit.py, stock_prediction_streamlit.py, profitability_analysis_streamlit.py, basket_analysis_streamlit.py")
            return

        if not st.session_state.data_loaded:
            st.warning("⚠️ Los datos no han sido cargados aún.")
            if st.button("🔄 Cargar Datos", type="primary", use_container_width=True):
                self.load_data()
        else:
            self.show_data_summary()

    def show_reporte_producto(self):
        if not MODULES_AVAILABLE:
            st.error("❌ Módulos no disponibles.")
            return
        if not st.session_state.data_loaded:
            st.error("❌ Primero debe cargar los datos desde la página de inicio.")
            return
        try:
            ProductReportStreamlit(st.session_state.df).show_interface()
        except Exception as e:
            st.error(f"Error en módulo de reporte de producto: {e}")

    def show_prediccion_ventas(self):
        if not MODULES_AVAILABLE:
            st.error("❌ Módulos no disponibles.")
            return
        if not st.session_state.data_loaded:
            st.error("❌ Primero debe cargar los datos desde la página de inicio.")
            return
        try:
            SalesPredictionStreamlit(st.session_state.df).show_interface()
        except Exception as e:
            st.error(f"Error en módulo de predicción de ventas: {e}")
            st.exception(e)

    def show_stock_existente(self):
        if not MODULES_AVAILABLE:
            st.error("❌ Módulos no disponibles.")
            return
        if not st.session_state.data_loaded or st.session_state.stock_predictor is None:
            st.error("❌ Primero debe cargar los datos desde la página de inicio.")
            return
        try:
            st.session_state.stock_predictor.show_existing_product_interface()
        except Exception as e:
            st.error(f"Error en módulo de stock existente: {e}")
            st.exception(e)

    def show_stock_nuevo(self):
        if not MODULES_AVAILABLE:
            st.error("❌ Módulos no disponibles.")
            return
        if not st.session_state.data_loaded or st.session_state.stock_predictor is None:
            st.error("❌ Primero debe cargar los datos desde la página de inicio.")
            return
        try:
            st.session_state.stock_predictor.show_new_product_interface()
        except Exception as e:
            st.error(f"Error en módulo de stock nuevo: {e}")
            st.exception(e)

    def show_model_performance_page(self):
        if not MODULES_AVAILABLE:
            st.error("❌ Módulos no disponibles.")
            return
        if not st.session_state.data_loaded or st.session_state.stock_predictor is None:
            st.error("❌ Primero debe cargar los datos desde la página de inicio.")
            return
        try:
            st.session_state.stock_predictor.show_model_performance()
        except Exception as e:
            st.error(f"Error en rendimiento de modelos de stock: {e}")
            st.exception(e)

    def show_rentabilidad(self):
        if not MODULES_AVAILABLE:
            st.error("❌ Módulos no disponibles.")
            return
        if not st.session_state.data_loaded:
            st.error("❌ Primero debe cargar los datos desde la página de inicio.")
            return
        try:
            ProfitabilityAnalysisStreamlit(st.session_state.df).show_interface()
        except Exception as e:
            st.error(f"Error en módulo de rentabilidad: {e}")
            st.exception(e)

    def show_cesta_compra(self):
        if not MODULES_AVAILABLE:
            st.error("❌ Módulos no disponibles.")
            return
        if not st.session_state.data_loaded:
            st.error("❌ Primero debe cargar los datos desde la página de inicio.")
            return
        try:
            basket = BasketAnalysisStreamlit(st.session_state.df)
            basket.prepare_data()
            meses_disponibles = basket.show_available_months()

            if not meses_disponibles:
                st.warning("⚠️ No hay meses disponibles para analizar.")
                return

            # Diccionario: {mes_num: "Mes Nombre"}
            opciones = {num: basket.meses_nombres[num].capitalize() for num in meses_disponibles}

            # Selectbox para elegir el mes
            mes_nombre_seleccionado = st.selectbox("Seleccione el mes a analizar:", list(opciones.values()))
            mes_numero = [k for k, v in opciones.items() if v == mes_nombre_seleccionado][0]

            # Ejecutar el análisis
            analisis_exitoso = basket.analyze_month(mes_numero)

            if analisis_exitoso:
                basket.generate_marketing_insights(mes_numero)

        except Exception as e:
            st.error(f"Error en módulo de análisis de cesta: {e}")
            st.exception(e)

    def run(self):
        """Método principal que ejecuta la aplicación"""
        self.show_header()
        selected_page = self.show_sidebar_menu()

        if selected_page == "inicio":
            self.show_inicio()
        elif selected_page == "reporte_producto":
            self.show_reporte_producto()
        elif selected_page == "prediccion_ventas":
            self.show_prediccion_ventas()
        elif selected_page == "stock_existente":
            self.show_stock_existente()
        elif selected_page == "stock_nuevo":
            self.show_stock_nuevo()
        elif selected_page == "rendimiento_stock":
            self.show_model_performance_page()
        elif selected_page == "rentabilidad":
            self.show_rentabilidad()
        elif selected_page == "cesta_compra":
            self.show_cesta_compra()


# EJECUTAR LA APLICACIÓN
def main():
    """Función principal"""
    app = MinimarketStreamlitApp()
    app.run()

# Esta es la forma correcta de ejecutar una app de Streamlit
if __name__ == "__main__":
    main()