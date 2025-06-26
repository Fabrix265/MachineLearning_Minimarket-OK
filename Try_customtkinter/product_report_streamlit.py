#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de reporte de productos para Streamlit
Genera reportes detallados de ventas por producto con interfaz gráfica
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime


class ProductReportStreamlit:
    def __init__(self, df):
        self.df = df
        self.meses_abrev = {
            1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr',
            5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Ago',
            9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
        }
        
    def get_products_list(self):
        """Obtiene la lista de productos disponibles"""
        return sorted(self.df['Producto'].unique())
    
    def filter_product_data(self, nombre_producto):
        """Filtra los datos por producto específico"""
        df_producto = self.df[self.df['Producto'] == nombre_producto]
        
        if df_producto.empty:
            raise ValueError(f"No se encontraron datos para el producto: {nombre_producto}")
        
        return df_producto
    
    def generate_sales_summary(self, df_producto):
        """Genera resumen de ventas por año y mes"""
        resumen = df_producto.groupby(['Año', 'Mes']).agg({
            'PrecioTotal': 'sum',
            'Cantidad': 'sum'
        }).reset_index()
        
        resumen['MesNombre'] = resumen['Mes'].map(self.meses_abrev)
        resumen = resumen.sort_values(['Año', 'Mes'])
        
        return resumen
    
    def create_sales_line_chart(self, resumen, nombre_producto):
        """Crea gráfico de líneas de ventas por año"""
        fig = px.line(
            resumen, 
            x='MesNombre', 
            y='PrecioTotal',
            color='Año',
            title=f'Evolución de Ventas - {nombre_producto}',
            labels={
                'PrecioTotal': 'Ventas (S/.)',
                'MesNombre': 'Mes',
                'Año': 'Año'
            },
            markers=True,
            height=500
        )
        
        # Personalizar el gráfico
        fig.update_traces(line=dict(width=3), marker=dict(size=8))
        fig.update_layout(
            title_font_size=16,
            title_x=0.5,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            legend_title_font_size=14,
            template='plotly_white',
            hovermode='x unified'
        )
        
        # Agregar valores en los puntos
        for trace in fig.data:
            trace.textposition = "top center"
            trace.texttemplate = "S/. %{y:,.0f}"
            trace.textfont = dict(size=10)
        
        return fig
    
    def create_comparative_bar_chart(self, resumen, nombre_producto):
        """Crea gráfico de barras comparativo por años"""
        fig = px.bar(
            resumen,
            x='MesNombre',
            y='PrecioTotal',
            color='Año',
            title=f'Comparativo de Ventas por Mes - {nombre_producto}',
            labels={
                'PrecioTotal': 'Ventas (S/.)',
                'MesNombre': 'Mes',
                'Año': 'Año'
            },
            height=500,
            text='PrecioTotal'
        )
        
        # Personalizar el gráfico
        fig.update_traces(texttemplate='S/. %{text:,.0f}', textposition='outside')
        fig.update_layout(
            title_font_size=16,
            title_x=0.5,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            legend_title_font_size=14,
            template='plotly_white'
        )
        
        return fig
    
    def create_units_chart(self, resumen, nombre_producto):
        """Crea gráfico de unidades vendidas"""
        fig = px.bar(
            resumen,
            x='MesNombre',
            y='Cantidad',
            color='Año',
            title=f'Unidades Vendidas - {nombre_producto}',
            labels={
                'Cantidad': 'Unidades',
                'MesNombre': 'Mes',
                'Año': 'Año'
            },
            height=450,
            text='Cantidad'
        )
        
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(
            title_font_size=16,
            title_x=0.5,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            legend_title_font_size=14,
            template='plotly_white'
        )
        
        return fig
    
    def create_payment_methods_chart(self, df_producto, nombre_producto):
        """Crea gráfico de métodos de pago"""
        payment_counts = df_producto['MétodoPago'].value_counts()
        
        fig = px.pie(
            values=payment_counts.values,
            names=payment_counts.index,
            title=f'Distribución de Métodos de Pago - {nombre_producto}',
            height=450
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont_size=12
        )
        
        fig.update_layout(
            title_font_size=16,
            title_x=0.5,
            template='plotly_white'
        )
        
        return fig
    
    def create_monthly_trend_heatmap(self, df_producto, nombre_producto):
        """Crea mapa de calor de tendencias mensuales"""
        # Crear tabla pivote para el heatmap
        heatmap_data = df_producto.groupby(['Año', 'Mes'])['PrecioTotal'].sum().reset_index()
        pivot_data = heatmap_data.pivot(index='Año', columns='Mes', values='PrecioTotal').fillna(0)
        
        # Renombrar columnas con nombres de meses
        pivot_data.columns = [self.meses_abrev[mes] for mes in pivot_data.columns]
        
        fig = px.imshow(
            pivot_data,
            labels=dict(x="Mes", y="Año", color="Ventas (S/.)"),
            title=f'Mapa de Calor de Ventas - {nombre_producto}',
            color_continuous_scale='Blues',
            height=400
        )
        
        fig.update_layout(
            title_font_size=16,
            title_x=0.5,
            template='plotly_white'
        )
        
        return fig
    
    def generate_statistics(self, df_producto):
        """Genera estadísticas del producto"""
        stats = {
            'ventas_totales': df_producto['PrecioTotal'].sum(),
            'unidades_vendidas': df_producto['Cantidad'].sum(),
            'precio_promedio': df_producto['PrecioUnitario'].mean(),
            'precio_min': df_producto['PrecioUnitario'].min(),
            'precio_max': df_producto['PrecioUnitario'].max(),
            'primera_venta': df_producto['Fecha'].min(),
            'ultima_venta': df_producto['Fecha'].max(),
            'categoria': df_producto['Categoría'].iloc[0],
            'total_transacciones': len(df_producto),
            'venta_promedio_transaccion': df_producto['PrecioTotal'].mean(),
            'cantidad_promedio_transaccion': df_producto['Cantidad'].mean()
        }
        
        # Calcular días activos
        dias_activos = (stats['ultima_venta'] - stats['primera_venta']).days + 1
        stats['dias_activos'] = dias_activos
        stats['promedio_diario'] = stats['ventas_totales'] / dias_activos if dias_activos > 0 else 0
        
        return stats
    
    def show_statistics_cards(self, stats):
        """Muestra las estadísticas en tarjetas"""
        # Fila 1: Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "💰 Ventas Totales",
                f"S/. {stats['ventas_totales']:,.2f}",
                delta=f"S/. {stats['promedio_diario']:.2f}/día"
            )
        
        with col2:
            st.metric(
                "📦 Unidades Vendidas",
                f"{stats['unidades_vendidas']:,}",
                delta=f"{stats['cantidad_promedio_transaccion']:.1f} por transacción"
            )
        
        with col3:
            st.metric(
                "🛒 Total Transacciones",
                f"{stats['total_transacciones']:,}",
                delta=f"S/. {stats['venta_promedio_transaccion']:.2f} promedio"
            )
        
        with col4:
            st.metric(
                "📅 Días Activos",
                f"{stats['dias_activos']:,}",
                delta=None
            )
        
        # Fila 2: Información de precios y categoría
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.info(f"**Precio Promedio**\nS/. {stats['precio_promedio']:.2f}")
        
        with col2:
            st.info(f"**Rango de Precios**\nS/. {stats['precio_min']:.2f} - S/. {stats['precio_max']:.2f}")
        
        with col3:
            st.info(f"**Categoría**\n{stats['categoria']}")
        
        with col4:
            st.info(f"**Período**\n{stats['primera_venta'].strftime('%d/%m/%Y')} - {stats['ultima_venta'].strftime('%d/%m/%Y')}")
    
    def show_interface(self):
        """Muestra la interfaz principal del reporte"""
        
        # Selector de producto
        productos = self.get_products_list()
        
        st.markdown("### 📋 Seleccionar Producto")
        
        # Crear dos columnas para el selector
        col1, col2 = st.columns([3, 1])
        
        with col1:
            producto_seleccionado = st.selectbox(
                "Seleccione un producto para analizar:",
                productos,
                key="producto_selector"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Espaciado
            generar_reporte = st.button("📊 Generar Reporte", type="primary")
        
        # Mostrar información del producto seleccionado
        if producto_seleccionado:
            df_filtrado = self.filter_product_data(producto_seleccionado)
            registros = len(df_filtrado)
            st.info(f"**Producto seleccionado:** {producto_seleccionado} | **Registros encontrados:** {registros:,}")
        
        # Generar reporte si se presiona el botón o si ya hay un producto seleccionado
        if producto_seleccionado and (generar_reporte or 'ultimo_producto' in st.session_state and st.session_state.ultimo_producto == producto_seleccionado):
            
            # Guardar el último producto para mantener el reporte visible
            st.session_state.ultimo_producto = producto_seleccionado
            
            try:
                with st.spinner('Generando reporte...'):
                    
                    # Filtrar datos del producto
                    df_producto = self.filter_product_data(producto_seleccionado)
                    
                    # Generar resumen de ventas
                    resumen = self.generate_sales_summary(df_producto)
                    
                    # Generar estadísticas
                    stats = self.generate_statistics(df_producto)
                
                st.success(f"✅ Reporte generado para: **{producto_seleccionado}**")
                
                # Mostrar estadísticas
                st.markdown("### 📊 Resumen Estadístico")
                self.show_statistics_cards(stats)
                
                st.markdown("---")
                
                # Gráficos en pestañas
                st.markdown("### 📈 Análisis Gráfico")
                
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📈 Evolución de Ventas",
                    "📊 Comparativo por Mes", 
                    "📦 Unidades Vendidas",
                    "💳 Métodos de Pago",
                    "🔥 Mapa de Calor"
                ])
                
                with tab1:
                    fig_line = self.create_sales_line_chart(resumen, producto_seleccionado)
                    st.plotly_chart(fig_line, use_container_width=True)
                    
                    # Insights automáticos
                    mes_mejor = resumen.loc[resumen['PrecioTotal'].idxmax()]
                    mes_peor = resumen.loc[resumen['PrecioTotal'].idxmin()]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"🔝 **Mejor mes:** {mes_mejor['MesNombre']} {mes_mejor['Año']} - S/. {mes_mejor['PrecioTotal']:,.2f}")
                    with col2:
                        st.warning(f"📉 **Mes más bajo:** {mes_peor['MesNombre']} {mes_peor['Año']} - S/. {mes_peor['PrecioTotal']:,.2f}")
                
                with tab2:
                    if len(resumen['Año'].unique()) > 1:
                        fig_bar = self.create_comparative_bar_chart(resumen, producto_seleccionado)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        st.info("ℹ️ Se necesitan datos de múltiples años para el análisis comparativo.")
                
                with tab3:
                    fig_units = self.create_units_chart(resumen, producto_seleccionado)
                    st.plotly_chart(fig_units, use_container_width=True)
                    
                    # Mostrar correlación entre ventas y unidades
                    correlacion = resumen['PrecioTotal'].corr(resumen['Cantidad'])
                    if correlacion > 0.7:
                        st.success(f"📈 **Alta correlación** ventas-unidades: {correlacion:.2f}")
                    elif correlacion > 0.4:
                        st.warning(f"📊 **Correlación moderada** ventas-unidades: {correlacion:.2f}")
                    else:
                        st.info(f"📉 **Baja correlación** ventas-unidades: {correlacion:.2f}")
                
                with tab4:
                    fig_payment = self.create_payment_methods_chart(df_producto, producto_seleccionado)
                    st.plotly_chart(fig_payment, use_container_width=True)
                    
                    # Mostrar estadísticas de métodos de pago
                    payment_stats = df_producto['MétodoPago'].value_counts()
                    st.markdown("**Resumen de Métodos de Pago:**")
                    for metodo, cantidad in payment_stats.items():
                        porcentaje = (cantidad / len(df_producto)) * 100
                        st.write(f"• {metodo}: {cantidad:,} transacciones ({porcentaje:.1f}%)")
                
                with tab5:
                    if len(resumen['Año'].unique()) > 1:
                        fig_heatmap = self.create_monthly_trend_heatmap(df_producto, producto_seleccionado)
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                        
                        st.markdown("**Interpretación del Mapa de Calor:**")
                        st.write("• 🔵 **Azul más intenso**: Mayores ventas")
                        st.write("• 🔵 **Azul más claro**: Menores ventas")
                        st.write("• ⚪ **Blanco**: Sin ventas registradas")
                    else:
                        st.info("ℹ️ Se necesitan datos de múltiples años para el mapa de calor.")
                
                # Sección de datos tabulares
                st.markdown("---")
                st.markdown("### 📋 Datos Detallados")
                
                with st.expander("Ver datos mensuales"):
                    # Mostrar tabla de resumen mensual
                    resumen_display = resumen.copy()
                    resumen_display['Ventas (S/.)'] = resumen_display['PrecioTotal'].apply(lambda x: f"S/. {x:,.2f}")
                    resumen_display['Unidades'] = resumen_display['Cantidad'].apply(lambda x: f"{x:,}")
                    
                    st.dataframe(
                        resumen_display[['Año', 'MesNombre', 'Ventas (S/.)', 'Unidades']].rename(columns={'MesNombre': 'Mes'}),
                        use_container_width=True,
                        hide_index=True
                    )
                
                with st.expander("Ver transacciones recientes"):
                    # Mostrar últimas 10 transacciones
                    transacciones_recientes = df_producto.sort_values('Fecha', ascending=False).head(10)
                    transacciones_display = transacciones_recientes[['Fecha', 'Cantidad', 'PrecioUnitario', 'PrecioTotal', 'MétodoPago']].copy()
                    transacciones_display['Fecha'] = transacciones_display['Fecha'].dt.strftime('%d/%m/%Y')
                    transacciones_display['PrecioUnitario'] = transacciones_display['PrecioUnitario'].apply(lambda x: f"S/. {x:.2f}")
                    transacciones_display['PrecioTotal'] = transacciones_display['PrecioTotal'].apply(lambda x: f"S/. {x:.2f}")
                    
                    st.dataframe(
                        transacciones_display,
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Botón para descargar reporte
                st.markdown("---")
                st.markdown("### 💾 Exportar Datos")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Preparar datos para descarga - Resumen
                    csv_resumen = resumen.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar Resumen CSV",
                        data=csv_resumen,
                        file_name=f"resumen_{producto_seleccionado.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Preparar datos para descarga - Transacciones
                    csv_transacciones = df_producto.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar Transacciones CSV",
                        data=csv_transacciones,
                        file_name=f"transacciones_{producto_seleccionado.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    # Preparar reporte estadístico
                    reporte_stats = pd.DataFrame([stats]).T
                    reporte_stats.columns = ['Valor']
                    csv_stats = reporte_stats.to_csv()
                    st.download_button(
                        label="📥 Descargar Estadísticas CSV",
                        data=csv_stats,
                        file_name=f"estadisticas_{producto_seleccionado.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"❌ Error al generar el reporte: {e}")
                st.exception(e)  # Para debugging