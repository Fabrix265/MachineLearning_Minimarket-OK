#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de procesamiento de datos
Maneja la carga, limpieza y preprocesamiento de datos
"""

import pandas as pd
import numpy as np
import os

class DataProcessor:
    def __init__(self):
        self.df = None
    
    def load_data(self, file_path):
        """
        Carga el archivo CSV de ventas
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo {file_path} no existe")
        
        try:
            df = pd.read_csv(file_path)
            print(f" Archivo cargado: {len(df)} registros")
            return df
        except Exception as e:
            raise Exception(f"Error al leer el archivo: {e}")
    
    def clean_data(self, df):
        """
        Limpia los datos eliminando columnas innecesarias
        """
        # Columnas a eliminar
        columnas_a_dropear = ['Nombre', 'DNI']
        
        # Eliminar columnas si existen
        columnas_existentes = [col for col in columnas_a_dropear if col in df.columns]
        if columnas_existentes:
            df = df.drop(columns=columnas_existentes)
            print(f" Columnas eliminadas: {columnas_existentes}")
        
        return df
    
    def preprocess_dates(self, df):
        """
        Procesa las fechas y extrae información temporal
        """
        # Convertir fecha a datetime
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        # Extraer información temporal
        df["Año"] = df["Fecha"].dt.year
        df["Mes"] = df["Fecha"].dt.month
        df["Trimestre"] = df["Fecha"].dt.quarter
        df["DiaSemana"] = df["Fecha"].dt.dayofweek
        
        print(" Fechas procesadas y variables temporales creadas")
        return df
    
    def detect_outliers(self, df):
        """
        Detecta y reporta outliers en las variables numéricas
        """
        outliers_info = {}
        
        # Variables numéricas a analizar
        numeric_cols = ['Cantidad', 'PrecioUnitario', 'PrecioTotal']
        
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Límites para outliers
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Contar outliers
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outliers_info[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        # Reportar outliers
        print("\n DETECCIÓN DE OUTLIERS:")
        for col, info in outliers_info.items():
            print(f"  {col}: {info['count']} outliers ({info['percentage']:.1f}%)")
        
        return outliers_info
    
    def generate_summary(self, df):
        """
        Genera un resumen estadístico de los datos
        """
        print("\n RESUMEN DE DATOS:")
        print(f"  Total de registros: {len(df):,}")
        print(f"  Productos únicos: {df['Producto'].nunique()}")
        print(f"  Categorías únicas: {df['Categoría'].nunique()}")
        print(f"  Período: {df['Fecha'].min().date()} a {df['Fecha'].max().date()}")
        print(f"  Ventas totales: S/. {df['PrecioTotal'].sum():,.2f}")
        print(f"  Unidades vendidas: {df['Cantidad'].sum():,}")
        
        # Top 5 productos más vendidos
        top_productos = df.groupby('Producto')['Cantidad'].sum().sort_values(ascending=False).head(5)
        print("\nTOP 5 PRODUCTOS MÁS VENDIDOS:")
        for i, (producto, cantidad) in enumerate(top_productos.items(), 1):
            print(f"  {i}. {producto}: {cantidad:,} unidades")
    
    def load_and_preprocess_data(self, file_path):
        """
        Función principal que ejecuta todo el pipeline de procesamiento
        """
        print(" Iniciando procesamiento de datos...")
        
        # 1. Cargar datos
        df = self.load_data(file_path)
        
        # 2. Limpiar datos
        df = self.clean_data(df)
        
        # 3. Procesar fechas
        df = self.preprocess_dates(df)
        
        # 4. Detectar outliers
        self.detect_outliers(df)
        
        # 5. Generar resumen
        self.generate_summary(df)
        
        # Guardar referencia
        self.df = df
        
        print("\nProcesamiento completado exitosamente\n")
        return df
    
    def get_products_list(self, df):
        """
        Retorna lista de productos únicos ordenada
        """
        return sorted(df['Producto'].unique())
    
    def get_categories_list(self, df):
        """
        Retorna lista de categorías únicas ordenada
        """
        return sorted(df['Categoría'].unique())
    
    def get_data_info(self, df):
        """
        Retorna información básica del dataset
        """
        return {
            'total_records': len(df),
            'unique_products': df['Producto'].nunique(),
            'unique_categories': df['Categoría'].nunique(),
            'date_range': (df['Fecha'].min(), df['Fecha'].max()),
            'total_sales': df['PrecioTotal'].sum(),
            'total_units': df['Cantidad'].sum()
        }