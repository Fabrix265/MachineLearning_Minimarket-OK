#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Predicción de Ventas Futuras
Análisis de tendencias y predicción usando regresión lineal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

class SalesPrediction:
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
        print("Evaluando modelos de predicción...")
        
        X = self.df[["ProductoCod", "CategoriaCod", "Mes", "PrecioUnitario", "Año"]]
        y = self.df["PrecioTotal"]
        
        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Modelos a evaluar
        modelos = {
            "LinearRegression": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        }
        
        print("\nResultados de evaluación:")
        print("-" * 50)
        
        for nombre, modelo in modelos.items():
            modelo.fit(X_train, y_train)
            pred = modelo.predict(X_test)
            
            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            r2 = r2_score(y_test, pred)
            
            print(f"\nModelo: {nombre}")
            print(f" MAE:  {mae:.2f}")
            print(f" RMSE: {rmse:.2f}")
            print(f" R²:   {r2:.2f}")
    
    def show_monthly_sales_by_year(self):
        """Muestra gráficos de ventas mensuales por año"""
        print("\nGenerando gráficos de ventas mensuales...")
        
        # Agrupar por año y mes
        ventas_mensuales = self.df.groupby(["Año", "Mes"])["PrecioTotal"].sum().reset_index()
        ventas_mensuales["Fecha"] = pd.to_datetime(
            ventas_mensuales["Año"].astype(str) + "-" + 
            ventas_mensuales["Mes"].astype(str) + "-01"
        )
        
        # Gráfico por año
        años = ventas_mensuales["Año"].unique()
        
        for año in años:
            datos_año = ventas_mensuales[ventas_mensuales["Año"] == año]
            
            plt.figure(figsize=(10, 6))
            plt.plot(datos_año["Mes"], datos_año["PrecioTotal"], 
                    marker="o", linewidth=2, markersize=8)
            plt.title(f"Ventas Totales por Mes - {año}", fontsize=14)
            plt.xlabel("Mes", fontsize=12)
            plt.ylabel("Ventas Totales (S/)", fontsize=12)
            plt.xticks(range(1, 13))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def show_annual_sales(self):
        """Muestra gráfico de ventas anuales"""
        print("\nGenerando gráfico de ventas anuales...")
        
        ventas_anuales = self.df.groupby("Año")["PrecioTotal"].sum().reset_index()
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(ventas_anuales["Año"].astype(str), 
                      ventas_anuales["PrecioTotal"], 
                      color='skyblue', edgecolor='navy')
        
        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'S/ {height:,.0f}', ha='center', va='bottom', fontsize=10)
        
        plt.title("Ventas Totales por Año", fontsize=14)
        plt.xlabel("Año", fontsize=12)
        plt.ylabel("Ventas Totales (S/)", fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def predict_future_sales(self, months_ahead=12):
        """Predice ventas futuras usando regresión lineal"""
        print(f"\nGenerando predicción para los próximos {months_ahead} meses...")
        
        # Filtrar datos desde abril 2024 en adelante
        ventas_filtradas = self.df[(self.df["Año"] == 2024) & (self.df["Mes"] >= 4)]
        
        if ventas_filtradas.empty:
            print("No hay datos suficientes desde abril 2024 para la predicción.")
            return
        
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
        
        # Gráfico de predicción
        plt.figure(figsize=(12, 6))
        
        # Ventas históricas
        fechas_historicas = pd.to_datetime(
            ventas_mensuales["Año"].astype(str) + "-" + 
            ventas_mensuales["Mes"].astype(str) + "-01"
        )
        plt.plot(fechas_historicas, ventas_mensuales["PrecioTotal"], 
                marker='o', label="Ventas Históricas", linewidth=2)
        
        # Predicciones
        plt.plot(df_prediccion["Fecha"], df_prediccion["VentasPredichas"], 
                marker='s', linestyle='--', color='red', 
                label="Ventas Predichas", linewidth=2)
        
        plt.title("Predicción de Ventas Futuras (Regresión Lineal)", fontsize=14)
        plt.xlabel("Fecha", fontsize=12)
        plt.ylabel("Ventas Totales (S/)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Mostrar predicciones numéricas
        print("\nPredicciones detalladas:")
        print("-" * 40)
        for i, row in df_prediccion.iterrows():
            fecha_str = row["Fecha"].strftime("%B %Y")
            print(f"{fecha_str}: S/ {row['VentasPredichas']:,.2f}")
        
        return df_prediccion
    
    def show_predictions(self):
        """Función principal que ejecuta todo el análisis"""
        print("Iniciando análisis de predicción de ventas...")
        
        # Evaluar modelos
        self.evaluate_models()
        
        # Mostrar gráficos históricos
        self.show_monthly_sales_by_year()
        self.show_annual_sales()
        
        # Predicción futura
        predicciones = self.predict_future_sales()
        
        print("\n Análisis de predicción de ventas completado")
        return predicciones