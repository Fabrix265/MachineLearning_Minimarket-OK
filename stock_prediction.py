#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Predicción de Stock
Incluye predicción para productos existentes y nuevos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import sklearn
from packaging import version

class StockPrediction:
    def __init__(self, df):
        self.df = df.copy()
        self.df_grouped_existing = None
        self.df_grouped_new = None
        self.encoder_existing = None
        self.model_existing = None
        self.le_cat = None
        self.model_new = None
        self.prepare_data()
        
    def prepare_data(self):
        """Prepara los datos para ambos tipos de predicción"""
        # Para productos existentes
        self.prepare_existing_products_data()
        # Para productos nuevos
        self.prepare_new_products_data()
        
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
        features = self.df_grouped_existing[['Producto', 'Mes', 'Categoría', 'MétodoPago']]
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
        
    def show_available_products(self):
        """Muestra productos disponibles"""
        print("\nProductos disponibles para predicción:")
        print("-" * 50)
        productos = sorted(self.df_grouped_existing['Producto'].unique())
        for i, producto in enumerate(productos, 1):
            print(f"{i:2d}. {producto}")
        return productos
        
    def show_available_categories(self):
        """Muestra categorías disponibles"""
        print("\nCategorías disponibles:")
        print("-" * 30)
        categorias = sorted(self.le_cat.classes_)
        for i, categoria in enumerate(categorias, 1):
            print(f"{i}. {categoria}")
        return categorias
        
    def predict_existing_product(self):
        """Predice stock para producto existente"""
        print("\n=== PREDICCIÓN DE STOCK - PRODUCTO EXISTENTE ===")
        
        # Mostrar productos disponibles
        productos = self.show_available_products()
        
        try:
            # Obtener entrada del usuario
            print(f"\nIngrese el nombre del producto tal como aparece en la lista:")
            producto_input = input("Producto: ").strip()
            
            if producto_input not in productos:
                print(f" Producto '{producto_input}' no encontrado.")
                return
                
            mes_input = int(input("Ingrese el número del mes (1-12): "))
            if mes_input < 1 or mes_input > 12:
                print(" Mes debe estar entre 1 y 12.")
                return
            
            # Realizar predicción
            if producto_input not in self.df_grouped_existing['Producto'].values:
                print(f" Producto '{producto_input}' no encontrado en datos de entrenamiento.")
                return
                
            fila = self.df_grouped_existing[self.df_grouped_existing['Producto'] == producto_input].iloc[0]
            
            prod_df = pd.DataFrame({
                'Producto': [producto_input],
                'Categoría': [fila['Categoría']],
                'MétodoPago': [fila['MétodoPago']],
                'Mes': [mes_input],
                'PrecioUnitario': [fila['PrecioUnitario']],
                'PrecioTotal': [fila['PrecioTotal']],
            })
            
            encoded_input = self.encoder_existing.transform(prod_df[['Producto', 'Categoría', 'MétodoPago']])
            encoded_input_df = pd.DataFrame(encoded_input, columns=self.encoder_existing.get_feature_names_out())
            input_data = pd.concat([encoded_input_df, prod_df[['Mes', 'PrecioUnitario', 'PrecioTotal']].reset_index(drop=True)], axis=1)
            
            pred_log = self.model_existing.predict(input_data)[0]
            pred = np.expm1(pred_log)
            
            # Mostrar resultado
            print("\n" + "="*60)
            print(f" RESULTADO DE PREDICCIÓN")
            print("="*60)
            print(f"Producto: {producto_input}")
            print(f"Mes: {mes_input}")
            print(f"Stock recomendado: {round(pred)} unidades")
            print("="*60)
            
            # Mostrar métricas del modelo
            print(f"\n Métricas del modelo:")
            print(f"MAE:  {self.mae_existing:.2f}")
            print(f"RMSE: {self.rmse_existing:.2f}") 
            print(f"R²:   {self.r2_existing:.2f}")
            
        except ValueError as e:
            print(f" Error en entrada: {e}")
        except Exception as e:
            print(f" Error inesperado: {e}")
    
    def validate_trimester_month(self, mes, trimestre):
        """Valida coherencia mes-trimestre"""
        trimestre_calculado = ((mes - 1) // 3) + 1
        if trimestre != trimestre_calculado:
            raise ValueError(f"Trimestre {trimestre} no corresponde al mes {mes} (debería ser {trimestre_calculado}).")
    
    def predict_new_product(self):
        """Predice stock para producto nuevo"""
        print("\n=== PREDICCIÓN DE STOCK - PRODUCTO NUEVO ===")
        
        # Mostrar categorías disponibles
        categorias = self.show_available_categories()
        
        try:
            # Obtener entrada del usuario
            print(f"\nIngrese los datos del nuevo producto:")
            nombre_producto = input("Nombre del producto: ").strip()
            
            print("\nSeleccione la categoría:")
            categoria_input = input("Categoría: ").strip().lower()
            
            if categoria_input not in [cat.lower() for cat in categorias]:
                print(f" Categoría '{categoria_input}' no encontrada.")
                print("Categorías disponibles:", ", ".join(categorias))
                return
            
            precio_unitario = float(input("Precio unitario (S/): "))
            mes_input = int(input("Mes (1-12): "))
            
            if mes_input < 1 or mes_input > 12:
                print(" Mes debe estar entre 1 y 12.")
                return
            
            # Calcular trimestre automáticamente
            trimestre_calculado = ((mes_input - 1) // 3) + 1
            
            # Encontrar categoría exacta
            categoria_exacta = next((cat for cat in categorias if cat.lower() == categoria_input), None)
            
            # Realizar predicción
            categoria_encoded = self.le_cat.transform([categoria_exacta])[0]
            cat_price_prom = self.df_grouped_new[self.df_grouped_new['Categoria_encoded'] == categoria_encoded]['PrecioUnitario'].mean()
            precio_relativo = precio_unitario / cat_price_prom if cat_price_prom > 0 else 1.0
            
            X_new = pd.DataFrame({
                'Categoria_encoded': [categoria_encoded],
                'PrecioUnitario': [precio_unitario],
                'Precio_relativo': [precio_relativo],
                'Mes_num': [mes_input],
                'Trimestre': [trimestre_calculado]
            })
            
            pred = self.model_new.predict(X_new)[0]
            stock_recomendado = max(round(pred), 0)
            
            # Mostrar resultado
            print("\n" + "="*60)
            print(f" RESULTADO DE PREDICCIÓN")
            print("="*60)
            print(f"Producto: {nombre_producto}")
            print(f"Categoría: {categoria_exacta}")
            print(f"Precio unitario: S/ {precio_unitario}")
            print(f"Mes: {mes_input} (Trimestre {trimestre_calculado})")
            print(f"Stock recomendado: {stock_recomendado} unidades")
            print("="*60)
            
            # Mostrar métricas del modelo
            print(f"\n Métricas del modelo:")
            print(f"MAE:  {self.mae_new:.2f}")
            print(f"RMSE: {self.rmse_new:.2f}")
            print(f"R²:   {self.r2_new:.2f}")
            
        except ValueError as e:
            print(f" Error en entrada: {e}")
        except Exception as e:
            print(f" Error inesperado: {e}")
    
    def show_model_performance(self):
        """Muestra gráficos de rendimiento de los modelos"""
        # Gráfico para modelo de productos existentes
        features = self.df_grouped_existing[['Producto', 'Mes', 'Categoría', 'MétodoPago']]
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
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_test_exp, y_pred, alpha=0.6, color='navy')
        plt.plot([y_test_exp.min(), y_test_exp.max()], [y_test_exp.min(), y_test_exp.max()], 'r--')
        plt.xlabel("Cantidad Real")
        plt.ylabel("Cantidad Predicha")
        plt.title("Productos Existentes - Predicción vs Real")
        plt.grid(True, alpha=0.3)
        
        # Gráfico para modelo de productos nuevos
        features_new = ['Categoria_encoded', 'PrecioUnitario', 'Precio_relativo', 'Mes_num', 'Trimestre']
        X_new = self.df_grouped_new[features_new]
        y_new = self.df_grouped_new['Cantidad']
        
        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
        y_pred_new = self.model_new.predict(X_test_new)
        
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=y_test_new, y=y_pred_new, alpha=0.6)
        plt.plot([y_new.min(), y_new.max()], [y_new.min(), y_new.max()], 'r--')
        plt.xlabel("Cantidad Real")
        plt.ylabel("Cantidad Predicha")
        plt.title("Productos Nuevos - Predicción vs Real")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()