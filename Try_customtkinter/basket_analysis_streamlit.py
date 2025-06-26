#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Análisis de Cesta de Compra
Análisis de patrones de compra y generación de promociones usando Market Basket Analysis
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

class BasketAnalysisStreamlit:
    def __init__(self, df):
        self.df = df.copy()
        self.transactions = None
        self.frequent_itemsets = None
        self.rules = None
        
        # Diccionario de nombres de meses
        self.meses_nombres = {
            1: 'enero', 2: 'febrero', 3: 'marzo', 4: 'abril',
            5: 'mayo', 6: 'junio', 7: 'julio', 8: 'agosto',
            9: 'septiembre', 10: 'octubre', 11: 'noviembre', 12: 'diciembre'
        }
        
    def prepare_data(self):
        """Prepara los datos para el análisis"""
        print("Preparando datos para análisis de cesta de compra...")
        
        # Convertir fecha y filtrar año 2024
        self.df['Fecha'] = pd.to_datetime(self.df['Fecha'])
        self.df = self.df[self.df['Fecha'].dt.year == 2024].copy()
        
        if self.df.empty:
            raise ValueError("No hay datos disponibles del año 2024.")
            
        # Extraer información temporal
        self.df['Mes'] = self.df['Fecha'].dt.to_period('M')
        self.df['MesNombre'] = self.df['Fecha'].dt.month
        
        print(f" Datos preparados: {len(self.df)} registros del año 2024")
        
    def show_available_months(self):
        """Muestra los meses disponibles para análisis"""
        meses_con_datos = sorted(self.df['MesNombre'].unique())
        
        print("\n" + "="*50)
        print("          MESES DISPONIBLES (2024)")
        print("="*50)
        
        for i, mes_num in enumerate(meses_con_datos, 1):
            transacciones_mes = len(self.df[self.df['MesNombre'] == mes_num]['Fecha'].unique())
            print(f"{i:2d}. {self.meses_nombres[mes_num].capitalize()} "
                  f"({transacciones_mes} días con ventas)")
        
        return meses_con_datos
        
    def get_month_input(self, meses_disponibles):
        """Obtiene la selección del mes del usuario"""
        while True:
            print(f"\nIngrese el número del mes a analizar (1-{len(meses_disponibles)}):")
            try:
                seleccion = int(input("Opción: ").strip())
                if 1 <= seleccion <= len(meses_disponibles):
                    mes_numero = meses_disponibles[seleccion - 1]
                    return mes_numero
                else:
                    print(f"Por favor ingrese un número entre 1 y {len(meses_disponibles)}")
            except ValueError:
                print("Por favor ingrese un número válido")
                
    def analyze_month(self, mes_numero):
        """Analiza las transacciones de un mes específico"""
        datos_mes = self.df[self.df['MesNombre'] == mes_numero]
        mes_nombre = self.meses_nombres[mes_numero]
        
        print(f"\n" + "="*60)
        print(f"    ANÁLISIS DE CESTA DE COMPRA - {mes_nombre.upper()} 2024")
        print("="*60)
        
        # Crear transacciones agrupadas por fecha
        transacciones_mes = datos_mes.groupby('Fecha')['Producto'].apply(list).tolist()
        
        print(f" Resumen del análisis:")
        print(f"   • Período: {mes_nombre} 2024")
        print(f"   • Días con ventas: {len(transacciones_mes)}")
        print(f"   • Total de registros: {len(datos_mes)}")
        print(f"   • Productos únicos: {datos_mes['Producto'].nunique()}")
        
        if len(transacciones_mes) < 3:
            print("\n Pocas transacciones disponibles para generar análisis significativo.")
            print("   Se requieren al menos 3 días con ventas.")
            return False
            
        return self.perform_market_basket_analysis(transacciones_mes, mes_nombre)
        
    def perform_market_basket_analysis(self, transacciones, mes_nombre):
        """Ejecuta el análisis de market basket"""
        print(f"\n Ejecutando análisis de patrones de compra...")
        
        try:
            # Codificar transacciones
            te = TransactionEncoder()
            te_ary = te.fit(transacciones).transform(transacciones)
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
            
            print(f"   • Productos codificados: {len(te.columns_)}")
            
            # Calcular soporte mínimo dinámico
            min_support = max(0.15, 2 / len(transacciones))
            print(f"   • Soporte mínimo: {min_support:.2%}")
            
            # Encontrar itemsets frecuentes
            self.frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
            
            if self.frequent_itemsets.empty:
                print("\n No se encontraron patrones frecuentes de compra.")
                print("   Intente con un mes que tenga más transacciones.")
                return False
                
            print(f"   • Patrones frecuentes encontrados: {len(self.frequent_itemsets)}")
            
            # Generar reglas de asociación
            self.rules = association_rules(self.frequent_itemsets, 
                                         metric="confidence", 
                                         min_threshold=0.3)
            
            # Filtrar reglas triviales
            self.rules = self.rules[self.rules['confidence'] < 1.0]
            
            if self.rules.empty:
                print("\n No se generaron reglas de asociación significativas.")
                self.show_frequent_products(df_encoded, mes_nombre)
                return False
                
            print(f"   • Reglas de asociación generadas: {len(self.rules)}")
            
            # Mostrar resultados
            self.show_analysis_results(mes_nombre)
            self.show_detailed_rules()
            self.show_frequent_products(df_encoded, mes_nombre)
            
            return True
            
        except Exception as e:
            print(f" Error en el análisis: {e}")
            return False
            
    def show_analysis_results(self, mes_nombre):
        """Muestra los resultados principales del análisis"""
        print(f"\n" + "="*60)
        print(f"           PROMOCIONES RECOMENDADAS - {mes_nombre.upper()}")
        print("="*60)
        
        # Ordenar reglas por confianza y tomar las mejores
        top_rules = self.rules.sort_values(['confidence', 'lift'], ascending=False).head(5)
        
        if len(top_rules) == 0:
            print("No se encontraron promociones recomendadas.")
            return
            
        for i, (_, rule) in enumerate(top_rules.iterrows(), 1):
            antecedent = ', '.join(list(rule['antecedents']))
            consequent = ', '.join(list(rule['consequents']))
            confidence = rule['confidence'] * 100
            lift = rule['lift']
            support = rule['support'] * 100
            
            print(f"\n PROMOCIÓN {i}:")
            print(f"   Si compra: {antecedent}")
            print(f"   Sugerir:   {consequent}")
            print(f"   Confianza: {confidence:.1f}% | Lift: {lift:.2f} | Soporte: {support:.1f}%")
            
            # Interpretación del lift
            if lift > 1.5:
                interpretacion = "Combinación muy fuerte"
            elif lift > 1.2:
                interpretacion = "Combinación interesante"
            else:
                interpretacion = "Combinación moderada"
            
            print(f"   {interpretacion}")
            
    def show_detailed_rules(self):
        """Muestra análisis detallado de las reglas"""
        print(f"\n" + "="*60)
        print("              ANÁLISIS DETALLADO DE REGLAS")
        print("="*60)
        
        print("\n MÉTRICAS DE LAS REGLAS:")
        print("-" * 50)
        
        # Estadísticas generales
        print(f"• Total de reglas generadas: {len(self.rules)}")
        print(f"• Confianza promedio: {self.rules['confidence'].mean():.1%}")
        print(f"• Lift promedio: {self.rules['lift'].mean():.2f}")
        print(f"• Soporte promedio: {self.rules['support'].mean():.1%}")
        
        # Top reglas por diferentes métricas
        print(f"\nTOP 3 POR CONFIANZA:")
        top_confidence = self.rules.nlargest(3, 'confidence')
        
        for i, (_, rule) in enumerate(top_confidence.iterrows(), 1):
            ant = ', '.join(list(rule['antecedents']))
            con = ', '.join(list(rule['consequents']))
            print(f"   {i}. {ant} → {con} ({rule['confidence']:.1%})")
            
        print(f"\nTOP 3 POR LIFT:")
        top_lift = self.rules.nlargest(3, 'lift')
        
        for i, (_, rule) in enumerate(top_lift.iterrows(), 1):
            ant = ', '.join(list(rule['antecedents']))
            con = ', '.join(list(rule['consequents']))
            print(f"   {i}. {ant} → {con} (lift: {rule['lift']:.2f})")
            
    def show_frequent_products(self, df_encoded, mes_nombre):
        """Muestra productos más frecuentes del mes"""
        print(f"\n" + "="*60)
        print(f"         PRODUCTOS MÁS VENDIDOS - {mes_nombre.upper()}")
        print("="*60)
        
        # Calcular frecuencia de cada producto
        product_freq = df_encoded.mean().sort_values(ascending=False)
        
        print("\nTOP 10 PRODUCTOS MÁS FRECUENTES:")
        print("-" * 40)
        
        for i, (producto, freq) in enumerate(product_freq.head(10).items(), 1):
            print(f"{i:2d}. {producto:<25} {freq:.1%}")
            
        # Productos que aparecen en menos del 5% de transacciones
        rare_products = product_freq[product_freq < 0.05]
        if len(rare_products) > 0:
            print(f"\nProductos de baja frecuencia ({len(rare_products)}):")
            print("    (Aparecen en menos del 5% de las transacciones)")
            for producto in rare_products.head(5).index:
                print(f"    • {producto}")
            if len(rare_products) > 5:
                print(f"    ... y {len(rare_products) - 5} más")
                
    def generate_marketing_insights(self, mes_numero):
        """Genera insights para marketing"""
        mes_nombre = self.meses_nombres[mes_numero]
        
        print(f"\n" + "="*60)
        print(f"           INSIGHTS PARA MARKETING - {mes_nombre.upper()}")
        print("="*60)
        
        if self.rules is None or len(self.rules) == 0:
            print("No hay suficientes datos para generar insights de marketing.")
            return
            
        # Productos que más aparecen como antecedentes (productos ancla)
        all_antecedents = []
        for _, rule in self.rules.iterrows():
            all_antecedents.extend(list(rule['antecedents']))
            
        antecedent_counts = pd.Series(all_antecedents).value_counts()
        
        print(f"\nPRODUCTOS ANCLA (generan más ventas cruzadas):")
        for i, (producto, count) in enumerate(antecedent_counts.head(5).items(), 1):
            print(f"   {i}. {producto} (aparece en {count} reglas)")
            
        # Productos que más aparecen como consecuentes (productos sugeridos)
        all_consequents = []
        for _, rule in self.rules.iterrows():
            all_consequents.extend(list(rule['consequents']))
            
        consequent_counts = pd.Series(all_consequents).value_counts()
        
        print(f"\nPRODUCTOS PARA PROMOCIONAR (más sugeridos):")
        for i, (producto, count) in enumerate(consequent_counts.head(5).items(), 1):
            print(f"   {i}. {producto} (sugerido en {count} reglas)")
            
        # Recomendaciones estratégicas
        print(f"\nESTRATEGIAS RECOMENDADAS:")
        print("   • Colocar productos ancla en ubicaciones estratégicas")
        print("   • Crear bundles con productos frecuentemente asociados")
        print("   • Implementar descuentos en segunda compra")
        print("   • Desarrollar campañas de productos complementarios")
        
    def run_analysis(self):
        """Ejecuta el análisis completo de cesta de compra"""
        try:
            self.prepare_data()
            meses_disponibles = self.show_available_months()
            
            if not meses_disponibles:
                print("No hay datos disponibles para análisis.")
                return
                
            mes_seleccionado = self.get_month_input(meses_disponibles)
            
            if self.analyze_month(mes_seleccionado):
                # Si el análisis fue exitoso, mostrar opciones adicionales
                self.generate_marketing_insights(mes_seleccionado)
                
                print(f"\n" + "="*60)
                print("¿Desea analizar otro mes? (s/n): ", end="")
                if input().lower().startswith('s'):
                    self.run_analysis()
                    
        except Exception as e:
            print(f"Error en análisis de cesta: {e}")


if __name__ == "__main__":
    # Código de prueba
    print("Módulo de Análisis de Cesta de Compra - Modo de prueba")