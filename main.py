#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Gestión de Minimarket - Machine Learning
Archivo principal con menú de opciones
"""

import os
import sys
from data_processor import DataProcessor
from product_report import ProductReport
from sales_prediction import SalesPrediction
from stock_prediction import StockPrediction
from profitability_analysis import ProfitabilityAnalysis
from basket_analysis import BasketAnalysis


class MinimarketSystem:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Carga y procesa los datos iniciales"""
        try:
            print("Cargando datos...")
            self.df = self.data_processor.load_and_preprocess_data('ventas_minimarket.csv')
            print(" Datos cargados correctamente")
        except Exception as e:
            print(f"Error al cargar datos: {e}")
            sys.exit(1)
    
    def show_main_menu(self):
        """Muestra el menú principal"""
        print("\n" + "="*60)
        print("           SISTEMA DE GESTIÓN MINIMARKET")
        print("="*60)
        print("1. Reporte de producto")
        print("2. Predicción de ventas futuras")
        print("3. Predicción de stock para producto existente")
        print("4. Predicción de stock para producto nuevo")
        print("5. Evaluación de rentabilidad")
        print("6. Análisis de cesta de compra")
        print("0. Salir")
        print("="*60)
    
    def run(self):
        """Ejecuta el sistema principal"""
        while True:
            self.show_main_menu()
            try:
                opcion = input("\nSeleccione una opción: ").strip()
                
                if opcion == '1':
                    self.reporte_producto()
                elif opcion == '2':
                    self.prediccion_ventas()
                elif opcion == '3':
                    self.prediccion_stock_existente()
                elif opcion == '4':
                    self.prediccion_stock_nuevo()
                elif opcion == '5':
                    self.evaluacion_rentabilidad()
                elif opcion == '6':
                    self.analisis_cesta()
                elif opcion == '0':
                    print("\n¡Gracias por usar el sistema!")
                    break
                else:
                    print("Opción no válida. Intente nuevamente.")
                    
            except KeyboardInterrupt:
                print("\n\n¡Hasta luego!")
                break
            except Exception as e:
                print(f"Error inesperado: {e}")
    
    def reporte_producto(self):
        """Módulo de reporte de producto"""
        print("\n" + "-"*50)
        print("           REPORTE DE PRODUCTO")
        print("-"*50)
        
        try:
            report = ProductReport(self.df)
            report.generate_report()
        except Exception as e:
            print(f"Error en reporte: {e}")
    
    def prediccion_ventas(self):
        """Módulo de predicción de ventas futuras"""
        print("\n" + "-"*50)
        print("        PREDICCIÓN DE VENTAS FUTURAS")
        print("-"*50)
        
        try:
            predictor = SalesPrediction(self.df)
            predictor.show_predictions()
        except Exception as e:
            print(f"Error en predicción: {e}")
    
    def prediccion_stock_existente(self):
        """Módulo de predicción de stock para producto existente"""
        print("\n" + "-"*50)
        print("    PREDICCIÓN DE STOCK - PRODUCTO EXISTENTE")
        print("-"*50)
        
        try:
            stock_pred = StockPrediction(self.df)
            stock_pred.predict_existing_product()
        except Exception as e:
            print(f"Error en predicción: {e}")
    
    def prediccion_stock_nuevo(self):
        """Módulo de predicción de stock para producto nuevo"""
        print("\n" + "-"*50)
        print("      PREDICCIÓN DE STOCK - PRODUCTO NUEVO")
        print("-"*50)
        
        try:
            stock_pred = StockPrediction(self.df)
            stock_pred.predict_new_product()
        except Exception as e:
            print(f"Error en predicción: {e}")
    
    def evaluacion_rentabilidad(self):
        """Módulo de evaluación de rentabilidad"""
        print("\n" + "-"*50)
        print("         EVALUACIÓN DE RENTABILIDAD")
        print("-"*50)
        
        try:
            analizador = ProfitabilityAnalysis(self.df)
            analizador.prepare_data()
            analizador.perform_clustering()
            analizador.show_cluster_analysis()
            analizador.show_all_products_by_cluster()
            analizador.plot_clusters()
        except Exception as e:
            print(f"Error en análisis de rentabilidad: {e}")
    
    def analisis_cesta(self):
        """Módulo de análisis de cesta de compra"""
        print("\n" + "-"*50)
        print("        ANÁLISIS DE CESTA DE COMPRA")
        print("-"*50)
        
        try:
            basket = BasketAnalysis(self.df)
            basket.run_analysis()
        except Exception as e:
            print(f"Error en análisis de cesta: {e}")


if __name__ == "__main__":
    system = MinimarketSystem()
    system.run()