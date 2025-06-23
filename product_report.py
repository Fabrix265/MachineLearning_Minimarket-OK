#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de reporte de productos
Genera reportes detallados de ventas por producto
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ProductReport:
    def __init__(self, df):
        self.df = df
        self.meses_abrev = {
            1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr',
            5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Ago',
            9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
        }
    
    def show_products_list(self):
        """Muestra la lista de productos disponibles"""
        productos = sorted(self.df['Producto'].unique())
        print(f"\n PRODUCTOS DISPONIBLES ({len(productos)} productos):")
        print("-" * 50)
        
        # Mostrar en columnas para mejor visualización
        for i, producto in enumerate(productos, 1):
            print(f"{i:2d}. {producto}")
        
        return productos
    
    def get_product_input(self):
        """Solicita al usuario que ingrese el nombre del producto"""
        productos = self.show_products_list()
        
        while True:
            try:
                print("\n Ingrese el producto a analizar:")
                print("   (Puede escribir el número o el nombre completo)")
                entrada = input("- ").strip()
                
                # Si ingresa un número
                if entrada.isdigit():
                    num = int(entrada)
                    if 1 <= num <= len(productos):
                        return productos[num - 1]
                    else:
                        print(f" Número inválido. Debe ser entre 1 y {len(productos)}")
                        continue
                
                # Si ingresa el nombre
                producto_encontrado = None
                for producto in productos:
                    if entrada.lower() == producto.lower():
                        producto_encontrado = producto
                        break
                
                if producto_encontrado:
                    return producto_encontrado
                else:
                    print(" Producto no encontrado. Intente nuevamente.")
                    print(" Tip: Puede usar el número del producto o escribir el nombre exacto")
                    
            except ValueError:
                print(" Entrada inválida. Intente nuevamente.")
    
    def filter_product_data(self, nombre_producto):
        """Filtra los datos por producto específico"""
        df_producto = self.df[self.df['Producto'].str.lower() == nombre_producto.lower()]
        
        if df_producto.empty:
            raise ValueError(f"No se encontraron datos para el producto: {nombre_producto}")
        
        return df_producto
    
    def generate_sales_summary(self, df_producto, nombre_producto):
        """Genera resumen de ventas por año y mes"""
        resumen = df_producto.groupby(['Año', 'Mes'])['PrecioTotal'].sum().reset_index()
        resumen['MesNombre'] = resumen['Mes'].map(self.meses_abrev)
        resumen = resumen.sort_values(['Año', 'Mes'])
        
        return resumen
    
    def plot_sales_by_year(self, resumen, nombre_producto):
        """Genera gráficos de líneas por año"""
        años_unicos = resumen['Año'].unique()
        
        for año in años_unicos:
            datos_año = resumen[resumen['Año'] == año]
            
            plt.figure(figsize=(12, 6))
            ax = sns.lineplot(data=datos_año, x='MesNombre', y='PrecioTotal', 
                             marker='o', linewidth=2.5, markersize=8)
            
            # Etiquetas sobre cada punto
            for i, row in datos_año.iterrows():
                ax.text(row['MesNombre'], row['PrecioTotal'] + max(datos_año['PrecioTotal']) * 0.02, 
                       f'S/. {row["PrecioTotal"]:.0f}',
                       color='black', ha='center', fontsize=10, fontweight='bold')
            
            plt.title(f" Ventas de '{nombre_producto}' - Año {año}", fontsize=14, fontweight='bold')
            plt.xlabel("Mes", fontsize=12)
            plt.ylabel("Ventas Totales (S/.)", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def plot_comparative_sales(self, resumen, nombre_producto):
        """Genera gráfico comparativo por años"""
        if len(resumen['Año'].unique()) > 1:
            plt.figure(figsize=(14, 8))
            ax = sns.barplot(data=resumen, x='MesNombre', y='PrecioTotal', 
                           hue='Año', palette='viridis')
            
            # Etiquetas numéricas sobre cada barra
            for p in ax.patches:
                height = p.get_height()
                if height > 0:  # Solo mostrar si hay valor
                    ax.annotate(f'S/. {height:.0f}',
                               (p.get_x() + p.get_width() / 2, height),
                               ha='center', va='bottom', fontsize=9)
            
            plt.title(f" Comparativo de Ventas - '{nombre_producto}'", fontsize=14, fontweight='bold')
            plt.xlabel("Mes", fontsize=12)
            plt.ylabel("Ventas Totales (S/.)", fontsize=12)
            plt.legend(title='Año', title_fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    def plot_units_sold(self, df_producto, nombre_producto):
        """Genera gráfico de unidades vendidas"""
        cant_vendida = df_producto.groupby(['Año', 'Mes'])['Cantidad'].sum().reset_index()
        cant_vendida['MesNombre'] = cant_vendida['Mes'].map(self.meses_abrev)
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=cant_vendida, x='MesNombre', y='Cantidad', 
                        hue='Año', palette='Set2')
        
        # Etiquetas sobre barras
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                           (p.get_x() + p.get_width() / 2, height),
                           ha='center', va='bottom', fontsize=9)
        
        plt.title(f" Unidades Vendidas - '{nombre_producto}'", fontsize=14, fontweight='bold')
        plt.xlabel("Mes", fontsize=12)
        plt.ylabel("Cantidad de Unidades", fontsize=12)
        if len(cant_vendida['Año'].unique()) > 1:
            plt.legend(title='Año')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_payment_methods(self, df_producto, nombre_producto):
        """Genera gráfico de métodos de pago"""
        plt.figure(figsize=(8, 8))
        
        # Contar métodos de pago
        payment_counts = df_producto['MétodoPago'].value_counts()
        
        # Crear gráfico de pie
        colors = plt.cm.Set3(np.linspace(0, 1, len(payment_counts)))
        wedges, texts, autotexts = plt.pie(payment_counts.values, 
                                          labels=payment_counts.index,
                                          autopct='%1.1f%%', 
                                          startangle=90,
                                          colors=colors)
        
        # Mejorar legibilidad
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.title(f" Métodos de Pago - '{nombre_producto}'", fontsize=14, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def generate_statistics(self, df_producto, nombre_producto):
        """Genera estadísticas del producto"""
        stats = {
            'ventas_totales': df_producto['PrecioTotal'].sum(),
            'unidades_vendidas': df_producto['Cantidad'].sum(),
            'precio_promedio': df_producto['PrecioUnitario'].mean(),
            'primera_venta': df_producto['Fecha'].min(),
            'ultima_venta': df_producto['Fecha'].max(),
            'categoria': df_producto['Categoría'].iloc[0],
            'total_transacciones': len(df_producto)
        }
        
        return stats
    
    def print_statistics(self, stats, nombre_producto):
        """Imprime las estadísticas del producto"""
        print("\n" + "="*60)
        print(f" RESUMEN ESTADÍSTICO - '{nombre_producto.upper()}'")
        print("="*60)
        print(f" Ventas totales:        S/. {stats['ventas_totales']:,.2f}")
        print(f" Unidades vendidas:     {stats['unidades_vendidas']:,}")
        print(f" Precio promedio:       S/. {stats['precio_promedio']:.2f}")
        print(f"  Categoría:            {stats['categoria']}")
        print(f" Total transacciones:   {stats['total_transacciones']:,}")
        print(f" Primera venta:         {stats['primera_venta'].strftime('%d/%m/%Y')}")
        print(f" Última venta:          {stats['ultima_venta'].strftime('%d/%m/%Y')}")
        
        # Calcular días activos
        dias_activos = (stats['ultima_venta'] - stats['primera_venta']).days + 1
        print(f" Días activos:          {dias_activos}")
        
        if dias_activos > 0:
            promedio_diario = stats['ventas_totales'] / dias_activos
            print(f" Promedio diario:       S/. {promedio_diario:.2f}")
        
        print("="*60)
    
    def generate_report(self):
        """Función principal que genera el reporte completo"""
        try:
            # 1. Solicitar producto
            nombre_producto = self.get_product_input()
            
            print(f"\n Generando reporte para: '{nombre_producto}'")
            print(" Procesando datos...")
            
            # 2. Filtrar datos
            df_producto = self.filter_product_data(nombre_producto)
            
            # 3. Generar resumen de ventas
            resumen = self.generate_sales_summary(df_producto, nombre_producto)
            
            # 4. Generar gráficos
            print(" Generando gráficos...")
            self.plot_sales_by_year(resumen, nombre_producto)
            self.plot_comparative_sales(resumen, nombre_producto)
            self.plot_units_sold(df_producto, nombre_producto)
            self.plot_payment_methods(df_producto, nombre_producto)
            
            # 5. Generar estadísticas
            stats = self.generate_statistics(df_producto, nombre_producto)
            self.print_statistics(stats, nombre_producto)
            
            print("\n Reporte generado exitosamente")
            
        except Exception as e:
            print(f" Error al generar el reporte: {e}")
            raise