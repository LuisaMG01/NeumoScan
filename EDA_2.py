"""
NeumoScan Project - Análisis Exploratorio de Datos (2 Datasets)
Team: HackIAdos | Data and Evaluation Lead

Este módulo proporciona funcionalidad EDA comprensiva para datasets de detección de neumonía.
Optimizado para 2 datasets principales debido a limitaciones de espacio.
Compatible con descargas de KaggleHub.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

#Configuración estilo de gráficos

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']


#0. Configuración Inicial: Descarga Datasets.
import kagglehub

print("NeumoScan EDA - Optimizado para 2 Datasets Principales")
print("-"*60)

datasets_paths = {}

print("0. Descargando datasets principales:")
print("-" * 40)

# Dataset principal 1 - Chest X-Ray Images (Pneumonia)
print("Dataset 1: Chest X-Ray Images (Pneumonia)")
try:
    datasets_paths['primary_pneumonia'] = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print(f"Primary Dataset 1 completado: {datasets_paths['primary_pneumonia']}")
except Exception as e:
    print(f"Error descargando dataset 1: {e}")
    print("Intentando continuar...")

print("\n")

# Dataset principal 2 - Felipe Salazar Pneumonia Dataset
print("Dataset 2: Felipe Salazar Pneumonia Dataset")
try:
    datasets_paths['felipe_pneumonia'] = kagglehub.dataset_download("felipesalazarn/neumonia-dataset")
    print(f"Primary Dataset 2 completado: {datasets_paths['felipe_pneumonia']}")
except Exception as e:
    print(f"Error descargando dataset 2: {e}")
    print("Intentando continuar...")

print(f"\nConfiguración completa.")
print(f"Datasets activos: {len([p for p in datasets_paths.values() if 'kagglehub' in str(p)])}")

print("-"*60)


#1. Diagnóstico para Datasets
def diagnose_datasets():
    print("\n1. Diagnóstico de Datasets")
    print("-"*50)
    
    for name, path in datasets_paths.items():
        print(f"\n{name}:")
        print(f"  Ruta: {path}")
        
        if os.path.exists(str(path)):
            print(f"  Estado: Existe")
            
            # Listar contenido del directorio principal
            try:
                contents = os.listdir(str(path))
                print(f"  Contenido ({len(contents)} elementos):")
                for item in contents[:10]:  # Mostrar primeros 10
                    item_path = os.path.join(str(path), item)
                    if os.path.isdir(item_path):
                        # Contar archivos en subdirectorio
                        try:
                            sub_contents = os.listdir(item_path)
                            # Filter out macOS metadata files
                            valid_images = [f for f in sub_contents if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('._')]
                            image_count = len(valid_images)
                            print(f"    📁 {item}.")
                        except:
                            print(f"    📁 {item} no accesible.")
                    else:
                        print(f" {item}")
                        
            except Exception as e:
                print(f"  Error listando contenido: {e}.")
        else:
            print(f"  Estado: No existe.")

diagnose_datasets()


#2. Búsqueda de Archivos
def find_real_image_folders():
    print("\n2. Exploración de Carpetas - Resumen")
    print("-"*50)
    
    for name, path in datasets_paths.items():
        print(f"\n{name}:")
        total_folders = 0
        total_images = 0
        
        if os.path.exists(str(path)):
            for root, dirs, files in os.walk(str(path)):
                if '__MACOSX' in root:
                    continue

                valid_images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('._')]
                
                if len(valid_images) > 50:
                    total_folders += 1
                    total_images += len(valid_images)
        
        print(f"  Total: {total_folders} carpetas con {total_images} imágenes.\n")

find_real_image_folders()

#4. EDA Datasets

class DatasetPneumoniaEDA:
    def __init__(self):
        self.datasets = {}
        self.all_dataset_info = {}
        self.comparison_stats = {}
        
    def add_dataset(self, name, path, description=""):
        if not os.path.exists(path):
            print(f"Ruta del dataset no encontrada: {path}")
            return False
            
        self.datasets[name] = {
            'path': path,
            'description': description,
            'dataset_info': {},
            'image_stats': {}
        }

        print(f"Dataset añadido: {name}.")
        print(f"   Ruta: {path}")
        if description:
            print(f"   Descripción: {description}.\n")
        return True
    
    def scan_dataset_structure(self, dataset_name=None):
        datasets_to_scan = [dataset_name] if dataset_name else list(self.datasets.keys())
        
        for ds_name in datasets_to_scan:
            if ds_name not in self.datasets:
                print(f"Dataset {ds_name} no encontrado.")
                continue
                
            #print(f"\nEscaneando Dataset: {ds_name}.")
            
            data_path = self.datasets[ds_name]['path']
            dataset_info = {}
            
            for root, dirs, files in os.walk(data_path):
                # Skip macOS metadata directories
                if '__MACOSX' in root:
                    continue
                    
                level = root.replace(data_path, '').count(os.sep)
                indent = ' ' * 2 * level
                dir_name = os.path.basename(root)
       #         print(f"{indent}{dir_name}/")
                
                # Count images in each directory
                image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('._')]
                if image_files and len(image_files) > 10:  # Only count directories with substantial images
                    subindent = ' ' * 2 * (level + 1)
    #                print(f"{subindent}{len(image_files)} imágenes")
                    
                    # Store directory info for medical classes
                    if dir_name.upper() in ['NORMAL', 'PNEUMONIA', 'BACTERIAL', 'VIRAL', 'COVID', '0', '1']:
                        # Map numeric labels to meaningful names for Felipe dataset
                        class_name = dir_name.upper()
                        if class_name == '0':
                            class_name = 'NORMAL'
                        elif class_name == '1':
                            class_name = 'PNEUMONIA'
                        
                        dataset_info[class_name] = {
                            'path': root,
                            'count': len(image_files),
                            'files': image_files[:15]  # Store first 15 filenames
                        }
            
            self.datasets[ds_name]['dataset_info'] = dataset_info
            self.all_dataset_info[ds_name] = dataset_info
    
    def analyze_single_dataset_distribution(self, dataset_name):
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} no encontrado.")
            return None
            
        dataset_info = self.datasets[dataset_name]['dataset_info']
        
        if not dataset_info:
            print(f"No se encontró información de clases para {dataset_name}.")
            return None
        
        print(f"\nAnálisis de Distribución de Clases - {dataset_name}")
        print("-"*50)
        
        # Crear dataframe resumen
        classes = list(dataset_info.keys())
        counts = [dataset_info[cls]['count'] for cls in classes]
        total = sum(counts)
        
        df_summary = pd.DataFrame({
            'Clase': classes,
            'Cantidad': counts,
            'Porcentaje': [count/total*100 for count in counts]
        })
        
        print(df_summary.to_string(index=False))
        
        # Visualización
        plt.figure(figsize=(15, 6))
        
        # Subplot 1: Gráfico de barras
        plt.subplot(1, 2, 1)
        colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6'][:len(classes)]
        bars = plt.bar(classes, counts, color=colors, alpha=0.8)
        plt.title(f'Distribución de Clases - {dataset_name}', fontsize=14, fontweight='bold')
        plt.ylabel('Número de Imágenes')
        plt.xlabel('Clase')
        plt.xticks(rotation=45)
        
        # Añadir etiquetas de valor en las barras
        max_count = np.max(counts) if counts else 0
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_count*0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Subplot 2: Gráfico circular
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=classes, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title(f'Distribución de Clases - {dataset_name}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return df_summary
    
    def compare_datasets(self):
        
        if len(self.all_dataset_info) < 2:
            print("Se necesitan al menos 2 datasets para comparar.")
            return
        
        # Crear dataframe de comparación
        comparison_data = []
        
        for ds_name, dataset_info in self.all_dataset_info.items():
            total_images = sum(info['count'] for info in dataset_info.values())
            
            for class_name, info in dataset_info.items():
                comparison_data.append({
                    'Dataset': ds_name,
                    'Clase': class_name,
                    'Cantidad': info['count'],
                    'Porcentaje': (info['count']/total_images)*100,
                    'Total_Imágenes': total_images
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        print("Resumen:")
        print(df_comparison.to_string(index=False))
        
        # Visualizaciones
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Comparación de tamaños de datasets
        dataset_totals = df_comparison.groupby('Dataset')['Total_Imágenes'].first()
        colors = ['#3498db', '#e74c3c']
        axes[0,0].bar(range(len(dataset_totals)), dataset_totals.values, color=colors)
        axes[0,0].set_title('Comparación de Tamaños de Datasets', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('Número de Imágenes')
        axes[0,0].set_xticks(range(len(dataset_totals)))
        axes[0,0].set_xticklabels(dataset_totals.index, rotation=45, ha='right')
        axes[0,0].grid(True, alpha=0.3)
        
        # Añadir etiquetas de valor
        max_count = np.max(dataset_totals.values)
        for i, (name, count) in enumerate(dataset_totals.items()):
            axes[0,0].text(i, count + max_count*0.01, 
                          f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # Distribución general de clases
        class_counts = df_comparison.groupby('Clase')['Cantidad'].sum()
        axes[0,1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
                      startangle=90, colors=['#3498db', '#e74c3c', '#f39c12', '#2ecc71'][:len(class_counts)])
        axes[0,1].set_title('Distribución Combinada de Clases', fontsize=14, fontweight='bold')
        
        # Distribución de clases por dataset (lado a lado)
        pivot_data = df_comparison.pivot_table(index='Clase', columns='Dataset', 
                                               values='Cantidad', fill_value=0)
        pivot_data.plot(kind='bar', ax=axes[1,0], color=['#3498db', '#e74c3c'])
        axes[1,0].set_title('Cantidad de Clases por Dataset', fontsize=14, fontweight='bold')
        axes[1,0].set_ylabel('Número de Imágenes')
        axes[1,0].legend(title='Dataset')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Comparación Normal vs Patológico
        df_comparison['Clase_Binaria'] = df_comparison['Clase'].apply(
            lambda x: 'NORMAL' if x == 'NORMAL' else 'PATOLÓGICO'
        )
        binary_summary = df_comparison.groupby(['Dataset', 'Clase_Binaria'])['Cantidad'].sum().unstack(fill_value=0)
        
        if len(binary_summary.columns) >= 2:
            binary_summary.plot(kind='bar', ax=axes[1,1], color=['#3498db', '#e74c3c'])
            axes[1,1].set_title('Normal vs Patológico por Dataset', fontsize=14, fontweight='bold')
            axes[1,1].set_ylabel('Número de Imágenes')
            axes[1,1].legend(title='Clase')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return df_comparison
    
    def analyze_sample_images(self, dataset_name, samples_per_class=5):
        """Analizar imágenes muestra de un dataset"""
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} no encontrado")
            return
        
        dataset_info = self.datasets[dataset_name]['dataset_info']
        
        print(f"\nAnálisis de Imágenes Muestra - {dataset_name}.")
        print("-"*40)
        
        image_stats = {}
        
        for class_name, info in dataset_info.items():
            print(f"\nClase: {class_name}")
            print("-" * 30)
            
            sample_files = info['files'][:samples_per_class]
            image_dimensions = []
            file_sizes = []
            
            for img_file in sample_files:
                img_path = os.path.join(info['path'], img_file)
                
                try:
                    # Obtener dimensiones de imagen
                    with Image.open(img_path) as img:
                        width, height = img.size
                        image_dimensions.append((width, height))
                    
                    # Obtener tamaño de archivo
                    file_size = os.path.getsize(img_path) / 1024  # KB
                    file_sizes.append(file_size)
                    
                except Exception as e:
                    print(f"Error procesando {img_file}: {e}")
            
            if image_dimensions:
                widths, heights = zip(*image_dimensions)
                avg_width, avg_height = np.mean(widths), np.mean(heights)
                avg_file_size = np.mean(file_sizes)
                
                print(f"  Dimensiones promedio: {avg_width:.0f} x {avg_height:.0f}")
                print(f"  Rango de dimensiones: {min(widths)}x{min(heights)} a {max(widths)}x{max(heights)}")
                print(f"  Tamaño promedio de archivo: {avg_file_size:.1f} KB")
                
                # Almacenar estadísticas
                image_stats[class_name] = {
                    'avg_width': avg_width,
                    'avg_height': avg_height,
                    'avg_file_size': avg_file_size,
                    'dimension_std': (np.std(widths), np.std(heights))
                }
        
        self.datasets[dataset_name]['image_stats'] = image_stats
    
    def visualize_sample_images(self, dataset_name, samples_per_class=3):
        """Mostrar imágenes muestra de un dataset"""
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} no encontrado")
            return
        
        dataset_info = self.datasets[dataset_name]['dataset_info']
        
        print(f"\nVisualizando Imágenes Muestra - {dataset_name}.")
        print("-"*50)
        
        num_classes = len(dataset_info)
        if num_classes == 0:
            print("No se encontraron clases en el dataset")
            return
        
        fig, axes = plt.subplots(num_classes, samples_per_class, 
                                figsize=(samples_per_class*4, num_classes*3))
        
        # Manejar casos de una sola clase o una sola muestra
        if num_classes == 1 and samples_per_class == 1:
            axes = np.array([[axes]])
        elif num_classes == 1:
            axes = axes.reshape(1, -1)
        elif samples_per_class == 1:
            axes = axes.reshape(-1, 1)
        
        images_loaded = 0
        
        for i, (class_name, info) in enumerate(dataset_info.items()):
            # Filter out macOS metadata files
            valid_files = [f for f in info['files'] if not f.startswith('._')]
            sample_files = valid_files[:samples_per_class]
            
            if not sample_files:
                print(f"  No hay archivos válidos para la clase {class_name}")
                continue
            
            for j, img_file in enumerate(sample_files):
                img_path = os.path.join(info['path'], img_file)
                
                try:
                    # Intentar cargar imagen con PIL primero
                    with Image.open(img_path) as pil_img:
                        img_array = np.array(pil_img)
                        
                        # Determinar el eje correcto
                        if num_classes == 1:
                            current_ax = axes[j]
                        else:
                            current_ax = axes[i, j]
                        
                        # Mostrar imagen
                        if len(img_array.shape) == 3:  # Color
                            current_ax.imshow(img_array)
                        else:  # Escala de grises
                            current_ax.imshow(img_array, cmap='gray')
                        
                        current_ax.set_title(f'{class_name}\n{img_file[:20]}...', fontsize=10)
                        current_ax.axis('off')
                        images_loaded += 1
                        
                except Exception as e:
                    print(f"Error cargando {img_file}: {e}")
                    # Mostrar mensaje de error en el subplot
                    if num_classes == 1:
                        current_ax = axes[j]
                    else:
                        current_ax = axes[i, j]
                    
                    current_ax.text(0.5, 0.5, f'Error cargando\n{img_file[:15]}...', 
                                   ha='center', va='center', transform=current_ax.transAxes,
                                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
                    current_ax.axis('off')
        
        plt.suptitle(f'Imágenes Muestra - {dataset_name} ({images_loaded} cargadas)', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        if images_loaded == 0:
            print("⚠ No se pudieron cargar imágenes. Verificando rutas...")
            # Diagnóstico adicional
            for class_name, info in dataset_info.items():
                valid_files = [f for f in info['files'] if not f.startswith('._')]
                if valid_files:
                    sample_path = os.path.join(info['path'], valid_files[0])
                    print(f"  {class_name}: {sample_path}")
                    if os.path.exists(sample_path):
                        print(f"    ✓ Archivo existe")
                    else:
                        print(f"    ✗ Archivo no encontrado")
                else:
                    print(f"  {class_name}: No hay archivos válidos (solo metadata de macOS)")
    
    def create_augmentation_strategy(self):
        
        # Analizar desbalance general de clases
        all_class_counts = {}
        total_images = 0
        
        for ds_name, dataset_info in self.all_dataset_info.items():
            for class_name, info in dataset_info.items():
                if class_name not in all_class_counts:
                    all_class_counts[class_name] = 0
                all_class_counts[class_name] += info['count']
                total_images += info['count']
        
        print("\nDistribución Combinada de Clases:")
        print("-" * 40)
        for class_name, count in all_class_counts.items():
            percentage = (count / total_images) * 100
            print(f"  {class_name}: {count:,} ({percentage:.1f}%)")
        
        # Calcular ratios de desbalance
        normal_count = all_class_counts.get('NORMAL', 0)
        pathological_counts = {k: v for k, v in all_class_counts.items() if k != 'NORMAL'}
        total_pathological = sum(pathological_counts.values())
        
        print(f"\nAnálisis de Desbalance:")
        print(f"  NORMAL: {normal_count:,}")
        print(f"  PATOLÓGICO: {total_pathological:,}")
        if normal_count > 0:
            print(f"  Ratio: {total_pathological/normal_count:.2f}:1")
        
        # Estrategia de aumentación
        print(f"\nData Augmentation Pipeline")
        print("-"*40)
        
        augmentation_plan = {
            'Data Augmentation Médico': {
                'brightness_contrast': 'Brillo ±20%, contraste 0.8-1.2',
                'rotation': '±15° para variaciones de posicionamiento del paciente',
                'zoom': '0.9-1.1 para variaciones de distancia',
                'horizontal_flip': 'Probabilidad 50% para orientación'
            },
            'Data Augmentation avanzado': {
                'elastic_deformation': 'Simular variaciones anatómicas',
                'gaussian_noise': 'Mejorar robustez al ruido de imagen',
                'histogram_equalization': 'Probabilidad 30% para mejorar contraste'
            }
        }
        
        for category, techniques in augmentation_plan.items():
            print(f"\n{category}:")
            for technique, description in techniques.items():
                print(f"  - {technique.replace('_', ' ').title()}: {description}")
        
        max_class_count = np.max(list(all_class_counts.values())) if all_class_counts else 0
        
        print("\n")
        print(f"Factores de Data Augmentation")
        print("-"*50)
        
        for class_name, count in all_class_counts.items():
            factor = max_class_count / count
            needed = max_class_count - count
            print(f"{class_name}:")
            print(f"  Actual: {count:,}")
            print(f"  Objetivo: {max_class_count:,}")
            print(f"  Factor: {factor:.2f}x")
            print(f"  Imágenes adicionales necesarias: {needed:,}")
            print()
        
        # Visualización
        self._plot_augmentation_strategy(all_class_counts, max_class_count)
    
    def _plot_augmentation_strategy(self, all_class_counts, max_class_count):
        """Graficar visualización de estrategia de aumentación"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Distribución actual vs objetivo
        classes = list(all_class_counts.keys())
        current_counts = list(all_class_counts.values())
        target_counts = [max_class_count] * len(classes)
        
        x = np.arange(len(classes))
        width = 0.35
        
        ax1.bar(x - width/2, current_counts, width, label='Actual', color='#e74c3c', alpha=0.7)
        ax1.bar(x + width/2, target_counts, width, label='Objetivo (Aumentado)', color='#2ecc71', alpha=0.7)
        
        ax1.set_title('Distribución Actual vs Objetivo de Clases', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Número de Imágenes')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Visualización de factores de aumentación
        factors = [max_class_count / count if count > 0 else 0 for count in current_counts]
        colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71'][:len(classes)]
        
        bars = ax2.bar(classes, factors, color=colors, alpha=0.7)
        ax2.set_title('Factores de Aumentación Requeridos', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Factor de Aumentación')
        ax2.set_xticklabels(classes, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Añadir etiquetas de valor
        for bar, factor in zip(bars, factors):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{factor:.1f}x', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def generate_final_report(self):
        print(f"\nReporte | Conclusiones")
        print("-"*50)
        
        # Estado de datasets
        available_datasets = len(self.datasets)
        print(f"Estado de Datasets:")
        print(f"   Datasets analizados: {available_datasets}.")
        print(f"   Completación del análisis: 100%")
        
        # Estadísticas totales
        total_images = sum(
            sum(info['count'] for info in dataset_info.values())
            for dataset_info in self.all_dataset_info.values()
        )
        
        print(f"\nVisión General:")
        print(f"  Total de imágenes: {total_images:,}.")
        print(f"  Número de datasets: {len(self.datasets)}.")
        
        # Resumen por dataset
        print(f"\nResumen de Datasets:")
        for ds_name, dataset_info in self.all_dataset_info.items():
            ds_total = sum(info['count'] for info in dataset_info.values())
            classes = list(dataset_info.keys())
            print(f"  {ds_name}:")
            print(f"    Imágenes: {ds_total:,}")
            print(f"    Clases: {len(classes)} ({', '.join(classes)})")
        

        print(f"\nAnálisis Completo! Listo para la siguiente fase.")
        print(f"Próximos pasos: Preprocesamiento → Desarrollo del Modelo → Evaluación")

print("Clase Dataset definida exitosamente.")

#5. Inicialización y configuración de los Datasets

# Inicializar EDA optimizado
dual_eda = DatasetPneumoniaEDA()

print("Configurando los datasets:")
print("-"*50)

# Configurar datasets disponibles
datasets_added = 0

# Dataset 1: Chest X-Ray Images (Pneumonia)
if 'primary_pneumonia' in datasets_paths and 'kagglehub' in str(datasets_paths['primary_pneumonia']):
    if dual_eda.add_dataset(
        name="Primary_Pneumonia",
        path=datasets_paths['primary_pneumonia'],
        description="Chest X-Ray Images (Pneumonia) - Paul Mooney | 5,863 imágenes con NORMAL, BACTERIAL, VIRAL"
    ):
        datasets_added += 1

# Dataset 2: Felipe Salazar Dataset
if 'felipe_pneumonia' in datasets_paths and 'kagglehub' in str(datasets_paths['felipe_pneumonia']):
    if dual_eda.add_dataset(
        name="Felipe_Pneumonia",
        path=datasets_paths['felipe_pneumonia'],
        description="Neumonia Dataset - Felipe Salazar | Etiquetado mejorado para diferenciación bacterial/viral"
    ):
        datasets_added += 1

print(f"\nConfiguración completa.")
print(f"Datasets activos: {datasets_added}/2")

if datasets_added >= 1:
    print("\n")
else:
    print("No se encontraron datasets válidos")
    print("Verifique que las descargas se completaron correctamente.")

#3. Análisis de estructura Datasets

if datasets_added >= 1:
    print("3. Análisis de estructura Datasets")
    print("-"*50)
    
    # Escanear estructura de todos los datasets
    dual_eda.scan_dataset_structure()

#Análisis individual.

if datasets_added >= 1:
    print(" 3.1 Análisis Individual")
    print("-"*40)
    
    # Analizar cada dataset individualmente
    for dataset_name in dual_eda.datasets.keys():
        print(f"\nAnalizando: {dataset_name}")
        df_analysis = dual_eda.analyze_single_dataset_distribution(dataset_name)

# Comparación entre Datasets

if datasets_added >= 2:
    print("\n")
    print("3.2 Comparación entre Datasets")
    print("-"*50)
    
    # Comparar todos los datasets
    df_comparison = dual_eda.compare_datasets()
elif datasets_added == 1:
    print("\nSolo 1 dataset disponible - saltando comparación")

#4. Análisis de Imágenes

if datasets_added >= 1:
    print("\n")
    print("4. Análisis de Imágenes")
    print("-"*50)
    
    # Analizar muestras de cada dataset
    for dataset_name in dual_eda.datasets.keys():
        print(f"\nAnálisis de Muestras: {dataset_name}.")
        dual_eda.analyze_sample_images(dataset_name, samples_per_class=8)

#Visualización de imágenes.

if datasets_added >= 1:
    print("\n")
    print("5. Visualización de Imágenes")
    print("-"*50)
    
    # Visualizar muestras de cada dataset
    for dataset_name in dual_eda.datasets.keys():
        print(f"\nMuestras Visuales: {dataset_name}.")
        dual_eda.visualize_sample_images(dataset_name, samples_per_class=4)

#6. Data Augmentation

if datasets_added >= 1:
    print("\n")
    print("6. Data Augmentation")
    print("-"*50)
    
    # Crear estrategia de aumentación basada en todos los datasets
    dual_eda.create_augmentation_strategy()

#7. Reporte/Conclusiones

def generate_final_2dataset_report():
    print(f"\nConclusiones")
    
    # Información sobre datasets disponibles
    available_datasets = len(dual_eda.datasets)
    expected_datasets = 2
    
    print(f"Estado del Proyecto:")
    print(f"   Datasets esperados: {expected_datasets}")
    print(f"   Datasets disponibles: {available_datasets}")
    print(f"   Completación: {(available_datasets/expected_datasets)*100:.1f}%")
    
    # Estado por categoría
    primary_datasets = ["Primary_Pneumonia", "Felipe_Pneumonia"]
    
    print(f"\nDATASETS PRINCIPALES ({len([d for d in primary_datasets if d in dual_eda.datasets])}/2):")
    for ds in primary_datasets:
        status = "Disponible" if ds in dual_eda.datasets else "Faltante"
        print(f"   • {ds}: {status}")
    
    # Generar estadísticas comprensivas
    if dual_eda.all_dataset_info:
        total_images = sum(
            sum(info['count'] for info in dataset_info.values())
            for dataset_info in dual_eda.all_dataset_info.values()
        )
        
        print(f"\nVisión General del Dataset:")
        print(f"  Total de imágenes: {total_images:,}")
        print(f"  Número de datasets analizados: {len(dual_eda.datasets)}")
        
        # Resumen por dataset
        print(f"\nResumen de Datasets:")
        for ds_name, dataset_info in dual_eda.all_dataset_info.items():
            ds_total = sum(info['count'] for info in dataset_info.values())
            classes = list(dataset_info.keys())
            print(f"  {ds_name}:")
            print(f"    Imágenes: {ds_total:,}")
            print(f"    Clases: {len(classes)} ({', '.join(classes)})")
    
    
    if available_datasets < expected_datasets:
        print(f"\nESTADO DE DATASETS FALTANTES:")
        missing = expected_datasets - available_datasets
        print(f"   {missing} dataset(s) aún necesitan ser configurados")
        print("   Análisis puede continuar con datasets disponibles")
    
    print(f"\nPróximos pasos")
    print("1. Pipeline de Preprocesamiento de Datos")
    print("2. Desarrollo de Arquitectura del Modelo")
    print("3. Framework de Evaluación")
    print("4. Optimización del Modelo")
    
    print(f"\nEDA Multi-Dataset Completo.")
    print(f"Listo para la siguiente fase: Preprocesamiento de datos y desarrollo del modelo")

# Generar reporte final específico para el proyecto NeumoScan
if datasets_added >= 1:
    generate_final_2dataset_report()

#Conclusiones

print("Análisis de datasets completado exitosamente")
print(f"Datasets procesados: {datasets_added}.")

print("\nResumen de Funcionalidades:")
print("-" * 50)
if datasets_added >= 1:
    print("- Descarga y configuración de datasets")
    print("- Escaneo de estructura de directorios")
    print("- Análisis de distribución de clases")
    if datasets_added >= 2:
        print("- Comparación entre datasets")
    print("- Análisis de imágenes muestra")
    print("- Visualización de imágenes")
    print("- Estrategia de aumentación de datos")
    print("- Reporte final comprensivo")
else:
    print("- No se pudieron procesar datasets")
    print("  Verifique conexión a internet y credenciales de Kaggle")

print("\nArchivos para continuar:")
print("-" * 40)
print("- 02_data_preprocessing.py (Preprocesamiento)")
print("- 03_data_augmentation.py (Aumentación)")  
print("- 04_model_architecture.py (Arquitectura del modelo)")
print("- 05_training_pipeline.py (Pipeline de entrenamiento)")
print("- 06_evaluation_metrics.py (Métricas de evaluación)")

print("Estado: EDA Fase 1 Completa.")
