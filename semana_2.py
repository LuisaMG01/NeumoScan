"""
NeumoScan Project - Binary Classification Evaluation Metrics
Team: HackIAdos  - Week 2

Este módulo implementa la división estratificada del dataset y métricas de evaluación
para clasificación binaria de neumonía (Normal vs Pneumonia).
Compatible con EDA_2.py y estructura del Action Plan.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import LabelEncoder
import cv2
from PIL import Image
from collections import Counter
import json
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

class BinaryPneumoniaEvaluator:
    """
    Evaluador comprensivo para clasificación binaria de neumonía
    Implementa Task 1.2.2 según Action Plan - Semana 2
    """
    
    def __init__(self, data_path=None, random_state=42):

        self.data_path = data_path
        self.random_state = random_state
        self.image_paths = []
        self.labels = []
        self.binary_labels = []
        self.splits = {}
        self.label_encoder = LabelEncoder()
        
        # según el action plan definimos unas métricas objetivo 
        self.target_metrics = {
            'binary_accuracy': 0.90,  # 90-95% según action plan
            'sensitivity': 0.95,      # >= 95% crítico para evitar falsos negativos
            'specificity': 0.85,      # >= 85%
            'f1_score': 0.88,         # >= 0.88
            'precision': 0.85,        # >= 85%
            'auc_roc': 0.90          # >= 0.90
        }
        
    def load_dataset_from_eda_structure(self, eda_dataset_info):      
        for dataset_name, dataset_info in eda_dataset_info.items():
            print(f"\nProcesando {dataset_name}:")
            
            for class_name, info in dataset_info.items():
                class_path = info['path']
                files = info['files']
                
                valid_files = [f for f in files if not f.startswith('._') and 
                             f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                print(f"  Clase {class_name}: {len(valid_files)} imágenes válidas")
                
                for img_file in valid_files:
                    img_path = os.path.join(class_path, img_file)
                    if os.path.exists(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(class_name)
        
        # convertir a etiquetas binarias (0: NORMAL, 1: PNEUMONIA)
        self.binary_labels = []
        for label in self.labels:
            if label == 'NORMAL':
                self.binary_labels.append(0)
            else:  # PNEUMONIA, BACTERIAL, VIRAL
                self.binary_labels.append(1)
        
        print(f"  Total de imágenes: {len(self.image_paths)}")
        print(f"  Distribución de clases: {Counter(self.labels)}")
        print(f"  Distribución binaria: Normal: {self.binary_labels.count(0)}, Pneumonia: {self.binary_labels.count(1)}")
        
        return self.image_paths, self.binary_labels
    
    def load_dataset_paths_manual(self, base_path):
        """
        Args:
            base_path (str): Path base del dataset
        """
        
        # estructura train/ o directa
        possible_paths = [
            os.path.join(base_path, 'train'),
            base_path
        ]
        
        data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                if any(d.upper() in ['NORMAL', 'PNEUMONIA'] for d in subdirs):
                    data_path = path
                    break
        
        if not data_path:
            raise ValueError(f"No se encontró estructura válida en {base_path}")
        
        for class_folder in os.listdir(data_path):
            class_path = os.path.join(data_path, class_folder)
            
            if os.path.isdir(class_path) and class_folder.upper() in ['NORMAL', 'PNEUMONIA']:
                print(f"  Procesando {class_folder}...")
                
                image_files = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('._')]
                
                for image_file in image_files:
                    image_path = os.path.join(class_path, image_file)
                    self.image_paths.append(image_path)
                    self.labels.append(class_folder.upper())
        
        # convwrtimos a etiquetas binarias
        self.binary_labels = [0 if label == 'NORMAL' else 1 for label in self.labels]
        
        print(f"Dataset cargado: {len(self.image_paths)} imágenes")
        print(f"Distribución: {Counter(self.labels)}")
        
        return self.image_paths, self.binary_labels
    
    def stratified_train_test_split(self, test_size=0.3, val_size=0.15):
        print(f"\nRealizando división estratificada del dataset...")
        print(f"  Train: {(1-test_size)*100:.0f}%, Test: {test_size*100:.0f}%")
        
        # primera división: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.image_paths,
            self.binary_labels,
            test_size=test_size,
            stratify=self.binary_labels,
            random_state=self.random_state
        )
        
        # seg division: train vs validation
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp,
                y_temp,
                test_size=val_size_adjusted,
                stratify=y_temp,
                random_state=self.random_state
            )
        else:
            X_train, X_val, y_train, y_val = X_temp, [], y_temp, []
        
        # divisiones
        self.splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        # estadística
        print(f"\nEstadísticas de división:")
        print(f"  Entrenamiento: {len(X_train):,} imágenes ({len(X_train)/len(self.image_paths):.1%})")
        if val_size > 0:
            print(f"  Validación:    {len(X_val):,} imágenes ({len(X_val)/len(self.image_paths):.1%})")
        print(f"  Test:          {len(X_test):,} imágenes ({len(X_test)/len(self.image_paths):.1%})")
        
        # estratificación
        train_dist = Counter(y_train)
        test_dist = Counter(y_test)
        
        print(f"\nVerificación de distribución de clases:")
        print(f"  Entrenamiento: Normal: {train_dist[0]} ({train_dist[0]/len(y_train):.1%}), Pneumonia: {train_dist[1]} ({train_dist[1]/len(y_train):.1%})")
        print(f"  Test:          Normal: {test_dist[0]} ({test_dist[0]/len(y_test):.1%}), Pneumonia: {test_dist[1]} ({test_dist[1]/len(y_test):.1%})")
        
        if val_size > 0:
            val_dist = Counter(y_val)
            print(f"  Validación:    Normal: {val_dist[0]} ({val_dist[0]/len(y_val):.1%}), Pneumonia: {val_dist[1]} ({val_dist[1]/len(y_val):.1%})")
        
        return self.splits
    
    def save_data_splits(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        # divisiones como archivos json
        splits_info = {}
        for split_name, data in self.splits.items():
            splits_info[split_name] = data
        
        with open(os.path.join(output_dir, 'data_splits.json'), 'w') as f:
            json.dump(splits_info, f, indent=2)
        
        #estadísticas resumen
        summary = {
            'total_images': len(self.image_paths),
            'train_size': len(self.splits['X_train']),
            'val_size': len(self.splits['X_val']) if self.splits['X_val'] else 0,
            'test_size': len(self.splits['X_test']),
            'class_distribution': dict(Counter(self.labels)),
            'binary_distribution': {'Normal': self.binary_labels.count(0), 'Pneumonia': self.binary_labels.count(1)},
            'random_state': self.random_state,
            'target_metrics': self.target_metrics
        }
        
        with open(os.path.join(output_dir, 'split_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Divisiones de datos guardadas en: {output_dir}")
    
    def calculate_binary_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Args:
            y_true (array): Etiquetas verdaderas
            y_pred (array): Etiquetas predichas
            y_pred_proba (array): Probabilidades predichas (opcional)
        Returns:
            dict: Diccionario con todas las métricas
        """
        metrics = {}
        
        # metricas
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary')
        metrics['recall'] = recall_score(y_true, y_pred, average='binary')
        metrics['sensitivity'] = metrics['recall']  # recall pero para binario
        metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')
        
        # matriz de confusión
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Valor Predictivo Negativo
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Valor Predictivo Positivo
        
        # ROC y AUC
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            metrics['auc_roc'] = auc(fpr, tpr)
            
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['auc_pr'] = average_precision_score(y_true, y_pred_proba)
        
        # métricas específicas médicas
        metrics['true_positives'] = tp
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        
        return metrics
    
    def generate_confusion_matrix(self, y_true, y_pred, class_names=['Normal', 'Pneumonia'], 
                                save_path=None, figsize=(8, 6)):

        cm = confusion_matrix(y_true, y_pred)
        
        # Crear figura
        plt.figure(figsize=figsize)
        
        # Calcular porcentajes
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Crear heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Cantidad'})
        
        plt.title('Matriz de confusión - Clasificación binaria de Neumonía', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Etiqueta predicha', fontsize=12)
        plt.ylabel('Etiqueta verdadera', fontsize=12)
        
        # Añadir anotaciones de porcentaje
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j+0.5, i+0.7, f'({cm_percent[i,j]:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Matriz de confusión guardada: {save_path}")
        
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None, figsize=(8, 6)):

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=figsize)
        
        # Graficar curva ROC
        plt.plot(fpr, tpr, color='#e74c3c', lw=2, 
                label=f'Curva ROC (AUC = {roc_auc:.3f})')
        
        # Línea de referencia diagonal
        plt.plot([0, 1], [0, 1], color='#34495e', lw=2, linestyle='--', 
                label='Clasificador aleatorio')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de falsos positivos (1 - especificidad)', fontsize=12)
        plt.ylabel('Tasa de verdaderos positivos (sensibilidad)', fontsize=12)
        plt.title('Curva ROC - Clasificación binaria de Neumonía', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Curva ROC guardada: {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, save_path=None, figsize=(8, 6)):
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=figsize)
        
        # Graficar curva PR
        plt.plot(recall, precision, color='#3498db', lw=2,
                label=f'Curva PR (AP = {avg_precision:.3f})')
        
        # Añadir línea base
        baseline = np.sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='#34495e', linestyle='--', 
                   label=f'Línea Base (AP = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (sensibilidad)', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Curva Precision-Recall - Clasificación binaria de Neumonía', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Curva PR guardada: {save_path}")
        
        plt.show()
    
    def create_metrics_comparison(self, metrics_dict, save_path=None, figsize=(15, 10)):
        metrics_to_plot = ['accuracy', 'sensitivity', 'specificity', 'f1_score', 'precision']
        if 'auc_roc' in metrics_dict:
            metrics_to_plot.append('auc_roc')
        
        current_values = [metrics_dict.get(metric, 0) for metric in metrics_to_plot]
        target_values = [self.target_metrics.get(f'binary_{metric}' if metric == 'accuracy' else metric, 0) 
                        for metric in metrics_to_plot]
        metric_names = [metric.replace('_', ' ').title() for metric in metrics_to_plot]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, current_values, width, label='Actual', 
                       color='#3498db', alpha=0.8)
        bars2 = ax1.bar(x + width/2, target_values, width, label='Objetivo del action plam', 
                       color='#e74c3c', alpha=0.8)
        
        ax1.set_xlabel('Métricas')
        ax1.set_ylabel('Puntuación')
        ax1.set_title('Rendimiento actual vs Objetivos del action plan', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metric_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
        angles += angles[:1] 
        
        current_values_radar = current_values + [current_values[0]]
        target_values_radar = target_values + [target_values[0]]
        
        ax2 = plt.subplot(2, 2, 2, projection='polar')
        ax2.plot(angles, current_values_radar, 'o-', linewidth=2, label='Actual', color='#3498db')
        ax2.fill(angles, current_values_radar, alpha=0.25, color='#3498db')
        ax2.plot(angles, target_values_radar, 'o-', linewidth=2, label='Objetivo', color='#e74c3c')
        ax2.fill(angles, target_values_radar, alpha=0.25, color='#e74c3c')
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metric_names)
        ax2.set_ylim(0, 1)
        ax2.set_title('Gráfico Radar de Rendimiento', fontweight='bold', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        if all(key in metrics_dict for key in ['true_positives', 'true_negatives', 
                                             'false_positives', 'false_negatives']):
            cm_data = np.array([
                [metrics_dict['true_negatives'], metrics_dict['false_positives']],
                [metrics_dict['false_negatives'], metrics_dict['true_positives']]
            ])
            
            sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Pneumonia'], 
                       yticklabels=['Normal', 'Pneumonia'], ax=ax3)
            ax3.set_title('Matriz de Confusión', fontweight='bold')
            ax3.set_xlabel('Predicho')
            ax3.set_ylabel('Real')
        
        medical_metrics = {
            'Sensibilidad\n(Recall)': metrics_dict.get('sensitivity', 0),
            'Especificidad': metrics_dict.get('specificity', 0),
            'VPP\n(Precision)': metrics_dict.get('ppv', metrics_dict.get('precision', 0)),
            'VPN': metrics_dict.get('npv', 0)
        }
        
        colors = []
        thresholds = [0.95, 0.85, 0.85, 0.85] 
        for i, (metric, value) in enumerate(medical_metrics.items()):
            if value >= thresholds[i]:
                colors.append('#2ecc71')  
            elif value >= thresholds[i] - 0.05:
                colors.append('#f39c12')  
            else:
                colors.append('#e74c3c')  
        
        bars = ax4.bar(medical_metrics.keys(), medical_metrics.values(), color=colors, alpha=0.8)
        ax4.set_title('Métricas de relevancia médica', fontweight='bold')
        ax4.set_ylabel('Puntuación')
        ax4.set_ylim(0, 1.1)
        ax4.grid(True, alpha=0.3)
        
        # Añadir líneas de objetivo
        for i, threshold in enumerate(thresholds):
            ax4.axhline(y=threshold, color='red', linestyle='--', alpha=0.5)
        
        # Añadir etiquetas de valor
        for bar, value in zip(bars, medical_metrics.values()):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparación de métricas guardada: {save_path}")
        
        plt.show()
    
    def generate_classification_report(self, y_true, y_pred, class_names=['Normal', 'Pneumonia'],
                                     save_path=None):
        report = classification_report(y_true, y_pred, target_names=class_names, 
                                     digits=4, output_dict=False)
        
        print("Reporte detallado de la clasificación:")
        print("=" * 50)
        print(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write("NeumoScan - Reporte de clasificación binaria\n")
                f.write("=" * 50 + "\n\n")
                f.write(report)
            print(f"Reporte de clasificación guardado: {save_path}")
        
        return report
    
    def cross_validation_setup(self, cv_folds=5):
        print(f"\nConfiguración de validación cruzada (K-Fold = {cv_folds})")
        print("=" * 50)
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_results = {
            'fold_accuracies': [],
            'fold_f1_scores': [],
            'fold_sensitivities': [],
            'fold_specificities': [],
            'fold_info': []
        }
        
        print("Configuración de Folds de validación cruzada:")
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.image_paths, self.binary_labels), 1):
            train_dist = Counter([self.binary_labels[i] for i in train_idx])
            val_dist = Counter([self.binary_labels[i] for i in val_idx])
            
            fold_info = {
                'fold': fold,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'train_normal': train_dist[0],
                'train_pneumonia': train_dist[1],
                'val_normal': val_dist[0],
                'val_pneumonia': val_dist[1]
            }
            
            cv_results['fold_info'].append(fold_info)
            
            print(f"  Fold {fold}: Train: {len(train_idx)} | Val: {len(val_idx)}")
            print(f"    Train dist: Normal: {train_dist[0]}, Pneumonia: {train_dist[1]}")
            print(f"    Val dist:   Normal: {val_dist[0]}, Pneumonia: {val_dist[1]}")
        
        print(f"\nMarco de validación cruzada listo para entrenamiento del modelo")
        return cv_results, skf
    
    def evaluate_model_performance(self, y_true, y_pred, y_pred_proba=None, 
                                 output_dir="results/binary_metrics/"):
        print("\nEvaluación Completa del Rendimiento del Modelo")
        print("=" * 60)
        os.makedirs(output_dir, exist_ok=True)
        metrics = self.calculate_binary_metrics(y_true, y_pred, y_pred_proba)
        
        print(f"\nMétricas Principales:")
        print(f"  Accuracy:      {metrics['accuracy']:.4f} (Objetivo: {self.target_metrics['binary_accuracy']:.3f})")
        print(f"  Sensitivity:   {metrics['sensitivity']:.4f} (Objetivo: {self.target_metrics['sensitivity']:.3f})")
        print(f"  Specificity:   {metrics['specificity']:.4f} (Objetivo: {self.target_metrics['specificity']:.3f})")
        print(f"  Precision:     {metrics['precision']:.4f} (Objetivo: {self.target_metrics['precision']:.3f})")
        print(f"  F1-Score:      {metrics['f1_score']:.4f} (Objetivo: {self.target_metrics['f1_score']:.3f})")
        if 'auc_roc' in metrics:
            print(f"  AUC-ROC:       {metrics['auc_roc']:.4f} (Objetivo: {self.target_metrics['auc_roc']:.3f})")
               
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        self.generate_confusion_matrix(y_true, y_pred, save_path=cm_path)
        
        if y_pred_proba is not None:
            roc_path = os.path.join(output_dir, 'roc_curve.png')
            self.plot_roc_curve(y_true, y_pred_proba, save_path=roc_path)
            
            pr_path = os.path.join(output_dir, 'precision_recall_curve.png')
            self.plot_precision_recall_curve(y_true, y_pred_proba, save_path=pr_path)
        
        metrics_path = os.path.join(output_dir, 'metrics_comparison.png')
        self.create_metrics_comparison(metrics, save_path=metrics_path)
        
        report_path = os.path.join(output_dir, 'classification_report.txt')
        self.generate_classification_report(y_true, y_pred, save_path=report_path)
        
        metrics_json_path = os.path.join(output_dir, 'metrics_summary.json')
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self._analyze_target_compliance(metrics)
        
        print(f"\nResultados guardados en: {output_dir}")
        return metrics
    
    def _analyze_target_compliance(self, metrics):
        print(f"\nAnálisis de cumplimiento de los objetivos:")
        print("-" * 55)
        
        compliance_report = []
        
        metric_mappings = {
            'Accuracy': ('accuracy', 'binary_accuracy'),
            'Sensitivity': ('sensitivity', 'sensitivity'),
            'Specificity': ('specificity', 'specificity'),
            'Precision': ('precision', 'precision'),
            'F1-Score': ('f1_score', 'f1_score'),
            'AUC-ROC': ('auc_roc', 'auc_roc')
        }
        
        objectives_met = 0
        total_objectives = 0
        
        for name, (metric_key, target_key) in metric_mappings.items():
            if metric_key in metrics and target_key in self.target_metrics:
                current_value = metrics[metric_key]
                target_value = self.target_metrics[target_key]
                met = current_value >= target_value
                
                status = "CUMPLIDO" if met else "NO CUMPLIDO"
                difference = current_value - target_value
                
                print(f"  {name:12}: {current_value:.4f} | Objetivo: {target_value:.3f} | {status}")
                print(f"               Diferencia: {difference:+.4f}")
                
                compliance_report.append({
                    'metric': name,
                    'current': current_value,
                    'target': target_value,
                    'met': met,
                    'difference': difference
                })
                
                if met:
                    objectives_met += 1
                total_objectives += 1
        #rwsumen
        compliance_rate = objectives_met / total_objectives if total_objectives > 0 else 0
        print(f"\nResumen de Cumplimiento:")
        print(f"  Objetivos cumplidos: {objectives_met}/{total_objectives} ({compliance_rate:.1%})")
        
        if compliance_rate >= 0.8:
            print(f"  Estado: EXCELENTE - Modelo cumple con expectativas del Action Plan")
        elif compliance_rate >= 0.6:
            print(f"  Estado: BUENO - Modelo cerca de los objetivos")
        else:
            print(f"  Estado: REQUIERE MEJORA - Ajustes necesarios")
        
        print(f"\nRecomendaciones:")
        for item in compliance_report:
            if not item['met']:
                if item['metric'] == 'Sensitivity':
                    print(f"  - Mejorar {item['metric']}: Crítico para evitar falsos negativos en diagnóstico")
                elif item['metric'] == 'Specificity':
                    print(f"  - Mejorar {item['metric']}: Reducir falsos positivos")
                else:
                    print(f"  - Mejorar {item['metric']}: Gap de {abs(item['difference']):.3f}")
    
    def generate_week2_summary_report(self, output_dir="results/binary_metrics/"):
        print(f"\nGenerando Reporte Resumen - Semana 2: Data Split and Evaluation")
        print("=" * 70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        total_images = len(self.image_paths)
        class_distribution = Counter(self.labels)
        binary_distribution = Counter(self.binary_labels)
        
        report_lines = [
            "NeumoScan - Reporte Semana 2: Data Split and Evaluation",
            "=" * 60,
            f"Team: HackIAdos | Data and Evaluation Lead",
            f"Fecha: Semana 2 del proyecto",
            "",
            "RESUMEN EJECUTIVO:",
            f"- Tarea completada: 1.2.2 - Data Split and Evaluation",
            f"- Dataset procesado: {total_images:,} imágenes",
            f"- División implementada: 70% train, 30% test (estratificada)",
            f"- Framework de evaluación: Configurado para modelo binario",
            "",
            "DISTRIBUCIÓN DEL DATASET:",
            f"- Total de imágenes: {total_images:,}",
            f"- Clases originales: {dict(class_distribution)}",
            f"- Distribución binaria: Normal: {binary_distribution[0]}, Pneumonia: {binary_distribution[1]}",
            "",
            "DIVISIÓN DE DATOS REALIZADA:"
        ]
        
        if self.splits:
            train_size = len(self.splits['X_train'])
            test_size = len(self.splits['X_test'])
            val_size = len(self.splits['X_val']) if self.splits['X_val'] else 0
            
            report_lines.extend([
                f"- Entrenamiento: {train_size:,} imágenes ({train_size/total_images:.1%})",
                f"- Test: {test_size:,} imágenes ({test_size/total_images:.1%})",
                f"- Validación: {val_size:,} imágenes ({val_size/total_images:.1%})" if val_size > 0 else "- Validación: No configurada",
                f"- Estratificación: Mantenida en todas las divisiones"
            ])
        
        report_lines.extend([
            "",
            "OBJETIVOS DE RENDIMIENTO ESTABLECIDOS:",
            f"- Accuracy Binaria: ≥{self.target_metrics['binary_accuracy']:.0%}",
            f"- Sensitivity (Recall): ≥{self.target_metrics['sensitivity']:.0%}",
            f"- Specificity: ≥{self.target_metrics['specificity']:.0%}",
            f"- F1-Score: ≥{self.target_metrics['f1_score']:.2f}",
            f"- Precision: ≥{self.target_metrics['precision']:.0%}",
            f"- AUC-ROC: ≥{self.target_metrics['auc_roc']:.2f}",
            "",
            "FRAMEWORK DE EVALUACIÓN CONFIGURADO:",
            "- Métricas de clasificación binaria: Implementadas",
            "- Matriz de confusión: Lista para generación",
            "- Curvas ROC y Precision-Recall: Configuradas",
            "- Validación cruzada: Framework preparado",
            "- Análisis de cumplimiento: Automatizado",
            "",
            "PRÓXIMOS PASOS (Semana 3):",
            "1. Implementar arquitectura del modelo CNN",
            "2. Aplicar transfer learning con ResNet18/EfficientNet",
            "3. Entrenar modelo binario (Normal vs Pneumonia)",
            "4. Evaluar rendimiento usando este framework",
            "",
            "ARCHIVOS GENERADOS:",
            "- data_splits.json: División de datos",
            "- split_summary.json: Resumen de estadísticas",
            "- metrics_framework.py: Framework de evaluación",
            ""        
            ])
        
        report_path = os.path.join(output_dir, 'week2_summary_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        for line in report_lines:
            print(line)
        
        print(f"\nReporte de Semana 2 guardado en: {report_path}")

def main():
    """
    Función principal para demostrar el pipeline de evaluación - Semana 2
    """    
    evaluator = BinaryPneumoniaEvaluator(random_state=42)
    
    print("- Task 1.2.2: Dsta split and evaluation")
    print("- División: 70% train, 30% test (estratificada)")
    print("- Métricas binarias: Normal vs Pneumonia")
    print("- Framework de evaluación: Preparado")
    
    print(f"\nObjetivos de rendimiento establecidos:")
    for metric, target in evaluator.target_metrics.items():
        print(f"  {metric}: ≥{target:.3f}")
    
    print(f"\nFramework listo para:")
    print("1. Cargar dataset desde EDA_2.py")
    print("2. División estratificada de datos")
    print("3. Evaluación comprensiva del modelo")
    print("4. Análisis de cumplimiento de objetivos")
    
    print(f"\nPróximo paso: Integrar con arquitectura del modelo (Semana 3)")
    
    return evaluator

if __name__ == "__main__":
    evaluator = main()