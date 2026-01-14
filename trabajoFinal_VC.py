!pip install mtcnn
from google.colab import drive
drive.mount('/content/drive')

from mtcnn.mtcnn import MTCNN
import cv2
import dlib
import numpy as np
import os
import time
from google.colab.patches import cv2_imshow
import math
import pandas as pd
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from skimage import exposure

#pip install mtcnn
# Configuración de estilo para matplotlib
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

# Inicialización de detectores
detector1 = MTCNN()
detector2 = dlib.get_frontal_face_detector()
modelFile = "/content/drive/MyDrive/UNI/face_detection/models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "/content/drive/MyDrive/UNI/face_detection/models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
classifier2 = cv2.CascadeClassifier('/content/drive/MyDrive/UNI/face_detection/models/haarcascade_frontalface2.xml')
lbp_face_cascade = cv2.CascadeClassifier('/content/drive/MyDrive/UNI/face_detection/models/lbpcascade_frontalface_improved.xml')
images = os.listdir('/content/drive/MyDrive/UNI/face_detection/faces/face_dark')



# Conteo REAL de imágenes
ground_truth_manual = {
    #'image01.jpg': 1,
    #'image02.jpg': 3,
    #'image03.jpg': 3,
    #'image04.jpg': 2,
    #'image05.jpg': 3,
    #'image06.jpg': 2,
    #'image07.jpg': 1,
    #'image08.jpg': 2,
    #'image09.jpg': 2,
    'image10.jpg': 2,
    #'image11.jpg': 6,
    #'image12.jpg': 3,
    'image13.jpg': 3,
    'image14.jpg': 7,
    'image15.jpg': 4,
    'image16.jpg': 2,
    '2015_06246': 3,
    '2015_06247': 1,
    '2015_06248': 2,
    '2015_06249': 1,
    '2015_06250': 1,
    '2015_06253': 1,
    '2015_06254': 1,
    '2015_06255': 5,
    '2015_06256': 3,
    '2015_06257': 1,
    '2015_06258': 5,
    '2015_06259': 2,
    '2015_06260': 3,
    '2015_06261': 1,
    '2015_06262': 4,
    '2015_06263': 1,
    '2015_06264': 1,
    '2015_06265': 1,
    '2015_06266': 1,
    '2015_06267': 1,
    '2015_06268': 1,
    '2015_06269': 2,
    '2015_06270': 1,
    '2015_06271': 1,
    '2015_06272': 3,
    '2015_06273': 2,
    '2015_06274': 1,
    '2015_06275': 3,
    '2015_06276': 27,
    '2015_06277': 2,
    '2015_06278': 1,
    '2015_06279': 6,
    '2015_06280': 2,
    '2015_06281': 4,
    '2015_06282': 4,
    '2015_06283': 3,
    '2015_06284': 8,
    '2015_06285': 5,
    '2015_06286': 25,
    '2015_06287': 6,
    '2015_06288': 1,
    '2015_06289': 6,
    '2015_06290': 2,
    '2015_06291': 2,
    '2015_06292': 1,
    '2015_06293': 5,
    '2015_06294': 4,
    '2015_06295': 1,
    '2015_06296': 10,
    '2015_06297': 4,
    #'image17.jpg': 15,
    #'image18.jpg': 26,
    #'image19.jpg': 24,
    #'image20.jpg': 5,
}

# Listas para almacenar resultados globales
resultados_globales = []

def crear_resumen_ejecucion():

    if not resultados_globales:
        return

    df_global = pd.DataFrame(resultados_globales)

    # Métricas
    resumen = {
        'Método': [],
        'Tiempo Promedio (s)': [],
        'Total Caras Detectadas': [],
        'Imágenes Procesadas': [],
        'Caras por Imagen': []
    }

    metodos = df_global['Método'].unique()
    for metodo in metodos:
        datos_metodo = df_global[df_global['Método'] == metodo]
        resumen['Método'].append(metodo)
        resumen['Tiempo Promedio (s)'].append(round(datos_metodo['Tiempo (segundos)'].mean(), 4))
        resumen['Total Caras Detectadas'].append(datos_metodo['Caras detectadas'].sum())
        resumen['Imágenes Procesadas'].append(len(datos_metodo))
        resumen['Caras por Imagen'].append(round(datos_metodo['Caras detectadas'].mean(), 2))

    df_resumen = pd.DataFrame(resumen)

    # Mostrar resumen
    print("="*80)
    print("RESUMEN EJECUTIVO - COMPARATIVA DE MÉTODOS DE DETECCIÓN")
    print("="*80)
    display(df_resumen.style.background_gradient(cmap='Blues'))
    print("="*80)

    # Gráficas comparativas
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))


    # Gráfica 1: Tiempos promedio
    sns.barplot(data=df_resumen, x='Método', y='Tiempo Promedio (s)', ax=ax1, color='#1f77b4')
    ax1.set_title('Tiempo Promedio de Ejecución por Método')
    ax1.set_ylabel('Tiempo (segundos)')
    ax1.tick_params(axis='x', rotation=45)


    # Gráfica 2: Caras detectadas promedio
    sns.barplot(data=df_resumen, x='Método', y='Caras por Imagen', ax=ax2, color='#ff7f0e')
    ax2.set_title('Promedio de Caras Detectadas por Imagen')
    ax2.set_ylabel('Número de Caras')
    ax2.tick_params(axis='x', rotation=45)

    # Gráfica 3: Eficiencia (caras/tiempo)
    df_resumen['Eficiencia'] = df_resumen['Caras por Imagen'] / df_resumen['Tiempo Promedio (s)']
    sns.barplot(data=df_resumen, x='Método', y='Eficiencia', ax=ax3, color='#2ca02c')
    ax3.set_title('Eficiencia (Caras Detectadas por Segundo)')
    ax3.set_ylabel('Caras/Segundo')
    ax3.tick_params(axis='x', rotation=45)

    # Gráfica 4: Distribución de tiempos
    tiempos_data = []
    for metodo in metodos:
        tiempos = df_global[df_global['Método'] == metodo]['Tiempo (segundos)']
        for tiempo in tiempos:
            tiempos_data.append({'Método': metodo, 'Tiempo': tiempo})

    df_tiempos = pd.DataFrame(tiempos_data)
    sns.boxplot(data=df_tiempos, x='Método', y='Tiempo', ax=ax4, color='#d62728')
    ax4.set_title('Distribución de Tiempos de Ejecución')
    ax4.set_ylabel('Tiempo (segundos)')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def calcular_metricas_reales(detecciones, ground_truth):
    """
    Calcula métricas de evaluación Precision, Recall, F1-Score, Accuracy

    Args:
        detecciones: número de caras detectadas por el algoritmo
        ground_truth: número real de caras en la imagen

    Returns:
        dict con las métricas calculadas
    """
    # True Positives: mínimo entre detectadas y reales
    tp = min(detecciones, ground_truth)

    # False Positives: detecciones extra más allá de las reales
    fp = max(0, detecciones - ground_truth)

    # False Negatives: caras reales no detectadas
    fn = max(0, ground_truth - detecciones)

    # Cálculo de métricas
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    accuracy = tp / ground_truth if ground_truth > 0 else 0

    return {
        'True Positives': tp,
        'False Positives': fp,
        'False Negatives': fn,
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1_score, 4),
        'Accuracy': round(accuracy, 4)
    }


def mostrar_metricas_avanzadas_paper():
    """
    Muestra métricas completas según el formato del paper
    """
    if not resultados_con_metricas:
        print("No hay datos de métricas para mostrar")
        return

    df_metricas = pd.DataFrame(resultados_con_metricas)

    # Métricas agregadas por método
    resumen_metricas = []

    for metodo in df_metricas['Método'].unique():
        datos_metodo = df_metricas[df_metricas['Método'] == metodo]

        resumen_metricas.append({
            'Método': metodo,
            'Precision Promedio': round(datos_metodo['Precision'].mean(), 4),
            'Recall Promedio': round(datos_metodo['Recall'].mean(), 4),
            'F1-Score Promedio': round(datos_metodo['F1-Score'].mean(), 4),
            'Accuracy Promedio': round(datos_metodo['Accuracy'].mean(), 4),
            'Tiempo Promedio (s)': round(datos_metodo['Tiempo (s)'].mean(), 4),
            'Total Caras Detectadas': datos_metodo['Caras Detectadas'].sum(),
            'Total True Positives': datos_metodo['True Positives'].sum(),
            'Total False Positives': datos_metodo['False Positives'].sum(),
            'Total False Negatives': datos_metodo['False Negatives'].sum()
        })

    df_resumen = pd.DataFrame(resumen_metricas)

    print("="*100)
    print("MÉTRICAS DE EVALUACIÓN ")
    print("="*100)

    # Aplicar formato estilo paper
    styled_metrics = df_resumen.style\
        .format({
            'Precision Promedio': '{:.4f}',
            'Recall Promedio': '{:.4f}',
            'F1-Score Promedio': '{:.4f}',
            'Accuracy Promedio': '{:.4f}',
            'Tiempo Promedio (s)': '{:.4f}'
        })\
        .background_gradient(subset=['Precision Promedio', 'Recall Promedio', 'F1-Score Promedio'],
                           cmap='YlGnBu')\
        .background_gradient(subset=['Tiempo Promedio (s)'], cmap='YlOrRd_r')\
        .set_properties(**{
            'text-align': 'center',
            'border': '1px solid black'
        })

    display(styled_metrics)

    # Gráficas comparativas como en el paper
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Comparación de Precision, Recall, F1-Score
    metrics_to_plot = ['Precision Promedio', 'Recall Promedio', 'F1-Score Promedio']
    x = np.arange(len(df_resumen))
    width = 0.25

    for i, metric in enumerate(metrics_to_plot):
        ax1.bar(x + i*width, df_resumen[metric], width, label=metric, alpha=0.8)

    ax1.set_xlabel('Método')
    ax1.set_ylabel('Score')
    ax1.set_title('Comparación: Precision, Recall y F1-Score\n(Como en el paper)')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(df_resumen['Método'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Accuracy vs Tiempo
    ax2.scatter(df_resumen['Tiempo Promedio (s)'], df_resumen['Accuracy Promedio'], s=100)
    for i, row in df_resumen.iterrows():
        ax2.annotate(row['Método'],
                    (row['Tiempo Promedio (s)'], row['Accuracy Promedio']),
                    xytext=(5, 5), textcoords='offset points')
    ax2.set_xlabel('Tiempo Promedio (s)')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Relación: Tiempo vs Accuracy\n(Análisis de eficiencia)')
    ax2.grid(True, alpha=0.3)

    # 3. Métricas de detección
    detection_metrics = ['Total True Positives', 'Total False Positives', 'Total False Negatives']
    bottom = np.zeros(len(df_resumen))

    for metric in detection_metrics:
        ax3.bar(df_resumen['Método'], df_resumen[metric], label=metric, bottom=bottom, alpha=0.8)
        bottom += df_resumen[metric]

    ax3.set_title('Análisis de Detecciones: TP, FP, FN')
    ax3.set_ylabel('Número de Detecciones')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()

    # 4. Radar chart de rendimiento
    ax4 = fig.add_subplot(2, 2, 4, polar=True)

    metrics_radar = ['Precision Promedio', 'Recall Promedio', 'F1-Score Promedio', 'Accuracy Promedio']
    angles = np.linspace(0, 2*np.pi, len(metrics_radar), endpoint=False).tolist()
    angles += angles[:1]

    for i, metodo in enumerate(df_resumen['Método']):
        valores = df_resumen[df_resumen['Método'] == metodo][metrics_radar].values[0].tolist()
        valores += valores[:1]
        ax4.plot(angles, valores, 'o-', linewidth=2, label=metodo)
        ax4.fill(angles, valores, alpha=0.1)

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(['Precision', 'Recall', 'F1-Score', 'Accuracy'])
    ax4.set_title('Rendimiento Comparativo\n(Radar Chart)')
    ax4.legend(bbox_to_anchor=(1.3, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    return df_resumen

def visualizar_resultados_individuales(imagen_nombre, resultados, imagenes, titulos):
    """Visualiza resultados individuales para cada imagen"""

    # Crear DataFrame para esta imagen
    df_individual = pd.DataFrame(resultados)

    # Aplicar estilo al DataFrame
    styled_df = df_individual.style\
    .bar(subset=['Tiempo (segundos)'], color='#7EB09B')\
    .bar(subset=['Caras detectadas'], color='#6A9FB5')\
    .set_properties(**{
        'background-color': '#FFFFFF',
        'border': '1px solid #E1E8ED',
        'text-align': 'center',
        'color': '#2C3E50'
    })\
    .set_table_styles([{
        'selector': 'th',
        'props': [('background-color', '#4A6572'),
                 ('color', 'white'),
                 ('font-weight', 'bold'),
                 ('border', '1px solid #34495E')]
    }, {
        'selector': 'td',
        'props': [('border', '1px solid #E1E8ED')]
    }, {
        'selector': 'tr:hover',
        'props': [('background-color', '#F8FBFE')]
    }])


    # Mostrar resultados
    print("="*70)
    print(f"   ANÁLISIS DETALLADO: {imagen_nombre}")
    print("="*70)
    display(styled_df)

    # Encontrar el mejor método por criterios
    mejor_tiempo = df_individual.loc[df_individual['Tiempo (segundos)'].idxmin()]
    mejor_deteccion = df_individual.loc[df_individual['Caras detectadas'].idxmax()]

    print("\n**OBSERVACIÓN:**")
    print(f"   • Más rápido: {mejor_tiempo['Método']} ({mejor_tiempo['Tiempo (segundos)']}s)")
    print(f"   • Más detecciones: {mejor_deteccion['Método']} ({mejor_deteccion['Caras detectadas']} caras)")

    # Visualización de imágenes
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Comparativa de Métodos - {imagen_nombre}', fontsize=16, fontweight='bold')

    colores = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

    for idx, (ax, img, titulo, color) in enumerate(zip(axes.flat, imagenes, titulos, colores)):
        # Convertir BGR a RGB para visualización
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(f'{titulo}\n({resultados["Caras detectadas"][idx]} caras, {resultados["Tiempo (segundos)"][idx]}s)',
                    fontweight='bold', color=color, fontsize=12)
        ax.axis('off')

        # Añadir recuadro informativo
        info_text = f"Caras: {resultados['Caras detectadas'][idx]}\nTiempo: {resultados['Tiempo (segundos)'][idx]}s"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()
#************************************************************************************************************
def diagnostico_metricas():
    """
    Función de diagnóstico para encontrar inconsistencias en las métricas
    """
    print("INICIANDO DIAGNÓSTICO DE MÉTRICAS")
    print("="*80)

    # Contadores globales
    total_imagenes = 0
    diagnostico_detallado = []

    # Revisar cada imagen procesada
    for image in images:
        if image.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join('/content/drive/MyDrive/UNI/face_detection/faces/face_dark', image)
            img = cv2.imread(img_path)

            if img is None:
                continue

            total_imagenes += 1

            # Verificar si tiene ground truth
            tiene_gt = image in ground_truth_manual
            gt_caras = ground_truth_manual[image] if tiene_gt else "Sin GT"

            # Buscar esta imagen en resultados_globales
            datos_imagen = [r for r in resultados_globales if r['Imagen'] == image]

            if datos_imagen:
                # Encontrar detecciones de MTCNN para esta imagen
                mtcnn_datos = [r for r in datos_imagen if r['Método'] == 'MTCNN']
                if mtcnn_datos:
                    caras_mtcnn = mtcnn_datos[0]['Caras detectadas']

                    diagnostico_detallado.append({
                        'Imagen': image,
                        'Tiene_GT': tiene_gt,
                        'GT_Caras': gt_caras,
                        'MTCNN_Detectadas': caras_mtcnn,
                        'En_Metricas_Paper': tiene_gt
                    })

                    # Mostrar advertencias específicas
                    if tiene_gt and caras_mtcnn > gt_caras + 2:
                        print(f"⚠️  POSIBLE SOBREDETECCIÓN: {image}")
                        print(f"    GT: {gt_caras} | MTCNN: {caras_mtcnn} | Diferencia: {caras_mtcnn - gt_caras}")

    # Análisis agregado
    print(f"\n📊 RESUMEN DIAGNÓSTICO:")
    print(f"   Total imágenes procesadas: {total_imagenes}")
    print(f"   Imágenes con Ground Truth: {len([d for d in diagnostico_detallado if d['Tiene_GT']])}")
    print(f"   Imágenes en métricas paper: {len([d for d in diagnostico_detallado if d['En_Metricas_Paper']])}")

    # Comparar totals entre ambas tablas
    total_caras_ejecutivo = sum([r['Caras detectadas'] for r in resultados_globales if r['Método'] == 'MTCNN'])
    total_caras_paper = sum([d['MTCNN_Detectadas'] for d in diagnostico_detallado if d['En_Metricas_Paper']])

    print(f"\n🔍 COMPARACIÓN DE TOTALES:")
    print(f"   Tabla Ejecutivo - Total caras MTCNN: {total_caras_ejecutivo}")
    print(f"   Tabla Paper - Total caras MTCNN: {total_caras_paper}")
    print(f"   Diferencia: {total_caras_ejecutivo - total_caras_paper}")

    # Mostrar detalles de algunas imágenes problemáticas
    print(f"\n📋 MUESTRA DE DETALLES (primeras 5 imágenes):")
    for i, detalle in enumerate(diagnostico_detallado[:5]):
        print(f"   {i+1}. {detalle['Imagen']}: GT={detalle['GT_Caras']}, MTCNN={detalle['MTCNN_Detectadas']}")

# Función de diagnóstico específico para el cálculo de métricas
def diagnostico_calculo_metricas():
    """
    Diagnóstico específico del cálculo de Precision, Recall, F1-Score
    """
    print("\n🔧 DIAGNÓSTICO CÁLCULO MÉTRICAS")
    print("="*80)

    # Simular cálculo con datos reales
    print("Ejemplo de cálculo para una imagen:")
    print("Si GT=3 y MTCNN detecta=5:")

    gt = 3
    detectadas = 5

    tp = min(detectadas, gt)          # = 3
    fp = max(0, detectadas - gt)      # = 2
    fn = max(0, gt - detectadas)      # = 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # = 3/5 = 0.6
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0     # = 3/3 = 1.0

    print(f"   TP={tp}, FP={fp}, FN={fn}")
    print(f"   Precision = {precision:.2f}, Recall = {recall:.2f}")

    # Verificar el código actual
    print(f"\n📝 REVISIÓN CÓDIGO ACTUAL:")
    print("   Verificar que 'detecciones' en calcular_metricas_reales()")
    print("   sea el valor BRUTO (len(faces1)), no procesado")

# Agregar diagnóstico al bucle principal
def agregar_diagnostico_bucle(image, faces1, gt_caras):
    """
    Función para agregar diagnóstico en tiempo real durante el procesamiento
    """
    print(f"🔍 DIAGNÓSTICO - {image}:")
    print(f"   GT: {gt_caras}, MTCNN detectadas: {len(faces1)}")

    # Calcular métricas en tiempo real para verificación
    tp = min(len(faces1), gt_caras)
    fp = max(0, len(faces1) - gt_caras)

    precision_real = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"   TP: {tp}, FP: {fp}")
    print(f"   Precision real-time: {precision_real:.4f}")

    if fp > 0:
        print(f"   ⚠️  FALSOS POSITIVOS DETECTADOS: {fp}")

    print("   " + "-"*50)
def verificar_consistencia_datos():
    """
    Verifica la consistencia entre diferentes estructuras de datos
    """
    print("\n📋 VERIFICACIÓN CONSISTENCIA DATOS")
    print("="*80)

    # Verificar resultados_globales vs ground_truth
    imagenes_con_datos = set([r['Imagen'] for r in resultados_globales])
    imagenes_con_gt = set(ground_truth_manual.keys())

    print(f"Imágenes en resultados_globales: {len(imagenes_con_datos)}")
    print(f"Imágenes en ground_truth: {len(imagenes_con_gt)}")
    print(f"Intersección: {len(imagenes_con_datos & imagenes_con_gt)}")

    # Imágenes faltantes
    faltantes_gt = imagenes_con_datos - imagenes_con_gt
    if faltantes_gt:
        print(f"Imágenes SIN ground truth: {len(faltantes_gt)}")
        for img in list(faltantes_gt)[:3]:  # Mostrar solo 3
            print(f"   - {img}")
def ejecutar_diagnostico_completo():
    """
    Ejecuta todas las funciones de diagnóstico
    """
    diagnostico_metricas()
    diagnostico_calculo_metricas()
    verificar_consistencia_datos()

#************************************************************************************************************
# Funciones de rotación-transformación-Mejorar brillo-contraste
def rotar_imagen(imagen, angulo):
     # Obtiene dimensiones de la imagen
    alto, ancho = imagen.shape[:2]

    # Calcula el centro de la imagen
    centro = (ancho // 2, alto // 2)

    # Obtiene la matriz de rotación
    matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo, 1.0)

    # Calcula el coseno y seno del ángulo
    coseno = abs(matriz_rotacion[0, 0])
    seno = abs(matriz_rotacion[0, 1])

    # Calcula nuevas dimensiones para evitar recorte
    nuevo_ancho = int((alto * seno) + (ancho * coseno))
    nuevo_alto = int((alto * coseno) + (ancho * seno))

    # Ajusta la matriz de rotación para el cambio de dimensiones
    matriz_rotacion[0, 2] += (nuevo_ancho / 2) - centro[0]
    matriz_rotacion[1, 2] += (nuevo_alto / 2) - centro[1]

    # Aplica la rotación
    imagen_rotada = cv2.warpAffine(imagen, matriz_rotacion, (nuevo_ancho, nuevo_alto))

    return imagen_rotada, matriz_rotacion


def transformar_coordenadas(puntos, matriz_rotacion):
    # Convierte los puntos a formato homogéneo (x, y, 1)
    puntos_homogeneos = np.array([[x, y, 1] for x, y in puntos])

    # Calcula la matriz inversa de rotación
    try:
        # Para transformaciones afines, necesitamos la inversa
        matriz_rotacion_inv = cv2.invertAffineTransform(matriz_rotacion)
    except:
        # Fallback: cálculo manual de la inversa para transformaciones afines
        a = matriz_rotacion[0, 0]
        b = matriz_rotacion[0, 1]
        tx = matriz_rotacion[0, 2]
        c = matriz_rotacion[1, 0]
        d = matriz_rotacion[1, 1]
        ty = matriz_rotacion[1, 2]

        det = a * d - b * c
        if det != 0:
            a_inv = d / det
            b_inv = -b / det
            c_inv = -c / det
            d_inv = a / det
            tx_inv = (-d * tx + b * ty) / det
            ty_inv = (c * tx - a * ty) / det

            matriz_rotacion_inv = np.array([[a_inv, b_inv, tx_inv],
                                          [c_inv, d_inv, ty_inv]])
        else:
            # Si no es invertible, retorna matriz identidad
            matriz_rotacion_inv = np.array([[1, 0, 0], [0, 1, 0]])

    # Aplica la transformación inversa
    puntos_transformados = []
    for punto in puntos_homogeneos:
        x_nuevo = matriz_rotacion_inv[0, 0] * punto[0] + matriz_rotacion_inv[0, 1] * punto[1] + matriz_rotacion_inv[0, 2]
        y_nuevo = matriz_rotacion_inv[1, 0] * punto[0] + matriz_rotacion_inv[1, 1] * punto[1] + matriz_rotacion_inv[1, 2]
        puntos_transformados.append((int(x_nuevo), int(y_nuevo)))

    return puntos_transformados

def es_imagen_claveBaja(img, umbral_brillo=60):
    """Detecta automáticamente si una imagen es oscura"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brillo_promedio = np.mean(gray)
    return brillo_promedio < umbral_brillo

def preprocesar_imagen_01(img):
    """Mejora específicamente imágenes oscuras"""
    #1. Ecualización CLAHE (mejor para contraste)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(lab[:,:,0])#Contrast Limited Adaptive Histogram Equalization(CLAHE)
    img_ecualizada = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


    # 2. Ajuste de gamma para brillo
    gamma = 1.5
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_constrastada = cv2.LUT(img_ecualizada, table)

    return img_constrastada

def preprocesar_imagen_02(img):
    K = img.copy()
    Khsv = cv2.cvtColor(K, cv2.COLOR_BGR2HSV)
    Khsv[:,:,2] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(Khsv[:,:,2])
    img_ecualizada2 = cv2.cvtColor(Khsv, cv2.COLOR_HSV2BGR)

    return img_ecualizada2

def preprocesar_imagen_03(img):
    gain = 1
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    log_rgb = exposure.adjust_log(img_rgb, gain=gain)
    img_ecualizada3 = cv2.cvtColor(log_rgb, cv2.COLOR_RGB2BGR)

    return img_ecualizada3

def preprocesar_imagen_04(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gamma_corrected = exposure.adjust_gamma(img_rgb, 0.50)
    img_ecualizada4 = cv2.cvtColor(img_gamma_corrected, cv2.COLOR_RGB2BGR)

    return img_ecualizada4

def preprocesar_imagen_05(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brillo_medio = np.mean(img_gray)
    brillo_min = img_gray.min()
    brillo_max = img_gray.max()

    if brillo_medio < 30: #Imágenes casi negras, detalles no visibles
        gamma = 0.35  # Muy oscura
        clip_limit = 4.0  # CLAHE
    elif brillo_medio <= 60: #Imágenes claramente oscuras pero con algunos detalles
        gamma = 0.50  # Moderadamente oscura
        clip_limit = 3.0
    else:
        gamma = 1.0  # No es subexpuesta
        clip_limit = 1.5
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gamma_rgb = exposure.adjust_gamma(img, gamma=gamma)
    img_gamma_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8)).apply(lab[:,:,0])#Contrast Limited Adaptive Histogram Equalization(CLAHE)
    img_ecualizada = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img_ecualizada


#*************************************************************************************************************
# BUCLE PRINCIPAL
resultados_con_metricas = []
for image in images:
    if image.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(os.path.join('/content/drive/MyDrive/UNI/face_detection/faces/face_dark', image))
    # img = cv2.resize(img, None, fx=2, fy=2)
        if img is None:
                print("No se puede leer")
                continue
        if es_imagen_claveBaja(img):
            print(f"Imagen clave baja detectada: {image} - Aplicando preprocesamiento")
            img_original = img.copy()  # Guardar original para comparación
            img = preprocesar_imagen_01(img)
        else:
            print(f"Imagen normal: {image}")
        height, width = img.shape[:2]
        img0 = img.copy()# para lbp
        img1 = img.copy()# Para Dlib
        img2 = img.copy()# Para DNN
        img3 = img.copy()# Para Haar
        img4 = img.copy()# Para Dlib con rotación iteracion2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# Para LBP, Dlib y Haar
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# Para MTCNN

        tiempos = []
        caras_detectadas = []



        #===============================================================
        # detect faces in the image - MTCNN

        inicio_mtcnn = time.time()

        faces1 = detector1.detect_faces(img_rgb)
        fin_mtcnn = time.time()
        tiempo_mtcnn = fin_mtcnn - inicio_mtcnn
        tiempos.append(tiempo_mtcnn)
        caras_detectadas.append(len(faces1))

        if image in ground_truth_manual:
            gt_caras = ground_truth_manual[image]

            agregar_diagnostico_bucle(image, faces1, gt_caras)
            metricas_mtcnn = calcular_metricas_reales(len(faces1), gt_caras)
            if metricas_mtcnn['False Positives'] == 0 and len(faces1) > gt_caras:
                print(f"   ❗ ALERTA: FP=0 pero detecciones({len(faces1)}) > GT({gt_caras})")
                print(f"     Revisar función calcular_metricas_reales()")


        #*********************************************************************

        #DBLIB
        inicio_dlib = time.time()
        faces2 = detector2(gray, 2)#Dlib
        fin_dlib = time.time()
        tiempo_dlib = fin_dlib - inicio_dlib
        tiempos.append(tiempo_dlib)
        caras_detectadas.append(len(faces2))

        #Inicio Iteracion2
        #DBLIB con rotación
        inicio_dlib_rot = time.time()
        # Rota la imagen x grados
        angulo_rotacion = 15 # Rotación (+)Antihoraria/(-)horario
        gray_rotado, matriz_rotacion = rotar_imagen(gray, angulo_rotacion)
        # Detecta caras en la imagen rotada
        faces2_rot = detector2(gray_rotado, 2)

        # Transforma las coordenadas de las detecciones a la imagen original
        caras_dlib_rot = []
        for face in faces2_rot:
            # Obtiene las coordenadas del rectángulo en la imagen rotada
            x_rot = face.left()
            y_rot = face.top()
            x1_rot = face.right()
            y1_rot = face.bottom()

            # Puntos de las esquinas del rectángulo
            puntos_rotados = [(x_rot, y_rot), (x1_rot, y1_rot)]

            # Transforma las coordenadas a la imagen original
            puntos_originales = transformar_coordenadas(puntos_rotados, matriz_rotacion)

            # Almacena las coordenadas transformadas
            caras_dlib_rot.append({
                'left': min(puntos_originales[0][0], puntos_originales[1][0]),
                'top': min(puntos_originales[0][1], puntos_originales[1][1]),
                'right': max(puntos_originales[0][0], puntos_originales[1][0]),
                'bottom': max(puntos_originales[0][1], puntos_originales[1][1])
            })

        fin_dlib_rot = time.time()
        tiempo_dlib_rot = fin_dlib_rot - inicio_dlib_rot
        tiempos.append(tiempo_dlib_rot)
        caras_detectadas.append(len(caras_dlib_rot))
        #Fin Iteracion2

        #OpenCV DNN
        inicio_dnn = time.time()
        #Preparar blob para DNN
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),
                                    1.0, (300, 300), (104.0, 117.0, 123.0))

        net.setInput(blob)
        faces3 = net.forward()
        fin_dnn = time.time()
        tiempo_dnn = fin_dnn - inicio_dnn
        tiempos.append(tiempo_dnn)

        caras_dnn = 0
        for i in range (faces3.shape[2]):
          confidence = faces3[0, 0, i, 2]
          if confidence > 0.5:
            caras_dnn += 1
        caras_detectadas.append(caras_dnn)


        #Haar Cascades
        inicio_haar = time.time()
        faces4 = classifier2.detectMultiScale(img)
        fin_haar = time.time()
        tiempo_haar = fin_haar - inicio_haar
        tiempos.append(tiempo_haar)
        caras_detectadas.append(len(faces4))

        #===============================================================
        #LBP
        Inicio_lbp = time.time()
        faces0 = lbp_face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=1, minSize=(10, 10), maxSize=(100, 100))
        fin_lbp = time.time()
        tiempo_lbp = fin_lbp - Inicio_lbp
        tiempos.append(tiempo_lbp)
        caras_detectadas.append(len(faces0))
        #===============================================================

        # Después de calcular todos los tiempos y detecciones:
        resultados_imagen = {
            "Método": [
                "MTCNN", "DLib", "DLib Rot", "OpenCV DNN", "Haar", "LBP"
            ],
            "Tiempo (segundos)": [
                round(tiempo_lbp, 4),
                round(tiempo_mtcnn, 4),
                round(tiempo_dlib, 4),
                round(tiempo_dlib_rot, 4),
                round(tiempo_dnn, 4),
                round(tiempo_haar, 4),
                #round(tiempo_haar_lighting, 4)
            ],
            "Caras detectadas": [
                len(faces1), len(faces2), len(caras_dlib_rot),
                caras_dnn, len(faces4), len(faces0)

            ]
        }

        # Almacenar para análisis global
        for i, metodo in enumerate(resultados_imagen["Método"]):
            resultados_globales.append({
                "Imagen": image,
                "Método": metodo,
                "Tiempo (segundos)": resultados_imagen["Tiempo (segundos)"][i],
                "Caras detectadas": resultados_imagen["Caras detectadas"][i]
            })

        # Visualización individual mejorada
        #LBP
        for result in faces0:
          # LBP - retorna diccionario con 'box'
            x, y, w, h = result
            x1, y1 = x + w, y + h
            cv2.rectangle(img0, (x, y), (x1, y1), (0, 0, 255), 2)

        #MTCNN
        for result in faces1:
          # MTCNN - retorna diccionario con 'box'
            x, y, w, h = result['box']
            x1, y1 = x + w, y + h
            cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)

        #DLIB
        for result in faces2:
          # Dlib - retorna objetos rectangle
            x = result.left()
            y = result.top()
            x1 = result.right()
            y1 = result.bottom()
            cv2.rectangle(img1, (x, y), (x1, y1), (0, 0, 255), 2)

        #Inicio Iteracion2
        # DLIB (CON ROTACIÓN) - Dibuja rectángulos azules para las detecciones con rotación
        for result in caras_dlib_rot:
            x = result['left']
            y = result['top']
            x1 = result['right']
            y1 = result['bottom']
            cv2.rectangle(img4, (x, y), (x1, y1), (255, 0, 0), 2)  # Color azul para diferenciar
        #Fin Iteracion2


        #OPENCV DNN
        for i in range(faces3.shape[2]):
            confidence = faces3[0, 0, i, 2]
            if confidence > 0.5:
              # DNN - requiere procesamiento de coordenadas
                box = faces3[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x, y, x1, y1) = box.astype("int")
                cv2.rectangle(img2, (x, y), (x1, y1), (0, 0, 255), 2)

        #HAAR
        for result in faces4:
          # Haar - retorna (x, y, w, h)
            x, y, w, h = result
            x1, y1 = x + w, y + h
            cv2.rectangle(img3, (x, y), (x1, y1), (0, 0, 255), 2)


        if image in ground_truth_manual:
            gt_caras = ground_truth_manual[image]

            # Calcular métricas para cada método
            
            metricas_mtcnn = calcular_metricas_reales(len(faces1), gt_caras)
            metricas_dlib = calcular_metricas_reales(len(faces2), gt_caras)
            metricas_dlib_rot = calcular_metricas_reales(len(caras_dlib_rot), gt_caras)
            metricas_dnn = calcular_metricas_reales(caras_dnn, gt_caras)
            metricas_haar = calcular_metricas_reales(len(faces4), gt_caras)
            metricas_lbp = calcular_metricas_reales(len(faces0), gt_caras)


            # Almacenar resultados
            metodos_metricas = [                
                ('MTCNN', metricas_mtcnn, tiempo_mtcnn),
                ('DLib', metricas_dlib, tiempo_dlib),
                ('DLib Rot', metricas_dlib_rot, tiempo_dlib_rot),
                ('OpenCV DNN', metricas_dnn, tiempo_dnn),
                ('Haar', metricas_haar, tiempo_haar),
                ('LBP', metricas_lbp, tiempo_lbp)

            ]

            for metodo, metricas, tiempo in metodos_metricas:
                resultados_con_metricas.append({
                    'Imagen': image,
                    'Método': metodo,
                    'Caras Reales': gt_caras,
                    'Caras Detectadas': metricas['True Positives'] + metricas['False Positives'],
                    'True Positives': metricas['True Positives'],
                    'False Positives': metricas['False Positives'],
                    'False Negatives': metricas['False Negatives'],
                    'Precision': metricas['Precision'],
                    'Recall': metricas['Recall'],
                    'F1-Score': metricas['F1-Score'],
                    'Accuracy': metricas['Accuracy'],
                    'Tiempo (s)': tiempo
                })


        
        cv2.imwrite(os.path.join('/content/drive/MyDrive/UNI/face_detection/faces2', 'mtcnn', image), img)
        cv2.imwrite(os.path.join('/content/drive/MyDrive/UNI/face_detection/faces2', 'dlib', image), img1)
        cv2.imwrite(os.path.join('/content/drive/MyDrive/UNI/face_detection/faces2', 'dlib_rot', image), img4)
        cv2.imwrite(os.path.join('/content/drive/MyDrive/UNI/face_detection/faces2', 'dnn', image), img2)
        cv2.imwrite(os.path.join('/content/drive/MyDrive/UNI/face_detection/faces2', 'haar', image), img3)
        cv2.imwrite(os.path.join('/content/drive/MyDrive/UNI/face_detection/faces2', 'lbp', image), img0)


        imagenes = [img, img1, img4, img2, img3, img0]
        titulos = ["MTCNN", "DLib", "DLib Rot", "OpenCV DNN", "Haar", "LBP"]

        visualizar_resultados_individuales(image, resultados_imagen, imagenes, titulos)


crear_resumen_ejecucion()
df_resultados_finales = mostrar_metricas_avanzadas_paper()
ejecutar_diagnostico_completo()
