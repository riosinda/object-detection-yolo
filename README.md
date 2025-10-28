# Object Detection with YOLOv5 - SKU-110K Dataset

Este proyecto implementa detección de objetos utilizando YOLOv5 en el dataset SKU-110K, un conjunto de datos de productos de tiendas retail con más de 110,000 anotaciones.

## 📋 Tabla de Contenidos
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Descarga del Dataset](#descarga-del-dataset)
- [Configuración de YOLOv5](#configuración-de-yolov5)
- [Exploración y Preparación de Datos](#exploración-y-preparación-de-datos)
- [Entrenamiento del Modelo](#entrenamiento-del-modelo)
- [Evaluación](#evaluación)
- [Estructura del Proyecto](#estructura-del-proyecto)

## 🔧 Requisitos

- Python 3.12
- CUDA 12.6+ (para entrenamiento con GPU)
- 13.6 GB de espacio libre en disco para el dataset
- Credenciales de AWS (para descarga del dataset)

## 📦 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/object-detection-yolo.git
cd object-detection-yolo
```

### 2. Crear un entorno virtual (recomendado)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

El archivo `requirements.txt` incluye todas las dependencias necesarias:
- PyTorch y TorchVision (con soporte CUDA)
- OpenCV
- Pandas, NumPy, Matplotlib, Seaborn
- Ultralytics
- Boto3 (para descarga de AWS)
- Y más...

## 📥 Descarga del Dataset

### Configurar credenciales de AWS

El dataset SKU-110K se descarga desde un bucket de AWS S3. Necesitas configurar tus credenciales:

1. Crea un archivo `.env` en la raíz del proyecto:

```bash
# .env
AWS_ACCESS_KEY_ID=tu_access_key_aqui
AWS_SECRET_ACCESS_KEY=tu_secret_key_aqui
```

### Ejecutar script de descarga

```bash
python src/download_data.py
```

Este script descargará automáticamente:
- Imágenes de entrenamiento, validación y test
- Anotaciones en formato CSV
- Estructura completa del dataset en `data/SKU110K_dataset/`

**Nota:** La descarga puede tomar varios minutos dependiendo de tu conexión (13.6 GB).

## 🔨 Configuración de YOLOv5

### 1. Clonar el repositorio de YOLOv5

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
cd ..
```

### 2. Crear archivo de configuración del dataset

Crea el archivo `yolov5/data/dataset.yaml` con la siguiente configuración:

```yaml
path: ../data/SKU110K_dataset

train: images/train
val: images/val
test: images/test

names:
  1: object
  0: empty
```

**Notas importantes:**
- Asegúrate de que las rutas coincidan con la estructura de tu dataset
- Si usaste el script `src/download_data.py`, el dataset estará en `data/SKU110K_dataset/`
- Las etiquetas (labels) deben estar en formato YOLO en las carpetas correspondientes

### 3. Verificar estructura del dataset

Tu estructura de carpetas debe verse así:

```
object-detection-yolo/
├── data/
│   └── SKU110K_dataset/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── annotations/
└── yolov5/
    └── data/
        └── dataset.yaml  # Tu archivo de configuración
```

## 📊 Exploración y Preparación de Datos

Ejecuta los notebooks de Jupyter en orden para explorar y preparar los datos:

### 1. Análisis Exploratorio de Datos (EDA)

```bash
jupyter notebook notebooks/1_EDA.ipynb
```

Este notebook incluye:
- Análisis de distribución de objetos
- Visualización de imágenes de muestra
- Estadísticas del dataset
- Análisis de tamaños de bounding boxes

### 2. Preparación de Datos

```bash
jupyter notebook notebooks/2_Prepare_data.ipynb
```

Este notebook realiza:
- Conversión de anotaciones CSV al formato YOLO
- Validación de etiquetas
- División de datos (train/val/test)
- Creación de archivos de configuración

### 3. Test del Modelo Entrenado

```bash
jupyter notebook notebooks/3_Test_trained_model.ipynb
```

Este notebook permite:
- Cargar el modelo entrenado
- Realizar inferencias
- Visualizar predicciones
- Evaluar métricas de rendimiento

## 🚀 Entrenamiento del Modelo

### Entrenamiento Básico

Una vez completada la configuración, entrena el modelo con YOLOv5:

```bash
cd yolov5
python train.py --img 640 --batch 16 --epochs 3 --dataset.yaml --weights yolov5s.pt
```

### Parámetros del entrenamiento

- `--data`: Ruta al archivo de configuración del dataset
- `--weights`: Pesos preentrenados (yolov5s.pt, yolov5m.pt, yolov5l.pt, yolov5x.pt)
- `--img`: Tamaño de imagen (640, 1280, etc.)
- `--batch`: Tamaño del batch (ajustar según tu GPU)
- `--epochs`: Número de épocas
- `--name`: Nombre del experimento

## 📁 Estructura del Proyecto

```
object-detection-yolo/
├── config/                      # Archivos de configuración
│   ├── main.yaml
│   ├── model/
│   └── process/
├── data/                        # Datasets
│   └── SKU110K_dataset/
│       ├── images/
│       ├── labels/
│       └── annotations/
├── models/                      # Modelos entrenados
│   └── best.pt
├── notebooks/                   # Jupyter notebooks
│   ├── 1_EDA.ipynb
│   ├── 2_Prepare_data.ipynb
│   └── 3_Test_trained_model.ipynb
├── src/                         # Código fuente
│   ├── download_data.py         # Script de descarga del dataset
│   ├── process.py               # Procesamiento de datos
│   ├── train_model.py           # Entrenamiento
│   └── utils.py                 # Utilidades
├── tests/                       # Tests unitarios
├── yolov5/                      # Repositorio de YOLOv5 (clonar)
├── requirements.txt             # Dependencias
└── README.md                    # Este archivo
```

## 📚 Referencias

- [YOLOv5 Documentation](https://docs.ultralytics.com/yolov5/)
- [SKU-110K Dataset Paper](https://arxiv.org/abs/1904.00853)
- [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)

## 📄 Licencia

Este proyecto es para fines educativos. El dataset SKU-110K y YOLOv5 tienen sus propias licencias.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request para sugerencias o mejoras.
