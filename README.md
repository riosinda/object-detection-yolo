# Object Detection with YOLO

Object detection project using YOLO models (YOLOv5 and YOLOv11) for the SKU-110K dataset. This project implements a product detection system for retail shelves, capable of distinguishing between objects and empty spaces.

## 📋 Description

This project implements a complete pipeline for training and evaluating YOLO-based object detection models. The main objective is to detect products in retail shelf images, classifying between:
- **Object**: Presence of products on the shelf
- **Empty**: Spaces without products

The project uses the SKU-110K dataset, which contains over 11,000 retail shelf images with bounding box annotations.

## ✨ Features

- 🚀 Implementation with YOLOv5 and YOLOv11 (Ultralytics)
- 📊 Complete exploratory data analysis (EDA)
- 🔄 Automated data processing pipeline
- ☁️ Automatic dataset download from AWS S3
- 🎯 Annotation conversion to YOLO format
- 📈 Model training and evaluation
- 🧪 Interactive notebooks for experimentation
- ⚙️ Flexible configuration system with Hydra

## 📁 Project Structure

```
object-detection-yolo/
├── src/                    # Python source code
│   ├── download_data.py   # Dataset download from S3
│   ├── process.py          # Data processing
│   ├── train_model.py      # Model training
│   └── utils.py            # Utilities and helper functions
├── notebooks/              # Jupyter notebooks
│   ├── 0_Test_YOLOv11.ipynb
│   ├── 1_EDA.ipynb         # Exploratory analysis
│   ├── 2_Prepare_data.ipynb # Data preparation
│   ├── 3_Test_trained_model.ipynb # Model testing
│   ├── 4_YOLO11.ipynb
│   └── dataset.yaml        # Dataset configuration
├── data/                   # Project data
│   └── SKU110K_dataset/    # SKU-110K dataset
│       ├── images/         # Images (train/val/test)
│       ├── labels/         # YOLO labels (train/val/test)
│       └── annotations/    # Original CSV annotations
├── models/                 # Trained models
│   ├── best.pt
│   ├── best_yolov5_v5x.pt
│   └── without_newlabels.pt
├── config/                 # Configuration files
│   ├── main.yaml
│   ├── model/              # Model configurations
│   └── process/            # Processing configurations
├── yolov5/                 # YOLOv5 repository
├── tests/                  # Unit tests
├── docs/                   # Additional documentation
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA (optional, for GPU acceleration)
- AWS Credentials (to download the dataset)

### Installation Steps

1. **Clone the repository** (if applicable):
```bash
git clone <repository-url>
cd object-detection-yolo
```

2. **Create a virtual environment**:
```bash
python -m venv .venv
```

3. **Activate the virtual environment**:
   - Windows:
   ```bash
   .venv\Scripts\activate
   ```
   - Linux/Mac:
   ```bash
   source .venv/bin/activate
   ```

4. **Install dependencies**:
```bash
pip install -r requirements.txt
```

5. **Configure AWS credentials** (to download the dataset):
   - Create a `.env` file in the project root:
   ```
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   ```

## 📖 Usage

### 1. Download the Dataset

To download the SKU-110K dataset from AWS S3:

```bash
python src/download_data.py
```

### 2. Exploratory Data Analysis

Open the `notebooks/1_EDA.ipynb` notebook to explore the dataset:
- Class distribution
- Bounding box statistics
- Sample visualizations

### 3. Prepare the Data

Run `notebooks/2_Prepare_data.ipynb` to:
- Convert CSV annotations to YOLO format
- Organize images and labels into directories
- Add new labels if necessary

### 4. Train a Model

To train a YOLO model:

```bash
python src/train_model.py
```

Or use YOLOv5 directly:

```bash
cd yolov5
python train.py --img 640 --batch 16 --epochs 100 --data ../notebooks/dataset.yaml --weights yolov5s.pt
```

### 5. Test the Trained Model

Open `notebooks/3_Test_trained_model.ipynb` to:
- Load a trained model
- Perform inference on test images
- Visualize results

## 🗂️ Dataset

### SKU-110K Dataset

The project uses the SKU-110K dataset, which contains:
- **Train**: ~8,500 images
- **Validation**: ~600 images  
- **Test**: ~2,900 images

Each image contains bounding box annotations in format:
- **Original CSV**: `image_name, x1, y1, x2, y2, class, image_width, image_height`
- **YOLO format**: `class_id x_center y_center width height` (normalized)

### Classes

- `0`: empty
- `1`: object (product)

## ⚙️ Configuration

The project uses YAML files for configuration:

- `config/main.yaml`: Main configuration
- `config/model/`: Specific model configurations
- `config/process/`: Processing configurations
- `notebooks/dataset.yaml`: Dataset configuration for YOLO

## 🛠️ Main Functions

### `src/utils.py`

- `convert_to_yolo_format()`: Converts CSV annotations to YOLO format
- `order_images()`: Organizes images into train/val/test directories
- `add_new_labels()`: Adds new labels to the dataset
- `fix_label_names()`: Fixes label file names

### `src/download_data.py`

- `download_dataset()`: Downloads the complete dataset from AWS S3

## 📊 Available Models

The project includes several trained models in `models/`:
- `best.pt`: Best general model
- `best_yolov5_v5x.pt`: Trained YOLOv5x model
- `without_newlabels.pt`: Model without new labels

## 🔧 Main Dependencies

- `ultralytics`: YOLO framework
- `torch`: PyTorch for deep learning
- `opencv-python`: Image processing
- `pandas`: Data manipulation
- `matplotlib`, `seaborn`: Visualization
- `boto3`: AWS S3 integration
- `tqdm`: Progress bars

See `requirements.txt` for the complete list.

## 📝 Notebooks

1. **0_Test_YOLOv11.ipynb**: Initial tests with YOLOv11
2. **1_EDA.ipynb**: Dataset exploratory analysis
3. **2_Prepare_data.ipynb**: Data preparation and conversion
4. **3_Test_trained_model.ipynb**: Trained model evaluation
5. **4_YOLO11.ipynb**: YOLOv11 experimentation

## 🤝 Contributing

Contributions are welcome. Please:
1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project uses the SKU-110K dataset. Please review the `data/SKU110K_dataset/LICENSE.txt` file for more information about the dataset license.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics) for YOLOv5 and YOLOv11
- The creators of the SKU-110K dataset
- AnyoneAI for support and resources

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This project requires AWS credentials to download the dataset. Make sure to have environment variables or the `.env` file configured before running `download_data.py`.
