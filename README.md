🫁 Pneumonia Detection and Segmentation using EfficientNet-B0 & U-Net

A deep learning project for automatic pneumonia detection and lung infection segmentation from chest X-ray images.

The project combines:

EfficientNet-B0 → Binary classification (Normal / Pneumonia)
U-Net → Segmentation of infected regions
Streamlit → Interactive web application for inference
📌 Project Overview

This project follows a two-stage pipeline:

Step 1 — Classification

An EfficientNet-B0 model is trained to classify chest X-ray images into:

Normal
Pneumonia

If pneumonia is detected, the image is passed to the segmentation model.

Step 2 — Segmentation

A U-Net architecture is used to identify and segment pneumonia-affected regions.

The output includes:

Original X-ray image
Predicted mask
Overlay visualization
🏗️ Project Architecture
Chest X-Ray
      │
      ▼
┌─────────────────┐
│ EfficientNet-B0 │
└─────────────────┘
      │
      ▼
 Normal ? ───► End

      │

 Pneumonia
      │
      ▼

┌──────────┐
│  U-Net   │
└──────────┘
      │
      ▼

Segmentation Mask
      │
      ▼

Overlay Visualization
📂 Repository Structure
.
├── app.py                     # Streamlit application
├── main.py                    # Local inference script
│
├── efficientnet_model.py      # EfficientNet model
├── train_efficient.py         # Classification training
├── dataset_efficientnet.py    # Classification dataset loader
│
├── UNET_model.py              # U-Net architecture
├── unet_train.py             # Segmentation training
├── dataset_unet.py            # Segmentation dataset loader
│
├── moyenne_std_data.py        # Dataset normalization
├── affichage.py               # Visualization functions
│
├── model.pth                  # Trained EfficientNet weights
├── best_model.pth             # Trained U-Net weights
│
└── README.md
🧠 Models
EfficientNet-B0

Used for:

Binary classification
Transfer learning
Feature extraction

Modifications:

Frozen backbone
Custom classifier layer

Output:
-Normal
-Pneumonia

U-Net : Custom implementation with:

Encoder
Bottleneck
Decoder
Skip Connections

Input:

3 × 128 × 128

Output:

1 × 128 × 128

(segmentation mask)


📊 Metrics
Classification

The following metrics are computed:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
Segmentation

The following metrics are used:

Dice Score
IoU (Intersection over Union)
BCE + Dice Loss
📉 Loss Functions
Classification
CrossEntropyLoss()
Segmentation

Combined loss:

Loss = BCEWithLogitsLoss + DiceLoss

This combination improves:

Pixel-wise prediction
Small lesion detection
Segmentation quality
📚 Dataset Structure
Classification Dataset
data_effi/
│
├── train/
│   ├── normal/
│   └── pneumonia/
│
└── val/
    ├── normal/
    └── pneumonia/
Segmentation Dataset
dataset/
│
├── train/
│   ├── image/
│   └── mask/
│
└── val/
    ├── image/
    └── mask/
⚙️ Installation

Clone the repository:

git clone https://github.com/your_username/pneumonia-detection-unet.git

cd pneumonia-detection-unet

Create a virtual environment:

python -m venv venv

Activate it:

Windows
venv\Scripts\activate
Linux / Mac
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt
🚀 Training
Train EfficientNet
python train_efficient.py

Outputs:

model.pth
training_log.csv
Train U-Net
python unet_train.py

Outputs:

best_model.pth
🖥️ Run the Application

Launch Streamlit:

streamlit run app.py

Upload a chest X-ray image.

The application will:

Predict Normal / Pneumonia
Generate segmentation mask
Display overlay visualization

📸 Example Output
Classification
Prediction : Pneumonia
Segmentation
Original Image
Predicted Mask
Overlay


🔬 Technologies Used
-Python
-PyTorch
-Torchvision
-Streamlit
-OpenCV
-NumPy
-Pandas
-Matplotlib
-Seaborn
-Pillow
-Scikit-Learn

🎯 Future Improvements
-Train deeper U-Net architectures
-Add Attention U-Net
-Integrate Grad-CAM for explainability
-Deploy using Docker
-CI/CD integration with GitHub Actions
-Cloud deployment (AWS / Azure)
-Multi-class lung disease classification

👨‍💻 Author

Lyes Bouaoun

Master's Student in Artificial Intelligence & Data Science

Areas of interest:

Deep Learning
Computer Vision
Graph Neural Networks (GNN)
Data Engineering
MLOps

📜 License

This project is developed for educational and research purposes.
