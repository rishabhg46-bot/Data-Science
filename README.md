🧠 Hybrid ResNet-50 + CatBoost Framework for Crowd Counting (MALL Dataset)

A deep learning + boosted regression hybrid pipeline for accurate crowd counting using the MALL dataset.
The project combines ResNet-50 feature extraction, deep regression, and CatBoost to model non-linear patterns in crowd density, outperforming classical regression methods.
(Full report reference: 

Updated_ds_rishabh

)

⭐ Project Highlights

Uses ResNet-50 (ImageNet pretrained) for extracting high-level visual features.

Experiments with two prediction heads:

ResNet-50 Regression Head

CatBoost Regressor trained on ResNet embeddings

Achieves strong performance on the MALL dataset (2000 frames, 13–53 people per frame).

Includes full evaluation: MAE, MSE, R², correlation heatmap, model comparison, and training curves.

Lightweight, easy to train, and suitable for real-time deployment.

📂 Dataset — MALL Dataset

2000 surveillance frames (640×480 resolution)

Ground-truth head counts provided in mall_gt.mat

Crowd count range: 13 to 53, mean ≈ 31

Challenges include: occlusion, perspective distortion, illumination changes

⚙️ Methodology

The pipeline consists of the following stages:

1️⃣ Preprocessing

Resize images to 224×224

Normalize using ImageNet statistics

Optional augmentations:

Horizontal flip

Color jitter

Random crop

2️⃣ Feature Extraction (ResNet-50)

Two routes:

✔ Route A – End-to-End Deep Regression

Replace FC layer ⟶ single neuron for count prediction
Use MSE loss + Adam optimizer

✔ Route B – Hybrid ML Route

Extract embeddings from ResNet-50

Train CatBoost Regressor (200–500 trees)

Predict counts using gradient boosting

CatBoost handles non-linear relationships more robustly, often outperforming direct regression.

📉 Training Curve — ResNet50

![Training](attachment:Screenshot 2025-08-17 011752.png)

The training/validation loss curves show stable optimization over epochs.

📈 Predicted vs True Counts Scatter Plot

A strong linear correlation indicates good generalization in the regression model.

📊 Model Comparison (ResNet-50 vs CatBoost)

![Comparison](attachment:Screenshot 2025-08-23 122049.png)

Metric	ResNet-50	CatBoost
MSE	9.927	8.854
MAE	2.519	2.320
R²	0.805	0.827

👉 CatBoost slightly outperforms deep regression, showing advantages of hybrid modeling.

🔥 Correlation Heatmap

![Heatmap](attachment:Screenshot 2025-09-04 000708.png)

Correlation coefficient ≈ 0.905, demonstrating strong linear alignment between predicted & actual counts.

🧪 Evaluation Metrics

MAE = 2.320 (best)

MSE = 8.854

R² Score = 0.826

RMSE = sqrt(MSE)

🏗️ Project Structure
📦 crowd-counting-hybrid
 ┣ 📂 dataset/
 ┣ 📂 src/
 ┃ ┣ resnet_model.py
 ┃ ┣ catboost_model.py
 ┃ ┣ train.py
 ┃ ┣ utils.py
 ┣ 📊 results/
 ┣ README.md
 ┗ requirements.txt

▶️ How to Run
1. Install dependencies
pip install -r requirements.txt

2. Train ResNet-50
python train_resnet.py

3. Extract Embeddings
python extract_embeddings.py

4. Train CatBoost Regressor
python train_catboost.py

📚 Tech Stack

Python

PyTorch

CatBoost

NumPy / Pandas / Matplotlib / Seaborn

OpenCV

📝 References

Full literature review included in report
(Full citations available here: 

Updated_ds_rishabh

)

🚀 Future Improvements

Add density-map based learning (CSRNet, DM-Count)

Use perspective normalization

Deploy a lightweight model (MobileNet-V3, YOLO-Nano)

Multi-task: detection + counting

❤️ Contributors

Rishabh Gupta
(Open to collaboration!)
