# 🌋 GeoCaverns — Cave Ecosystem Explorer
 
> An interactive Machine Learning web application for geospatial cave ecosystem analysis, biodiversity prediction, and tourist accessibility classification — built with Python and Streamlit.
 
---
 
## 📌 Overview
 
GeoCaverns is a fully deployed, multi-module ML dashboard that allows users to explore a global cave dataset through **7 interactive modules** — without writing a single line of code. From clustering caves by geographic region to detecting anomalous cave records and visualising 3D depth surfaces, the app covers the complete data science pipeline in a single interactive interface.
 
---
 
## 🖥️ Live Demo
 
> Run the app locally using the instructions below.
> Deployable on [Streamlit Cloud](https://streamlit.io/cloud) with no additional configuration.
 
---
 
## 📂 Project Structure
 
```
GeoCaverns/
│
├── app.py                              # Main Streamlit application
├── GeoCaverns_Cave_Dataset_Synthetic.xlsx  # Dataset
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```
 
---
 
## 🗂️ Dataset
 
| Property | Details |
|---|---|
| **File** | `GeoCaverns_Cave_Dataset_Synthetic.xlsx` |
| **Type** | Synthetic global cave records |
| **Key Features** | Latitude, Longitude, Depth (m), Length (km), Cave Type, Access Difficulty, Species Found, Tourist Accessible, Discovery Year |
| **Target (Regression)** | `Depth (m)` — continuous |
| **Target (Classification)** | `Tourist Accessible` — binary (Yes / No) |
 
> **Note:** This dataset is synthetically generated. Feature correlations are minimal by design. Model performance reflects the data structure, not implementation errors.
 
---
 
## ⚙️ Features — 7 ML Modules
 
### 📊 1. View Data
- Raw dataset preview (first 200 rows)
- Engineered feature display
- Column schema and dataset shape
---
 
### 🔬 2. Regression — Predict Cave Depth
Compares **5 regression models** to predict `Depth (m)` from cave features.
 
| Model | Purpose |
|---|---|
| `DummyRegressor (mean)` | Baseline — must be beaten |
| `Linear Regression` | Simple linear benchmark |
| `Random Forest` | Ensemble, non-linear |
| `Gradient Boosting` | Boosted trees — typically strongest |
| `KNN Regressor` | Distance-based |
 
**Key Design Decisions:**
- `StandardScaler` applied — fit on train, transform on test only (no data leakage)
- `KFold` cross-validation with user-controlled fold count (3–10)
- Optional `GridSearchCV` hyperparameter tuning (toggle)
- Automated diagnosis when no model beats the baseline
- Actual vs Predicted scatter, Residual histogram, Feature Importance charts
---
 
### 🎯 3. Classification — Predict Tourist Accessibility
Reframes the problem to a more learnable binary target — predicting whether a cave is tourist accessible.
 
| Model | Purpose |
|---|---|
| `DummyClassifier (most_frequent)` | Baseline |
| `Logistic Regression` | Linear decision boundary |
| `Random Forest Classifier` | Ensemble, feature importances |
| `KNN Classifier` | Distance-based |
 
**Key Design Decisions:**
- `StratifiedKFold` — preserves class balance in each fold
- `stratify=y` in train/test split
- Confusion matrix heatmap per model
- Full `classification_report` output
- Model comparison table sorted by Hold-out Accuracy
---
 
### 🗺️ 4. Depth Interpolation — 3D Surface
Estimates cave depth across the full geographic grid using spatial interpolation.
 
- `scipy.griddata` with `method='linear'` for smooth interpolation
- Nearest-neighbor fallback fills NaN regions outside data's convex hull
- Interactive `plotly go.Surface` 3D chart
- User-controlled grid resolution (40 × 40 to 200 × 200)
---
 
### 🔵 5. Clustering — Geographic Grouping
Groups caves by geographic and physical features using three algorithms.
 
| Algorithm | Key Parameter | Best For |
|---|---|---|
| `KMeans` | k (2–12) | Equal-size geographic zones |
| `DBSCAN` | eps, min_samples | Density-based clusters + noise detection |
| `Agglomerative` | k (2–12) | Hierarchical geographic structure |
 
- All features scaled with `StandardScaler` before clustering
- 3D scatter (Lat × Lon × Depth) or 2D fallback
- Cluster size summary
---
 
### ⚠️ 6. Anomaly Detection — Flag Unusual Caves
Identifies cave records that deviate significantly from the rest.
 
| Method | How It Works |
|---|---|
| `Isolation Forest` | Isolates anomalies via random recursive splitting |
| `Local Outlier Factor` | Compares local density to k nearest neighbors |
 
- User-controlled contamination % (0.1%–10%)
- StandardScaler applied before detection
- 3D scatter with anomalies highlighted in red
- Anomaly count displayed
---
 
### 🌐 7. Visualisations — Maps & Charts
- **Folium Map** — MarkerCluster with popup details per cave (with Plotly fallback)
- **Box Plot** — Depth distribution by Cave Type
- **Scatter Plot** — Species Found vs Depth coloured by Access Difficulty
---
 
## 🛠️ Tech Stack
 
| Category | Library |
|---|---|
| **Web App** | Streamlit |
| **Data** | Pandas, NumPy |
| **ML Models** | Scikit-learn |
| **Deep Learning** | TensorFlow / Keras |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Geospatial** | Folium, streamlit-folium |
| **Interpolation** | SciPy |
 
---
 
## 🚀 Getting Started
 
### 1. Clone the repository
```bash
git clone https://github.com/snesa2567-arch/GeoCaverns.git
cd GeoCaverns
```
 
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
 
### 3. Run the app
```bash
streamlit run app.py
```
 
The app will open in your browser at `http://localhost:8501`
 
---
 
## 📦 Requirements
 
```txt
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
plotly
scipy
folium
streamlit-folium
openpyxl
```
 
Install all at once:
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn tensorflow plotly scipy folium streamlit-folium openpyxl
```
 
---
 
## 🔧 Key Engineering Decisions
 
### ✅ @st.cache_data
Both `load_data()` and `build_features()` are cached. Streamlit re-runs the full script on every user interaction — caching prevents re-reading the Excel file and re-running KMeans feature engineering on every slider movement.
 
### ✅ Feature Engineering (`build_features`)
12 derived spatial features created from raw coordinates:
 
| Feature | Formula | Why |
|---|---|---|
| `Lat_x_Lon` | Latitude × Longitude | Combined geographic zone |
| `Lat_squared` | Latitude² | Non-linear distance from equator |
| `Lon_squared` | Longitude² | Non-linear east/west position |
| `Lat_abs` | \|Latitude\| | Distance from equator (positive) |
| `Lat_bin` | pd.cut (6 bins) | Latitude zone label |
| `Lon_bin` | pd.cut (6 bins) | Longitude zone label |
| `Region_Cluster` | KMeans (k=8) | Geographic region ID |
| `Dist_from_equator` | \|Latitude\| | Climate zone proxy |
| `Dist_from_meridian` | \|Longitude\| | Continental position proxy |
| `Log_Length` | log1p(Length) | Compress right-skewed distribution |
| `Log_Depth` | log1p(Depth) | Compress right-skewed distribution |
 
### ✅ No Data Leakage
- `StandardScaler`: `fit_transform` on train only, `transform` on test
- `GridSearchCV`: tuning done only on training fold within CV
- Baseline model included in every comparison
### ✅ Honest Results Communication
- All models achieve ~50% classification accuracy (expected for synthetic data)
- Automated check flags when no model beats the `DummyClassifier` baseline
- App explains the diagnosis — data limitation, not code failure
---
 
## 📊 Results
 
### Regression (Predict Cave Depth)
> All models perform near baseline — expected for synthetic data with no real depth-location correlation.
 
| Model | CV R² | Hold-out R² |
|---|---|---|
| Baseline (Mean) | ~0.00 | ~0.00 |
| Linear Regression | ~0.01 | ~0.01 |
| Random Forest | ~0.02 | ~0.02 |
| Gradient Boosting | ~0.02 | ~0.02 |
| KNN Regressor | ~0.01 | ~0.01 |
 
### Classification (Predict Tourist Accessibility)
> ~50% accuracy across all models — consistent with a balanced synthetic target.
 
| Model | CV Accuracy | Hold-out Accuracy |
|---|---|---|
| Baseline (Most Frequent) | ~0.50 | ~0.50 |
| Logistic Regression | ~0.50 | ~0.51 |
| KNN Classifier | ~0.51 | ~0.50 |
| Random Forest | ~0.51 | ~0.50 |
 
> **With real cave data** where accessibility genuinely correlates with access difficulty and cave type, expected accuracy: **75–85%**
 
---
 
## 🔍 What Would Improve Results
 
| Improvement | Expected Impact |
|---|---|
| Replace synthetic data with real cave records | Models learn genuine geographic-depth relationships |
| Add SHAP values | Per-prediction explainability |
| Deploy on Streamlit Cloud | Public URL accessible without local setup |
| Add time-series features (discovery year trends) | Temporal pattern analysis |
| Add more domain features (geology type, rock formation) | Richer feature space |
 
---
 
## 📁 Module Summary
 
```
app.py
├── load_data()          → Validate, clean, encode, cache dataset
├── build_features()     → Engineer 12 spatial features, cache result
│
├── "📊 View Data"       → Preview raw + engineered data
├── "🔬 Regression"      → 5 models, CV, scaling, tuning, diagnosis
├── "🎯 Classification"  → 4 models, StratifiedKFold, confusion matrix
├── "🗺️ Interpolation"   → SciPy griddata + Plotly 3D surface
├── "🔵 Clustering"      → KMeans / DBSCAN / Agglomerative
├── "⚠️ Anomaly"         → Isolation Forest / LOF
└── "🌐 Visualizations"  → Folium map + Plotly charts
```
 
---
 
## 👤 Author
 
**S. Nesa Sankaran**  
M.Sc Data Science — Vellore Institute of Technology, Chennai  
B.Sc Statistics — PSG College of Arts and Science, Coimbatore
 
[![GitHub](https://img.shields.io/badge/GitHub-snesa2567--arch-181717?style=flat&logo=github)](https://github.com/snesa2567-arch)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-S--Nesa--Sankaran-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/s-nesa-sankaran-5563b1259)
[![Email](https://img.shields.io/badge/Email-S.nesa2567@gmail.com-D14836?style=flat&logo=gmail)](mailto:S.nesa2567@gmail.com)
 
---
 
## 📄 License
 
This project is open source and available under the [MIT License](LICENSE).
 
---
 
## 🙏 Acknowledgements
 
- [Streamlit](https://streamlit.io) — for the web app framework
- [Scikit-learn](https://scikit-learn.org) — for ML models and evaluation
- [Plotly](https://plotly.com) — for interactive 3D visualisations
- [Folium](https://python-visualization.github.io/folium/) — for geospatial maps
- [SciPy](https://scipy.org) — for spatial interpolation
---
 
*Built as part of a data science portfolio demonstrating end-to-end ML product development — from data pipeline and feature engineering to model evaluation and live deployment.*
