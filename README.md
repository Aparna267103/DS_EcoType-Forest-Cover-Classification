🌲 EcoType: Forest Cover Classification using Machine Learning
📌 Project Overview

EcoType is a Machine Learning project that predicts the type of forest cover based on environmental and cartographic features such as elevation, slope, soil type, hillshade values, hydrology distance, and wilderness area information.
The project aims to support environmental monitoring, forest resource management, wildfire risk assessment, and land cover mapping.

🚀 Features
 * Data Cleaning & Preprocessing
* Exploratory Data Analysis (EDA)
* Outlier Detection & Skewness Treatment
* Feature Engineering
* Class Imbalance Handling using SMOTE
* Multiple Machine Learning Models
* Hyperparameter Tuning
* Feature Importance Visualization
* Streamlit Web Application Deployment
  
📊 Dataset Information
* Dataset Name: Forest Cover Type Dataset
* Rows: 1,45,891
* Columns: 13
* Target Variable: Cover_Type
* Classes: 7 Forest Cover Types

🔹 Features Used
* Elevation
* Aspect
* Slope
* Horizontal Distance to Hydrology
* Vertical Distance to Hydrology
* Horizontal Distance to Roadways
* Hillshade (9am, Noon, 3pm)
* Horizontal Distance to Fire Points
* Wilderness Area
* Soil Type
  
🛠 Technologies Used
* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)
* Streamlit
* Joblib
  
🤖 Machine Learning Models
The following classification models were trained and evaluated:
* Random Forest Classifier
* Decision Tree Classifier
* Logistic Regression
* K-Nearest Neighbors (KNN)
* XGBoost Classifier
 
📈 Exploratory Data Analysis (EDA)
EDA includes:

* Missing Value Analysis
* Duplicate Record Detection
* Histogram Visualization
* Boxplot for Outlier Detection
* Correlation Heatmap
* Scatterplot Analysis
* Pairplot Visualization

⚙️ Data Preprocessing
* Missing values handled using median imputation
* Duplicate rows removed
* Outliers handled using IQR Method
* Skewness corrected using log1p() transformation
* Feature Engineering performed:
  - Hydrology_Distance
  - Hillshade_mean

⚖️ Class Imbalance Handling
SMOTE (Synthetic Minority Oversampling Technique) was used to balance the training dataset before model training.

🧪 Model Evaluation Metrics

Models were evaluated using:
  * Accuracy Score
  * Confusion Matrix
  * Classification Report
    
🔧 Hyperparameter Tuning
GridSearchCV was used to optimize the Random Forest model parameters for better accuracy and generalization.

💾 Saved Files
The project saves:

  * forest_model.pkl → Trained Machine Learning Model
  * label_encoder.pkl → Label Encoder for inverse transformation
    
🌐 Streamlit Web Application
The Streamlit application allows users to:
  * Enter environmental feature values
  * Predict forest cover type instantly
  * View prediction results interactively

📂 Project Structure
EcoType_Project/
│
├── EDA.py
├── train.py
├── app.py
├── covertype.csv
├── forest_model.pkl
├── label_encoder.pkl
├── requirements.txt
└── README.md

▶️ How to Run the Project
1️⃣ Clone Repository
  git clone <your-github-repo-link>
  cd EcoType_Project

2️⃣ Install Required Libraries
  pip install -r requirements.txt
3️⃣ Run EDA File
  python EDA.py
4️⃣ Train the Model
  python train.py
5️⃣ Run Streamlit Application
  streamlit run app.py
  
📌 Future Improvements
  * Deploy using Streamlit Cloud / Render
  * Add advanced feature selection methods
  * Improve UI design
  * Add real-time geospatial visualization
  * Use Deep Learning models for comparison
    
📷 Sample Output
  * Forest Cover Type Prediction
  * Feature Importance Graph
  * Correlation Heatmap
  * Classification Metrics
    
📚 Skills Gained
  * Exploratory Data Analysis
  * Data Cleaning
  * Feature Engineering
  * Classification Algorithms
  * Hyperparameter Tuning
  * Model Deployment
  * Streamlit Application Development
    
👩‍💻 Author
Aparna V
