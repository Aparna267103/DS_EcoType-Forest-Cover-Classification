🌲 EcoType: Forest Cover Classification using Machine Learning
📌 Project Overview

EcoType is a Machine Learning project that predicts forest cover types based on environmental and cartographic features such as elevation, slope, soil type, and hydrology distance.
The project helps in environmental monitoring, forest management, wildfire assessment, and land-use planning.

🚀 Features
Data Cleaning & Preprocessing
Exploratory Data Analysis (EDA)
Feature Engineering
Multiple Classification Models
Model Evaluation & Comparison
Best Model Saving using Joblib
Streamlit Web Application for Predictions
📊 Dataset Information
Dataset Size: 1,45,891 rows × 13 columns
Target Variable: Cover_Type
Domain: Environmental & Geospatial Data
Main Features:
Elevation
Aspect
Slope
Soil Type
Wilderness Area
Hydrology Distance
Hillshade Values
Fire Point Distance
🛠 Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Streamlit
Joblib
XGBoost
🤖 Machine Learning Models Used
Random Forest Classifier
Decision Tree Classifier
Logistic Regression
K-Nearest Neighbors (KNN)
XGBoost Classifier
📈 Project Workflow
Data Collection
Data Cleaning
Exploratory Data Analysis
Feature Engineering
Model Training
Model Evaluation
Hyperparameter Tuning
Model Saving
Streamlit Deployment
▶️ How to Run the Project
1️⃣ Install Required Libraries
pip install -r requirements.txt
2️⃣ Run the Main File
python main.py
3️⃣ Run Streamlit App
streamlit run app.py
📦 Model Output

The best-performing model is saved as:

models/best_model.pkl
🌐 Streamlit App

The Streamlit application allows users to:

Enter environmental feature values
Predict forest cover type instantly
View prediction results interactively
📌 Future Improvements
Add Hyperparameter Tuning
Improve UI Design
Deploy on Streamlit Cloud / Render
Add Feature Importance Visualization
👩‍💻 Author

Developed as part of a Machine Learning Capstone Project.
