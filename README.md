## 🎓 Student Success Prediction System (Machine Learning Project)
**📌 Project Overview**

This project aims to predict student success status based on academic performance and study behavior using Machine Learning techniques.
It demonstrates a complete end-to-end ML workflow, including data preprocessing, visualization, supervised learning, unsupervised learning, and dimensionality reduction.

The project also allows real-time prediction by taking user input.

**🎯 Problem Statement**

- Educational institutions often want to identify:

- Which students are likely to succeed

- Which students need academic support

- Using historical student data, this project predicts whether a student is Successful or Not Successful.

**📊 Dataset Description**

The dataset (student_success.csv) contains the following features:

- Feature Name	Description
- attendance_percent	Student attendance percentage
- mid_exam_score	Mid-term exam score
- final_exam_score	Final exam score
- assignments_avg	Average assignment score
- study_hours_per_week	Weekly study hours
- success_status	Target variable (Successful / Not Successful)
**🧠 Machine Learning Techniques Used**
**✅ Supervised Learning**

- Logistic Regression
- Used to classify students as Successful or Not Successful

**✅ Unsupervised Learning**

KMeans Clustering
Groups students into performance-based clusters

**✅ Dimensionality Reduction**

- Principal Component Analysis (PCA)
- Reduces feature dimensions for visualization and analysis

**🔧 Technologies & Libraries**

- Python

- Pandas

- NumPy

- Matplotlib
  
- Seaborn

- Scikit-learn

**🔄 Project Workflow**

- Data Loading & Inspection

- Exploratory Data Analysis (EDA)

- Boxplot visualization

- Data Preprocessing

- Label Encoding

- Feature Scaling (StandardScaler)

- Train-Test Split

- Model Training

- Logistic Regression

- Model Evaluation

- Accuracy

- Precision

- Recall

- Confusion Matrix

- Clustering

- KMeans

- PCA Visualization

- User Input Prediction

**📈 Model Evaluation Metrics**

- Accuracy

- Precision

- Recall

- Confusion Matrix Visualization

- These metrics help evaluate how well the model classifies student success.

**📊 Visualizations Included**

- Boxplot: Attendance vs Success Status

- Confusion Matrix Heatmap

- KMeans Cluster Scatter Plot

- PCA 2D Scatter Plot

**🔮 Predicting Student Success (User Input)**

-The project allows users to input:

- Attendance %

- Exam scores

- Assignment average

- Study hours

- The trained model predicts the success status of the student.

## 📁 Project Structure
- ├── student_success.csv
- ├── student_success_prediction.py
- ├── README.md

**🚀 How to Run the Project**
pip install pandas numpy matplotlib seaborn scikit-learn
python student_success_prediction.py

**🧪 Future Improvements**

Use Pipeline to avoid data leakage

- Hyperparameter tuning

- Try other models (Random Forest, SVM)

- Deploy as a web app using Streamlit or Flask

**🏆 Conclusion**

- This project demonstrates:

- Strong understanding of Machine Learning fundamentals

- Ability to handle real-world data

- Use of multiple ML techniques in one system

- It is suitable for:

- Intermediate ML learners

- Portfolio projects

- Technical interviews


## 🚀 How to Run the Project
1. Clone the repository:

   git clone https://github.com/syedhassannaseem/student-success-analyser.git
