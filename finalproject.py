import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , recall_score , precision_score ,confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load the dataset

try:
    df = pd.read_csv("projects\\student_success.csv")
except FileNotFoundError:
    print("Error: The file 'student_success.csv' was not found.")
    exit()


# Display basic information about the dataset

print(f"info: {df.info()}")
print(f"shape: {df.shape}")
print(f"isnull: {df.isnull().sum()}")

# Visualize the relationship between attendance and success status

plt.figure(figsize=(8, 6))
ax = sns.boxplot(x="success_status", y="attendance_percent", data=df, palette="Set2")

# Add counts above each box
success_counts = df["success_status"].value_counts()
for i, count in enumerate(success_counts):
    ax.text(i, df["attendance_percent"].max() + 1, f"Count: {count}",
            horizontalalignment='center', fontsize=10, color='black')

plt.xlabel("Success Status")
plt.ylabel("Attendance Percent")
plt.title("Boxplot of Attendance Percent by Success Status with Counts")
plt.grid(True)
plt.savefig("Boxplot_Attendance_Success.png", dpi=300, bbox_inches="tight")
plt.show()


# Encode categorical variables

le = LabelEncoder()
df["success_status"] = le.fit_transform(df["success_status"])

# Feature Scaling

ss = StandardScaler()
features = ["attendance_percent","mid_exam_score","final_exam_score","assignments_avg","study_hours_per_week"]
df[features] = ss.fit_transform(df[features])

# Split the dataset

x = df[features]
y = df["success_status"]
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42)

# Train the Logistic Regression model

model = LogisticRegression()
model.fit(x_train , y_train)
y_pred = model.predict(x_test)

# Model Evaluation

print(f"Accuracy: {accuracy_score(y_test , y_pred)}")
print(f"Recall: {recall_score(y_test , y_pred)}")
print(f"Precision: {precision_score(y_test , y_pred)}")

cm = confusion_matrix(y_test , y_pred)

plt.figure(figsize=(8,8))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Confusion Matrix")
plt.xticks([0,1], ["Not Successful", "Successful"])
plt.yticks([0,1], ["Not Successful", "Successful"])
plt.ylabel('True label', fontsize=16 )
plt.xlabel('Predicted label',fontsize=16)
plt.savefig("Confusion_Matrix.png", dpi=300, bbox_inches="tight")
plt.show()



# k-Means Clustering

km = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Groups"] = km.fit_predict(x)  # 0,1,2
# Numeric cluster labels ko custom names mein map karo
df_plot = df.sample(400, random_state=42)
group_names = {
    0: "Weak",
    1: "Average",
    2: "Excellent"
}
df_plot["Groups_Name"] = df_plot["Groups"].map(group_names)

# Scatter plot: attendance vs final_exam_score with groups
for group_name in df_plot["Groups_Name"].unique():
    group_data = df_plot[df_plot["Groups_Name"] == group_name]
    plt.scatter(group_data["attendance_percent"], group_data["final_exam_score"], label=group_name, alpha=0.6)
# for centoriods
centers = km.cluster_centers_
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    s=300,
    marker="X",
    label="Centroids"
)
plt.xlabel("Attendance Percent")
plt.ylabel("Final Exam Score")
plt.title("KMeans Clustering: Student Success Groups")
plt.legend(    
    loc="upper left")
plt.grid(True)
plt.savefig("K-Means.png", dpi=300, bbox_inches="tight")
plt.show()

# Principal Component Analysis (PCA)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
pcs_df = pd.DataFrame(pca_data , columns=["PCA1","PCA2"])
print(pcs_df)
perce = pca.explained_variance_ratio_ * 100
np.round(perce,2)
plt.figure(figsize=(8,6))
plt.scatter(pcs_df["PCA1"], pcs_df["PCA2"], color='blue', alpha=0.7, label='[PCA1, PCA2]')
plt.xlabel(f"PCA1 - {perce[0]:.2f}% Variance")
plt.ylabel(f"PCA2 - {perce[1]:.2f}% Variance")
plt.title("PCA of Student Success Data")
plt.grid(True)
plt.legend()
plt.savefig("PCA.png", dpi=300, bbox_inches="tight")
plt.show()

# Predicting Success Status for New Data
print("------------- Predict Your Success Status: -------------")
try:
    attendance_percent = float(input("Enter Attendance Percent: "))
    mid_exam_score = float(input("Enter Mid Exam Score: ")) 
    final_exam_score = float(input("Enter Final Exam Score: "))
    assignments_avg = float(input("Enter Assignments Average Score: "))
    study_hours_per_week = float(input("Enter Study Hours per Week: "))
    input_data = pd.DataFrame([[attendance_percent, mid_exam_score, final_exam_score, assignments_avg, study_hours_per_week]],columns=features)
    input_data[features] = ss.transform(input_data[features])
    prediction = model.predict(input_data)
    result = le.inverse_transform(prediction)
    print(f"Predicted Success Status: {result[0]}")
except Exception as e:
    print(f"Error in input. Please enter valid numerical values.\n {e}")
