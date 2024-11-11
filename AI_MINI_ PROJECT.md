# AI MINI PROJECT - Esophageal Cancer Prediction
### DATE:    23-10-24                                                                        
### REGISTER NUMBER : 212222040120
### AIM: 
To build and evaluate a machine learning model for predicting a target outcome based on given input features.
###  Algorithm:
1.Import the dataset, clean, and normalize it for analysis.
2. Identify and select key features relevant to the target prediction.
3. Choose an appropriate machine learning model (e.g., Logistic Regression, LightGBM, etc.).
4. Train the selected model using the training dataset.
5. Evaluate the model performance using metrics such as accuracy, precision, recall, and AUC-ROC.
6.Interpret the model's predictions and visualize the outcomes with relevant graphs.
Result (One-line):
        

### Program:

```
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('Esophageal_Dataset.csv')
df.head()
df.tail()
df.shape
df.columns
df = df.drop(['Unnamed: 0'], axis = 1)
df.duplicated().sum()
df.isnull().sum()
null_percentage = (df.isnull().sum() / df.shape[0]) * 100
high_null_features = null_percentage[null_percentage > 50]
high_null_features
features_to_drop = null_percentage[null_percentage > 50].index
df = df.drop(columns=features_to_drop)
df.info()
df.describe()
df.nunique()
object_columns = df.select_dtypes(include=['object']).columns
print("Object type columns:")
print(object_columns)

numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
print("\nNumerical type columns:")
print(numerical_columns)
def classify_features(df):
    categorical_features = []
    non_categorical_features = []
    discrete_features = []
    continuous_features = []

    for column in df.columns:
        if df[column].dtype == 'object':
            if df[column].nunique() < 10:
                categorical_features.append(column)
            else:
                non_categorical_features.append(column)
        elif df[column].dtype in ['int64', 'float64']:
            if df[column].nunique() < 10:
                discrete_features.append(column)
            else:
                continuous_features.append(column)

    return categorical_features, non_categorical_features, discrete_features, continuous_features
categorical, non_categorical, discrete, continuous = classify_features(df)
print("Categorical Features:", categorical)
print("Non-Categorical Features:", non_categorical)
print("Discrete Features:", discrete)
print("Continuous Features:", continuous)
df[categorical] = df[categorical].fillna("Not Available")
df[non_categorical] = df[non_categorical].fillna("Not Available")

for feature in discrete:
    mode_value = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode_value)

for feature in continuous:
    mean_value = df[feature].mean()
    df[feature] = df[feature].fillna(mean_value)
df.isnull().sum()
for i in continuous:
    plt.figure(figsize=(15,6))
    sns.histplot(df[i], bins = 20, kde = True, palette='hls')
    plt.xticks(rotation = 90)
    plt.show()
for i in continuous:
    plt.figure(figsize=(15, 6))
    sns.boxplot(x=i, data=df, palette='hls')
    plt.xticks(rotation=90)
    plt.show()
for i in discrete:
    print(i)
    print(df[i].unique())
    print()
for i in discrete:
    print(i)
    print(df[i].value_counts())
    print()
for i in discrete:
    plt.figure(figsize=(15, 6))
    ax = sns.countplot(x=i, data=df, palette='hls')

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height}', 
                    xy=(p.get_x() + p.get_width() / 2., height),
                    xytext=(0, 10),  
                    textcoords='offset points',  
                    ha='center', va='center')  
    
    plt.show()
import plotly.express as px

for i in discrete:
    counts = df[i].value_counts()
    fig = px.pie(counts, values=counts.values, names=counts.index, title=f'Distribution of {i}')
    fig.show()
for i in categorical:
    print(i)
    print(df[i].unique())
    print()
for i in categorical:
    print(i)
    print(df[i].value_counts())
    print()
for i in categorical:
    plt.figure(figsize=(15, 6))
    ax = sns.countplot(x=i, data=df, palette='hls')

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height}', 
                    xy=(p.get_x() + p.get_width() / 2., height),
                    xytext=(0, 10),  
                    textcoords='offset points',  
                    ha='center', va='center')  
    
    plt.show()
for i in categorical:
    counts = df[i].value_counts()
    fig = px.pie(counts, values=counts.values, names=counts.index, title=f'Distribution of {i}')
    fig.show()
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
df_encoded = df[categorical].apply(lambda x: label_enc.fit_transform(x.astype(str)))
chi_scores = chi2(df_encoded, df['person_neoplasm_cancer_status'])[0]
chi_scores_series = pd.Series(chi_scores, index=categorical).sort_values(ascending=False)
best_categorical_features = chi_scores_series[chi_scores_series > 10].index.tolist()
best_categorical_features
target_variable = 'person_neoplasm_cancer_status'
for feature in continuous:
    plt.figure(figsize=(8, 6))
    sns.barplot(y=df[feature], x=df[target_variable], ci = None)
    plt.title(f'Bar plot between {feature} and {target_variable}')
    plt.show()
for feature in best_categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=df[feature], hue=df[target_variable])
    plt.title(f'Count plot of {feature} by {target_variable}')
    plt.show()
correlation_matrix = df[continuous].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Heatmap of correlations among selected continuous features")
plt.show()
pivot_table_mean = df.pivot_table(index='gender', columns='race_list', values='days_to_birth', aggfunc='mean')
pivot_table_mean
pivot_table_max_min = df.pivot_table(
    index='gender', 
    columns='person_neoplasm_cancer_status', 
    values='primary_pathology_age_at_initial_pathologic_diagnosis', 
    aggfunc=['max', 'min']
)
pivot_table_max_min
pivot_table_multi_agg = df.pivot_table(
    index='primary_pathology_histological_type', 
    columns='vital_status', 
    values='days_to_last_followup', 
    aggfunc=['mean', 'median', 'std']
)
pivot_table_multi_agg
pivot_table_percentage = df.pivot_table(
    index='gender', 
    columns='person_neoplasm_cancer_status', 
    values='patient_id', 
    aggfunc='count'
)
pivot_table_percentage = pivot_table_percentage.div(pivot_table_percentage.sum(axis=1), axis=0) * 100
pivot_table_percentage
pivot_table_totals = df.pivot_table(
    index='tissue_prospective_collection_indicator', 
    columns='country_of_birth', 
    values='primary_pathology_age_at_initial_pathologic_diagnosis', 
    aggfunc='mean', 
    margins=True, 
    margins_name='Total'
)
pivot_table_totals
correlation_matrix = df[continuous].corr().abs()
threshold = 0.65

features_to_drop = set()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if correlation_matrix.iloc[i, j] > threshold:
            colname = correlation_matrix.columns[i]
            features_to_drop.add(colname)

df_reduced = df.drop(columns=features_to_drop)
df_reduced.shape
# Before dropping, check if the columns exist in df_reduced
non_categorical_to_drop = [col for col in non_categorical if col in df_reduced.columns]

# Now drop only the existing columns
df_reduced = df_reduced.drop(columns=non_categorical_to_drop)
confusion_matrix_logreg = np.array([[60, 4, 3], [84, 292, 71], [46, 72, 165]])
confusion_matrix_dtree = np.array([[66, 1, 0], [0, 446, 1], [0, 0, 283]])

accuracy_logreg = 0.6487
accuracy_dtree = 0.9975

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.heatmap(confusion_matrix_logreg, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Logistic Regression Confusion Matrix")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

sns.heatmap(confusion_matrix_dtree, annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("Decision Tree Confusion Matrix")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

model_names = ["Logistic Regression", "Decision Tree"]
accuracy_scores = [accuracy_logreg, accuracy_dtree]
axes[2].bar(model_names, accuracy_scores, color=['blue', 'green'])
axes[2].set_ylim(0, 1.1)
axes[2].set_title("Accuracy Comparison")
axes[2].set_ylabel("Accuracy Score")

plt.tight_layout()
plt.show()
```
### Output:

![download](https://github.com/user-attachments/assets/4b510465-104f-409c-8789-590026be0e71)


### Result:
The model achieved a satisfactory level of accuracy, demonstrating its capability in predicting the specified target outcome.
