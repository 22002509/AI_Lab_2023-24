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

import os
import numpy as np
import pandas as pd

# Upload files manually
from google.colab import files
uploaded = files.upload()

# The files will be in the current directory
for dirname, _, filenames in os.walk('/content'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

!pip install wolta

df=pd.read_csv('Esophageal_Dataset.csv')
df.head()
df.shape

from wolta.data_tools import col_types

types = col_types(df, print_columns=True)


from wolta.data_tools import seek_null

seeked = seek_null(df, print_columns=True)

from wolta.data_tools import unique_amounts

unique_amounts(df)

will_del = ['Unnamed: 0',
           'patient_barcode',
           'tissue_source_site',
           'patient_id',
           'bcr_patient_uuid',
           'informed_consent_verified',
           'icd_o_3_site',
           'icd_o_3_histology',
           'icd_10',
           'tissue_prospective_collection_indicator',
           'tissue_retrospective_collection_indicator']


from wolta.feature_tools import list_deletings

df = list_deletings(df, extra=will_del, null_tolerance=10)

seeked = seek_null(df, print_columns=True)

types = col_types(df, print_columns=True)

seeked = seek_null(df, print_columns=True)

df['height'] = df['height'].fillna(np.nanmean(df['height'].values))
df['weight'] = df['weight'].fillna(np.nanmean(df['weight'].values))
df['tobacco_smoking_history'] = df['tobacco_smoking_history'].fillna(np.nanmean(df['tobacco_smoking_history'].values))



df['primary_pathology_year_of_initial_pathologic_diagnosis'] = df['primary_pathology_year_of_initial_pathologic_diagnosis'].astype(str)


df['country_of_procurement'] = df['country_of_procurement'].fillna('UNKNOWN')
df['alcohol_history_documented'] = df['alcohol_history_documented'].fillna('UNKNOWN')
df['primary_pathology_esophageal_tumor_cental_location'] = df['primary_pathology_esophageal_tumor_cental_location'].fillna('UNKNOWN')
df['primary_pathology_esophageal_tumor_involvement_sites'] = df['primary_pathology_esophageal_tumor_involvement_sites'].fillna('UNKNOWN')
df['primary_pathology_year_of_initial_pathologic_diagnosis'] = df['primary_pathology_year_of_initial_pathologic_diagnosis'].fillna('UNKNOWN')
df['primary_pathology_initial_pathologic_diagnosis_method'] = df['primary_pathology_initial_pathologic_diagnosis_method'].fillna('UNKNOWN')
df['primary_pathology_primary_lymph_node_presentation_assessment'] = df['primary_pathology_primary_lymph_node_presentation_assessment'].fillna('UKNOWN')


from wolta.data_tools import make_numerics

df['person_neoplasm_cancer_status'], outs = make_numerics(df['person_neoplasm_cancer_status'], space_requested=True)

print(outs)
outs = list(outs)
print(outs)

types = col_types(df)
loc = 0

for col in df.columns:
    if types[loc] == 'str':
        df[col] = make_numerics(df[col])
    
    loc += 1

df.describe()



from wolta.data_tools import stat_sum

stat_sum(df,
        ['max', 'min', 'width', 'var', 'med'])


df['person_neoplasm_cancer_status'].value_counts().plot(kind='pie')


y = df['person_neoplasm_cancer_status'].values
del df['person_neoplasm_cancer_status']
X = df.values
del df



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
del X, y

from collections import Counter

print(Counter(y_train))
print(Counter(y_test))


from wolta.model_tools import compare_models

results = compare_models('clf',
                        ['ada', 'cat', 'lbm', 'raf', 'ext', 'dtr', 'rdg', 'per'],
                        ['acc', 'precision', 'f1'],
                        X_train, y_train, X_test, y_test,
                        get_result=True)


from wolta.model_tools import get_best_model

model = get_best_model(results, 'acc', 'clf', X_train, y_train, behavior='max-best')
y_pred = model.predict(X_test)

from sklearn.metrics import classification_report as rep

print(rep(y_test, y_pred))


from sklearn.metrics import confusion_matrix as conf
from sklearn.metrics import ConfusionMatrixDisplay as cmd

cm = conf(y_test, y_pred)
disp = cmd(confusion_matrix=cm, display_labels=outs)
disp.plot()

```








### Output:

![image](https://github.com/user-attachments/assets/a6cb90c3-19c5-4903-bbb8-c2c984409518)



### Result:
The model achieved a satisfactory level of accuracy, demonstrating its capability in predicting the specified target outcome.
