import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import hamming_loss, classification_report
import pandas as pd

df = pd.read_csv('medicine_dataset.csv')

categorical_columns = ['Chemical Class', 'Therapeutic Class', 'Action Class']
df[categorical_columns] = df[categorical_columns].fillna('Unknown')

# 2. Replace NaN in binary column 'Habit Forming' with most frequent value
df['Habit Forming'] = df['Habit Forming'].fillna(df['Habit Forming'].mode()[0])

# 3. Replace NaN in side effect columns with 0
side_effect_columns = [col for col in df.columns if 'sideEffect' in col]
df[side_effect_columns] = df[side_effect_columns].fillna('')

def preprocess_data(data):
    # Input features
    features = ['Chemical Class', 'Therapeutic Class', 'Action Class', 'Habit Forming']
    X = data[features]
    
    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X)

    # Target labels (side effects)
    side_effect_columns = ['sideEffect0']
    y = data[side_effect_columns]

    return X_encoded, y

X, y = preprocess_data(df)
n_class  = len(y['sideEffect0'].unique())
le = LabelEncoder()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

X_train = X_train.rename(
    columns={fname: f"feature_{i}" for i, fname in enumerate(list(X_train.columns))}
)
X_test = X_test.rename(
    columns={fname: f"feature_{i}" for i, fname in enumerate(list(X_test.columns))}
)

model = XGBClassifier(num_class=n_class, objective='multi:softmax')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print (classification_report(y_test, y_pred))