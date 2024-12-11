import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultiOutputClassifier(RandomForestClassifier(random_state=42))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print (classification_report(y_test, y_pred))