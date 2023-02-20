# Setup plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.deep_learning_intro.ex6 import *

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
import warnings
warnings.filterwarnings('ignore')
import os
from tensorflow import keras
from tensorflow.keras import layers
import sys

sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/models/DNN", "")
sys.path.insert(1, os.path.join(dirname, "general_classes"))

from DataPreparation import DataPreparation
from Performance import Performance

path = dirname + "/data/train.csv"
df = pd.read_csv(path)
print(df.head())
X = df.copy()
y = X.pop("Lead")



features_num = [
    "Number words female", "Total words", "Number of words lead",
    "Difference in words lead and co-lead", "Number of male actors",
    "Year", "Number of female actors", "Number words male",
    "Gross", "Mean Age Male", "Mean Age Female", "Age Lead", "Age Co-Lead"
]

for col in features_num:
    X[col] = pd.to_numeric(X[col])

transformer_num = make_pipeline(
    SimpleImputer(strategy="constant"), # there are a few missing values
    StandardScaler(),
)

preprocessor = make_column_transformer(
    (transformer_num, features_num),
)

# stratify - make sure classes are evenlly represented across splits
X_train, X_valid, y_train, y_valid = \
    train_test_split(X, y, stratify=y, train_size=0.75)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)

input_shape = [X_train.shape[1]]


model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid'),
])


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)


early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)


history = model.fit(
    X_train, y_train,
    batch_size=200,
    epochs=200,
    callbacks=[early_stopping]
)
