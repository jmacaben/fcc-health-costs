# Some of the given code may have been changed for use outside of a notebook environment

# Cell 1 (given)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import requests
import os

# Cell 2 (given)
url = "https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv"
filename = "insurance.csv"

if not os.path.exists(filename):
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"Downloaded '{filename}'")
else:
    print(f"'{filename}' already exists. Skipping download.")

dataset = pd.read_csv(filename)
print(dataset.tail())

# Cell 3
# Convert categorical data to numbers
dataset["sex"] = pd.Categorical(dataset["sex"]).codes
dataset["smoker"] = pd.Categorical(dataset["smoker"]).codes
dataset["region"] = pd.Categorical(dataset["region"]).codes

# Use 80% of the data as the train_dataset and 20% of the data as the test_dataset
from sklearn.model_selection import train_test_split

train_dataset, test_dataset, train_labels, test_labels = train_test_split(
    dataset.drop("expenses", axis=1),  
    dataset["expenses"],              
    test_size=0.2,                    
    random_state=42                   
)

# Create a Normalization layer
# Helps model converge faster
normalizer = layers.Normalization()
normalizer.adapt(np.array(train_dataset))

# Create a Sequential model
model = keras.Sequential([
    normalizer,
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mae',
    metrics=['mae', 'mse']
)
model.build()
model.summary()

history = model.fit(
    train_dataset,
    train_labels,
    epochs=100
)

# Cell 4 (given)
# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)
