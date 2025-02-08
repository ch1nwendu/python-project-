# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# CTEC402 - Project 1
# Chinwendu Imegwu
from tensorflow import keras  # import keras from tensorflow
from tensorflow.keras import layers  # import layers from tensorflow keras
from sklearn.metrics import classification_report  # import Classification Report from sklearn
from sklearn.model_selection import train_test_split  # import Train Test Split from sklearn
from sklearn.preprocessing import StandardScaler  # import Standard Scaler from sklearn
from sklearn.ensemble import IsolationForest  # import Isolation Forest from sklearn
from matplotlib.ticker import ScalarFormatter  # import ScalarFormatter from matplotlib
import pandas as pd  # import pandas as the name pd
import matplotlib.pyplot as plt  # import pyplot from matplotlib

# reading in the data from the comma separated value file
data = pd.read_csv('synthetic_network_traffic.csv')

# oversamples the 'Anomaly' class to balance the class distribution
anomaly_data = data[data['IsAnomaly'] == 1]
oversampled_data = pd.concat([data, anomaly_data], axis=0)

# splits the dataset into features and labels
X = oversampled_data.drop(columns=['IsAnomaly'])  # Features
y = oversampled_data['IsAnomaly']  # Labels

# splits the dataset into training, validation and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# creates and fits the isolation forest model
isolation_forest = IsolationForest(contamination=0.1, random_state=42)
isolation_forest.fit(X_train)

# predicts anomalies using the isolation forest model
y_pred_iforest = isolation_forest.predict(X_test)
y_pred_iforest = (y_pred_iforest == -1)  # Convert -1 (anomaly) to 1, 1 (normal) to 0

# creates the deep learning model
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification, use 'sigmoid' for anomaly detection
])

# compiles the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# trains the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))

# evaluates the model on the test set
y_prediction = model.predict(X_test)
y_prediction = (y_prediction > 0.5)  # Apply threshold (adjust as needed)

# creates a classification report
report = classification_report(y_test, y_prediction, target_names=['Normal', 'Anomaly'], zero_division=1)

# creating a loss variable from the trained model
loss = history.history['loss']
# creating a validation loss variable from the trained model
validation_loss = history.history['val_loss']
# creating an accuracy variable from the trained model
accuracy = history.history['accuracy']
# creating a validation accuracy variable from the trained model
validation_accuracy = history.history['val_accuracy']
# creating an epochs variable to show the number of samples used
epochs = range(1, len(loss) + 1)

# setting the plot graph size
plt.figure(figsize=(12, 5))

# creating a subplot to display the training loss over epochs on the left of the two
plt.subplot(1, 2, 1)
# setting the epoch and loss variables for the plot over epochs
plt.plot(epochs, loss, 'bo-', label='Training Loss')
# setting the epoch and validation loss variables for the plot over epochs
plt.plot(epochs, validation_loss, 'ro-', label='Validation Loss')
# the title of the subplot
plt.title('Loss over Epochs')
# labeling the x-axis
plt.xlabel('Epochs')
# labeling the y-axis
plt.ylabel('Loss')
# creating a legend for the subplot
plt.legend()

# creating a subplot to display the training accuracy over epochs on the right of the two
plt.subplot(1, 2, 2)
# setting the epoch and accuracy variables for the plot over epochs
plt.plot(epochs, accuracy, 'bo-', label='Training Accuracy')
# setting the epoch and validation accuracy variables for the plot over epochs
plt.plot(epochs, validation_accuracy, 'ro-', label='Validation Accuracy')
# the title of the subplot
plt.title('Accuracy over Epochs')
# labeling the x-axis
plt.xlabel('Epochs')
# labeling the y-axis
plt.ylabel('Accuracy')
# setting the y-axis to use regular decimal form
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
# creating a legend for the subplot
plt.legend()

# prints the classification report
print(report)

# show both the accuracy and loss result graphs in the subplot
