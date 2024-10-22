import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime

# time recording
start_time = time.time()
current_time = datetime.datetime.now()
print("start time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))

# lateral intensity check
# 1 = LP + lateral
# 2 = Only LP
pred_type = 2

# number out simulated cube data files
file_num = 10

# directory
base_path = './3D QR cube/Simulated_cube/'
output_path = './3D QR cube/ML_learning_output/' # your output (trained model, confusion matrix, accuracy-loss graph, missclassified data)

test_file_name = f'random_side_LP_3_value_test' # Rename the name of your simulated cube file for test data to this
file_name = f'random_side_LP_3_value_' 

if pred_type == 1:
    confusion_name = 'DNN_confusion_matrix_side_LP'
    model_name = 'DNN_trained_model_side_LP'
    missclass_name = 'DNN_missclassified_side_LP'
    accuracy_name = f'DNN_3_accuracy-loss_per_iteration_side_LP'

elif pred_type == 2:
    confusion_name = 'DNN_confusion_matrix_LP'
    model_name = 'DNN_trained_model_LP'
    missclass_name = 'DNN_missclassified_LP'
    accuracy_name = f'DNN_3_accuracy-loss_per_iteration_LP'

file_path = [f"{base_path}{file_name}{i}.csv" for i in range(1,file_num+1)]
df = [pd.read_csv(file) for file in file_path]
data = pd.concat(df, ignore_index=True)
test_df = [pd.read_csv(f'{base_path}/{test_file_name}.csv')]
test_data = pd.concat(test_df, ignore_index=True)

# data preprocessing
label_num = 8
if pred_type == 1:
    features = data[['h(x,y,z)f', 'h(x,y,z)b', 'r1', 'r2', 'r3', 'l1', 'l2', 'l3', 'd1', 'd2', 'd3', 'u1', 'u2', 'u3']]
    test_features = test_data[['h(x,y,z)f', 'h(x,y,z)b', 'r1', 'r2', 'r3', 'l1', 'l2', 'l3', 'd1', 'd2', 'd3', 'u1', 'u2', 'u3']]

else:
    features = data[['h(x,y,z)f', 'h(x,y,z)b']]
    test_features = test_data[['h(x,y,z)f', 'h(x,y,z)b']]

features = features / 10000
test_features = test_features / 10000

labels = data['g(x,y,z)']
test_labels = test_data['g(x,y,z)']

# data splitting (train:validation=8:2)
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=12)
X_test = test_features
y_test = test_labels

# one-hot encoding
y_train = to_categorical(y_train, num_classes=label_num)
y_val = to_categorical(y_val, num_classes=label_num)
y_test = to_categorical(y_test, num_classes=label_num)

# neural network composing
nn_num = 64
model = Sequential([
    Dense(nn_num, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),  
    Dense(nn_num, activation='relu'),
    BatchNormalization(),  
    Dense(nn_num, activation='relu'),
    BatchNormalization(),  
    Dense(nn_num, activation='relu'),
    BatchNormalization(),  
    Dense(nn_num, activation='relu'),
    BatchNormalization(),  
    Dense(nn_num, activation='relu'),
    BatchNormalization(),  
    Dense(label_num, activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=1000, validation_data=(X_val, y_val))

# ----------------------------------------------------------------------------------------------------------
# model examination
y_pred = model.predict(X_test)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

history_df = pd.DataFrame(history.history)
result_csv_dir = f'{output_path}{accuracy_name}.csv'
history_df.to_csv(result_csv_dir, index=True)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"duration time: {elapsed_time}S")

# Plotting Accuracy and Loss Over Iteration
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()

# confusion matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
cm_df = pd.DataFrame(conf_matrix, index=[i for i in range(label_num)], columns=[i for i in range(label_num)])
cm_df.to_csv(f'{output_path}{confusion_name}.csv')

model.save(f'{output_path}{model_name}.h5')

print(f"Test Accuracy: {test_accuracy}")
print(f"Classification Report:\n{classification_report(y_test_classes, y_pred_classes)}")

# Export Misclassified Data
misclassified_indices = np.where(y_test_classes != y_pred_classes)[0]
misclassified_data = X_test.iloc[misclassified_indices].copy()
misclassified_data['True Label'] = y_test_classes[misclassified_indices]
misclassified_data['Predicted Label'] = y_pred_classes[misclassified_indices]
misclassified_data.to_csv(f'{output_path}{missclass_name}.csv', index=False)
print("Misclassified data saved")