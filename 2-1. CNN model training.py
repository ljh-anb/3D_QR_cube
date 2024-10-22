import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Cube data have to be previously process by "2-5.CNN input data preprocessing.py"
# number of simulated cube files
file_num = 10

# directory
base_path = './3D QR cube/Simulated_cube/'
output_path = './3D QR cube/ML_learning_output/'# your output (trained model, confusion matrix, accuracy-loss graph, missclassified data)

data_npy_path = f'{base_path}random side_LP_3 value_data'
label_npy_path = f'{base_path}random side_LP_3 value_label'
test_data_path = f'{base_path}random side_LP_3 value_data_test.npy' # Rename the name of your simulated cube file for test data to this
test_label_path = f'{base_path}random side_LP_3 value_label_test.npy' # Rename the name of your simulated cube file for test data to this

confusion_name = f'CNN_confusion_matrix_3_side_LP'
model_name = f'CNN_trained_model_3_side_LP'
accuracy_name = f'CNN_accuracy-loss_per_iteration_3_side_LP'

# data splitting
def split_dataset(images, labels, random_state=38):
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]
    num_train = int(0.8 * len(images))
    x_train, x_val = images[:num_train], images[num_train:]
    y_train, y_val = labels[:num_train], labels[num_train:]
    return (x_train, y_train), (x_val, y_val)

# dataset loading
def load_dataset_npy(data_npy_path, file_num, label_npy_path):
    all_images = []
    all_labels = []
    for i in range(1, file_num + 1):
        images_path = f"{data_npy_path}_{i}.npy"
        labels_path = f"{label_npy_path}_{i}.npy"
        images = np.load(images_path)
        images = images.astype('float32') / 10000
        labels = np.load(labels_path)
        all_images.append(images)
        all_labels.append(labels)
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0) 
    return all_images, all_labels

# test dataset loading
def load_test_dataset_npy(test_data_path, test_label_path):
    test_images = np.load(test_data_path).astype('float32') / 10000
    test_labels = np.load(test_label_path)
    return test_images, test_labels

images, labels = load_dataset_npy(data_npy_path, file_num, label_npy_path)
random_state_value = 2
(train_images, train_labels), (val_images, val_labels) = split_dataset(images, labels, random_state=random_state_value)
test_images, test_labels = load_test_dataset_npy(test_data_path, test_label_path)

# convolution layer composing
model = Sequential([
    InputLayer(input_shape=(5, 4, 4)),
    Conv2D(10, kernel_size=(1, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(10, kernel_size=(1, 3), activation='relu', padding='same'),
    BatchNormalization(),        
    Conv2D(20, kernel_size=(1, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(20, kernel_size=(1, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(40, kernel_size=(1, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(40, kernel_size=(1, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(80, kernel_size=(1, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(80, kernel_size=(1, 3), activation='relu', padding='same'),
    BatchNormalization(),       
    Flatten(),
    Dense(64, activation='relu'),
    Dense(8, activation='softmax') 
])

model.compile(optimizer=Adam(learning_rate=0.0005),loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(train_images, to_categorical(train_labels), validation_data=(val_images, to_categorical(val_labels)), 
                    epochs=50, batch_size=1000, verbose = 1)
model.save(f'{output_path}{model_name}.h5')

# model examination
test_loss, test_accuracy = model.evaluate(test_images, to_categorical(test_labels))
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

# Confusion Matrix
class_num = 8
cm = confusion_matrix(test_labels, predicted_classes)
cm_df = pd.DataFrame(cm, index=[i for i in range(class_num)], columns=[i for i in range(class_num)])
cm_df.to_csv(f'{output_path}{confusion_name}.csv')

# Accuracy, Loss
history_df = pd.DataFrame(history.history)
result_csv_dir = f'{output_path}{accuracy_name}.csv'
history_df.to_csv(result_csv_dir, index=True)

# result visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy per Iteration')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss per Iteration')
plt.legend()
plt.show()

print(f"Test Accuracy: {test_accuracy}")
print(f"Test F-Score: {classification_report(test_labels, predicted_classes)}")
