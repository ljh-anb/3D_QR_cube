import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import datetime
import joblib

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
    confusion_name = 'kNN_confusion_matrix_side_LP'
    model_name = 'kNN_trained_model_side_LP'
    missclass_name = 'kNN_missclassified_side_LP'

elif pred_type == 2:
    confusion_name = 'kNN_confusion_matrix_LP'
    model_name = 'kNN_trained_model_LP'
    missclass_name = 'kNN_missclassified_LP'

file_path = [f"{base_path}{file_name}{i}.csv" for i in range(1,file_num+1)]
df = [pd.read_csv(file) for file in file_path]
data = pd.concat(df, ignore_index=True)
test_df = [pd.read_csv(f'{base_path}{test_file_name}.csv')]
test_data = pd.concat(test_df, ignore_index=True)

# data preprocessing
if pred_type == 1:
    X = data[['h(x,y,z)f', 'h(x,y,z)b', 'r1', 'r2', 'r3', 'l1', 'l2', 'l3', 'd1', 'd2', 'd3', 'u1', 'u2', 'u3']]
    y = data['g(x,y,z)']

else:
    X = data[['h(x,y,z)f', 'h(x,y,z)b']]
    y = data['g(x,y,z)']

# data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
if pred_type == 1:
    X_test = test_data[['h(x,y,z)f', 'h(x,y,z)b', 'r1', 'r2', 'r3', 'l1', 'l2', 'l3', 'd1', 'd2', 'd3', 'u1', 'u2', 'u3']]
    y_test = test_data['g(x,y,z)']

else:
    X_test = test_data[['h(x,y,z)f', 'h(x,y,z)b']]
    y_test = test_data['g(x,y,z)']

X_test = X_test / 10000
X_train = X_train / 10000

# model training
knn = KNeighborsClassifier(n_neighbors=1001)
knn.fit(X_train, y_train)

joblib.dump(knn, f'{output_path}/{model_name}.joblib')
print('Trained model has been saved to knn_model.joblib')

# model examination
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# confusion matrix
label_num = 8
cm_df = pd.DataFrame(conf_matrix, index=[i for i in range(label_num)], columns=[i for i in range(label_num)])
cm_df.to_csv(f'{output_path}/{confusion_name}.csv')

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# missclassified data
misclassified_indices = y_test != y_pred
misclassified_data = X_test[misclassified_indices].copy()
misclassified_data['True Label'] = y_test[misclassified_indices]
misclassified_data['Predicted Label'] = y_pred[misclassified_indices]
misclassified_data.to_csv(f'{output_path}/{missclass_name}.csv', index=False)


end_time = time.time()
elapsed_time = end_time - start_time
print(f"duration time: {elapsed_time}초")