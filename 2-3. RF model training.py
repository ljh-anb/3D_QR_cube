import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import joblib
import time
import datetime
import numpy as np

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
file_name = f'random_side_LP_3_value' 

if pred_type == 1:
    confusion_name = f'RF_confusion_matrix_3_side_LP'
    model_name = 'RF_trained_model_side_LP'
    missclass_name = 'RF_missclassified_side_LP'
else:
    confusion_name = f'RF_confusion_matrix_3_LP'
    model_name = 'RF_trained_model_LP'
    missclass_name = 'RF_missclassified_LP'

file_path = [f"{base_path}{file_name}_{i}.csv" for i in range(1,file_num+1)]
train_df = [pd.read_csv(file) for file in file_path]
train_data = pd.concat(train_df, ignore_index=True)
test_df = pd.read_csv(f'{base_path}{test_file_name}.csv')
test_data = pd.DataFrame(test_df)

# data preprocessing
if pred_type == 1:
    X_train = train_data[['h(x,y,z)f', 'h(x,y,z)b', 'r1', 'r2', 'r3', 'l1', 'l2', 'l3', 'd1', 'd2', 'd3', 'u1', 'u2', 'u3']]
    X_test = test_data[['h(x,y,z)f', 'h(x,y,z)b', 'r1', 'r2', 'r3', 'l1', 'l2', 'l3', 'd1', 'd2', 'd3', 'u1', 'u2', 'u3']]
else:
    X_train = train_data[['h(x,y,z)f', 'h(x,y,z)b']]
    X_test = test_data[['h(x,y,z)f', 'h(x,y,z)b']]

X_train = X_train / 10000
X_test = X_test / 10000
y_train = train_data['g(x,y,z)']
y_test = test_data['g(x,y,z)']

# model training
rf_model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=10, min_samples_split=2, min_samples_leaf=1, verbose = 1, random_state=2)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, f'{output_path}{model_name}.pkl')
print('model saved')

# model examination
y_test_pred = rf_model.predict(X_test)

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
pd.DataFrame(conf_matrix, index=rf_model.classes_, columns=rf_model.classes_).to_csv(f'{output_path}{confusion_name}.csv')
print('confusion table saved')

class_names = [str(cls) for cls in rf_model.classes_]
end_time = time.time()

# missclassified data
misclassified_indices = np.where(y_test != y_test_pred)[0]
misclassified_data = X_test.iloc[misclassified_indices].copy()
misclassified_data['True Label'] = y_test[misclassified_indices]
misclassified_data['Predicted Label'] = y_test_pred[misclassified_indices]
misclassified_data.to_csv(f'{output_path}{missclass_name}.csv', index=False)
print("Misclassified data saved")

print(f'duration time : {end_time - start_time} seconds')
print("Test Set Final Evaluation:")
print(f"Classification Report:\n{classification_report(y_test, y_test_pred, target_names=class_names)}")
