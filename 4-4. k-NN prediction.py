import numpy as np
import pandas as pd
import joblib
import time
import datetime
from sklearn.metrics import confusion_matrix

# Intensities of actual cube data have to be extracted and reformed by "Data processing and CNN visualize.py" 
# number of simulated cube file / simulated = 1
file_num = 5

# simulated cube prediction = 0 / actual cube prediction = 1
external = 1

# lateral intensity check
# 1 = LP + lateral
# 2 = Only LP
pred_type = 2

# ---------------------------------------------------------directory-------------------------------------------------------
simulated_cube_dir = './3D QR cube/Simulated_cube/' # reformed actual cube data have be put this folder
model_dir = './3D QR cube/ML_learning_output/'
array_pred_dir = './3D QR cube/ML_prediction_output/'
# -------------------------------------------------DO NOT CHANGE HERE------------------------------------------------------
v = 3
if external == 0:
    simulated_output = f'random_side_LP_3_value_test'
else:
    simulated_output = f'rd 3 side_LP'

if pred_type == 1 :
    model_name = 'kNN_trained_model_side_LP'
    confus_name = 'KNN_confusion_matrix_side_LP_ext'
    if external == 0:
        array_name = 'kNN_array_pred_side_LP'
    else:
        array_name = 'kNN_array_pred_side_LP_ext'
elif pred_type == 2:
    model_name = 'kNN_trained_model_LP'
    confus_name = 'KNN_confusion_matrix_side_LP_ext'
    if external ==0:
        array_name = 'kNN_array_pred_LP'
    else:
        array_name = 'kNN_array_pred_LP_ext'
#------------------------------------------------------------------
#------------------------------------------------------------------
start_time = time.time()
current_time = datetime.datetime.now()
print("start time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))

# 1. loading saved model
model = joblib.load(f'{model_dir}/{model_name}.joblib')

# 2. loading cube data
df = []
if external == 0:
    single_data = pd.read_csv(f'{simulated_cube_dir}/{simulated_output}.csv')
    df.append(single_data)
else:
    for a in range(1,file_num+1):
        single_data = pd.read_csv(f'{simulated_cube_dir}/{simulated_output}_{a}.csv')
        df.append(single_data)
data = pd.concat(df, ignore_index=True)

# 3. data preprocessing
if pred_type == 1:
    features = ['h(x,y,z)f', 'h(x,y,z)b', 'r1', 'r2', 'r3', 'l1', 'l2', 'l3', 'd1', 'd2', 'd3', 'u1', 'u2', 'u3']
elif pred_type == 2:
    features = ['h(x,y,z)f', 'h(x,y,z)b']
    
X_new = data[features]
X_new = X_new / 10000

# 4. prediction
predictions_proba = model.predict_proba(X_new)

# 5. returning predicted label and confidence scores
predictions = np.argmax(predictions_proba, axis=1)
data['Predicted Label'] = predictions
confidence_scores = np.max(predictions_proba, axis=1)
data['Confidence Score'] = confidence_scores
for i in range(int(2**int(v))):
    data[f'Class {i}'] = predictions_proba[:, i]
    
if external == 1:
    df = []
    for a in range(1,file_num+1):
        single_data = pd.read_csv(f'{simulated_cube_dir}designated_g_value_{a}.csv')
        df.append(single_data)
    ext_data = pd.concat(df, ignore_index=True)
    true_labels = ext_data['g(x,y,z)']
    predicted_labels = data['Predicted Label']

    cm = confusion_matrix(true_labels, predicted_labels)
    cm_df = pd.DataFrame(cm, index=[i for i in range(8)], columns=[i for i in range(8)])
    cm_df.to_csv(f'{array_pred_dir}/{confus_name}.csv')

# 6. prediction results save
data.to_csv(f'{array_pred_dir}/{array_name}.csv', index=False)

end_time = time.time()

print(f'Duration time : {end_time - start_time}S')
print(f'Prediction complete')