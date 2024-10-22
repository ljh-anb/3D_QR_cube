import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import datetime
from sklearn.metrics import confusion_matrix

# Intensities of actual cube data have to be extracted and reformed by "Data processing and CNN visualize.py" 
# number of actual cubes
file_num = 5

# simulated cube prediction = 0 / actual cube prediction = 1
external = 1

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

model_name = f'CNN_trained_model_3_side_LP'

if external == 0:
    array_name = 'CNN_array_pred_side_LP'
else:
    array_name = 'CNN_array_pred_side_LP_ext'

confus_name = 'CNN_confusion_matrix_side_LP_ext'
#-------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------

start_time = time.time()
current_time = datetime.datetime.now()
print("start time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))

# 1. loading saved model
def load_and_predict(model_dir, model_name, data):
    model = load_model(f'{model_dir}/{model_name}.h5')
    predictions = model.predict(data)
    return predictions

# 2. loading cube data
df = []
if external == 0:
    single_data = pd.read_csv(f'{simulated_cube_dir}{simulated_output}.csv')
    df.append(single_data)
else:
    for a in range(1,file_num+1):
        single_data = pd.read_csv(f'{simulated_cube_dir}{simulated_output}_{a}.csv')
        df.append(single_data)
data = pd.concat(df, ignore_index=True)

# 3. data preprocessing
if external == 0:
    transformed_array = np.load(f'{simulated_cube_dir}random side_LP_3 value_data_test.npy')
    transformed_array = transformed_array.astype('float32')/10000
    
elif external == 1:
    transformed = []
    for index,row in data.iterrows():
        matrix = np.zeros((5,4,4))

        # Intensity
        matrix[:, :, 0] = [[row['h(x,y,z)f'], row['h(x,y,z)f'], row['h(x,y,z)f'], row['h(x,y,z)f']],
                            [row['r1'], row['l1'], row['u1'], row['d1']],
                            [row['r2'], row['l2'], row['u2'], row['d2']],
                            [row['r3'], row['l3'], row['u3'], row['d3']],
                            [row['h(x,y,z)b'], row['h(x,y,z)b'], row['h(x,y,z)b'], row['h(x,y,z)b']],]
            
        # Direction (F:1000/L:2000/R:3000/U:4000/D:5000/B:6000)
        matrix[:, :, 3] = [[1000, 1000, 1000, 1000],
                            [2000, 3000, 4000, 5000],
                            [2000, 3000, 4000, 5000],
                            [2000, 3000, 4000, 5000],
                            [6000, 6000, 6000, 6000]]
                
        # Sequence
        matrix[:, :, 1] = [[1000, 1000, 1000, 1000],
                            [2000, 2000, 2000, 2000],
                            [3000, 3000, 3000, 3000],
                            [4000, 4000, 4000, 4000],
                            [5000, 5000, 5000, 5000]]
        
        # Layer Index
        if row['x'] == 0 :
            matrix[:, :, 2] = [[1000*row['y'], 1000*row['y'], 1000*row['z'], 1000*row['z']],
                                [1000*row['y'], 1000*row['y'], 1000*row['z'], 1000*row['z']],
                                [1000*row['y'], 1000*row['y'], 1000*row['z'], 1000*row['z']],
                                [1000*row['y'], 1000*row['y'], 1000*row['z'], 1000*row['z']],
                                [1000*row['y'], 1000*row['y'], 1000*row['z'], 1000*row['z']]]
            
        elif row['y'] == 0 :
            matrix[:, :, 2] = [[1000*(4-row['x']), 1000*(4-row['x']), 1000*row['z'], 1000*row['z']],
                                [1000*(4-row['x']), 1000*(4-row['x']), 1000*row['z'], 1000*row['z']],
                                [1000*(4-row['x']), 1000*(4-row['x']), 1000*row['z'], 1000*row['z']],
                                [1000*(4-row['x']), 1000*(4-row['x']), 1000*row['z'], 1000*row['z']],
                                [1000*(4-row['x']), 1000*(4-row['x']), 1000*row['z'], 1000*row['z']]]
        
        elif row['z'] == 0 :
            matrix[:, :, 2] = [[1000*row['x'], 1000*row['x'], 1000*row['y'], 1000*row['y']],
                                [1000*row['x'], 1000*row['x'], 1000*row['y'], 1000*row['y']],
                                [1000*row['x'], 1000*row['x'], 1000*row['y'], 1000*row['y']],
                                [1000*row['x'], 1000*row['x'], 1000*row['y'], 1000*row['y']],
                                [1000*row['x'], 1000*row['x'], 1000*row['y'], 1000*row['y']]]
            
        elif row['x'] == 4 :
            matrix[:, :, 2] = [[1000*(4 - row['y']), 1000*(4 - row['y']), 1000*(4 - row['z']), 1000*(4 - row['z'])],
                                [1000*(4 - row['y']), 1000*(4 - row['y']), 1000*(4 - row['z']), 1000*(4 - row['z'])],
                                [1000*(4 - row['y']), 1000*(4 - row['y']), 1000*(4 - row['z']), 1000*(4 - row['z'])],
                                [1000*(4 - row['y']), 1000*(4 - row['y']), 1000*(4 - row['z']), 1000*(4 - row['z'])],
                                [1000*(4 - row['y']), 1000*(4 - row['y']), 1000*(4 - row['z']), 1000*(4 - row['z'])]]
            
        elif row['y'] == 4 :
            matrix[:, :, 2] = [[1000*row['x'], 1000*row['x'], 1000*(4 - row['z']), 1000*(4 - row['z'])],
                                [1000*row['x'], 1000*row['x'], 1000*(4 - row['z']), 1000*(4 - row['z'])],
                                [1000*row['x'], 1000*row['x'], 1000*(4 - row['z']), 1000*(4 - row['z'])],
                                [1000*row['x'], 1000*row['x'], 1000*(4 - row['z']), 1000*(4 - row['z'])],
                                [1000*row['x'], 1000*row['x'], 1000*(4 - row['z']), 1000*(4 - row['z'])]]
        
        elif row['z'] == 4 :
            matrix[:, :, 2] = [[1000*(4 - row['x']), 1000*(4 - row['x']), 1000*(4 - row['y']), 1000*(4 - row['y'])],
                                [1000*(4 - row['x']), 1000*(4 - row['x']), 1000*(4 - row['y']), 1000*(4 - row['y'])],
                                [1000*(4 - row['x']), 1000*(4 - row['x']), 1000*(4 - row['y']), 1000*(4 - row['y'])],
                                [1000*(4 - row['x']), 1000*(4 - row['x']), 1000*(4 - row['y']), 1000*(4 - row['y'])],
                                [1000*(4 - row['x']), 1000*(4 - row['x']), 1000*(4 - row['y']), 1000*(4 - row['y'])]]

        transformed.append(matrix)
    transformed_array = np.array(transformed)
    transformed_array = transformed_array.astype('float32')/10000

# 4. prediction
predictions_proba = load_and_predict(model_dir, model_name, transformed_array)

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
data.to_csv(f'{array_pred_dir}{array_name}.csv', index=False)

end_time = time.time()

print(f'Duration time : {end_time - start_time}S')
print(f'Prediction complete')