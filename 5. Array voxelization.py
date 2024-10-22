import numpy as np
import pandas as pd
import os
import csv
import time
import datetime

# lateral intensity check
# 1 = LP + lateral / 2 = Only LP
pred_type = 2

# type of cube data
# simulated = 0 / actual = 1
external = 1

# ---------------------------------------------------------directory-------------------------------------------------------
model_dir = './3D QR cube/ML_learning_output/'
array_pred_dir = './3D QR cube/ML_prediction_output/'
voxel_pred_dir = './3D QR cube/Voxelization'
# ------------------------------------------------------DO NOT CHANGE HERE------------------------------------------------
v = 3
if external == 0:
    array_name = 'CNN_array_pred_side_LP'
    output_file_name = 'CNN_voxel_pred_side_LP'
else:
    array_name = 'CNN_array_pred_side_LP_ext'
    output_file_name = 'CNN_voxel_pred_side_LP_ext'

# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

#------------------------------------------------------array voxelization------------------------------------------
start_time = time.time()
current_time = datetime.datetime.now()
print("start time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))

df = pd.read_csv(f'{array_pred_dir}/{array_name}.csv')

# reading array type
def g(x, y, z, df):
    result = df[(df['x'] == x) & (df['y'] == y) & (df['z'] == z)]
    if not result.empty:
        return result['Predicted Label'].iloc[0]
    else:
        return None

# functions of glow score calculation
def pa3(x,y,z,df):
    result = df[(df['x'] == x) & (df['y'] == y) & (df['z'] == z)]
    if not result.empty:
        a = result['Class 1'].iloc[0] + result['Class 3'].iloc[0] + result['Class 5'].iloc[0] + result['Class 7'].iloc[0]
        return a
    else:
        return None
    
def pb3(x,y,z,df):
    result = df[(df['x'] == x) & (df['y'] == y) & (df['z'] == z)]
    if not result.empty:
        a = result['Class 2'].iloc[0] + result['Class 3'].iloc[0] + result['Class 6'].iloc[0] + result['Class 7'].iloc[0]
        return a
    else:
        return None
    
def pc3(x,y,z,df):
    result = df[(df['x'] == x) & (df['y'] == y) & (df['z'] == z)]
    if not result.empty:
        a = result['Class 4'].iloc[0] + result['Class 5'].iloc[0] + result['Class 6'].iloc[0] + result['Class 7'].iloc[0]
        return a
    else:
        return None

def single_edge(x, y, z, size):
    return (
        (x in [0, size - 1] and y not in [0, size - 1] and z not in [0, size - 1]) or
        (y in [0, size - 1] and x not in [0, size - 1] and z not in [0, size - 1]) or
        (z in [0, size - 1] and x not in [0, size - 1] and y not in [0, size - 1])
        )

# -------------------------------------------------------------------------------------------------------
# saving glow score list
f_proba = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
p_a = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
p_b = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
p_c = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
p_d = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
p_x = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
p_y = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
p_z = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]

chunk_size = 6*(v**2)
total_chunks = len(df) // chunk_size
for i in range(total_chunks):
    start_index = i*chunk_size
    end_index = start_index + chunk_size
    chunk_df = df.iloc[start_index:end_index]
    chunk_data = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
    confid_a = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
    confid_b = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
    confid_c = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
    for x in range(v+2):
        for y in range(v+2):
            for z in range(v+2):
                chunk_data[x][y][z] = g(x,y,z,chunk_df)
                confid_a[x][y][z] = pa3(x,y,z,chunk_df)
                confid_b[x][y][z] = pb3(x,y,z,chunk_df)
                confid_c[x][y][z] = pc3(x,y,z,chunk_df)

    for x in range(v+2):
        for y in range(v+2):
            for z in range(v+2):
                if single_edge(x, y, z, v+2):
                    prob_a = confid_a[x][y][z]
                    prob_b = confid_b[x][y][z]
                    prob_c = confid_c[x][y][z]
                    p_a[x][y][z] = prob_a
                    p_b[x][y][z] = prob_b
                    p_c[x][y][z] = prob_c

    for x in range(1,v+1):
        for y in range(1,v+1):
            for z in range(1,v+1):
                px1 = p_a[0][y][z] + p_c[v+1][y][z]
                px2 = p_b[0][y][z] + p_b[v+1][y][z]
                px3 = p_c[0][y][z] + p_a[v+1][y][z]
                py1 = p_a[x][0][z] + p_c[x][v+1][z]
                py2 = p_b[x][0][z] + p_b[x][v+1][z]
                py3 = p_c[x][0][z] + p_a[x][v+1][z]
                pz1 = p_a[x][y][0] + p_c[x][y][v+1]
                pz2 = p_b[x][y][0] + p_b[x][y][v+1]
                pz3 = p_c[x][y][0] + p_a[x][y][v+1]
                p_x[1][y][z] = px1
                p_x[2][y][z] = px2
                p_x[3][y][z] = px3
                p_y[x][1][z] = py1
                p_y[x][2][z] = py2
                p_y[x][3][z] = py3
                p_z[x][y][1] = pz1
                p_z[x][y][2] = pz2
                p_z[x][y][3] = pz3

    for x in range(1,v+1):
        for y in range(1,v+1):
            for z in range(1,v+1):
                prob = p_x[x][y][z] + p_y[x][y][z] + p_z[x][y][z]
                f_proba[x][y][z] = prob/6

    # saving voxelized voxels as csv files
    output_file_path = os.path.join(f'{voxel_pred_dir}', f'{output_file_name}.csv')

    with open(output_file_path, 'a', newline='') as output_file:
        output_writer = csv.writer(output_file)
        if i == 0:
            output_writer.writerow(['x', 'y', 'z', 'probability'])
        for x in range(1,v+1):
            for y in range(1,v+1):
                for z in range(1,v+1):                
                    output_writer.writerow([x, y, z, f_proba[x][y][z]])
    end_time = time.time()
    print(f"\rProgress: {i+1}/{total_chunks} ({((i+1)/total_chunks)*100:.2f}%) / Time consumed : {np.round(end_time - start_time, 1)} seconds", end='')

end_time = time.time()
print(f'Duration time: {(np.round((end_time - start_time),1)//60)} minutes {(np.round((end_time - start_time)%60),1)}seconds')