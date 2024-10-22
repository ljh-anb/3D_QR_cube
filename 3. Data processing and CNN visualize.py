from PIL import Image
import numpy as np
import pandas as pd
import os
import csv
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Whether to use the visualization of the cube predicted by the CNN model without voxelization (on:1 / off:2)
each_side_view = 1

# Which block to turn on (on:1 / off:0)
intensity_reading = 1
intensity_reforming = 1
prediction = 1
reconstruction = 1
visualization = 1

# directory
image_dir = './3D QR cube/Image/' # All six images should have the same name, and the numbering should be labeled in the order of +y/+x/-y/-x/+z/-z
mask_dir = './3D QR cube/mask/'
output_csv_dir = './3D QR cube/Intensity_extraction/'

image_name = 'flatten_image' # put your image name
mask_name = 'Mask'

model_dir = './3D QR cube/ML_learning_output/'
pred_dir = './3D QR cube/ML_prediction_output/'

# -------------------------------------------------DO NOT CHANGE HERE------------------------------------------------------
v = 3
output_csv_name = 'external_intensity_side_LP'
intensity1 = f'{output_csv_name}_1'
intensity2 = f'{output_csv_name}_2'
intensity3 = f'{output_csv_name}_3'
intensity4 = f'{output_csv_name}_4'
intensity5 = f'{output_csv_name}_5'
intensity6 = f'{output_csv_name}_6'

ext_output = 'rd 3 side_LP'

model_name = 'CNN_trained_model_3_side_LP'
confid_name = 'CNN_prediction_side_LP'

output_file_name_1 = 'CNN_prediction_side_LP_1'
output_file_name_2 = 'CNN_prediction_side_LP_2'
output_file_name_3 = 'CNN_prediction_side_LP_3'
output_file_name_4 = 'CNN_prediction_side_LP_4'
output_file_name_5 = 'CNN_prediction_side_LP_5'
output_file_name_6 = 'CNN_prediction_side_LP_6'
output_file_name_7 = 'CNN_prediction_side_LP_total'
# ----------------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------intensity reading block------------------------------------------------------
# The intensities from images of six faces of the 3D QR cube are read here

if intensity_reading == 1:
    def calculate_ordered_pixel_averages(image_dir, image_name, mask_dir, mask_name, output_csv_dir, output_csv_name, i):
        image_path = f'{image_dir}{image_name}_{i}.tif'
        mask_path = f'{mask_dir}{mask_name}.tif'
        output_csv_path = f'{output_csv_dir}{output_csv_name}_{i}.csv'

        image = Image.open(image_path)
        mask = Image.open(mask_path)
        image_np = np.array(image)
        mask_np = np.array(mask)
        if mask_np.shape != image_np.shape:
            mask_resized = np.resize(mask_np, (image_np.shape[0], image_np.shape[1]))
        else:
            mask_resized = mask_np
            
        # mask area setting
        # 3 6 9
        # 2 5 8
        # 1 4 7
            
        # 4 8 12 16
        # 3 7 11 15
        # 2 6 10 14
        # 1 5  9 13

        grid_rows, grid_cols = v, v
        height, width = mask_resized.shape
        cell_height, cell_width = height // grid_rows, width // grid_cols
            
        grid_area_numbers = np.zeros_like(mask_resized)
        for row in range(grid_rows):
            for col in range(grid_cols):
                grid_area_numbers[row*cell_height:(row+1)*cell_height, col*cell_width:(col+1)*cell_width] = v - row + col * grid_rows
            
        # calculating the average pixel value in each area
        ordered_area_pixel_averages = {}
        for area_number in range(1, grid_rows*grid_cols + 1):
            area_mask = grid_area_numbers == area_number
            valid_mask = area_mask & (mask_resized > 0)
            area_pixels = image_np[valid_mask]
                
            ordered_area_pixel_averages[area_number] = round(np.mean(area_pixels)) if area_pixels.size > 0 else 0
            
        df_ordered_area_pixel_averages = pd.DataFrame(list(ordered_area_pixel_averages.items()), columns=['Area Number', 'Pixel Value'])
        df_ordered_area_pixel_averages.to_csv(output_csv_path, index=False)

    for i in range(1,7):
        calculate_ordered_pixel_averages(image_dir, image_name, mask_dir, mask_name, output_csv_dir, output_csv_name, i)

    print('intensity reading complete')
# ------------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------intensity reforming block-----------------------------------------------------
# The intensities read in "intensity reading block" are reformed to be suitable for input data preprocessing

if intensity_reforming == 1:
    df1 = pd.read_csv(f'{output_csv_dir}/{intensity1}.csv')
    df2 = pd.read_csv(f'{output_csv_dir}/{intensity2}.csv')
    df3 = pd.read_csv(f'{output_csv_dir}/{intensity3}.csv')
    df4 = pd.read_csv(f'{output_csv_dir}/{intensity4}.csv')
    df5 = pd.read_csv(f'{output_csv_dir}/{intensity5}.csv')
    df6 = pd.read_csv(f'{output_csv_dir}/{intensity6}.csv')

    # reading intensities according to the coordinates 
    def h(x, y, z):
        if y == 0:
            df = df1  
            result = df[df['Area Number']== v*(x-1)+z]   
            if not result.empty:
                return result['Pixel Value'].iloc[0]
            else:
                return None 

        if x == 0:
            df = df2
            result = df[df['Area Number']== v*(v-y)+z]   
            if not result.empty:
                return result['Pixel Value'].iloc[0]
            else:
                return None
                
        if y == v+1 :
            df = df3     
            result = df[df['Area Number']== v*(v-x)+z]   
            if not result.empty:
                return result['Pixel Value'].iloc[0]
            else:
                return None

        if x == v+1 :
            df = df4        
            result = df[df['Area Number']== v*(y-1)+z]   
            if not result.empty:
                return result['Pixel Value'].iloc[0]
            else:
                return None
            
        if z == 0 :
            df = df5
            result = df[df['Area Number']== v*(v-x)+y]
            if not result.empty:
                return result['Pixel Value'].iloc[0]
        
        if z == v+1 :
            df = df6
            result = df[df['Area Number']== v*(x-1)+y]
            if not result.empty:
                return result['Pixel Value'].iloc[0]

    # reform the intensities   
    def side_LP_3(x, y, z):
        if x == 0:
            h_g_value = (h(0,y,z), h(v+1,y,z), 
                        h(1,0,z), h(2,0,z), h(3,0,z),                    
                        h(1,v+1,z), h(2,v+1,z), h(3,v+1,z),
                        h(1,y,0), h(2,y,0), h(3,y,0),
                        h(1,y,v+1), h(2,y,v+1), h(3,y,v+1))

        elif y == 0:
            h_g_value = (h(x,0,z), h(x,v+1,z),
                        h(v+1,1,z), h(v+1,2,z), h(v+1,3,z),                     
                        h(0,1,z), h(0,2,z), h(0,3,z),
                        h(x,1,0), h(x,2,0), h(x,3,0),
                        h(x,1,v+1), h(x,2,v+1), h(x,3,v+1))
            
        elif z == 0:
            h_g_value = (h(x,y,0), h(x,y,v+1),
                            h(0,y,1), h(0,y,2), h(0,y,3),
                            h(v+1,y,1), h(v+1,y,2), h(v+1,y,3),
                            h(x,0,1), h(x,0,2), h(x,0,3),
                            h(x,v+1,1), h(x,v+1, 2), h(x,v+1,3))

        elif x == v+1:
            h_g_value = (h(v+1,y,z), h(0,y,z),     
                        h(3,v+1,z), h(2,v+1,z), h(1,v+1,z),                                     
                        h(3,0,z), h(2,0,z), h(1,0,z),
                        h(3,y,0), h(2,y,0), h(1,y,0),
                        h(3,y,v+1), h(2,y,v+1), h(1,y,v+1))

        elif y == v+1:
            h_g_value = (h(x,v+1,z), h(x,0,z),
                        h(0,3,z), h(0,2,z), h(0,1,z),                     
                        h(v+1,3,z), h(v+1,2,z), h(v+1,1,z),
                        h(x,3,0), h(x,2,0), h(x,1,0),
                        h(x,3,v+1), h(x,2,v+1), h(x,1,v+1))
            
        elif z == v+1:
            h_g_value = (h(x,y,v+1), h(x,y,0),
                            h(v+1,y,3), h(v+1,y,2), h(v+1,y,1),
                            h(0,y,3), h(0,y,2), h(0,y,1),
                            h(x,0,3), h(x,0,2), h(x,0,1),
                            h(x,v+1,3), h(x,v+1,2), h(x,v+1,1))
            
        else:
            h_g_value = (None, None, None, None, None, None, None, None, None, None, None, None, None, None)
        return h_g_value

    def single_edge(x, y, z, size):
        return (
            (x in [0, size - 1] and y not in [0, size - 1] and z not in [0, size - 1]) or
            (y in [0, size - 1] and x not in [0, size - 1] and z not in [0, size - 1]) or
            (z in [0, size - 1] and x not in [0, size - 1] and y not in [0, size - 1])
            )

    output_file = os.path.join(output_csv_dir,f'{ext_output}.csv')

    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Writing header
        csv_writer.writerow(['x', 'y', 'z', 'h(x,y,z)f', 'h(x,y,z)b', 'r1', 'r2', 'r3', 'l1', 'l2', 'l3', 'd1', 'd2', 'd3', 'u1', 'u2', 'u3'])
        
        # Writing values for different x, y, z combinations
        for x in range(v + 2):
            for y in range(v + 2):
                for z in range(v + 2):
                        if single_edge(x, y, z, v+2):
                            h_g_values = side_LP_3(x, y, z)
                            if None not in h_g_values: 
                                csv_writer.writerow([x, y, z, *h_g_values])

    print('reforming complete')
# --------------------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Prediction through trained model-----------------------------------------------
# The reformed data are preprocessed and used for prediction

if prediction == 1:
    # 1. model loading
    model = load_model(f'{model_dir}/{model_name}.h5')

    # 2. data loading
    data = pd.read_csv(f'{output_csv_dir}/{ext_output}.csv')

    # 3. data preprocessing
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

    # 4. array prediction
    predictions_proba = model.predict(transformed_array)

    # 5. returning predicted label and confidence scores
    predictions = np.argmax(predictions_proba, axis=1)
    data['Predicted Label'] = predictions
    confidence_scores = np.max(predictions_proba, axis=1)
    confidence_scores = np.round(confidence_scores, 2)
    data['Confidence Score'] = confidence_scores
    for i in range(int(2**int(v))):
        data[f'Class {i}'] = np.round(predictions_proba[:, i],2)  

    # 6. prediction results save
    data.to_csv(f'{pred_dir}/{confid_name}.csv', index=False)

    print('prediction complete')
# --------------------------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------predicted array reconstruction---------------------------------------------
# The arrays predicted before are divided into voxels here
# This block is to reconstruct a cube only by the array prediction of the model

if reconstruction == 1:
    # predicted file loading
    df = pd.read_csv(f'{pred_dir}{confid_name}.csv')

    # reading array type
    def g(x, y, z, df):
        result = df[(df['x'] == x) & (df['y'] == y) & (df['z'] == z)]
        if not result.empty:
            return result['Predicted Label'].iloc[0]
        else:
            return None

    # functions of deviding arrays into voxels
    def reconstruction_y_0(predict):
        binary = bin(predict)[2:]
        for i in range(len(binary)):
            f_value_1[x][i+1][z] = int(binary[len(binary)-i-1])

    def reconstruction_x_0(predict):
        binary = bin(predict)[2:]
        for i in range(len(binary)):
            f_value_2[i+1][y][z] = int(binary[len(binary)-i-1])

    def reconstruction_z_0(predict):
        binary = bin(predict)[2:]
        for i in range(len(binary)):
            f_value_3[x][y][i+1] = int(binary[len(binary)-i-1])

    def reconstruction_y_vn1(predict):
        binary = bin(predict)[2:]
        for i in range(len(binary)):
            f_value_4[x][v-i][z] = int(binary[len(binary)-i-1])

    def reconstruction_x_vn1(predict):
        binary = bin(predict)[2:]
        for i in range(len(binary)):
            f_value_5[v-i][y][z] = int(binary[len(binary)-i-1])

    def reconstruction_z_vn1(predict):
        binary = bin(predict)[2:]
        for i in range(len(binary)):
            f_value_6[x][y][v-i] = int(binary[len(binary)-i-1])
        
    def single_edge(x, y, z, size):
        return (
            (x in [0, size - 1] and y not in [0, size - 1] and z not in [0, size - 1]) or
            (y in [0, size - 1] and x not in [0, size - 1] and z not in [0, size - 1]) or
            (z in [0, size - 1] and x not in [0, size - 1] and y not in [0, size - 1])
            )

    # lists for saving types of voxels
    f_value_1 = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
    f_value_2 = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
    f_value_3 = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
    f_value_4 = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
    f_value_5 = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
    f_value_6 = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]

    # deviding arrays into voxels
    for x in range(v+2):
        for y in range(v+2):
            for z in range(v+2):
                if single_edge(x, y, z, v+2):
                    predicted = g(x, y, z, df)
                    if x == 0:
                        reconstruction_x_0(predicted)
                    elif x == v+1:
                        reconstruction_x_vn1(predicted)
                    elif y == 0:
                        reconstruction_y_0(predicted)
                    elif y == v+1:
                        reconstruction_y_vn1(predicted)
                    elif z == 0:
                        reconstruction_z_0(predicted)
                    elif z == v+1:
                        reconstruction_z_vn1(predicted)
                    else:
                        None
    # --------------------------------------------------------------------------------------------------------------------------------------

    # --------------------------------------------------------------array voxelization------------------------------------------------------
    # Array voxelization is executed here

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

    # saving glow score list
    f_value = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
    p_a = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
    p_b = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
    p_c = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
    p_d = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
    p_x = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
    p_y = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
    p_z = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]

    # glow score calculation
    for x in range(v+2):
        for y in range(v+2):
            for z in range(v+2):
                if single_edge(x, y, z, v+2):
                    prob_a = pa3(x,y,z,df)
                    prob_b = pb3(x,y,z,df)
                    prob_c = pc3(x,y,z,df)
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
                if prob >= 3:
                    f_value[x][y][z] = 1
                else:
                    f_value[x][y][z] = 0

    # saving devided voxels as csv files
    output_file_path_1 = os.path.join(pred_dir, f'{output_file_name_1}.csv')
    output_file_path_2 = os.path.join(pred_dir, f'{output_file_name_2}.csv')
    output_file_path_3 = os.path.join(pred_dir, f'{output_file_name_3}.csv')
    output_file_path_4 = os.path.join(pred_dir, f'{output_file_name_4}.csv')
    output_file_path_5 = os.path.join(pred_dir, f'{output_file_name_5}.csv')
    output_file_path_6 = os.path.join(pred_dir, f'{output_file_name_6}.csv')
    output_file_path_7 = os.path.join(pred_dir, f'{output_file_name_7}.csv')

    with open(output_file_path_1, 'w', newline='') as output_file_1:
        output_writer = csv.writer(output_file_1)
        output_writer.writerow(['x', 'y', 'z', 'f(x,y,z)'])
        for x in range(1,v+1):
            for y in range(1,v+1):
                for z in range(1,v+1):
                    output_writer.writerow([x, y, z, f_value_1[x][y][z]])

    with open(output_file_path_2, 'w', newline='') as output_file_2:
        output_writer = csv.writer(output_file_2)
        output_writer.writerow(['x', 'y', 'z', 'f(x,y,z)'])
        for x in range(1,v+1):
            for y in range(1,v+1):
                for z in range(1,v+1):
                    output_writer.writerow([x, y, z, f_value_2[x][y][z]])
                    
    with open(output_file_path_3, 'w', newline='') as output_file_3:
        output_writer = csv.writer(output_file_3)
        output_writer.writerow(['x', 'y', 'z', 'f(x,y,z)'])
        for x in range(1,v+1):
            for y in range(1,v+1):
                for z in range(1,v+1):                
                    output_writer.writerow([x, y, z, f_value_3[x][y][z]])

    with open(output_file_path_4, 'w', newline='') as output_file_4:
        output_writer = csv.writer(output_file_4)
        output_writer.writerow(['x', 'y', 'z', 'f(x,y,z)'])
        for x in range(1,v+1):
            for y in range(1,v+1):
                for z in range(1,v+1):                
                    output_writer.writerow([x, y, z, f_value_4[x][y][z]])

    with open(output_file_path_5, 'w', newline='') as output_file_5:
        output_writer = csv.writer(output_file_5)
        output_writer.writerow(['x', 'y', 'z', 'f(x,y,z)'])
        for x in range(1,v+1):
            for y in range(1,v+1):
                for z in range(1,v+1):                
                    output_writer.writerow([x, y, z, f_value_5[x][y][z]])

    with open(output_file_path_6, 'w', newline='') as output_file_6:
        output_writer = csv.writer(output_file_6)
        output_writer.writerow(['x', 'y', 'z', 'f(x,y,z)'])
        for x in range(1,v+1):
            for y in range(1,v+1):
                for z in range(1,v+1):                
                    output_writer.writerow([x, y, z, f_value_6[x][y][z]])

    with open(output_file_path_7, 'w', newline='') as output_file_7:
        output_writer = csv.writer(output_file_7)
        output_writer.writerow(['x', 'y', 'z', 'f(x,y,z)'])
        for x in range(1,v+1):
            for y in range(1,v+1):
                for z in range(1,v+1):                
                    output_writer.writerow([x, y, z, f_value[x][y][z]])
# --------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------Reconstructed Cube Visualization-----------------------------------------------
# Visualize the cubes

if visualization == 1:
    def cube_vertices(x, y, z, d):
        return [
            (x - d, y - d, z - d),
            (x + d, y - d, z - d),
            (x + d, y + d, z - d),
            (x - d, y + d, z - d),
            (x - d, y - d, z + d),
            (x + d, y - d, z + d),
            (x + d, y + d, z + d),
            (x - d, y + d, z + d)
        ]

    def draw_cube(x, y, z, f, d=0.5):
        if f == 1:
            vertices = cube_vertices(x, y, z, d)
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[4], vertices[7], vertices[3], vertices[0]]
            ]
            return Poly3DCollection(faces, alpha=0.8, linewidths=0.1, edgecolors='grey', facecolors='#FFA07A')
        else:
            vertices = cube_vertices(x, y, z, d)
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[4], vertices[7], vertices[3], vertices[0]]
            ]
            return Poly3DCollection(faces, alpha=0.2, linewidths=0.1, edgecolors='grey', facecolors='white')


    # -------------------------------------------------------------------------------------------------
    if each_side_view == 1 :

        # ---------- prediction from +y ----------
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for x in range(1, v+1):
            for y in range(1, v+1):
                for z in range(1, v+1):
                    f = f_value_1[x][y][z]
                    cube = draw_cube(x, y, z, f)
                    if cube:
                        ax.add_collection3d(cube)

        ax.set_xlim(0, v+1)
        ax.set_ylim(0, v+1)
        ax.set_zlim(0, v+1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.axis('off')

        plt.show()

        # ---------- prediction from +x ----------
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for x in range(1, v+1):
            for y in range(1, v+1):
                for z in range(1, v+1):
                    f = f_value_2[x][y][z]
                    cube = draw_cube(x, y, z, f)
                    if cube:
                        ax.add_collection3d(cube)

        ax.set_xlim(0, v+1)
        ax.set_ylim(0, v+1)
        ax.set_zlim(0, v+1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.axis('off')

        plt.show()

        # ---------- prediction from +z ----------
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for x in range(1, v+1):
            for y in range(1, v+1):
                for z in range(1, v+1):
                    f = f_value_6[x][y][z]
                    cube = draw_cube(x, y, z, f)
                    if cube:
                        ax.add_collection3d(cube)

        ax.set_xlim(0, v+1)
        ax.set_ylim(0, v+1)
        ax.set_zlim(0, v+1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.axis('off')

        plt.show()

        # ---------- prediction from -y ----------
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for x in range(1, v+1):
            for y in range(1, v+1):
                for z in range(1, v+1):
                    f = f_value_4[x][y][z]
                    cube = draw_cube(x, y, z, f)
                    if cube:
                        ax.add_collection3d(cube)

        ax.set_xlim(0, v+1)
        ax.set_ylim(0, v+1)
        ax.set_zlim(0, v+1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.axis('off')

        plt.show()

        # ---------- prediction from -x ----------
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for x in range(1, v+1):
            for y in range(1, v+1):
                for z in range(1, v+1):
                    f = f_value_5[x][y][z]
                    cube = draw_cube(x, y, z, f)
                    if cube:
                        ax.add_collection3d(cube)

        ax.set_xlim(0, v+1)
        ax.set_ylim(0, v+1)
        ax.set_zlim(0, v+1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.axis('off')

        plt.show()

        # ---------- prediction from -z ----------
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for x in range(1, v+1):
            for y in range(1, v+1):
                for z in range(1, v+1):
                    f = f_value_3[x][y][z]
                    cube = draw_cube(x, y, z, f)
                    if cube:
                        ax.add_collection3d(cube)

        ax.set_xlim(0, v+1)
        ax.set_ylim(0, v+1)
        ax.set_zlim(0, v+1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.axis('off')

        plt.show()

    # ----------- voxelized cube -----------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for x in range(1, v+1):
        for y in range(1, v+1):
            for z in range(1, v+1):
                f = f_value[x][y][z]
                cube = draw_cube(x, y, z, f)
                if cube:
                    ax.add_collection3d(cube)

    ax.set_xlim(0, v+1)
    ax.set_ylim(0, v+1)
    ax.set_zlim(0, v+1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.axis('off')

    plt.show()