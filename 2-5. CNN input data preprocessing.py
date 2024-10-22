import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os
from tqdm import tqdm

# directory
file_num = 11 # number of your simulated cube files
base_path = './3D QR cube/Simulated_cube/'
file_name = 'random_side_LP_3_value_'
output_name = 'random side_LP_3 value'

def process_file(file_path, base_path, output_name):
    data = pd.read_csv(file_path)

    # input data preprocessing
    transformed = []
    labels = []
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
        labels.append(row['g(x,y,z)'])
    transformed_array = np.array(transformed)
    labels_array = np.array(labels)

    # data save
    file_number = os.path.basename(file_path).split('_')[-1].replace('.csv', '')
    npy_data_path = f"{base_path}{output_name}_data_{file_number}.npy"
    np.save(npy_data_path, transformed_array)
    npy_label_path = f"{base_path}{output_name}_label_{file_number}.npy"
    np.save(npy_label_path, labels_array)

    return npy_data_path, npy_label_path

# multi-threading
def main():
    file_paths = [f"{base_path}{file_name}{m}.csv" for m in range(1, file_num + 1)]
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_file, file_path, base_path, output_name) for file_path in file_paths]
        for future in tqdm(futures, total=len(futures), desc="Processing Files"):
            result = future.result()
            print(f'Processed: {result}')

if __name__ == '__main__':
    main()