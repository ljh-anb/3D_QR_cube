import random
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# time recording
start_time = time.time()

# value output switch (on:1 / off:0)
fval = 1 # types of voxels (UCNPs or blank)
gval = 1 # types of arrays
hval = 0 # intensity value of an array relative to the front
side_LP_3 = 1 # LP and lateral intensities

# output size
n = 100 # number of cubes per a file
m = 11 # number of files

# directory setting

input_dir = './3D QR cube/range/'
range_name = 'ranges'
output_dir ='./3D QR cube/Simulated_cube'

# -------------------------------------------------DO NOT CHANGE HERE------------------------------------------------------
v = 3
f_name = 'random_f_value'
g_name = 'random_g_value'
h_name = 'random_h_value'
side_LP_3_name = 'random_side_LP_3_value'
# -------------------------------------------------------------------------------------------------------------------------

# range register
range_file_path = os.path.join(input_dir, f'{range_name}.csv')
ranges = {}
with open(range_file_path, mode='r') as csvfile:
    csvreader = list(csv.reader(csvfile))
    keys = csvreader[0]  # first low as key
    means = csvreader[1]  # second low as mean
    stds = csvreader[2]  # third low as standard deviation
    for i, key in enumerate(keys):
        ranges[int(key)] = (float(means[i])), float(stds[i])


# f_value : determine type of voxel in a cube (UCNPs or blank)

def f(x, y, z):
            if x in [0, v+1] or y in [0, v+1] or z in [0, v+1]:
                return int(0)
            else:
                return random.randint(0, 1)

def single_edge(x, y, z, size):
    return (
        (x in [0, size - 1] and y not in [0, size - 1] and z not in [0, size - 1]) or
        (y in [0, size - 1] and x not in [0, size - 1] and z not in [0, size - 1]) or
        (z in [0, size - 1] and x not in [0, size - 1] and y not in [0, size - 1])
        )

# g_value : type of an array determined by f_value
def g(x, y, z,f_values):
        if x == 0:
            return sum(2**n * f_values[n+1][y][z] for n in range(v))
        elif x == v+1:
            return sum(2**n * f_values[v-n][y][z] for n in range(v))
        elif y == 0:
            return sum(2**n * f_values[x][n+1][z] for n in range(v))
        elif y == v+1:
            return sum(2**n * f_values[x][v-n][z] for n in range(v))
        elif z == 0:
            return sum(2**n * f_values[x][y][n+1] for n in range(v))
        elif z == v+1:
            return sum(2**n * f_values[x][y][v-n] for n in range(v))

def create_file(task_details):
    repeat, n, output_dir, v, ranges = task_details
    for run in range(1, n+1):            
        f_values = [[[f(x, y, z) for z in range(v+2)] for y in range(v+2)] for x in range(v+2)]
        g_values = [[[g(x, y, z,f_values) for z in range(v+2)] for y in range(v+2)] for x in range(v+2)]
        h_values = [[[0 for _ in range(v+2)] for _ in range(v+2)] for _ in range(v+2)]
        for x in range(v+2):
            for y in range(v+2):
                for z in range(v+2):
                    mean, std = ranges.get(g_values[x][y][z], (0, 0))
                    h_value = int(round(np.random.normal(mean, std), 0))
                    h_values[x][y][z] = h_value
                    
        # f_value output
        if fval == 1 :
            f_output_file = os.path.join(output_dir, f'{f_name}_{repeat}.csv')
            with open(f_output_file, 'a', newline='') as f_file:
                f_writer = csv.writer(f_file)
                if run == 1:
                    f_writer.writerow(['x', 'y', 'z', 'f(x,y,z)'])
                for x in range(1,v+1):
                    for y in range(1,v+1):
                        for z in range(1,v+1):
                            f_writer.writerow([x, y, z, f_values[x][y][z]])
        else :
            None

        # g_value output
        if gval == 1 :
            g_output_file = os.path.join(output_dir, f'{g_name}_{repeat}.csv')
            with open(g_output_file, 'a', newline='') as g_file:
                g_writer = csv.writer(g_file)
                if run == 1:
                    g_writer.writerow(['x', 'y', 'z', 'g(x,y,z)'])
                for x in range(v+2):
                    for y in range(v+2):
                        for z in range(v+2):
                            if single_edge(x, y, z, v+2):
                                    g_writer.writerow([x, y, z, g_values[x][y][z]])
        else:
            None

        # h_value output / h_val : intensity value that are seen on surface
        if hval == 1:
            h_output_file = os.path.join(output_dir, f'{h_name}_{repeat}.csv')
            with open(h_output_file, 'a', newline='') as h_file:
                h_writer = csv.writer(h_file)
                if run == 1:
                    h_writer.writerow(['x', 'y', 'z', 'h(x,y,z)'])
                for x in range(v+2):
                    for y in range(v+2):
                        for z in range(v+2):
                            if single_edge(x, y, z, v+2):
                                h_writer.writerow([x, y, z, h_values[x][y][z]])
        else:
            None

        hg_10side_output_file = os.path.join(output_dir, f'{side_LP_3_name}_{repeat}.csv')
        with open(hg_10side_output_file, 'a', newline='') as hg_10side_file:
            hg_10side_writer = csv.writer(hg_10side_file)
            if run == 1:
                hg_10side_writer.writerow(['x', 'y', 'z', 
                                            'h(x,y,z)f', 'h(x,y,z)b', 
                                            'r1', 'r2', 'r3', 
                                            'l1', 'l2', 'l3', 
                                            'd1', 'd2', 'd3', 
                                            'u1', 'u2', 'u3', 
                                            'g(x,y,z)'])
            for x in range(v+2):
                for y in range(v+2):
                    for z in range(v+2):
                        if single_edge(x, y, z, v+2):
                            if x == 0:
                                hg_10side_writer.writerow([x, y, z, 
                                                        h_values[0][y][z], h_values[v+1][y][z], 
                                                        h_values[1][0][z], h_values[2][0][z], h_values[3][0][z],
                                                        h_values[1][v+1][z], h_values[2][v+1][z], h_values[3][v+1][z],
                                                        h_values[1][y][0], h_values[2][y][0], h_values[3][y][0],
                                                        h_values[1][y][v+1], h_values[2][y][v+1], h_values[3][y][v+1],
                                                        g_values[x][y][z]])

                            elif y == 0:
                                hg_10side_writer.writerow([x, y, z, 
                                                        h_values[x][0][z], h_values[x][v+1][z], 
                                                        h_values[v+1][1][z], h_values[v+1][2][z], h_values[v+1][3][z],
                                                        h_values[0][1][z], h_values[0][2][z], h_values[0][3][z],                                                                
                                                        h_values[x][1][0], h_values[x][2][0], h_values[x][3][0],
                                                        h_values[x][1][v+1], h_values[x][2][v+1], h_values[x][3][v+1],
                                                        g_values[x][y][z]])
                                
                            elif z == 0:
                                hg_10side_writer.writerow([x, y, z,
                                                            h_values[x][y][0], h_values[x][y][v+1],
                                                            h_values[0][y][1], h_values[0][y][2], h_values[0][y][3],
                                                            h_values[v+1][y][1], h_values[v+1][y][2], h_values[v+1][y][3],
                                                            h_values[x][0][1], h_values[x][0][2], h_values[x][0][3],
                                                            h_values[x][v+1][1], h_values[x][v+1][2], h_values[x][v+1][3],
                                                            g_values[x][y][z]])
                                
                            if x == v+1: 
                                hg_10side_writer.writerow([x, y, z, 
                                                        h_values[v+1][y][z], h_values[0][y][z], 
                                                        h_values[3][v+1][z], h_values[2][v+1][z], h_values[1][v+1][z],
                                                        h_values[3][0][z], h_values[2][0][z], h_values[1][0][z],
                                                        h_values[3][y][0], h_values[2][y][0], h_values[1][y][0],
                                                        h_values[3][y][v+1], h_values[2][y][v+1], h_values[1][y][v+1],
                                                        g_values[x][y][z]])

                            elif y == v+1:
                                hg_10side_writer.writerow([x, y, z, 
                                                        h_values[x][0][z], h_values[x][v+1][z], 
                                                        h_values[0][3][z], h_values[0][2][z], h_values[0][1][z], 
                                                        h_values[v+1][3][z], h_values[v+1][2][z], h_values[v+1][1][z],
                                                        h_values[x][3][0], h_values[x][2][0], h_values[x][1][0],
                                                        h_values[x][3][v+1], h_values[x][2][v+1], h_values[x][1][v+1],
                                                        g_values[x][y][z]])
                                
                            elif z == v+1:
                                hg_10side_writer.writerow([x, y, z,
                                                            h_values[x][y][0], h_values[x][y][v+1],
                                                            h_values[v+1][y][3], h_values[v+1][y][2], h_values[v+1][y][1],
                                                            h_values[0][y][3], h_values[0][y][2], h_values[0][y][1],
                                                            h_values[x][0][3], h_values[x][0][2], h_values[x][0][1],
                                                            h_values[x][v+1][3], h_values[x][v+1][2], h_values[x][v+1][1],
                                                            g_values[x][y][z]])
                                
                            else :
                                None

        print(f"\r진행도: {run}/{n} ({(run/n)*100:.2f}%)", end='')
    print(f'반복 {repeat}: {repeat}번째 파일 생성완료')

# multithreading
def append_to_files_parallel(n, m, output_dir, v, ranges):
    task_details = [(repeat, n, output_dir, v, ranges) for repeat in range(1, m+1)]
    
    with ProcessPoolExecutor(max_workers=5) as executor:
        results = executor.map(create_file, task_details)
        for result in results:
            print(result)

if __name__ == "__main__":
    start_time = time.time()

    append_to_files_parallel(n, m, output_dir, v, ranges)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
