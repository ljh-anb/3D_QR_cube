# 3D_QR_cube

## how to use the codes

1. monte carlo 3D QR simulation
- Decide how many cubes are going to be made.
- "n" is the number of cubes per a file
- "m" is the number of files
- n*m = total amount of simulated cubes

- "range" file is needed for monte carlo method
- They are mean values and standard deviations from reference intensity range
- Sample file is provided in supporting information

- Simulated cube files are saved in "Simulated_cube" folder


2. Machine learning model training
- Each ML models are trained by the simulated training data in "Simulated_cube" folder
- CNN model training specifically needs preprocessed training data by "2-5.CNN input data preprocessing.py"
- "2-5.CNN input data preprocessing.py" reforms the input data suitable for CNN training

- After reforming by "2-5.CNN input data preprocessing.py", test dataset needs to be seperated by its name both csv files and npy files.
- We produces n=10,000, m=11 simulated data, and the last data (which is numbered "11") was chosen for test data so that changed numbering to "test"

- Results (trained model, confusion matrix, accuracy/loss graph, missclassified data) are saved in "ML_learning_output" folder


3. Data processing and CNN visualize
- This code extracts the array luminescence intensities from the captured images of unknown cube.
- Also, this reconstructs and visualizes the structure of the cube.
- You can turn on/off the functions

- For array prediction, actual cube intensity data have to be extracted by "intensity_reading" and "intensity_reforming" block
- Intensity data from each actual cubes have to be extracted and saved with numbering (see example files of "rd 3 side_LP_" in "Sample/3D QR cube/Simulated_cube/)
- "rd 3 side_LP" files are extracted by this code and saved in "Intensity_extraction" folder, so they have to be copied from "Intensity_extraction" to "Simulated_Cube" for further use.

- Visualization through CNN model prediction and voxelization is possible.
- If you put six images of unknown cube, this code process the images to predict and reconstruct the structure of the cube, and visualize it.

- Prediction of an unknown 3D QR cube is done by this code.
- Next steps are for examination of the model's accuracy and voxelization ROC.


4. Prediction via Machine learning model
- Prediction accuracy of each models for actual cubes is measured through this code.
- Data for voxelization is made by this code.

- Putting numbers of actual cubes in "file_num". You don't need this when you predicts simulated cubes.
- Dataset used for simulated cubes prediction is the test dataset used in model training.

- You can get confusion matrix of actual cubes prediction here. One for simulated cubes is already done when training model.
- The results are saved in "ML_prediction_output" folder.

- Data acquired here are further used in voxelization.


5. Voxelization
- Array voxelization is done by this code.
- The results consist "probability", which is glow score in the paper, and ROC AUC is measured by this.
