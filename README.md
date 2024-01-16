### Pose classification

The purpose of My assignment was to classify the different poses based on the 33 key points of the body.


### Project Structure
# Data Collection:
The dataset is organized into 17 classes, each representing a specific pose category. The data is stored as numpy arrays in folders corresponding to each class.
Data Loading and Preprocessing:
The provided Python script loads the data, flattens the 3D arrays to 2D for compatibility with the chosen linear Support Vector Machine (SVM) model, and encodes class labels numerically using scikit-learn's LabelEncoder.

# Model Training:
The SVM model is selected for its simplicity and effectiveness in handling high-dimensional data. The training script employs scikit-learn's SVM implementation and reports key performance metrics.

# Model Evaluation:
The model is evaluated on a test dataset, and metrics such as accuracy and a detailed classification report are generated. These metrics provide insights into the model's performance across different pose categories.

# Model Deployment:
The trained SVM model, along with the label encoder, can be saved using the joblib library. This enables easy deployment for making predictions on new datasets.

# Usage:
# Prerequisites:
Ensure that the required libraries (e.g., NumPy, scikit-learn, joblib) are installed in your Python environment.
Loading the Trained Model:
Clone the Repository:git clone https://github.com/Kchilala/ML_assignment.git
Navigate to the Project file 
Move into the project file 
Load the Model:
In your Python script or Jupyter Notebook, use the following code to load the trained SVM model and the associated label encoder.
from joblib import load

# Load the trained SVM model
svm_model = load('path/to/body_scatch_svm_model.joblib')

# Load the label encoder
label_encoder = load('path/to/label_encoder.joblib')
Making Predictions on New Data:
Load New Data:
Prepare your new dataset, ensuring that it follows the same format as the data used during training. For each pose instance, you should have a numpy array containing the coordinates of 33 key points.
python
Copy code
# Example: Load new data (replace 'new_data.npy' with your file)
new_data = np.load('path/to/new_data.npy')
Preprocess the New Data:
Flatten the 3D arrays to 2D for compatibility with the SVM model and encode class labels if necessary.
python
Copy code
# Flatten the 3D arrays
new_data_flatten = new_data.reshape(new_data.shape[0], -1)

# Encode class labels if needed
# (skip if the new data already has numerical labels)
# y_new_encoded = label_encoder.transform(y_new)
Make Predictions:
Use the loaded SVM model to make predictions on the new data.
python
Copy code
# Make predictions
predictions = svm_model.predict(new_data_flatten)

# Decode numerical predictions back to original class labels
decoded_predictions = label_encoder.inverse_transform(predictions)
Interpret the Predictions:
The decoded_predictions variable now contains the predicted pose categories for each instance in your new data.
python
Copy code
# Print the predictions
print(decoded_predictions)

## Results:
The SVM model achieved excellent performance on the test dataset, with an accuracy of 100%. The classification report provides detailed metrics for each pose category, demonstrating the model's capability to generalize well.

## Future Improvements:
While the current model exhibits strong performance, there are opportunities for enhancement. Consider exploring more complex models, such as deep neural networks, to capture intricate relationships in pose data.



## Help

Any advise for common problems or issues.


## Authors

Keci chilala 

ex. Keci Chilala  
ex. [(https://www.linkedin.com/in/keci-chilala/)]

