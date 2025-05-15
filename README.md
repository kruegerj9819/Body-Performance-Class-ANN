# Body Performance Classification using an Artificial Neural Network
This project uses an **Artificial Neural Network (ANN)** to classify body performance into four distinct classes: **A**, **B**, **C**, and **D** (A being the most fit physically). The model is trained to predict the body performance class of individuals based on various physical attributes, and it aims to provide accurate predictions using deep learning techniques.

## Dataset
The dataset used for this project contains various physical attributes and performance measurements of individuals. The goal is to classify the body performance class based on these attributes. The dataset is split into two parts:
- **Training set**: `Train_BodyPerformance.csv`
- **Test set**: `Test_BodyPerformance.csv`
The dataset can be found on [Kaggle](https://www.kaggle.com/datasets/kukuroo3/body-performance-data).

### Class Labels:
- **A**: Excellent performance
- **B**: Good performance
- **C**: Average performance
- **D**: Poor performance

## Network Architecture
The **Artificial Neural Network** has 2 hidden layers. 
- The first hidden layer's parameters are as follows: 12 (input) &rarr; 32 &rarr; 16 (output). The activation function used is **Sigmoid**.
- The second hidden layer's parameters are as follows: 16 (input) &rarr; 32 &rarr; 4 (output). The activation function used is **ReLU**.

## Installation
To run this project, clone this repository and install the required dependencies. You can use the following command to install the necessary libraries:
```
pip install -r requirements.txt
```

### Requirements
- pandas
- numpy
- matplotlib
- torch
- scikit-learn
- tqdm

## How to Run
1. Clone this repository or download the necessary files.
2. Ensure that the datasets `Train_BodyPerformance.csv` and `Test_BodyPerformance.csv` are available in the same directory as the Python script.
3. Run the Python script `BodyHealth.py` to train and evaluate the model.

## Training the Network:
The model is set up to train for 100 epochs with a learning rate of 0.001. If you would like to save the model after training, uncomment the `save_file` parameter in the network declaration. During training:
- **Confusion Matrix**: The confusion matrix will be displayed on the validation set after the final epoch.
- **Accuracy**: The accuracy of the model on both the training and validation sets will be printed at the end of each epoch.

## Results
The script will output:
- The running loss for each epoch.
- The accuracy on the training and validation sets after each epoch.
- The confusion matrix for the final epoch to evaluate the model's performance.

### Example Output
```
Running loss for epoch 1 of 100: 0.6934
Accuracy on train set: 0.52
Accuracy on validation set: 0.51

Running loss for epoch 100 of 100: 0.2095
Accuracy on train set: 0.91
Accuracy on validation set: 0.87
```

### Confusion Matrix
A confusion matrix will be displayed at the end of the final epoch to assess the classification performance on the validation set.

## Interpretation
### Confusion Matrix
![body-performance-confusion-matrix](https://github.com/user-attachments/assets/5cec457d-8cf0-4e02-a79c-22f8314a9a45)

After training the model for 100 epochs, it seems to plateau at around 75% accuracy on the test set. It does extremely well at classifying the excellent and poor classes of body performance. It does struggle in the categories that are in between the two extremes, however. Increasing the number of layers and parameters in the network does significantly increase overfitting, which is not ideal. The dataset may be limiting the network from being able to distinguish the two middle classes, or they may be extremely difficult to distinguish while straying away from overfitting entirely. At the end of the day, 75% is still a very good accuracy for the network to achieve.
