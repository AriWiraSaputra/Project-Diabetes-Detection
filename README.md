# Diabetes Prediction App

![App Preview](app_preview.png)

## Table of Contents
- [Introduction](#introduction)
- [Usage](#usage)
- [Data Source](#data-source)
- [Machine Learning Model](#machine-learning-model)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [License](#license)
- [Contributors](#contributors)

## Introduction
The Diabetes Prediction App is a web application that uses machine learning to predict the likelihood of an individual having diabetes. It provides a user-friendly interface where users can input relevant medical information and obtain a prediction result.

## Usage
To use the app, follow these steps:
1. Clone the repository: ```git clone https://github.com/your-username/diabetes-prediction.git```
2. Navigate to the project directory: ```cd diabetes-prediction```
3. Install the required dependencies: ```pip install -r requirements.txt```
4. Start the app by running the following command: ```streamlit run app.py```
5. Once the app is running, you will see a form where you can enter the patient's information, such as the number of pregnancies, glucose concentration, blood pressure, etc.
6. Click on the "Predict" button to obtain the prediction result, which will indicate whether the patient is likely to have diabetes or not.

## Data Source
The dataset used for training the machine learning model is the Pima Indians Diabetes Database, which is publicly available from the UCI Machine Learning Repository. The dataset contains various features related to diabetes risk factors, such as age, BMI, and blood pressure.

## Machine Learning Model
The machine learning model used for diabetes prediction is Support Vector Machine (SVM). SVM is a supervised learning algorithm that analyzes data and builds a classification model to make predictions. The model has been trained using the scikit-learn library in Python.

## Dependencies
The following dependencies are required to run the app:
- Python 3.7
- Streamlit 0.80.0
- Pandas 1.2.4
- NumPy 1.20.2
- Scikit-learn 0.24.2
- Matplotlib 3.4.2
- Seaborn 0.11.1

You can install them using the command mentioned in the installation instructions.

## License
This project is licensed under the [MIT License](LICENSE).

## Contributors
- Ari Wira Saputra (ariewira072@gmail.com)

If you would like to contribute to this project, please fork the repository and submit a pull request with your proposed changes.
