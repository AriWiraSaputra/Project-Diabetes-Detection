import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_data():
    diabetes_df = pd.read_csv("diabetes.csv")
    return diabetes_df


# Preprocess the dataset
def preprocess_data(df):
    x = df.drop(columns='Outcome', axis=1)
    y = df['Outcome']
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x, y


# Train the SVM classifier
def train_model(x_train, y_train):
    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)
    return clf


# Make predictions
def predict(model, data):
    prediction = model.predict(data)
    return prediction


# Main app function
def app(diabetes_df):
    
    # Load the dataset
    df = load_data()

    # Preprocess the dataset
    x, y = preprocess_data(df)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Train the model
    model = train_model(x_train, y_train)

    st.title("Diabetes Prediction App")
    st.markdown("Diabetes Checker:")

    # Create sidebar input fields
    st.sidebar.title("Input Fields")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        pregnancies = st.slider("Number of times pregnant", 0, 20, 1)
        glucose = st.slider("Plasma glucose concentration", 0, 200, 100)
        bp = st.slider("Diastolic blood pressure", 0, 120, 80)
        skinthickness = st.slider("Triceps skin fold thickness", 0, 100, 20)
    with col2:
        insulin = st.slider("Insulin concentration", 0, 500, 100)
        bmi = st.slider("Body mass index", 0, 60, 25)
        dpf = st.slider("Diabetes pedigree function", 0.0, 2.0, 0.5, 0.1)
        age = st.slider("Age", 20, 100, 30)

    # Create a DataFrame from input data
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [bp],
        "SkinThickness": [skinthickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [dpf],
        "Age": [age]
    })

    # Preprocess the dataset
    X, y = preprocess_data(diabetes_df)

    # Train the model
    model = train_model(X, y)
    
    # Make the prediction
    prediction = predict(model, input_data)
    
    # Show the prediction
    st.markdown("### Prediction Result")
    if prediction == 0:
        st.markdown("# You do not have diabetes")
    else:
        st.markdown("# You have diabetes")

    # Show statistical description of the dataset
    st.subheader("Statistical Description")
    st.write(diabetes_df.describe())

    # PATIENT DATA
    diabetes_df = diabetes_df.append(input_data, ignore_index=True)
    user_data = input_data
    st.subheader('Patient Data')
    st.write(user_data)

    # VISUALISATIONS
    st.title('Visualised Patient Report')

    # MODEL
    rf  = RandomForestClassifier()
    rf.fit(x_train, y_train)
    user_result = rf.predict(user_data)

    # COLOR FUNCTION
    if user_result[0]==0:
        color = 'blue'
    else:
        color = 'red'

    # Age vs Pregnancies
    st.header('Pregnancy count Graph (Others vs Yours)')
    fig_preg = plt.figure()
    ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = df, hue = 'Outcome', palette = 'twilight_shifted')
    ax2 = sns.scatterplot(x = user_data['Age'], y = user_data['Pregnancies'], s = 150, color = color)
    plt.xticks(np.arange(10,100,5))
    plt.yticks(np.arange(0,20,2))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_preg)

    # Age vs Glucose
    st.header('Glucose Value Graph (Others vs Yours)')
    fig_glucose = plt.figure()
    ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='PuOr')
    ax4 = sns.scatterplot(x = user_data['Age'], y = user_data['Glucose'], s = 150, color = color)
    plt.xticks(np.arange(10,100,5))
    plt.yticks(np.arange(0,220,10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_glucose)

    # Age vs Bp
    st.header('Blood Pressure Value Graph (Others vs Yours)')
    fig_bp = plt.figure()
    ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = df, hue = 'Outcome', palette='seismic')
    ax6 = sns.scatterplot(x = user_data['Age'], y = user_data['BloodPressure'], s = 150, color = color)
    plt.xticks(np.arange(10,100,5))
    plt.yticks(np.arange(0,130,10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_bp)

    # Age vs St
    st.header('Skin Thickness Value Graph (Others vs Yours)')
    fig_st = plt.figure()
    ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = df, hue = 'Outcome', palette='tab20b')
    ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['SkinThickness'], s = 150, color = color)
    plt.xticks(np.arange(10,100,5))
    plt.yticks(np.arange(0,110,10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_st)

    # Age vs Insulin
    st.header('Insulin Value Graph (Others vs Yours)')
    fig_i = plt.figure()
    ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rocket')
    ax10 = sns.scatterplot(x = user_data['Age'], y = user_data['Insulin'], s = 150, color = color)
    plt.xticks(np.arange(10,100,5))
    plt.yticks(np.arange(0,900,50))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_i)

    # Age vs BMI
    st.header('BMI Value Graph (Others vs Yours)')
    fig_bmi = plt.figure()
    ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = df, hue = 'Outcome', palette='hsv')
    ax12 = sns.scatterplot(x = user_data['Age'], y = user_data['BMI'], s = 150, color = color)
    plt.xticks(np.arange(10,100,5))
    plt.yticks(np.arange(0,70,5))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_bmi)

    # Age vs Dpf
    st.header('DPF Value Graph (Others vs Yours)')
    fig_dpf = plt.figure()
    ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = df, hue = 'Outcome', palette='YlOrBr')
    ax14 = sns.scatterplot(x = user_data['Age'], y = user_data['DiabetesPedigreeFunction'], s = 150, color = color)
    plt.xticks(np.arange(10,100,5))
    plt.yticks(np.arange(0,3,0.2))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_dpf)

# Run the app
if __name__ == '__main__':
    diabetes_df = load_data()
    app(diabetes_df)
