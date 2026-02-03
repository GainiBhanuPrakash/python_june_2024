import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# --- Page Configuration ---
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# --- 1. Load Data ---
@st.cache_data
def load_data():
    # Load the dataset
    try:
        df = pd.read_csv('diabetes.csv')
        return df
    except FileNotFoundError:
        st.error("File 'diabetes.csv' not found. Please ensure it is in the same directory.")
        return None

df = load_data()

if df is not None:
    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    options = ["Exploratory Data Analysis (EDA)", "Data Preprocessing", "Data Visualization", "Machine Learning Model"]
    choice = st.sidebar.radio("Go to Section:", options)

    # --- Section 1: Exploratory Data Analysis (EDA) ---
    if choice == "Exploratory Data Analysis (EDA)":
        st.title(" Diabetes Prediction Application")
        st.title("Exploratory Data Analysis")
        st.write("An overview of the raw data and its structure.")

        # Show raw data
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        # Show shape
        st.write(f"**Dataset Dimensions:** {df.shape[0]} rows and {df.shape[1]} columns.")

        # Show Column Types
        st.subheader("Column Data Types")
        dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type']).astype(str)
        st.table(dtype_df)

        # Descriptive Statistics
        st.subheader("Descriptive Statistics")
        st.write(df.describe())

        # Value Counts for Target
        st.subheader("Target Variable Distribution (Diabetes)")
        st.write(df['Diabetes'].value_counts())

    # --- Section 2: Data Preprocessing ---
    elif choice == "Data Preprocessing":
        st.title(" Data Preprocessing")
        st.write("Preparing the data for the machine learning model.")

        # Check for missing values
        st.subheader("1. Missing Values Check")
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0])
        if missing_values.sum() == 0:
            st.success("No standard null values found in the dataset.")

        # Encoding Target Variable
        st.subheader("2. Label Encoding")
        st.write("The 'Diabetes' column contains categorical text ('pos', 'neg'). We convert this to numbers (1, 0).")
        
        le = LabelEncoder()
        df['Diabetes_Encoded'] = le.fit_transform(df['Diabetes'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original Target Column:**")
            st.write(df['Diabetes'].head())
        with col2:
            st.write("**Encoded Target Column:**")
            st.write(df['Diabetes_Encoded'].head())
            
        st.info(f"Encoding Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        # Store processed df in session state for other sections if needed, 
        # but for simplicity, we re-process locally or use the main df variable carefully.

    # --- Section 3: Data Visualization ---
    elif choice == "Data Visualization":
        st.title(" Data Visualization")
        st.write("Visualizing relationships and distributions.")

        # Encode for plotting purposes
        le = LabelEncoder()
        df['Diabetes_Encoded'] = le.fit_transform(df['Diabetes'])

        # 1. Correlation Heatmap
        st.subheader("Correlation Heatmap")
        st.write("Shows how different features correlate with the onset of Diabetes.")
        
        # Select only numeric columns for correlation
        numeric_df = df.select_dtypes(include=['float64', 'int64', 'int32'])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

        # 2. Glucose vs Diabetes
        st.subheader("Glucose Levels vs Diabetes Outcome")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='Diabetes', y='Glucose', data=df, palette='Set2', ax=ax2)
        st.pyplot(fig2)

        # 3. Age Distribution
        st.subheader("Age Distribution by Diabetes Class")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.histplot(data=df, x='Age', hue='Diabetes', kde=True, palette='seismic', ax=ax3)
        st.pyplot(fig3)

    # --- Section 4: Machine Learning Model ---
    elif choice == "Machine Learning Model":
        st.title(" Machine Learning Model")
        st.write("Supervised Learning Classification using **Random Forest**.")

        # Preprocessing (Encoding)
        le = LabelEncoder()
        df['Diabetes_Encoded'] = le.fit_transform(df['Diabetes'])

        # Feature Selection
        X = df.drop(['Diabetes', 'Diabetes_Encoded'], axis=1)
        y = df['Diabetes_Encoded']

        # Train/Test Split
        test_size = st.sidebar.slider("Test Set Size (%)", 10, 100) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        st.subheader("Model Training")
        st.write(f"Training on {X_train.shape[0]} samples, Testing on {X_test.shape[0]} samples.")

        # Train Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Metrics Display
        st.metric(label="Model Accuracy", value=f"{accuracy:.2%}")
        
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred, target_names=le.classes_))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax_cm)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig_cm)

        # --- Interactive Prediction ---
        st.markdown("---")
        st.subheader("try it yourself: Predict Diabetes")
        
        # Input form
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glucose", 0, 200, 110)
            bp = st.number_input("Blood Pressure", 0, 150, 70)
            skin = st.number_input("Skin Thickness", 0, 100, 20)
        with col2:
            insulin = st.number_input("Insulin", 0, 900, 79)
            bmi = st.number_input("BMI", 0.0, 70.0, 30.0)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            age = st.number_input("Age", 0, 120, 30)

        if st.button("Predict"):
            input_data = [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]]
            prediction = model.predict(input_data)
            result_text = le.inverse_transform(prediction)[0]
            
            if result_text == 'pos':
                st.error(f"The model predicts: **Positive (Diabetes)**")
            else:
                st.success(f"The model predicts: **Negative (No Diabetes)**")

else:
    st.warning("Awaiting Data Upload.")