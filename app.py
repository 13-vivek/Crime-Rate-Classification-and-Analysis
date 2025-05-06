import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Streamlit UI Setup
st.title("🔍 Crime Classification in India")
st.markdown("<hr style='border: 2px solid #ccc;'>", unsafe_allow_html=True)
st.write("This app allows you to classify crimes based on input features.")

# Upload Dataset
uploaded_file = st.file_uploader("📂 Upload Crime Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Drop unnecessary columns
    df.drop(columns=['Report Number', 'Crime Category', 'Crime Domain', 'Date Reported', 
                     'Date Case Closed', "Date of Occurrence", "Time of Occurrence"], inplace=True, errors='ignore')

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    # Define features and target
    X = df.drop(columns=["Crime_Label"])
    y = df["Crime_Label"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    st.sidebar.subheader("City codes")
    model_choice = st.sidebar.selectbox("Choose a model", 
                                        ["Logistic Regression", "SVM (Linear)", "SVM (RBF)", "SVM (Polynomial)", "Random Forest"])


    # Train and Evaluate Model
    model = None
    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_choice == "Logistic Regression":
        model = LogisticRegression()
    elif model_choice == "SVM (Linear)":
        model = SVC(kernel="linear")
    elif model_choice == "SVM (RBF)":
        model = SVC(kernel="rbf")
    elif model_choice == "SVM (Polynomial)":
        model = SVC(kernel="poly", degree=3)

    # Train the selected model
    if model is not None:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.markdown("<hr style='border: 2px solid #ccc;'>", unsafe_allow_html=True)
        st.subheader(f" Model Performance for {model_choice}")

        # Display accuracy in a centered box
        # Display accuracy in a centered box
        st.markdown(
            f"""
            <div style="text-align: center; padding: 15px; border-radius: 10px; 
                        background-color: #f8f9fa; font-size: 20px; font-weight: bold;">
                 Accuracy: {accuracy:.4f}
            </div>
            """, unsafe_allow_html=True
        )



        report_dict = classification_report(y_test, y_pred, output_dict=True)

        # Convert it to a DataFrame
        report_df = pd.DataFrame(report_dict).transpose()

        # Round values for better readability
        report_df = report_df.round(2)

        # Rename columns for better display
        report_df.columns = ["Precision", "Recall", "F1-Score", "Support"]

        # Convert DataFrame to a properly formatted table
        st.markdown("### 📝 Classification Report")
        st.dataframe(report_df.style.set_properties(**{'text-align': 'center'}))  # Ensures alignment




        # Confusion Matrix
        st.markdown("### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=["Non-Violent", "Violent"], yticklabels=["Non-Violent", "Violent"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        st.pyplot(fig)

    # User Input Prediction
    st.markdown("<hr style='border: 2px solid #ccc;'>", unsafe_allow_html=True)
    st.subheader("🔍 Predict Crime Type")

    col1, col2 = st.columns(2)  # Arrange input fields in two columns
    input_features = []

    for idx, col in enumerate(X.columns):
        with col1 if idx % 2 == 0 else col2:
            value = st.number_input(f"{col}", value=int(df[col].mean()))
            input_features.append(value)

    if st.button("Predict"):
        input_array = np.array(input_features).reshape(1, -1)
        input_array = scaler.transform(input_array)  # Standardize input

        if model is not None:
            prediction = model.predict(input_array)
            # st.success(f"### 🔍 Predicted Crime Label: {prediction[0]}")
            if {prediction[0]} == 0:
                st.success(f"### 🔍 The Crime is Non-Violent")
            else:
                st.success(f"### 🔍 The Crime is Violent")

        else:
            st.error("❌ Model is not trained. Please select and train a model first.")

st.markdown("<hr style='border: 2px solid #ccc;'>", unsafe_allow_html=True)
