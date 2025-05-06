#  Crime Classification Using Machine Learning

This project emphasizes the creation of a crime classification system using machine learning models such as **Random Forest**, **SVM**, and **Logistic Regression** to categorize crimes as **violent** or **non-violent** based on data from Indian crime records.

The goal is to preprocess and analyze crime datasets, apply and compare various machine learning models, and develop an **interactive Streamlit-based web application** for real-time crime prediction.

---

##  Project Objective

To build a reliable, data-driven solution that classifies crimes effectively to support law enforcement agencies in **crime prevention** and **decision-making**.

---

##  Key Features

- Real-time prediction of crime type (Violent / Non-Violent)  
- Web application with interactive input and visualization  
- Model performance comparison and insights for practical deployment  
- Data preprocessing techniques for handling real-world data

---

##  Technologies Used

- Python
- Jupyter Notebook
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- Matplotlib, Seaborn

---

##  Project Pipeline

### 1. **Data Collection & Preprocessing**
- Sourced from verified **police databases**
- Cleaned data to remove redundant and duplicate features like timestamps and report numbers  
- Handled missing values  
- Encoded categorical variables  
- Normalized numerical attributes

### 2. **Handling Imbalanced Data**
- Applied **undersampling techniques** to mitigate model bias toward frequently occurring crime types

### 3. **Model Training & Comparison**
- Trained and tested multiple ML models:  
  - Logistic Regression  
  - Support Vector Machines (SVM)  
  - Random Forest (Best performer)

### 4. **Model Evaluation**
- Evaluated models based on:  
  - **Accuracy**  
  - **Precision**  
  - **Recall**  
  - **F1-Score**  

### 5. **Interactive Web App**
- Built using **Streamlit**
- Allows users to input crime-related features and receive predictions instantly  
- Visualizes prediction results for user-friendly interpretation

---

##  Visual Insights

![Image1](https://github.com/user-attachments/assets/83805140-5ee3-4cf0-ab2f-eb9f30588ff6)  
![Image2](https://github.com/user-attachments/assets/60070cc0-6e12-4fb8-b605-c5a8252072b6)  
![Image3](https://github.com/user-attachments/assets/c7c1b0fd-b127-45b4-904d-64a35e444c6e)  
![Image4](https://github.com/user-attachments/assets/44cff63c-b63e-430a-a1c2-48221de66499)  
![Image5](https://github.com/user-attachments/assets/a84992ef-38aa-4768-8190-0585eadec0f3)  
![Image6](https://github.com/user-attachments/assets/41f6280b-6f61-4737-98be-a43e194c3b62)  
![Image7](https://github.com/user-attachments/assets/962c6a8d-be06-458e-b48a-579ae1184d3e)  
![Image8](https://github.com/user-attachments/assets/487b6639-6e90-4f01-b854-83a1d8fff24f)

---

##  Conclusion

After evaluating various machine learning models, **Random Forest** proved to be the most accurate and robust, especially when handling imbalanced datasets and complex crime patterns.

The model is deployed in an **interactive Streamlit app**, allowing users to input features and receive real-time predictions. Future work could incorporate:  
- More extensive and diverse datasets  
- Deep learning models  
- Additional crime-related attributes  

These enhancements can significantly boost the model's predictive power and help authorities implement **more effective crime-prevention strategies**.

---

