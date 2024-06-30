import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
st.set_page_config(
    page_title="Cancer Prediction",
    page_icon=":lungs:",
    layout="wide"
)
# Load the dataset
dataset = pd.read_csv(r"C:\Users\Parth\OneDrive\Desktop\Project\cancer patient data sets (2).csv")

dataset = dataset.drop('Patient Id', axis=1)
dataset = dataset.drop('index', axis=1)

# Preprocess the data
# dataset['Gender'] = dataset['Gender'].map({1: 'Male', 0: 'Female'})

label_encoder = LabelEncoder()
dataset['Level'] = label_encoder.fit_transform(dataset['Level'])

# Split the dataset
X = pd.get_dummies(dataset.drop(columns=['Level']), columns=['Gender'])
y = dataset['Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train the model
classifier = RandomForestClassifier(n_estimators=10, criterion="entropy")
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


# Create a sidebar for navigation
st.sidebar.header("Navigation")
nav_type = st.sidebar.selectbox("Select Page", ["Home", "Visualization",  "Make Prediction", "BMI Calculator","Feedback Submission"])

if nav_type == "Home":
    st.title("Welcome to Cancer Patient Data Analysis and Prediction")
    st.write("This application is designed to analyze, visualize and predict cancer.")
    st.write("You can select the type of visualization you want to see from the sidebar.")
    st.write("You can also evaluate the performance of the machine learning model used for prediction.")
    st.image(r"C:/Users/Parth/OneDrive/Desktop/Project/LungCancer_1400b.png", width=500)

    st.markdown("""
    ## What is Cancer?
    Cancer is a disease in which some of the body’s cells grow uncontrollably and spread to other parts of the body. Cancer can start almost anywhere in the human body, which is made up of trillions of cells.

    ## Symptoms of Cancer
    Cancer symptoms are quite varied and depend on where the cancer is located, where it has spread, and how big the tumor is. Some common symptoms of cancer include:
    - Fatigue
    - Lump or area of thickening that can be felt under the skin
    - Weight changes, including unintended loss or gain
    - Skin changes, such as yellowing, darkening or redness of the skin, sores that won't heal, or changes to existing moles
    - Changes in bowel or bladder habits
    - Persistent cough or trouble breathing
    - Difficulty swallowing
    - Hoarseness
    - Persistent indigestion or discomfort after eating
    - Persistent, unexplained muscle or joint pain
    - Persistent, unexplained fevers or night sweats
    - Unexplained bleeding or bruising

    ## Risk Factors
    While doctors have an idea of what can increase the risk of cancer, most cancers occur in people who don’t have any known risk factors. Factors known to increase your risk of cancer include:
    - Age
    - Habits
    - Family history
    - Health conditions
    - Environment

    ## Prevention
    There's no certain way to prevent cancer. But doctors have identified several ways of reducing your cancer risk, such as:
    - Stop smoking
    - Avoid excessive sun exposure
    - Eat a healthy diet
    - Exercise most days of the week
    - Maintain a healthy weight
    - Get vaccinated
    - Avoid risky behaviors
    - Get regular medical care

    ## Treatment
    Cancer treatment varies greatly depending on the type and stage of cancer. Common treatments include:
    - Surgery
    - Chemotherapy
    - Radiation therapy
    - Immunotherapy
    - Targeted drug therapy
    - Hormone therapy
    - Bone marrow transplant

    ## Conclusion
    Early detection and treatment can significantly improve the chances of successful treatment and recovery from cancer. Regular screenings and awareness of symptoms are crucial.
    """)


elif nav_type == "Visualization":
    # Create a sidebar for selecting the visualization
    st.sidebar.header("Select Visualization")
    vis_type = st.sidebar.selectbox("Select Visualization Type", ["Histogram", "Pie Chart", "Box Plot", "Scatter Plot", "Histogram (Multiple Columns)"])

    # Create a sidebar for selecting the column for visualization
    st.sidebar.header("Select Column")
    column = st.sidebar.selectbox("Select Column", dataset.columns)

    # Data visualization
    if vis_type == "Histogram":
        if dataset[column].dtype.kind in 'bifc':  # Check if the column is numeric
            plt.hist(dataset[column], bins=25)
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.title(f'{column} Distribution')
            st.pyplot()
        else:
            st.write("The selected column is not numeric. Please select a different column or visualization type.")
    elif vis_type == "Pie Chart":
        plt.pie(dataset[column].value_counts(), labels=dataset[column].unique(), autopct='%1.1f%%')
        plt.title(f'{column} Distribution')
        st.pyplot()

    elif vis_type == "Box Plot":
        plt.boxplot(dataset[column])
        plt.title(f'{column} Distribution')
        st.pyplot()
    elif vis_type == "Scatter Plot":
        if 'Age' in dataset.columns:
            plt.scatter(dataset['Age'], dataset[column])
            plt.xlabel('Age')
            plt.ylabel(column)
            plt.title(f'Age vs. {column}')
            st.pyplot()
        else:
            st.write("The 'Age' column is not found. Please select a different column or visualization type.")

    elif vis_type == "Histogram (Multiple Columns)":
        if 'Age' in dataset.columns and 'Air Pollution' in dataset.columns:
            dataset[[column, 'Age', 'Air Pollution']].hist(bins=20, figsize=(10, 6))
            st.pyplot()
        else:
            st.write("The 'Age' or 'Air Pollution' column is not found. Please select a different column or visualization type.")



elif nav_type == "Make Prediction":
    st.header("Make Prediction")

    # Create input fields for all features
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    air_pollution = st.number_input("Air Pollution", min_value=0, max_value=10, value=5)
    alcohol_use = st.number_input("Alcohol Use", min_value=0, max_value=10, value=5)
    dust_allergy = st.number_input("Dust Allergy", min_value=0, max_value=10, value=5)
    occupational_hazards = st.number_input("Occupational Hazards", min_value=0, max_value=10, value=5)
    genetic_risk = st.number_input("Genetic Risk", min_value=0, max_value=10, value=5)
    chronic_lung_disease = st.number_input("Chronic Lung Disease", min_value=0, max_value=10, value=5)
    balanced_diet = st.number_input("Balanced Diet", min_value=0, max_value=10, value=5)
    obesity = st.number_input("Obesity", min_value=0, max_value=10, value=5)
    smoking = st.number_input("Smoking", min_value=0, max_value=10, value=5)
    passive_smoker = st.number_input("Passive Smoker", min_value=0, max_value=10, value=5)
    chest_pain = st.number_input("Chest Pain", min_value=0, max_value=10, value=5)
    coughing_of_blood = st.number_input("Coughing of Blood", min_value=0, max_value=10, value=5)
    fatigue = st.number_input("Fatigue", min_value=0, max_value=10, value=5)
    weight_loss = st.number_input("Weight Loss", min_value=0, max_value=10, value=5)
    shortness_of_breath = st.number_input("Shortness of Breath", min_value=0, max_value=10, value=5)
    wheezing = st.number_input("Wheezing", min_value=0, max_value=10, value=5)
    swallowing_difficulty = st.number_input("Swallowing Difficulty", min_value=0, max_value=10, value=5)
    clubbing_of_finger_nails = st.number_input("Clubbing of Finger Nails", min_value=0, max_value=10, value=5)
    frequent_cold = st.number_input("Frequent Cold", min_value=0, max_value=10, value=5)
    dry_cough = st.number_input("Dry Cough", min_value=0, max_value=10, value=5)
    snoring = st.number_input("Snoring", min_value=0, max_value=10, value=5)

    # Create a dictionary for the input values
    input_data = {
        'Age': age,
        'Gender': gender,
        'Air Pollution': air_pollution,
        'Alcohol use': alcohol_use,
        'Dust Allergy': dust_allergy,
        'Occupational Hazards': occupational_hazards,
        'Genetic Risk': genetic_risk,
        'chronic Lung Disease': chronic_lung_disease,
        'Balanced Diet': balanced_diet,
        'Obesity': obesity,
        'Smoking': smoking,
        'Passive Smoker': passive_smoker,
        'Chest Pain': chest_pain,
        'Coughing of Blood': coughing_of_blood,
        'Fatigue': fatigue,
        'Weight Loss': weight_loss,
        'Shortness of Breath': shortness_of_breath,
        'Wheezing': wheezing,
        'Swallowing Difficulty': swallowing_difficulty,
        'Clubbing of Finger Nails': clubbing_of_finger_nails,
        'Frequent Cold': frequent_cold,
        'Dry Cough': dry_cough,
        'Snoring': snoring
    }

    # Convert the input data to a DataFrame and encode gender
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, columns=['Gender'])

    # Ensure the input_df has the same columns as the training data
    for col in X_train.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match the training data
    input_df = input_df[X_train.columns]

    # Make prediction
    prediction = classifier.predict(input_df)[0]
    prediction_label = label_encoder.inverse_transform([prediction])[0]

    st.title(f"The predicted cancer level is: {prediction_label}")


elif nav_type == "BMI Calculator":
    st.title("Body Mass Index Calculator")

    weight = st.number_input("Your weight (Kg): ")
    feet = st.number_input("Your height (feet): ")
    inches = st.number_input("Your height (inches): ")

    height_in_meters = (feet * 0.3048) + (inches * 0.0254)

    if weight and height_in_meters:
        bmi = round(weight / height_in_meters ** 2, 2)
        st.metric("BMI", bmi)

        if bmi < 18.5:
            st.write("You are underweight.")
        elif bmi < 25:
            st.write("You are normal weight.")
        elif bmi < 30:
            st.write("You are overweight.")
        else:
            st.write("You are obese.")
elif nav_type == "Feedback Submission":
    rating = st.slider("Rate our project", 1, 10)
    if rating <= 3:
        st.error("We'll work harder to improve!")
    elif rating <= 7:
        st.warning("We'll keep improving!")
    else:
        st.success("Thank you for your feedback!")
    st.title("Thankyou!!!!!")
