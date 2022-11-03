import pandas as pd
import joblib
import streamlit as st

featuress = pd.read_csv("Data.csv")
target = featuress.drop(columns=['Is_Success'])

st.set_page_config(page_title="Zomato  Prediction App", page_icon='icon.jfif')
st.title(' Zomato Prediction App', '\n')
st.subheader("Zomato  is an Indian multinational restaurant aggregator and food delivery company founded  in 2008. Zomato provides information, menus and user reviews of restaurants as well as food delivery options from partnerrestaurants in select cities. As of 2019, the service is available in 24 countries and in more than 10,000 cities.")

st.markdown(""" ##### The goal of this application is to classify the restaurant as successful or not based on the characteristics or data of this restaurant.
##### To predict whether the restaurant is successful or not, just follow these steps:
##### 1. Enter information describing the restaurant.
##### 2. Press the "Predict" button and wait for the result.
##### Author: Ahmed Mohamed Kamel ([GitHub](https://github.com/AhmedMohamedKamel2022))

""")

def user_input_features():

    st.sidebar.write('# Fill this form please..')

    online_order = st.sidebar.radio("Is online ordering is available in the restaurant or not ? ",
     options=(online_order for online_order in featuress.online_order.unique()))

    book_table = st.sidebar.radio("Is booking a table available or not ?", 
    options=(book_table for book_table in featuress.book_table.unique()))

    Type = st.sidebar.selectbox("What is the type of meal ?",
                                    options=(Type for Type in featuress.Type.unique()))

    rest_type = st.sidebar.selectbox("What is the restaurant type ?", 
                                    (rest_type for rest_type in featuress.rest_type.unique()))
    
    location = st.sidebar.selectbox("What is the location ?",
                                        options=(location for location in featuress.location.unique()))

    cuisines = st.sidebar.selectbox("What is kind of kitchen (food style) ?",
                                        options=(cuisines for cuisines in featuress.cuisines.unique()))                                  

    votes = st.sidebar.number_input('How many votes ?', 0, 10000)

    Cost = st.sidebar.number_input('How much does food cost ?', 0, 10000)
    
    data = {
        "online_order": online_order,
        "book_table": book_table,
        "Type": Type,
        "rest_type": rest_type,
        "location": location,
        "cuisines": cuisines,
        "votes": votes,
        "Cost2plates": Cost}

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

df = pd.concat([input_df,target],axis=0)

cols = ['online_order', 'book_table', 'location', 'rest_type', 'cuisines', 'Type']
for col in cols:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

RF_MODEL_PATH = joblib.load("model.h5")
RF_SCALER_PATH = joblib.load("scaler.h5")

scaled_data = RF_SCALER_PATH.transform(df)
prediction_proba = RF_MODEL_PATH.predict_proba(scaled_data)

if st.sidebar.button('Predict'):
    st.sidebar.success(f'# The probability of this restaurant succeeding is: {round(prediction_proba[0][1] * 100, 2)}%')