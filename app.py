import numpy as np
import streamlit as st
import joblib

model = joblib.load('best_fire_detection_model.pkl')
scalar = joblib.load("D:\\EDUNET-SHELL INTERNSHIP\\Deforestation detection\\scaler.pkl")

st.set_page_config(page_title="Fire type classification", layout='centered')
st.title("Fire type classification")
st.markdown("Predict fire type classification based on MODIS classification data")

brightness = st.number_input("Brightness", value=300.0)
bright_t31 = st.number_input("Brightness t31", value=290.0)
frp = st.number_input("Fire radiative power", value=15.0)
scan = st.number_input("Scan", value=1.0)
track = st.number_input("Track", value=1.0)
confidence = st.selectbox("Confidence level",["low","nominal","high"])

confidence_map = {"low":0,"nominal":1,"high":2}
confidence_val = confidence_map[confidence]

input_data = np.array([[brightness,bright_t31,frp,scan,track,confidence_val]])
scaled_input = scalar.transform(input_data)

if st.button("Predict fire type"):
    prediction = model.predict(scaled_input)[0]

    fire_types = {
        0: "Vegetation fire",
        2: "Other Static land source",
        3: "Offshore fire"
    }

    result = fire_types.get(prediction,"Unknown")
    st.success(f"**Predicted Fire Type**{result}")