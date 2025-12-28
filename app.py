import streamlit as st
import numpy as np
import joblib

# Load trained Decision Tree model
model = joblib.load("decision_model.pkl")

st.set_page_config(page_title="Airline Satisfaction", layout="wide")
st.title("✈️ Airline Customer Satisfaction Prediction")
st.write("Model: Decision Tree Classifier")

st.markdown("---")

# ================= INPUTS =================
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", 10, 100, 30)
travel_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
flight_distance = st.number_input("Flight Distance", 0, 5000, 500)

dep_delay = st.number_input("Departure Delay", 0, 500, 0)
arr_delay = st.number_input("Arrival Delay", 0, 500, 0)

time_conv = st.slider("Departure & Arrival Time Convenience", 0, 5, 3)
online_booking = st.slider("Ease of Online Booking", 0, 5, 3)
checkin = st.slider("Check-in Service", 0, 5, 3)
online_boarding = st.slider("Online Boarding", 0, 5, 3)
gate_location = st.slider("Gate Location", 0, 5, 3)
onboard_service = st.slider("On-board Service", 0, 5, 3)
seat_comfort = st.slider("Seat Comfort", 0, 5, 3)
leg_room = st.slider("Leg Room Service", 0, 5, 3)
cleanliness = st.slider("Cleanliness", 0, 5, 3)
food = st.slider("Food and Drink", 0, 5, 3)
inflight_service = st.slider("In-flight Service", 0, 5, 3)
wifi = st.slider("In-flight Wifi Service", 0, 5, 3)
entertainment = st.slider("In-flight Entertainment", 0, 5, 3)
baggage = st.slider("Baggage Handling", 0, 5, 3)

travel_type = st.selectbox("Type of Travel", ["Business", "Personal"])
customer_type = st.selectbox("Customer Type", ["First-time", "Returning"])

# ================= ENCODING =================
gender_val = 1 if gender == "Male" else 0
class_val = {"Eco": 0, "Eco Plus": 1, "Business": 2}[travel_class]
travel_type_val = 1 if travel_type == "Personal" else 0
customer_type_val = 1 if customer_type == "Returning" else 0

# ================= INPUT ARRAY (22 FEATURES) =================
input_data = np.array([[
    gender_val, age, class_val, flight_distance,
    dep_delay, arr_delay, time_conv, online_booking,
    checkin, online_boarding, gate_location, onboard_service,
    seat_comfort, leg_room, cleanliness, food,
    inflight_service, wifi, entertainment, baggage,
    travel_type_val, customer_type_val
]])

# ================= PREDICTION =================
if st.button("Predict Satisfaction"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ Passenger is SATISFIED")
    else:
        st.error("❌ Passenger is NOT SATISFIED")
