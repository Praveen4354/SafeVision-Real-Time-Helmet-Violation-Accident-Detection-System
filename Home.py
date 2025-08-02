import streamlit as st

# Sidebar setup for navigation
st.sidebar.title("🛡️ UrbanPulse")
page = st.sidebar.radio("Select a page", ("Helmet Detection", "Accident Detection"))

if page == "Helmet Detection":
    # Import and call your Helmet Detection page logic here
    import Helmet_Detection
    Helmet_Detection.run()  # Assuming the Helmet_Detection.py file has a `run()` function
    
elif page == "Accident Detection":
    # Import and call your Accident Detection page logic here
    import Accident_Detection
    Accident_Detection.run()  # Assuming the Accident_Detection.py file has a `run()` function
