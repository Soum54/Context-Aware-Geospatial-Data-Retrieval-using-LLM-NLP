import streamlit as st
import spacy
import requests
from geopy.geocoders import Nominatim
import openai

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize geolocators
nominatim_geolocator = Nominatim(user_agent="geoapiExercises")

# Function to perform named entity recognition (NER) and extract location
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
    return entities

# Function to geocode location using geopy and LocationIQ
def geocode_location(location_name, locationiq_api_key):
    try:
        location = nominatim_geolocator.geocode(location_name)
        if location:
            return (location.latitude, location.longitude)
    except:
        pass
    # Fallback to LocationIQ Geocoding API
    url = f"https://us1.locationiq.com/v1/search.php?key={locationiq_api_key}&q={location_name}&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data:
            return (data[0]['lat'], data[0]['lon'])
    return None

# Function to use OpenAI's GPT-4 for contextual understanding
def generate_summary(text, openai_api_key):
    openai.api_key = openai_api_key
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant that summarizes geospatial information."},
            {"role": "user", "content": f"Extract and summarize geospatial information from the following text:\n\n{text}"}
        ],
        max_tokens=150
    )
    return response.choices[0].message['content'].strip()

# Streamlit app
def main():
    st.title("Geospatial Entity Recognition and Summarization")

    # Input section
    st.header("Input Text")
    text = st.text_area("Enter your text here:", height=200)
    
    locationiq_api_key = st.text_input("Enter your LocationIQ API Key")
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

    if st.button("Process"):
        if text and locationiq_api_key and openai_api_key:
            # Extract entities
            entities = extract_entities(text)
            st.subheader("Extracted Entities")
            st.write(entities)

            # Geocode locations
            geocoded_locations = [(entity[0], geocode_location(entity[0], locationiq_api_key)) for entity in entities]
            st.subheader("Geocoded Locations")
            st.write(geocoded_locations)

            # Generate summary using OpenAI GPT-4
            summary = generate_summary(text, openai_api_key)
            st.subheader("Summary")
            st.write(summary)
        else:
            st.error("Please enter text, LocationIQ API key, and OpenAI API key.")

if __name__ == '__main__':
    main()
