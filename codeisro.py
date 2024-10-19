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
# Example texts from India
social_media_post = "Floods in Kerala have displaced thousands of people. Relief operations are ongoing."
news_article = "A cyclone hit the coastal areas of Odisha and West Bengal, causing severe damage to infrastructure and homes."

# Extract entities
entities_post = extract_entities(social_media_post)
entities_article = extract_entities(news_article)

# Geocode locations using Location API key
locationiq_api_key = 'your_location_api_key'  # Replace with your Location API key
geocoded_post = [(entity[0], geocode_location(entity[0], locationiq_api_key)) for entity in entities_post]
geocoded_article = [(entity[0], geocode_location(entity[0], locationiq_api_key)) for entity in entities_article]

# Print extracted and geocoded entities
print("Social Media Post Geocoded Entities:", geocoded_post)
print("News Article Geocoded Entities:", geocoded_article)

# Generate summary using OpenAI GPT-4
# Replace 'your_openai_api_key' with your actual OpenAI API key
openai_api_key = 'your_openai_api_key'
summary_post = generate_summary(social_media_post, openai_api_key)
summary_article = generate_summary(news_article, openai_api_key)

# Print summaries
print("\nSummary for Social Media Post:")
print(summary_post)
print("\nSummary for News Article:")
print(summary_article)
