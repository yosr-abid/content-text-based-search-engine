
# client elasticsearch 
#query request : Fuzzy = Floue 
import streamlit as st
from elasticsearch import Elasticsearch
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
from tensorflow.keras.preprocessing.image import load_img
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = joblib.load('D:\\indexation\\cbir\\pca_model.pkl')
scaler = joblib.load('D:\\indexation\\cbir\\scaler.pkl')

class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('predictions').output)

    def extract(self, img):
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
        return feature / np.linalg.norm(feature)  # Normalize
Fe=FeatureExtractor()

client = Elasticsearch("http://localhost:9200")
def extract(image):
    query_vector= Fe.extract(image)
    # query_vector = query_vector.reshape(1, -1)  # Assuming query_vector is a 1D numpy array

    # # Standardize the query vector (important for PCA)
    # query_vector = scaler.transform(query_vector)  # Using the same scaler you used for the training data

    # # Transform the query vector using PCA
    # query_vector = pca.transform(query_vector)
    # query_vector = query_vector.flatten()
    return query_vector

def fetch_cbir_images(query_vector, num_images):
    knn_query = {
        "knn": {
            "field": "Vector",
            "query_vector": query_vector,
            "k": num_images,
            "num_candidates": 1000
                },
        "fields": [ "ImageURL"]
    }
    results = client.search(index="test4", body=knn_query)
    hits = results['hits']['hits']
    image_urls = [hit['_source']['ImageURL'] for hit in hits]
    return image_urls


def fetch_hybrid_images(query_vector,input,num_images) :
        knn_query = {
        "knn": {
            "field": "Vector",
            "query_vector": query_vector,
            "k": 30,
            "num_candidates": 1000
                },
        "fields": [ "ImageURL"]
        }
        results = client.search(index="test4", body=knn_query)
        hits = results['hits']['hits']
        image_urls1 = [hit['_source']['ImageURL'] for hit in hits]
        request = {
        "query": {
            "fuzzy": {
                "Tags": input
            }
        }
        }
    
        results = client.search(index="test4", body=request, size=30)
        hits = results["hits"]["hits"]
    
        # Fetch image URLs from the "ImageURL" field
        image_urls = [hit["_source"]["ImageURL"] for hit in hits] 
        intersection = [value for value in image_urls1 if value in image_urls]
        if len(intersection)==0:
            return image_urls1[:num_images] 
        elif len(intersection)<num_images :
            return intersection
        else :
            return intersection[:num_images]


    


def fetch_initial_images(input, num_images):
    request = {
        "query": {
            "fuzzy": {
                "Tags": input
            }
        }
    }
    
    results = client.search(index="test4", body=request, size=num_images)
    hits = results["hits"]["hits"]
    
    # Fetch image URLs from the "ImageURL" field
    image_urls = [hit["_source"]["ImageURL"] for hit in hits]
    
    return image_urls


# def fetch_more_images(input, start_index, num_images):
#     request = {
#         "query": {
#             "fuzzy": {
#                 "Tags": input
#             }
#         }
#     }
    
#     results = client.search(index="test4", body=request, size=num_images)
#     hits = results["hits"]["hits"]
    
#     # Fetch image URLs from the "ImageURL" field
#     image_urls = [hit["_source"]["ImageURL"] for hit in hits]
    
#     return image_urls[start_index:start_index + num_images]

st.markdown('## Search Engine')
st.subheader('Made by Yosr Abid & Oussema Louhichi', divider='rainbow')
num_images = st.slider('pic the number of images you want', 1, 15, 4)
option = st.selectbox('choose the search engine you want',('Text based','Image based','Image and Text based'))
if option=='Text based' :
    input = st.text_input("Write a word ")
    search_button = st.button("Search")
    if search_button:
        st.session_state.images = fetch_initial_images(input, num_images)
    
elif  option=='Image based':
    uploaded_file = st.file_uploader("Choose a picture", type=["jpg", "png", "jpeg", "gif"])
    search_button = st.button("Search")
    if uploaded_file is not None:
        # Check if the uploaded file is an image
        if uploaded_file.type.startswith("image"):
            uploaded_image = Image.open(uploaded_file)
            print(type(uploaded_image))
            query_vector=extract(uploaded_image)
        else:
            st.write("File is not a valid image format.")
        if search_button:
            st.session_state.images =fetch_cbir_images(query_vector, num_images)
elif option=='Image and Text based' : 
    input = st.text_input("Write a word ")
    uploaded_file = st.file_uploader("Choose a picture", type=["jpg", "png", "jpeg", "gif"])
    search_button = st.button("Search")
    if uploaded_file is not None:
        # Check if the uploaded file is an image
        if uploaded_file.type.startswith("image"):
            uploaded_image = Image.open(uploaded_file)
            print(type(uploaded_image))
            query_vector=extract(uploaded_image)
        else:
            st.write("File is not a valid image format.")
        if search_button:
            st.session_state.images = fetch_hybrid_images(query_vector,input,num_images)
    


# Display the images
if hasattr(st.session_state, "images") and st.session_state.images:
    columns = st.columns(3)
    
    for i, image_url in enumerate(st.session_state.images):
        columns[i % 3].image(image_url)



# show_more_button = st.button("Show More")

# if show_more_button:
#     num_images_to_display = 3
#     additional_images = fetch_more_images(input, st.session_state.image_start_index, num_images_to_display)
#     if additional_images:
#         st.session_state.images.extend(additional_images)
#         st.session_state.image_start_index += num_images_to_display

