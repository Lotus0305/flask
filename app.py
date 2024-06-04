import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
import re
import nltk
from nltk.corpus import stopwords
from utils.build_model import train_and_save_model
from config.config import Config

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config.from_object(Config)

# MongoDB configuration
client = MongoClient(app.config['MONGO_URI'])
db = client[app.config['DB_NAME']]
novels_collection = db['novels']
categories_collection = db['categories']

# Load mappings from JSON
MODEL_DIR = 'models'
with open(os.path.join(MODEL_DIR, 'user_id_to_label.json'), 'r') as file:
    user_id_to_label = json.load(file)
with open(os.path.join(MODEL_DIR, 'novel_id_to_label.json'), 'r') as file:
    novel_id_to_label = json.load(file)
with open(os.path.join(MODEL_DIR, 'category_id_to_label.json'), 'r') as file:
    category_id_to_label = json.load(file)

model_path = os.path.join(MODEL_DIR, 'recommendation_model.h5')
model = tf.keras.models.load_model(model_path)

# Load SentenceTransformer model
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to convert IDs using JSON mapping
def encode_id(mapping, original_id):
    return mapping.get(str(original_id), -1)  # Return -1 if ID not found

def decode_id(mapping, encoded_id):
    reverse_mapping = {v: k for k, v in mapping.items()}
    return reverse_mapping.get(encoded_id, None)

# Utility function for preprocessing descriptions
def preprocess_description(description):
    description = description.lower()
    description = re.sub(r'[^a-zA-Z0-9\s]', '', description)
    description = re.sub(r'\s+', ' ', description).strip()
    words = description.split()
    filtered_description = ' '.join([word for word in words if word not in stop_words])
    return filtered_description

# Function to compute embedding using SentenceTransformer
def compute_embedding(row):
    preprocessed_description = preprocess_description(row['description'])
    combined_text = f"{row['name']} {row['author']} {row['category']} {preprocessed_description}"
    return sentence_model.encode(combined_text, convert_to_tensor=True)

# Endpoint to compute and store embeddings for novels
@app.route('/compute_embeddings', methods=['POST'])
def compute_and_store_embeddings():
    novels_df = pd.DataFrame(list(novels_collection.find({}, {"_id": 1, "name": 1, "author": 1, "category": 1, "description": 1})))
    novels_df['combined_embedding'] = novels_df.apply(compute_embedding, axis=1)
    novels_df['combined_embedding'] = novels_df['combined_embedding'].apply(lambda x: x.tolist())
    novels_df.to_pickle('models/novels_with_combined_embeddings.pkl')
    return jsonify({'status': 'Embeddings computed and stored successfully'}), 200

# Endpoint to search for novels based on text query
@app.route('/search_novel', methods=['GET'])
def search_novel():
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'Query is required'}), 400

    novels_df = pd.read_pickle('models/novels_with_combined_embeddings.pkl')
    query_embedding = sentence_model.encode(query, convert_to_tensor=True)
    novels_embeddings = torch.tensor(novels_df['combined_embedding'].tolist())
    similarities = cosine_similarity(query_embedding.unsqueeze(0), novels_embeddings)
    top_k_indices = similarities.argsort()[0][-10:][::-1]  # Limit to top 10 results
    results = novels_df.iloc[top_k_indices]
    results['category'] = results['category'].apply(lambda x: category_id_to_label[str(x)])
    
    result_list = results[['category', 'name']].to_dict(orient='records')
    for i, result in enumerate(result_list):
        result['_id'] = str(results.iloc[i]['_id'])

    return jsonify(result_list)

# Endpoint to recommend novels based on user ID
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400
    try:
        user_encoded = encode_id(user_id_to_label, user_id)
    except ValueError:
        return jsonify({'error': 'Invalid User ID format'}), 400

    if user_encoded == -1:
        return jsonify({'error': 'User ID not found'}), 404

    all_novels = list(novel_id_to_label.keys())
    novel_array = np.array([encode_id(novel_id_to_label, nid) for nid in all_novels])
    user_array = np.array([user_encoded] * len(novel_array))
    category_array = np.zeros(len(novel_array))  # Placeholder if categories are not used in prediction

    predictions = model.predict([user_array, novel_array, category_array])
    novel_predictions = list(zip(all_novels, predictions.flatten()))
    novel_predictions.sort(key=lambda x: x[1], reverse=True)
    top_novels = novel_predictions[:50]
    recommended_novels = [{'novel_id': int(nid), 'predicted_rating': float(rating)} for nid, rating in top_novels]

    return jsonify(recommended_novels)

# Endpoint to recommend novels based on a specific novel's ID
@app.route('/recommend_based_on_novel', methods=['GET'])
def recommend_based_on_novel():
    novel_id = request.args.get('novel_id')
    if not novel_id:
        return jsonify({'error': 'Novel ID is required'}), 400
    try:
        novel_encoded = encode_id(novel_id_to_label, novel_id)
    except ValueError:
        return jsonify({'error': 'Invalid Novel ID format'}), 400

    if novel_encoded == -1:
        return jsonify({'error': 'Novel ID not found'}), 404

    novel_embeddings = model.get_layer('novel_embedding_gmf').get_weights()[0]
    selected_novel_embedding = novel_embeddings[novel_encoded]
    similarities = np.dot(novel_embeddings, selected_novel_embedding) / (np.linalg.norm(novel_embeddings, axis=1) * np.linalg.norm(selected_novel_embedding))
    top_n_indices = similarities.argsort()[-11:][::-1]
    top_novels = [(int(decode_id(novel_id_to_label, idx)), float(similarities[idx])) for idx in top_n_indices if idx != novel_encoded]
    recommended_novels = [{'novel_id': novel_id, 'similarity_score': score} for novel_id, score in top_novels]

    return jsonify(recommended_novels)

# Endpoint to retrain the model
@app.route('/train', methods=['POST'])
def train_model():
    response = train_and_save_model()
    return jsonify({'status': 'Model trained successfully', 'mse': response}), 200

# Health check endpoint to ensure the service is running
@app.route('/check', methods=['GET'])
def health_check():
    return jsonify({'status': 'Service is up and running'}), 200

if __name__ == '__main__':
    app.run(debug=False)



