import os
from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np
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

# Load encoders and model paths
MODEL_DIR = 'models'
category_encoder_path = os.path.join(MODEL_DIR, 'category_encoder.pkl')
novel_encoder_path = os.path.join(MODEL_DIR, 'novel_encoder.pkl')
user_encoder_path = os.path.join(MODEL_DIR, 'user_encoder.pkl')
model_path = os.path.join(MODEL_DIR, 'recommendation_model.h5')

category_encoder = joblib.load(category_encoder_path)
novel_encoder = joblib.load(novel_encoder_path)
user_encoder = joblib.load(user_encoder_path)
model = tf.keras.models.load_model(model_path)

# Load SentenceTransformer model
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Fetch novels and categories data from MongoDB
novels_data = list(novels_collection.find({}, {"_id": 1, "name": 1, "author": 1, "category": 1, "description": 1}))
categories_data = list(categories_collection.find({}, {"_id": 1, "name": 1}))

# Create a DataFrame for novels and categories
novels_df = pd.DataFrame(novels_data)
categories_df = pd.DataFrame(categories_data)

# Create a dictionary to map category IDs to category names
category_dict = pd.Series(categories_df.name.values, index=categories_df._id).to_dict()

def preprocess_description(description):
    description = description.lower()
    description = re.sub(r'[^a-zA-Z0-9\s]', '', description)
    description = re.sub(r'\s+', ' ', description).strip()
    words = description.split()
    filtered_description = ' '.join([word for word in words if word not in stop_words])
    return filtered_description

def compute_embedding(row):
    preprocessed_description = preprocess_description(row['description'])
    combined_text = f"{row['name']} {row['author']} {row['category']} {preprocessed_description}"
    return sentence_model.encode(combined_text, convert_to_tensor=True)

@app.route('/compute_embeddings', methods=['POST'])
def compute_and_store_embeddings():
    novels_df['combined_embedding'] = novels_df.apply(compute_embedding, axis=1)
    novels_df['combined_embedding'] = novels_df['combined_embedding'].apply(lambda x: x.tolist())
    novels_df.to_pickle('data/novels_with_combined_embeddings.pkl')
    return jsonify({'status': 'Embeddings computed and stored successfully'}), 200

def load_embeddings():
    embeddings_path = 'data/novels_with_combined_embeddings.pkl'
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError("Embeddings file not found. Please run the /compute_embeddings endpoint first.")
    return pd.read_pickle(embeddings_path)

@app.route('/search_novel', methods=['GET'])
def search_novel():
    query = request.args.get('query')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400

    novels_df = load_embeddings()
    query_embedding = sentence_model.encode(query, convert_to_tensor=True)
    novels_embeddings = torch.tensor(novels_df['combined_embedding'].tolist())
    similarities = cosine_similarity(query_embedding.unsqueeze(0), novels_embeddings)
    top_k_indices = similarities.argsort()[0][-10:][::-1]  # Limit to top 10 results
    results = novels_df.iloc[top_k_indices]
    
    # Map category IDs to category names
    results['category'] = results['category'].map(category_dict)
    
    result_list = results[['category', 'name']].to_dict(orient='records')
    for i, result in enumerate(result_list):
        result['_id'] = str(results.iloc[i]['_id'])

    return jsonify(result_list)

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400
    try:
        user_id = int(user_id)
    except ValueError:
        return jsonify({'error': 'Invalid User ID format'}), 400

    if user_id not in user_encoder.classes_:
        return jsonify({'error': 'User ID not found'}), 404

    user_encoded = user_encoder.transform([user_id])[0]
    all_novels = novel_encoder.classes_
    user_array = np.array([user_encoded] * len(all_novels))
    novel_array = novel_encoder.transform(all_novels)
    category_array = np.zeros(len(all_novels))

    predictions = model.predict([user_array, novel_array, category_array])
    novel_predictions = list(zip(all_novels, predictions.flatten()))
    novel_predictions.sort(key=lambda x: x[1], reverse=True)
    top_novels = novel_predictions[:50]
    recommended_novels = [{'novel_id': int(novel_id), 'predicted_rating': float(rating)} for novel_id, rating in top_novels]

    return jsonify(recommended_novels)

@app.route('/recommend_based_on_novel', methods=['GET'])
def recommend_based_on_novel():
    novel_id = request.args.get('novel_id')
    
    if not novel_id:
        return jsonify({'error': 'Novel ID is required'}), 400
    try:
        novel_id = int(novel_id)
    except ValueError:
        return jsonify({'error': 'Invalid Novel ID format'}), 400

    if novel_id not in novel_encoder.classes_:
        return jsonify({'error': 'Novel ID not found'}), 404

    novel_encoded = novel_encoder.transform([novel_id])[0]
    novel_embeddings = model.get_layer('novel_embedding_gmf').get_weights()[0]
    selected_novel_embedding = novel_embeddings[novel_encoded]
    similarities = np.dot(novel_embeddings, selected_novel_embedding) / (np.linalg.norm(novel_embeddings, axis=1) * np.linalg.norm(selected_novel_embedding))
    top_n_indices = similarities.argsort()[-11:][::-1]
    top_novels = [(novel_encoder.inverse_transform([idx])[0], similarities[idx]) for idx in top_n_indices if idx != novel_encoded]
    recommended_novels = [{'novel_id': int(novel_id), 'similarity_score': float(score)} for novel_id, score in top_novels]

    return jsonify(recommended_novels)

@app.route('/train', methods=['POST'])
def train_model():
    response = train_and_save_model()
    return jsonify(response), 200

@app.route('/check', methods=['GET'])
def health_check():
    return jsonify({'status': 'up'}), 200

if __name__ == '__main__':
    app.run(debug=False)
