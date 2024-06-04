import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.build_model import train_and_save_model
from flask_cors import CORS

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('DB_NAME')

if not DB_NAME:
    raise ValueError("DB_NAME environment variable is not set or empty")

app = Flask(__name__)
CORS(app)
# Debugging prints
print(f"MONGO_URI: {MONGO_URI}")
print(f"DB_NAME: {DB_NAME}")

# MongoDB configuration
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
novels_collection = db['novels']
comments_collection = db['comments']

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

# Load novels data for TF-IDF
novels_data = list(novels_collection.find({}, {"_id": 1, "name": 1}))

# Create TF-IDF Vectorizer and fit it on novel names
novel_names = [novel['name'] for novel in novels_data]
vectorizer = TfidfVectorizer().fit(novel_names)
tfidf_matrix = vectorizer.transform(novel_names)

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    
    if user_id is None:
        return jsonify({'error': 'User ID is required'}), 400

    try:
        user_id = int(user_id)
    except ValueError:
        return jsonify({'error': 'Invalid User ID format'}), 400

    # Encode the user_id
    if user_id not in user_encoder.classes_:
        account_data = comments_collection.find_one({}, {'_id': 0, 'account': 1})
        user_id = account_data
        # return jsonify({'error': 'User ID not found', 'account_data': user_id}), 404
    user_encoded = user_encoder.transform([user_id])[0]

    # Get all novels
    all_novels = novel_encoder.classes_

    # Predict ratings for all novels
    user_array = np.array([user_encoded] * len(all_novels))
    novel_array = novel_encoder.transform(all_novels)
    category_array = np.zeros(len(all_novels))  # Placeholder if category is not available

    predictions = model.predict([user_array, novel_array, category_array])
    
    # Combine novels with predictions
    novel_predictions = list(zip(all_novels, predictions.flatten()))
    
    # Sort by predicted rating
    novel_predictions.sort(key=lambda x: x[1], reverse=True)

    # Select top N recommendations
    top_novels = novel_predictions[:50]

    # Convert novel IDs and ratings to regular Python integers and floats
    recommended_novels = [{'novel_id': int(novel_id), 'predicted_rating': float(rating)} for novel_id, rating in top_novels]

    return jsonify(recommended_novels)

@app.route('/recommend_based_on_novel', methods=['GET'])
def recommend_based_on_novel():
    novel_id = request.args.get('novel_id')
    
    if novel_id is None:
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

@app.route('/search_novel', methods=['GET'])
def search_novel():
    query = request.args.get('query')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400

    query_tfidf = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    top_n_indices = cosine_similarities.argsort()[-10:][::-1]
    top_novels = [(novels_data[idx]['_id'], novels_data[idx]['name'], cosine_similarities[idx]) for idx in top_n_indices]

    results = [{'novel_id': str(novel_id), 'name': name, 'similarity_score': float(score)} for novel_id, name, score in top_novels]

    return jsonify(results)

@app.route('/train', methods=['POST'])
def train_model():
    response = train_and_save_model()
    return jsonify(response), 200

# Health check endpoint
@app.route('/check', methods=['GET'])
def health_check():
    return jsonify({'status': 'up'}), 200

if __name__ == '__main__':
    app.run(debug=False)
