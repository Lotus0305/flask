import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import json
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from utils.build_model import train_and_save_model
from config.config import Config
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf

# Load stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load environment variables
load_dotenv()

# Flask app setup
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# MongoDB configuration
client = MongoClient(app.config['MONGO_URI'])
db = client[app.config['DB_NAME']]
novels_collection = db['novels']
categories_collection = db['categories']
comments_collection = db['comments']

# Load mappings from JSON
MODEL_DIR = 'models'
with open(os.path.join(MODEL_DIR, 'user_id_to_label.json'), 'r') as file:
    user_id_to_label = json.load(file)
with open(os.path.join(MODEL_DIR, 'novel_id_to_label.json'), 'r') as file:
    novel_id_to_label = json.load(file)
with open(os.path.join(MODEL_DIR, 'category_id_to_label.json'), 'r') as file:
    category_id_to_label = json.load(file)

model_path = os.path.join(MODEL_DIR, 'recommendation_model_with_description.h5')

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Function to convert IDs using JSON mapping
def encode_id(mapping, original_id):
    return mapping.get(str(original_id), -1)  # Return -1 if ID not found

def decode_id(mapping, encoded_id):
    reverse_mapping = {v: k for k, v in mapping.items()}
    return reverse_mapping.get(encoded_id, None)

# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load the TF-IDF vectorizer used during model training
tfidf = TfidfVectorizer(max_features=64)

# Load novels data
novels = pd.DataFrame(list(novels_collection.find()))
novels['description'] = novels['description'].apply(preprocess_text)
novels['combined'] = novels['description'] + " " + novels['name'] + " " + novels['category'].astype(str)
tfidf.fit(novels['combined'])

# Endpoint to recommend novels based on user ID
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    encoded_user_id = encode_id(user_id_to_label, user_id)
    
    if encoded_user_id == -1:
        return jsonify({'error': 'User ID not found'}), 400

    all_novel_ids = novels['_id'].unique()
    rated_novel_ids = comments_collection.find({'account': user_id}, {'novel': 1})
    rated_novel_ids = [comment['novel'] for comment in rated_novel_ids]
    novel_ids_to_rate = np.setdiff1d(all_novel_ids, rated_novel_ids)

    user_input = np.full(len(novel_ids_to_rate), encoded_user_id)
    novel_input = np.array([encode_id(novel_id_to_label, str(nid)) for nid in novel_ids_to_rate])
    category_input = novels.set_index('_id').loc[novel_ids_to_rate, 'category'].values
    category_input = np.array([encode_id(category_id_to_label, cid) for cid in category_input])
    description_input = tfidf.transform(novels.set_index('_id').loc[novel_ids_to_rate, 'description']).toarray()

    predicted_ratings = model.predict([user_input, novel_input, category_input, description_input])
    top_n_indices = np.argsort(predicted_ratings[:, 0])[-10:][::-1]
    top_n_novel_ids = novel_ids_to_rate[top_n_indices]
    top_n_novel_names = novels.set_index('_id').loc[top_n_novel_ids, 'name'].values

    recommendations = [{"id": str(nid), "name": name} for nid, name in zip(top_n_novel_ids, top_n_novel_names)]
    
    return jsonify(recommendations)

# Endpoint to recommend novels based on a specific novel's ID
@app.route('/recommend_based_on_novel', methods=['GET'])
def recommend_based_on_novel():
    novel_id = request.args.get('novel_id')
    encoded_novel_id = encode_id(novel_id_to_label, novel_id)
    
    if encoded_novel_id == -1:
        return jsonify({'error': 'Novel ID not found'}), 400

    novel_embedding = get_novel_embedding(encoded_novel_id)
    all_novel_ids = novels['_id'].unique()
    other_novel_ids = np.setdiff1d(all_novel_ids, [novel_id])

    similarities = []
    for other_novel_id in other_novel_ids:
        other_encoded_novel_id = encode_id(novel_id_to_label, str(other_novel_id))
        other_novel_embedding = get_novel_embedding(other_encoded_novel_id)
        similarity = cosine_similarity([novel_embedding], [other_novel_embedding])[0][0]
        similarities.append((other_novel_id, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_n_novel_ids = [sim[0] for sim in similarities[:10]]
    top_n_novel_names = novels.set_index('_id').loc[top_n_novel_ids, 'name'].values
    
    recommendations = [{"id": str(nid), "name": name} for nid, name in zip(top_n_novel_ids, top_n_novel_names)]
    
    return jsonify(recommendations)

@app.route('/search_novel', methods=['GET'])
def search_novel():
    query = request.args.get('query')
    preprocessed_query = preprocess_text(query)
    query_tfidf = tfidf.transform([preprocessed_query])
    
    novels['similarity'] = cosine_similarity(tfidf.transform(novels['combined']), query_tfidf).flatten()
    top_novels = novels.sort_values(by='similarity', ascending=False).head(10)
    
    results = [{"id": str(novel['_id']), "name": novel['name']} for _, novel in top_novels.iterrows()]
    
    return jsonify(results)

def get_novel_embedding(novel_id):
    novel_embedding_gmf = model.get_layer('novel_embedding_gmf').get_weights()[0][novel_id]
    novel_embedding_mlp = model.get_layer('novel_embedding_mlp').get_weights()[0][novel_id]
    novel_embedding = np.concatenate((novel_embedding_gmf, novel_embedding_mlp))
    return novel_embedding

# Endpoint to retrain the model
@app.route('/train', methods=['POST'])
def train_model():
    response = train_and_save_model()
    return jsonify({'status': 'Model trained successfully', 'mse': response}), 200

# Health check endpoint to ensure the service is running
@app.route('/check', methods=['GET'])
def health_check():
    return jsonify({'status': 'Service is up and running'}), 200

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000)
