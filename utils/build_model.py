import pandas as pd
import numpy as np
import tensorflow as tf
import json
import re
from dotenv import load_dotenv
import os

from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, Concatenate, Multiply, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

load_dotenv()

# Load environment variables
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('DB_NAME')

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

comments_collection = db['comments']
novels_collection = db['novels']

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def train_and_save_model():
    # Load data from MongoDB
    comments = pd.DataFrame(list(comments_collection.find()))
    novels = pd.DataFrame(list(novels_collection.find()))

    # Preprocess comments data
    comments = comments.rename(columns={
        '_id': 'commentId',
        'account': 'accountId',
        'novel': 'novelId',
        'rating': 'rating',
        'createdAt': 'timeStamp',
        'content': 'content'
    })[['accountId', 'novelId', 'rating', 'commentId']]

    # Preprocess novels data
    novels = novels.rename(columns={
        '_id': 'novelId',
        'name': 'name',
        'category': 'category',
        'description': 'description',
        'chapters': 'chapters',
        'views': 'views',
        'powerStone': 'powerStone',
        'imageUrl': 'imageUrl'
    })[['novelId', 'name', 'category', 'description']]

    # Drop missing values
    comments = comments.dropna(subset=['accountId', 'novelId', 'rating', 'commentId'])
    novels = novels.dropna(subset=['novelId', 'category', 'description'])

    # Convert data types
    comments = comments.astype({
        'accountId': np.int64,
        'novelId': np.int64,
        'rating': np.int64,
        'commentId': np.int64
    })
    novels = novels.astype({
        'novelId': np.int64,
        'category': str,
        'name': str,
        'description': str
    })

    # Label encoding
    user_encoder = LabelEncoder()
    novel_encoder = LabelEncoder()
    category_encoder = LabelEncoder()

    comments['user'] = user_encoder.fit_transform(comments['accountId'])
    comments['novel'] = novel_encoder.fit_transform(comments['novelId'])
    novels['category'] = category_encoder.fit_transform(novels['category'])

    # Tạo từ điển mã hóa và lưu trữ vào file JSON
    user_id_to_label = {original_id: encoded_id for original_id, encoded_id in zip(comments['accountId'], comments['user'])}
    novel_id_to_label = {original_id: encoded_id for original_id, encoded_id in zip(comments['novelId'], comments['novel'])}
    category_id_to_label = {original_id: encoded_id for original_id, encoded_id in zip(novels['category'], novels['category'])}

    os.makedirs('models', exist_ok=True)
    with open('models/user_id_to_label.json', 'w') as f:
        json.dump(user_id_to_label, f)
    with open('models/novel_id_to_label.json', 'w') as f:
        json.dump(novel_id_to_label, f)
    with open('models/category_id_to_label.json', 'w') as f:
        json.dump(category_id_to_label, f)

    # Merge comments and novels data
    merge = pd.merge(comments, novels[['novelId', 'category', 'description']], on='novelId', how='left')
    merge = merge.dropna()
    merge = merge.astype({
        'novelId': np.int64,
        'category': np.int64,
        'description': str,
        'accountId': np.int64,
        'rating': np.int64,
        'commentId': np.int64,
        'user': np.int64,
        'novel': np.int64,
    })

    # Apply text preprocessing
    merge['description'] = merge['description'].apply(preprocess_text)

    tfidf = TfidfVectorizer(max_features=64)

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(merge, test_size=0.2, random_state=42)

    train_users = train_data['user'].values
    train_novels = train_data['novel'].values
    train_categories = train_data['category'].values
    train_ratings = train_data['rating'].values
    train_descriptions = tfidf.fit_transform(train_data['description']).toarray()

    test_users = test_data['user'].values
    test_novels = test_data['novel'].values
    test_categories = test_data['category'].values
    test_ratings = test_data['rating'].values
    test_descriptions = tfidf.transform(test_data['description']).toarray()

    classes = np.unique(train_ratings)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_ratings)
    class_weight_dict = dict(enumerate(class_weights))

    # Model parameters
    embedding_dim = 64
    dropout_rate = 0.4
    dense_units = 64
    learning_rate = 0.0001
    reg_value = 0.01
    regularization = l2(reg_value)

    num_users = len(user_encoder.classes_)
    num_novels = len(novel_encoder.classes_)
    num_categories = len(category_encoder.classes_)
    num_description_features = train_descriptions.shape[1]

    # Model inputs
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    novel_input = Input(shape=(1,), dtype='int32', name='novel_input')
    category_input = Input(shape=(1,), dtype='int32', name='category_input')
    description_input = Input(shape=(num_description_features,), dtype='float32', name='description_input')

    # GMF part
    user_embedding_gmf = Embedding(input_dim=num_users, output_dim=embedding_dim, name='user_embedding_gmf')(user_input)
    novel_embedding_gmf = Embedding(input_dim=num_novels, output_dim=embedding_dim, name='novel_embedding_gmf')(novel_input)
    category_embedding_gmf = Embedding(input_dim=num_categories, output_dim=embedding_dim, name='category_embedding_gmf')(category_input)

    user_vec_gmf = Flatten()(user_embedding_gmf)
    novel_vec_gmf = Flatten()(novel_embedding_gmf)
    category_vec_gmf = Flatten()(category_embedding_gmf)

    gmf = Multiply()([user_vec_gmf, novel_vec_gmf, category_vec_gmf])

    # MLP part
    user_embedding_mlp = Embedding(input_dim=num_users, output_dim=embedding_dim, name='user_embedding_mlp')(user_input)
    novel_embedding_mlp = Embedding(input_dim=num_novels, output_dim=embedding_dim, name='novel_embedding_mlp')(novel_input)
    category_embedding_mlp = Embedding(input_dim=num_categories, output_dim=embedding_dim, name='category_embedding_mlp')(category_input)

    user_vec_mlp = Flatten()(user_embedding_mlp)
    novel_vec_mlp = Flatten()(novel_embedding_mlp)
    category_vec_mlp = Flatten()(category_embedding_mlp)

    # Include the description embedding in the MLP part
    mlp = Concatenate()([user_vec_mlp, novel_vec_mlp, category_vec_mlp, description_input])
    mlp = Dense(dense_units, kernel_regularizer=regularization)(mlp)
    mlp = LeakyReLU(negative_slope=0.1)(mlp)
    mlp = BatchNormalization()(mlp)
    mlp = Dropout(dropout_rate)(mlp)
    mlp = Dense(dense_units // 2, kernel_regularizer=regularization)(mlp)
    mlp = LeakyReLU(negative_slope=0.1)(mlp)
    mlp = BatchNormalization()(mlp)
    mlp = Dropout(dropout_rate)(mlp)
    mlp = Dense(dense_units // 4, kernel_regularizer=regularization)(mlp)
    mlp = LeakyReLU(negative_slope=0.1)(mlp)
    mlp = BatchNormalization()(mlp)

    neumf = Concatenate()([gmf, mlp])
    output = Dense(1)(neumf)

    model = Model(inputs=[user_input, novel_input, category_input, description_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mae', 'mape', tf.keras.metrics.RootMeanSquaredError()])

    # Early Stopping and ReduceLROnPlateau
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

    # Model Training
    history = model.fit([train_users, train_novels, train_categories, train_descriptions], train_ratings,
                        batch_size=256, epochs=200, verbose=1,
                        validation_data=([test_users, test_novels, test_categories, test_descriptions], test_ratings),
                        callbacks=[early_stopping, reduce_lr],
                        class_weight=class_weight_dict)

    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/recommendation_model_with_description.h5')
    print("Model saved successfully.")

    # Model Evaluation
    eval_results = model.evaluate([test_users, test_novels, test_categories, test_descriptions], test_ratings, verbose=1)
    print(f"Test Loss: {eval_results[0]}, Test MAE: {eval_results[1]}, Test MAPE: {eval_results[2]}, Test RMSE: {eval_results[3]}")

    return eval_results

if __name__ == "__main__":
    train_and_save_model()
