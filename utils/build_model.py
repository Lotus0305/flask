import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, Concatenate, Multiply, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pymongo import MongoClient
import json
import os
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder

load_dotenv()

# Load environment variables
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('DB_NAME')

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

comments_collection = db['comments']
novels_collection = db['novels']

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
    })[['novelId', 'name', 'category']]

    # Drop missing values
    comments = comments.dropna(subset=['accountId', 'novelId', 'rating', 'commentId'])
    novels = novels.dropna(subset=['novelId', 'category'])

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
        'name': str
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

    with open('models/user_id_to_label.json', 'w') as f:
        json.dump(user_id_to_label, f)
    with open('models/novel_id_to_label.json', 'w') as f:
        json.dump(novel_id_to_label, f)
    with open('models/category_id_to_label.json', 'w') as f:
        json.dump(category_id_to_label, f)

    # Merge comments and novels data
    merge = pd.merge(novels, comments, on='novelId', how='left')
    merge = merge.dropna()
    merge = merge.astype({
        'novelId': np.int64,
        'category': np.int64,
        'accountId': np.int64,
        'rating': np.int64,
        'commentId': np.int64,
        'user': np.int64,
        'novel': np.int64,
        'name': str
    })

    # Split data into training and testing sets
    train_data, test_data = train_test_split(merge, test_size=0.2, random_state=42)

    train_users = train_data['user'].values
    train_novels = train_data['novel'].values
    train_categories = train_data['category'].values
    train_ratings = train_data['rating'].values

    test_users = test_data['user'].values
    test_novels = test_data['novel'].values
    test_categories = test_data['category'].values
    test_ratings = test_data['rating'].values

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

    # Model inputs
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    novel_input = Input(shape=(1,), dtype='int32', name='novel_input')
    category_input = Input(shape=(1,), dtype='int32', name='category_input')

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

    mlp = Concatenate()([user_vec_mlp, novel_vec_mlp, category_vec_mlp])
    mlp = Dense(units=dense_units, kernel_regularizer=regularization)(mlp)
    mlp = LeakyReLU(alpha=0.01)(mlp)
    mlp = BatchNormalization()(mlp)
    mlp = Dropout(rate=dropout_rate)(mlp)
    mlp = Dense(units=dense_units // 2, kernel_regularizer=regularization)(mlp)
    mlp = LeakyReLU(alpha=0.01)(mlp)
    mlp = BatchNormalization()(mlp)
    mlp = Dropout(rate=dropout_rate)(mlp)
    mlp = Dense(units=dense_units // 4, kernel_regularizer=regularization)(mlp)
    mlp = LeakyReLU(alpha=0.01)(mlp)
    mlp = BatchNormalization()(mlp)

    # Concatenate GMF and MLP parts
    neu_mf = Concatenate()([gmf, mlp])
    prediction = Dense(units=1, activation='relu', kernel_regularizer=regularization)(neu_mf)

    model = Model(inputs=[user_input, novel_input, category_input], outputs=prediction)

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mean_squared_error'])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    # Train model
    history = model.fit(
        [train_users, train_novels, train_categories],
        train_ratings,
        validation_split=0.1,
        epochs=100,
        batch_size=256,
        callbacks=[early_stopping, reduce_lr]
    )

    # Save model
    model.save('models/recommendation_model.h5')
    print("Model saved successfully.")

    # Evaluate model
    loss, mse = model.evaluate([test_users, test_novels, test_categories], test_ratings)
    print(f'Test Mean Squared Error: {mse}')
    
    return mse

if __name__ == "__main__":
    train_and_save_model()
