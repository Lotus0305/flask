import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, Concatenate, Multiply, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pymongo import MongoClient
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('DB_NAME')

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

comments_collection = db['comments']
novels_collection = db['novels']

def train_and_save_model():
    comments = pd.DataFrame(list(comments_collection.find()))
    novels = pd.DataFrame(list(novels_collection.find()))

    comments = comments.rename(columns={
        '_id': 'commentId',
        'account': 'accountId',
        'novel': 'novelId',
        'rating': 'rating',
        'createdAt': 'timeStamp',
        'content': 'content'
    })[['accountId', 'novelId', 'rating', 'commentId']]

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

    comments = comments.dropna(subset=['accountId', 'novelId', 'rating', 'commentId'])
    novels = novels.dropna(subset=['novelId', 'category'])

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

    user_encoder = LabelEncoder()
    novel_encoder = LabelEncoder()
    category_encoder = LabelEncoder()

    comments['user'] = user_encoder.fit_transform(comments['accountId'])
    comments['novel'] = novel_encoder.fit_transform(comments['novelId'])
    novels['category'] = category_encoder.fit_transform(novels['category'])

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

    train_data, test_data = train_test_split(merge, test_size=0.2, random_state=42)

    train_users = train_data['user'].values
    train_novels = train_data['novel'].values
    train_categories = train_data['category'].values
    train_ratings = train_data['rating'].values

    test_users = test_data['user'].values
    test_novels = test_data['novel'].values
    test_categories = test_data['category'].values
    test_ratings = test_data['rating'].values

    embedding_dim = 64
    dropout_rate = 0.4
    dense_units = 64
    learning_rate = 0.0001
    reg_value = 0.01
    regularization = l2(reg_value)

    num_users = len(user_encoder.classes_)
    num_novels = len(novel_encoder.classes_)
    num_categories = len(category_encoder.classes_)

    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    novel_input = Input(shape=(1,), dtype='int32', name='novel_input')
    category_input = Input(shape=(1,), dtype='int32', name='category_input')

    user_embedding_gmf = Embedding(input_dim=num_users, output_dim=embedding_dim, name='user_embedding_gmf')(user_input)
    novel_embedding_gmf = Embedding(input_dim=num_novels, output_dim=embedding_dim, name='novel_embedding_gmf')(novel_input)
    category_embedding_gmf = Embedding(input_dim=num_categories, output_dim=embedding_dim, name='category_embedding_gmf')(category_input)

    user_vec_gmf = Flatten()(user_embedding_gmf)
    novel_vec_gmf = Flatten()(novel_embedding_gmf)
    category_vec_gmf = Flatten()(category_embedding_gmf)

    gmf = Multiply()([user_vec_gmf, novel_vec_gmf, category_vec_gmf])

    user_embedding_mlp = Embedding(input_dim=num_users, output_dim=embedding_dim, name='user_embedding_mlp')(user_input)
    novel_embedding_mlp = Embedding(input_dim=num_novels, output_dim=embedding_dim, name='novel_embedding_mlp')(novel_input)
    category_embedding_mlp = Embedding(input_dim=num_categories, output_dim=embedding_dim, name='category_embedding_mlp')(category_input)

    user_vec_mlp = Flatten()(user_embedding_mlp)
    novel_vec_mlp = Flatten()(novel_embedding_mlp)
    category_vec_mlp = Flatten()(category_embedding_mlp)

    mlp = Concatenate()([user_vec_mlp, novel_vec_mlp, category_vec_mlp])
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

    model = Model(inputs=[user_input, novel_input, category_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mae', 'mape', tf.keras.metrics.RootMeanSquaredError()])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

    history = model.fit([train_users, train_novels, train_categories], train_ratings,
                        batch_size=256, epochs=100, verbose=1,
                        validation_data=([test_users, test_novels, test_categories], test_ratings),
                        callbacks=[early_stopping, reduce_lr])

    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'recommendation_model.h5'))
    joblib.dump(user_encoder, os.path.join(model_dir, 'user_encoder.pkl'))
    joblib.dump(novel_encoder, os.path.join(model_dir, 'novel_encoder.pkl'))
    joblib.dump(category_encoder, os.path.join(model_dir, 'category_encoder.pkl'))

    eval_results = model.evaluate([test_users, test_novels, test_categories], test_ratings, verbose=1)
    response = {
        "Test Loss": eval_results[0],
        "Test MAE": eval_results[1],
        "Test MAPE": eval_results[2],
        "Test RMSE": eval_results[3]
    }

    return response
