import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load data
info_tourism = pd.read_csv(r"C:\Users\fabia\Documents\Tugas Bangkit\Capstone Deployment\tourism_with_id.csv")
tourism_rating = pd.read_csv(r"C:\Users\fabia\Documents\Tugas Bangkit\Capstone Deployment\tourism_rating.csv")

# Identify outliers and clip the "Price" column
Q1 = info_tourism['Price'].quantile(0.25)
Q3 = info_tourism['Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Clip outliers
info_tourism['Clipped_Price'] = info_tourism['Price'].clip(lower=lower_bound, upper=upper_bound)

# Normalize the clipped Price column
min_price = min(info_tourism['Clipped_Price'])
max_price = max(info_tourism['Clipped_Price'])
info_tourism['Normalized_Price'] = (info_tourism['Clipped_Price'] - min_price) / (max_price - min_price)

# Encode the City and Category columns
city_encoder = LabelEncoder()
info_tourism['City_Encoded'] = city_encoder.fit_transform(info_tourism['City'])
category_encoder = LabelEncoder()
info_tourism['Category_Encoded'] = category_encoder.fit_transform(info_tourism['Category'])

city_to_city_encoded = {x: i for i, x in enumerate(city_encoder.classes_)}
city_encoded_to_city = {i: x for i, x in enumerate(city_encoder.classes_)}
category_to_category_encoded = {x: i for i, x in enumerate(category_encoder.classes_)}
category_encoded_to_category = {i: x for i, x in enumerate(category_encoder.classes_)}

# Merge city_encoded, category_encoded, and Price into tourism_rating
tourism_rating = pd.merge(tourism_rating, info_tourism[['Place_Id', 'City_Encoded', 'Category_Encoded', 'Clipped_Price', 'Normalized_Price']], on='Place_Id', how='left')

# Create DataFrame for tourism
tourism_new = pd.DataFrame({
    "id": info_tourism.Place_Id.tolist(),
    "name": info_tourism.Place_Name.tolist(),
    "category": info_tourism.Category.tolist(),
    "description": info_tourism.Description.tolist(),
    "city": info_tourism.City.tolist(),
    "city_category": info_tourism[['City', 'Category']].agg(' '.join, axis=1).tolist(),
    "city_encoded": info_tourism['City_Encoded'].tolist(),
    "category_encoded": info_tourism['Category_Encoded'].tolist(),
    "lat": info_tourism['Lat'].tolist(),
    "lng": info_tourism['Long'].tolist(),
    "price": info_tourism['Price'].tolist(),
    "clipped_price": info_tourism['Clipped_Price'].tolist(),
    "normalized_price": info_tourism['Normalized_Price'].tolist()
})

# Collaborative Filtering Model
df = tourism_rating
user_ids = df.User_Id.unique().tolist()
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
place_ids = df.Place_Id.unique().tolist()
place_to_place_encoded = {x: i for i, x in enumerate(place_ids)}
place_encoded_to_place = {i: x for i, x in enumerate(place_ids)}

df['user'] = df.User_Id.map(user_to_user_encoded)
df['place'] = df.Place_Id.map(place_to_place_encoded)
num_users = len(user_to_user_encoded)
num_place = len(place_encoded_to_place)
num_cities = len(city_to_city_encoded)
num_categories = len(category_to_category_encoded)

# Normalize ratings
df['Place_Ratings'] = df['Place_Ratings'].values.astype(np.float32)
min_rating = min(df['Place_Ratings'])
max_rating = max(df['Place_Ratings'])

# Shuffle and split data
df = df.sample(frac=1, random_state=42)
x = df[['user', 'place', 'City_Encoded', 'Category_Encoded', 'Normalized_Price']].values
y = df['Place_Ratings'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = x[:train_indices], x[train_indices:], y[:train_indices], y[train_indices:]

# Define the RecommenderNet model
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_place, num_cities, num_categories, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = layers.Embedding(num_users, embedding_size,
                                               embeddings_initializer='he_normal',
                                               embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.user_bias = layers.Embedding(num_users, 1)
        self.place_embedding = layers.Embedding(num_place, embedding_size,
                                                embeddings_initializer='he_normal',
                                                embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.place_bias = layers.Embedding(num_place, 1)
        self.city_embedding = layers.Embedding(num_cities, embedding_size,
                                               embeddings_initializer='he_normal',
                                               embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.category_embedding = layers.Embedding(num_categories, embedding_size,
                                                   embeddings_initializer='he_normal',
                                                   embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.price_dense = layers.Dense(embedding_size)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        place_vector = self.place_embedding(inputs[:, 1])
        place_bias = self.place_bias(inputs[:, 1])
        city_vector = self.city_embedding(inputs[:, 2])
        category_vector = self.category_embedding(inputs[:, 3])
        price_vector = self.price_dense(tf.expand_dims(inputs[:, 4], axis=-1))

        dot_user_place = tf.tensordot(user_vector, place_vector, 2)
        dot_user_city = tf.tensordot(user_vector, city_vector, 2)
        dot_user_category = tf.tensordot(user_vector, category_vector, 2)
        dot_place_city = tf.tensordot(place_vector, city_vector, 2)
        dot_place_category = tf.tensordot(place_vector, category_vector, 2)
        dot_user_price = tf.tensordot(user_vector, price_vector, 2)
        dot_place_price = tf.tensord_user_price = tf.tensordot(place_vector, price_vector, 2)

        x = (dot_user_place + user_bias + place_bias +
             dot_user_city + dot_user_category +
             dot_place_city + dot_place_category +
             dot_user_price + dot_place_price)
        return tf.nn.sigmoid(x)

# Early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_root_mean_squared_error',
    patience=10,
    restore_best_weights=True
)

# Compile and train the model
model = RecommenderNet(num_users, num_place, num_cities, num_categories, 100)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])
history = model.fit(x=x_train, y=y_train, batch_size=8, epochs=100, validation_data=(x_val, y_val), callbacks=[early_stopping])

# Plot training history
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('Model Metrics')
plt.ylabel('Root Mean Squared Error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Generate recommendations
user_id = df.User_Id.sample(1).iloc[0]
place_visited_by_user = df[df.User_Id == user_id]
place_not_visited = tourism_new[~tourism_new['id'].isin(place_visited_by_user['Place_Id'].values)][['id', 'city_encoded', 'category_encoded', 'price', 'clipped_price', 'normalized_price']]
place_not_visited = place_not_visited.drop_duplicates()

place_not_visited_encoded = [[place_to_place_encoded.get(p_id), city, category, price, clipped_price, normalized_price] for p_id, city, category, price, clipped_price, normalized_price in place_not_visited.values]
user_encoder = user_to_user_encoded.get(user_id)
user_place_array = np.hstack(([[user_encoder]] * len(place_not_visited_encoded), place_not_visited_encoded))

ratings = model.predict(user_place_array).flatten()
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_place_ids = [place_encoded_to_place.get(place_not_visited_encoded[x][0]) for x in top_ratings_indices]

print('Showing recommendations for user: {}'.format(user_id))
print('===' * 9)
print('Places with high ratings from user')
print('----' * 8)
top_place_user = place_visited_by_user.sort_values(by='Place_Ratings', ascending=False).head(5).Place_Id.values
place_df_rows = tourism_new[tourism_new['id'].isin(top_place_user)]
print(place_df_rows[['name', 'category', 'city', 'description', 'price', 'lat', 'lng']])

print('----' * 8)
print('Top 10 place recommendations')
print('----' * 8)
recommended_place = tourism_new[tourism_new['id'].isin(recommended_place_ids)]
print(recommended_place[['name', 'category', 'city', 'description', 'price', 'lat', 'lng']])

# # Save the model
# model.save(r"C:\Users\fabia\Documents\Tugas Bangkit\Capstone Deployment\place_recommendation2.h5")