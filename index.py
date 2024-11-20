import orjson as json
import tensorflow as tf
import numpy as np
import re
import logging
from datetime import datetime
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Set up logging
logging.basicConfig(level=logging.INFO)

# Helper function to extract numeric value from a string
def extract_numeric_value(text):
    if isinstance(text, (str, int, float)):
        if isinstance(text, (int, float)):
            return int(text)  # Convert numbers directly to int
        numbers = re.findall(r'\d+', str(text))  # Ensure it's treated as a string
        return int(numbers[0]) if numbers else 0
    else:
        logging.warning(f"Unexpected input to extract_numeric_value: {text}")
        return 0

# Load courses from JSON with error handling
async def load_courses(file_path='./course.json'):
    try:
        with open(file_path, 'rb') as f:
            json_courses = json.loads(f.read())

        courses = []
        for d in json_courses:
            try:
                rating_str = str(d.get('Rating', '')).replace('stars', '').strip()
                rating = float(rating_str) if rating_str else 0.0
                viewers_str = str(d.get('Number of viewers', '')).strip()
                view = extract_numeric_value(viewers_str)

                courses.append({
                    'id': d.get('ID'),
                    'category': d.get('Category', 'Unknown'),
                    'rating': rating,
                    'view': view,
                    'title': d.get('Title', 'Unknown'),
                })
            except (ValueError, TypeError) as e:
                logging.warning(f"Error processing course {d.get('Title', 'Unknown')}: {e}")

        return courses
    except FileNotFoundError as e:
        logging.error(f"Course file not found: {e}")
        return []

# Load users from JSON
async def load_users(file_path='./user.json'):
    try:
        with open(file_path, 'rb') as f:
            return json.loads(f.read())
    except FileNotFoundError as e:
        logging.error(f"User file not found: {e}")
        return []

# Normalize features
def normalize_data(user_vectors, course_vectors):
    for user in user_vectors:
        user['vector'] = [float(x) for x in user['vector']]
    
    for course in course_vectors:
        course['vector'] = [float(x) for x in course['vector']]
    
    return user_vectors, course_vectors

# Preprocess the data
def preprocess_data(users, courses):
    user_vectors = [
        {
            'userId': user['id'],
            'vector': [
                1 if any(interest.lower() in course.get('category', '').lower() for interest in user['interests']) else 0
                for course in courses
            ],
            'interests': user['interests']
        }
        for user in users
    ]

    course_vectors = [
        {
            'courseId': course.get('id'),
            'category': course.get('category', '').lower(),
            'vector': [
                float(course.get('rating', 0)) / 5,
                extract_numeric_value(course.get('view', 0)) / 10000
            ]
        }
        for course in courses
    ]

    return normalize_data(user_vectors, course_vectors)

# Build the Neural Collaborative Filtering (NCF) model
def build_ncf_model(user_vector_length, course_vector_length):
    user_input = tf.keras.Input(shape=(user_vector_length,))
    course_input = tf.keras.Input(shape=(course_vector_length,))

    user_embedding = tf.keras.layers.Dense(8, kernel_regularizer=regularizers.l2(0.01))(user_input)
    user_embedding = tf.keras.layers.BatchNormalization()(user_embedding)
    user_embedding = tf.keras.layers.Activation('relu')(user_embedding)
    user_embedding = tf.keras.layers.Dropout(0.1)(user_embedding)

    course_embedding = tf.keras.layers.Dense(8, kernel_regularizer=regularizers.l2(0.01))(course_input)
    course_embedding = tf.keras.layers.BatchNormalization()(course_embedding)
    course_embedding = tf.keras.layers.Activation('relu')(course_embedding)
    course_embedding = tf.keras.layers.Dropout(0.1)(course_embedding)

    dot_product = tf.keras.layers.Dot(axes=1)([user_embedding, course_embedding])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dot_product)

    model = tf.keras.Model(inputs=[user_input, course_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Generate mock data for training
def generate_mock_data(user_vectors, course_vectors):
    xs_users, xs_courses, ys = [], [], []

    for user in user_vectors:
        normalized_interests = [interest.strip().lower() for interest in user['interests']]
        
        for course in course_vectors:
            is_positive = any(interest in course['category'] for interest in normalized_interests)

            xs_users.append(user['vector'])
            xs_courses.append(course['vector'])
            ys.append(1 if is_positive else 0)

    return {
        'xsUsers': tf.convert_to_tensor(xs_users, dtype=tf.float32),
        'xsCourses': tf.convert_to_tensor(xs_courses, dtype=tf.float32),
        'ys': tf.convert_to_tensor(ys, dtype=tf.float32)
    }

# Train the model
async def train_model(model, xs_users, xs_courses, ys):
    logging.info(f'Start training model at {datetime.now().strftime("%H:%M:%S")}')

    class_weight = {0: 1.0, 1: 5.0}
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
        EarlyStopping(monitor='val_loss', patience=10),
        ModelCheckpoint('./saved/best_model.keras', save_best_only=True, monitor='val_loss')
    ]

    history = model.fit(
        [xs_users, xs_courses], ys,
        epochs=50,
        batch_size=256,
        validation_split=0.2,
        class_weight=class_weight,
        callbacks=callbacks
    )

    logging.info(f"Final Training Loss: {history.history['loss'][-1]}")
    logging.info(f"Final Training Accuracy: {history.history['accuracy'][-1]}")

    # Calculate Precision, Recall, F1-Score
    y_pred = model.predict([xs_users, xs_courses])
    y_true = ys.numpy()
    y_pred_classes = (y_pred > 0.5).astype(int)

    precision = precision_score(y_true, y_pred_classes, zero_division=0)
    recall = recall_score(y_true, y_pred_classes, zero_division=0)
    f1 = f1_score(y_true, y_pred_classes, zero_division=0)

    logging.info(f'Precision: {precision}, Recall: {recall}, F1-Score: {f1}')
    logging.info("Number of positive predictions: %d", np.sum(y_pred_classes))

    plot_training_history(history)

# Plotting function for loss and accuracy
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Get course recommendations
def recommend_courses(model, user_vector, course_vectors):
    predictions = []
    for course_vector in course_vectors:
        prediction = model.predict([tf.convert_to_tensor([user_vector]), tf.convert_to_tensor([course_vector['vector']])])
        predictions.append({'courseId': course_vector['courseId'], 'title': course_vector['title'], 'score': prediction[0][0]})

    return sorted(predictions, key=lambda x: x['score'], reverse=True)[:5]

# Bootstrap the process
async def main():
    try:
        courses = await load_courses()
        users = await load_users()

        data = preprocess_data(users, courses)
        user_vectors, course_vectors = data

        mock_data = generate_mock_data(user_vectors, course_vectors)
        xs_users, xs_courses, ys = mock_data['xsUsers'], mock_data['xsCourses'], mock_data['ys']
    
        model = build_ncf_model(len(user_vectors[0]['vector']), len(course_vectors[0]['vector']))

        await train_model(model, xs_users, xs_courses, ys)

        # Get recommendations for the first user
        user = users[1]
        user_vector = next(u['vector'] for u in user_vectors if u['userId'] == user['id'])
        recommendations = recommend_courses(model, user_vector, course_vectors)

        print(f"Recommendations for {user['name']}:")
        for rec in recommendations:
            course = next(c for c in courses if c['id'] == rec['courseId'])
            print(f"- {course['title']} (Score: {rec['score']:.4f})")

    except Exception as err:
        logging.error(f"Error: {err}")

# Run the script
import asyncio
asyncio.run(main())
