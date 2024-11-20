import orjson as json
import tensorflow as tf
import numpy as np
from datetime import datetime
import re
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
from sklearn.metrics import precision_score, recall_score, f1_score

# Helper function to extract numeric value from a string
def extract_numeric_value(text):
    if isinstance(text, (str, int, float)):
        if isinstance(text, (int, float)):
            return int(text)  # Convert numbers directly to int
        numbers = re.findall(r'\d+', str(text))  # Ensure it's treated as a string
        return int(numbers[0]) if numbers else 0
    else:
        print(f"Unexpected input to extract_numeric_value: {text}")
        return 0

# Load courses from JSON with error handling
async def load_courses(file_path='./course.json'):
    with open(file_path, 'rb') as f:
        json_courses = json.loads(f.read())

    # Ensure the data is a list of dictionaries
    if not isinstance(json_courses, list):
        raise ValueError("Expected a list of courses")
    
    courses = []
    for i, d in enumerate(json_courses):
        if not isinstance(d, dict):
            print(f"Unexpected course data format at index {i}: {d}")  # Debugging line to catch bad format
            continue  # Skip invalid data
        try:
            rating_str = str(d.get('Rating', '')).replace('stars', '').strip()
            rating = float(rating_str) if rating_str else 0.0
            viewers_str = str(d.get('Number of viewers', '')).strip()
            view = extract_numeric_value(viewers_str)

            courses.append({
                'id': d.get('ID', f'unknown_id_{i}'),  # Provide a fallback ID if missing
                'category': d.get('Category', 'Unknown'),
                'rating': rating,
                'view': view,
                'title': d.get('Title', 'Unknown'),
            })
        except ValueError as e:
            print(f"Error processing course {d.get('Title', 'Unknown')} at index {i}: {e}")
        except Exception as e:
            print(f"Unexpected error processing course {d} at index {i}: {e}")  # Catch all errors

    return courses

# Load users from JSON
async def load_users(file_path='./user.json'):
    try:
        with open(file_path, 'rb') as f:
            json_users = json.loads(f.read())
        
        # Ensure the data is a list of dictionaries
        if not isinstance(json_users, list):
            raise ValueError("Expected a list of users")

        return json_users
    except FileNotFoundError as e:
        print(f"User file not found: {e}")
        return []
    except ValueError as e:
        print(f"Error loading user data: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error loading users: {e}")
        return []

# Preprocess the data
def preprocess_data(users, courses):
    user_vectors = []

    for i, user in enumerate(users):
        if not isinstance(user, dict):
            print(f"Unexpected user data format at index {i}: {user}")  # Debugging line to catch bad format
            continue  # Skip invalid data
        
        interests_encoded = [
            1 if any(interest.lower() in course.get('category', '').lower() for interest in user['interests']) else 0
            for course in courses
        ]
        trimmed_interests = interests_encoded[:len(courses)]
        user_vectors.append({
            'userId': user.get('id', f'unknown_user_id_{i}'),  # Provide a fallback ID if missing
            'vector': trimmed_interests,
            'interests': user.get('interests', [])
        })

    course_vectors = []
    for i, course in enumerate(courses):
        category = course.get('category')
        if not category:
            continue

        rating = float(course.get('rating', 0)) / 5
        viewers = extract_numeric_value(course.get('view', 0)) / 10000

        course_vectors.append({
            'courseId': course.get('id', f'unknown_course_id_{i}'),
            'category': category.lower(),
            'vector': [rating, viewers]
        })

    return {'userVectors': user_vectors, 'courseVectors': course_vectors}

# Build the Neural Collaborative Filtering (NCF) model
def build_ncf_model(user_vector_length, course_vector_length):
    user_input = tf.keras.Input(shape=(user_vector_length,))
    course_input = tf.keras.Input(shape=(course_vector_length,))

    # User embedding with Batch Normalization
    user_embedding = tf.keras.layers.Dense(8, kernel_regularizer=regularizers.l2(0.01))(user_input)
    user_embedding = tf.keras.layers.BatchNormalization()(user_embedding)
    user_embedding = tf.keras.layers.Activation('relu')(user_embedding)
    user_embedding = tf.keras.layers.Dropout(0.1)(user_embedding)

    # Course embedding with Batch Normalization
    course_embedding = tf.keras.layers.Dense(8, kernel_regularizer=regularizers.l2(0.01))(course_input)
    course_embedding = tf.keras.layers.BatchNormalization()(course_embedding)
    course_embedding = tf.keras.layers.Activation('relu')(course_embedding)
    course_embedding = tf.keras.layers.Dropout(0.1)(course_embedding)

    dot_product = tf.keras.layers.Dot(axes=1)([user_embedding, course_embedding])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dot_product)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model = tf.keras.Model(inputs=[user_input, course_input], outputs=output)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Generate mock data for training
def generate_mock_data(user_vectors, course_vectors):
    xs_users = []
    xs_courses = []
    ys = []

    for user in user_vectors:
        normalized_interests = [interest.strip().lower() for interest in user.get('interests', [])]
        
        # Iterate through all courses to determine interactions
        for course in course_vectors:
            is_positive = any(
                interest in course['category'].strip().lower() for interest in normalized_interests
            )

            xs_users.append(user['vector'])
            xs_courses.append(course['vector'])
            ys.append(1 if is_positive else 0)  # Positive if matches, otherwise negative

    return {
        'xsUsers': tf.convert_to_tensor(xs_users, dtype=tf.float32),
        'xsCourses': tf.convert_to_tensor(xs_courses, dtype=tf.float32),
        'ys': tf.convert_to_tensor(ys, dtype=tf.float32)
    }

# Train the model with early stopping and learning rate reduction
async def train_model(model, xs_users, xs_courses, ys):
    print(f'Start training model at {datetime.now().strftime("%H:%M:%S")}')

    class_weight = {0: 1.0, 1: 5.0}
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    checkpoint = ModelCheckpoint('./saved/best_model_2.keras', save_best_only=True, monitor='val_loss')

    history = model.fit(
        [xs_users, xs_courses],
        ys,
        epochs=50,
        batch_size=256,
        validation_split=0.2,
        class_weight=class_weight,
        callbacks=[reduce_lr, early_stopping, checkpoint]
    )

    # Save the trained model after initial training
    model.save('./saved/my_model_2.keras')

    # Calculate Precision, Recall, F1-Score
    y_pred = model.predict([xs_users, xs_courses])
    y_true = ys.numpy()

    # Rate Loss & Accuracy
    print(f"Final Training Loss: {history.history['loss'][-1]}")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]}")

    # Check the distribution of predicted values
    print("Predicted values:", y_pred.flatten())
    
    y_pred_classes = (y_pred > 0.5).astype(int)  # Threshold at 0.5

    # Inspect class predictions
    print("Predicted classes:", y_pred_classes.flatten())
    print("True labels:", y_true)

    # Add zero_division=0 to avoid UndefinedMetricWarning
    precision = precision_score(y_true, y_pred_classes, zero_division=0)
    recall = recall_score(y_true, y_pred_classes, zero_division=0)
    f1 = f1_score(y_true, y_pred_classes, zero_division=0)

    print(f'Precision: {precision}, Recall: {recall}, F1-Score: {f1}')

    # Check number of positive predictions
    print("Number of positive predictions:", np.sum(y_pred_classes))

    return history

# Retrain the model after saving
async def retrain_model(xs_users, xs_courses, ys):
    print("Retraining the model...")

    # Reload the saved model
    model = tf.keras.models.load_model('./saved/my_model_2.keras')

    # Optionally modify hyperparameters like learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),  # New learning rate
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Retrain the model
    history_retrain = model.fit(
        [xs_users, xs_courses],  # Use new or same data
        ys,
        epochs=50,  # Number of epochs for retraining
        batch_size=256,
        validation_split=0.2,
        class_weight={0: 1.0, 1: 5.0},
        callbacks=[
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
            EarlyStopping(monitor='val_loss', patience=5)
        ]
    )

    return history_retrain

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
        # Load and validate data
        courses = await load_courses()
        users = await load_users()

        # Ensure that the loaded data is valid
        if not courses or not users:
            print("No valid courses or users data found.")
            return

        data = preprocess_data(users, courses)
        user_vectors, course_vectors = data['userVectors'], data['courseVectors']

        mock_data = generate_mock_data(user_vectors, course_vectors)
        xs_users, xs_courses, ys = mock_data['xsUsers'], mock_data['xsCourses'], mock_data['ys']

        user_vector_length = len(user_vectors[0]['vector']) if user_vectors else 0
        course_vector_length = len(course_vectors[0]['vector']) if course_vectors else 0
        model = build_ncf_model(user_vector_length, course_vector_length)

        # Train the model
        results = await train_model(model, xs_users, xs_courses, ys)

        # Retrain the model after saving
        # await retrain_model(xs_users, xs_courses, ys)

        # Show training chart result
        plot_training_history(results)

        # Example recommendation for first user
        # recommendations = recommend_courses(model, user_vectors[1]['vector'], course_vectors)
        # print("Top 5 recommendations for user 1:", recommendations)

    except Exception as e:
        print(f"Error in main: {e}")

# Run the script
import asyncio
asyncio.run(main())
