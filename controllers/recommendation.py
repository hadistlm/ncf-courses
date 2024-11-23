import tensorflow as tf
import numpy as np
from datetime import datetime
from tensorflow import convert_to_tensor

# Get course recommendations
def recommend_courses(user_vector, course_vectors, model_location = './saved/best_model_2.keras'):
  print(f'Start calculate recommendation at {datetime.now().strftime("%H:%M:%S")}')
  print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

  try:
    # Load the trained model once
    model = tf.keras.models.load_model(model_location)

    # Extract all course vectors as a batch
    course_vectors_np = np.array([course['vector'] for course in course_vectors], dtype=np.float32)

    # Repeat the user vector for the entire batch of course vectors
    user_vector_np = np.repeat([user_vector], len(course_vectors), axis=0)

    # Make batch predictions
    predictions = model.predict([user_vector_np, course_vectors_np], batch_size=128)  # Use batching for efficiency

    # Combine predictions with course metadata
    recommendations = [
      {
        'courseId': course['courseId'],
        'title': course['title'],
        'link': course['link'],
        'score': score
      }
      for course, score in zip(course_vectors, predictions.flatten())  # Flatten predictions
    ]
  except Exception as e:
    print(f"Error in recommendation: {e}")

  print(f'End calculate recommendation at {datetime.now().strftime("%H:%M:%S")}')

  # Return top 5 predictions sorted by score
  return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:5]

def recommendation_display(recommendations):
  if not recommendations:
    print("There's no suitable recommendation data")
    return
  
  print("\nTop 5 recommendations for user id 5: ")
  for i, recommendation in enumerate(recommendations, start=1):
    print(f"Recommendation {i}:")
    print(f"  Course ID: {recommendation['courseId']}")
    print(f"  Title: {recommendation['title']}")
    print(f"  Link: {recommendation['link']}")
    print(f"  Score: {recommendation['score']:.4f}")
    print("-" * 30)  # Separator between entries