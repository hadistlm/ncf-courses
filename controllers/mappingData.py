import tensorflow as tf
import numpy as np
from datetime import datetime

# Generate sync data for training
def generate_sync_data(user_vectors, course_vectors):
  print(f'Start mapping data at {datetime.now().strftime("%H:%M:%S")}')

  # Extract user and course vectors as NumPy arrays
  user_vectors_np = np.array([user['vector'] for user in user_vectors], dtype=np.float32)
  course_vectors_np = np.array([course['vector'] for course in course_vectors], dtype=np.float32)
  
  # Compute interaction labels
  ys = []
  for user in user_vectors:
    user_interests = set([interest.strip().lower() for interest in user.get('interests', [])])
    labels = np.array([
      1 if course['category'] in user_interests else 0
      for course in course_vectors
    ], dtype=np.float32)
    ys.append(labels)
  
  # Flatten `ys` into a single vector
  ys_np = np.concatenate(ys)
  
  # Broadcast user and course vectors to create interaction pairs
  xs_users_np = np.repeat(user_vectors_np, len(course_vectors), axis=0)
  xs_courses_np = np.tile(course_vectors_np, (len(user_vectors), 1))

  print(f'End process data at {datetime.now().strftime("%H:%M:%S")}')
  
  # Convert back to TensorFlow tensors
  return {
    'xsUsers': tf.convert_to_tensor(xs_users_np, dtype=tf.float32),
    'xsCourses': tf.convert_to_tensor(xs_courses_np, dtype=tf.float32),
    'ys': tf.convert_to_tensor(ys_np, dtype=tf.float32)
  }
