import tensorflow as tf
from datetime import datetime

# Generate sync data for training
def generate_sync_data(user_vectors, course_vectors):
  print(f'Start mapping data at {datetime.now().strftime("%H:%M:%S")}')

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

  returned = {
    'xsUsers': tf.convert_to_tensor(xs_users, dtype=tf.float32),
    'xsCourses': tf.convert_to_tensor(xs_courses, dtype=tf.float32),
    'ys': tf.convert_to_tensor(ys, dtype=tf.float32)
  }

  print(f'End process data at {datetime.now().strftime("%H:%M:%S")}')

  return returned