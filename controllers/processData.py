from datetime import datetime
from helpers.sanitizeNum import extract_numeric_value

# Preprocess the data
def preprocess_data(users, courses):
  print(f'Start process data at {datetime.now().strftime("%H:%M:%S")}')

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
      'title': course.get('title'),
      'category': category.lower(),
      'vector': [rating, viewers]
    })

  print(f'End process data at {datetime.now().strftime("%H:%M:%S")}')

  return {'userVectors': user_vectors, 'courseVectors': course_vectors}