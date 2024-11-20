import tensorflow as tf
from tensorflow import convert_to_tensor

# Get course recommendations
def recommend_courses(user_vector, course_vectors, model_location = './saved/best_model_2.keras'):
  model = tf.keras.models.load_model(model_location)
  predictions = []
  for course_vector in course_vectors:
    prediction = model.predict([
      convert_to_tensor([user_vector]),
      convert_to_tensor([course_vector['vector']])
    ])
    predictions.append({
      'courseId': course_vector['courseId'],
      'title': course_vector['title'],
      'score': prediction[0][0]
    })

  return sorted(predictions, key=lambda x: x['score'], reverse=True)[:5]