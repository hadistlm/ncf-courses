import tensorflow as tf
from datetime import datetime

# Build the Neural Collaborative Filtering (NCF) model
def build_foundation_model(user_vectors, course_vectors, learningRT=1e-3, matrix_size = 6, layer_weight = 0.01):
  print(f'Start build model data at {datetime.now().strftime("%H:%M:%S")}')

  user_vector_length = len(user_vectors[0]['vector']) if user_vectors else 0
  course_vector_length = len(course_vectors[0]['vector']) if course_vectors else 0
  user_input = tf.keras.Input(shape=(user_vector_length,))
  course_input = tf.keras.Input(shape=(course_vector_length,))

  # User embedding with Batch Normalization
  user_embedding = tf.keras.layers.Dense(matrix_size, kernel_regularizer=tf.keras.regularizers.l2(layer_weight))(user_input)
  user_embedding = tf.keras.layers.BatchNormalization()(user_embedding)
  user_embedding = tf.keras.layers.Activation('relu')(user_embedding)
  user_embedding = tf.keras.layers.Dropout(0.1)(user_embedding)

  # Course embedding with Batch Normalization
  course_embedding = tf.keras.layers.Dense(matrix_size, kernel_regularizer=tf.keras.regularizers.l2(layer_weight))(course_input)
  course_embedding = tf.keras.layers.BatchNormalization()(course_embedding)
  course_embedding = tf.keras.layers.Activation('relu')(course_embedding)
  course_embedding = tf.keras.layers.Dropout(0.1)(course_embedding)

  dot_product = tf.keras.layers.Dot(axes=1)([user_embedding, course_embedding])
  output = tf.keras.layers.Dense(1, activation='sigmoid')(dot_product)

  optimizer = tf.keras.optimizers.Adam(learning_rate=learningRT)

  model = tf.keras.Model(inputs=[user_input, course_input], outputs=output)
  model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

  print(f'End build model data at {datetime.now().strftime("%H:%M:%S")}')

  return model