import tensorflow as tf
import numpy as np

from datetime import datetime
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import precision_score, recall_score, f1_score

# Train the model with early stopping and learning rate reduction
async def train_model(model, xs_users, xs_courses, ys, preferedBatch = 256):
  print(f'Start training model at {datetime.now().strftime("%H:%M:%S")}')

  try:
    class_weight = {0: 1.0, 1: 5.0}
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    checkpoint = ModelCheckpoint('./saved/best_model_2.keras', save_best_only=True, monitor='val_loss')

    history = model.fit(
      [xs_users, xs_courses],
      ys,
      epochs=50,
      batch_size=preferedBatch,
      validation_split=0.2,
      class_weight=class_weight,
      callbacks=[reduce_lr, early_stopping, checkpoint]
    )

    # Save the trained model after initial training
    model.save('./saved/best_model_2.keras')

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
  except Exception as e:
    print(f"Error in train model: {e}")

  print(f'End training model at {datetime.now().strftime("%H:%M:%S")}')

  return history

# Retrain the model after saving
async def retrain_model(xs_users, xs_courses, ys, learningRT=1e-3, preferedBatch = 256):
  print(f'Start re-training the model at {datetime.now().strftime("%H:%M:%S")}')

  try:
    # Reload the saved model
    model = tf.keras.models.load_model('./saved/best_model_2.keras')

    # Optionally modify hyperparameters like learning rate
    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=learningRT),  # New learning rate
      loss='binary_crossentropy',
      metrics=['accuracy']
    )

    # Retrain the model
    history_retrain = model.fit(
      [xs_users, xs_courses],  # Use new or same data
      ys,
      epochs=50,  # Number of epochs for retraining
      batch_size=preferedBatch,
      validation_split=0.2,
      class_weight={0: 1.0, 1: 5.0},
      callbacks=[
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
        EarlyStopping(monitor='val_loss', patience=5)
      ]
    )
  except Exception as e:
    print(f"Error in re-train model: {e}")

  print(f'End re-training the model at {datetime.now().strftime("%H:%M:%S")}')

  return history_retrain