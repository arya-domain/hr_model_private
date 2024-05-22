import pandas as pd
import numpy as np
import tensorflow as tf
from keras.losses import BinaryCrossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.preprocessor import *
from utils.encoders import *
from models.model import *
import os
import warnings
warnings.filterwarnings("ignore")

def sheets_merger(sheets, classes):
  all_dfs = []

  for sheet_name, df in sheets.items():
    df['Class'] = list(classes.score)
    all_dfs.append(df)

  merged_df = pd.concat(all_dfs, ignore_index=True)

  return merged_df.drop(columns=['Class']), merged_df['Class']


def trainer(train_sheets, y_train, test_sheets, y_test):
  
  smote = SMOTE(random_state=42)
  
  train_df, y_train = sheets_merger(train_sheets, y_train)
  test_df, y_test = sheets_merger(test_sheets, y_test)

  X, y_ = smote.fit_resample(train_df, y_train)

  X_train, y_train = np.array(X), np.array(y_)
  X_test, y_test = np.array(test_df), np.array(y_test)

  # Reshape
  # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
  # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
  y_train = y_train.reshape(-1, 1)
  y_test = y_test.reshape(-1, 1)

  print("X_train shape:", X_train.shape)
  print("X_test shape:", X_test.shape)
  print("y_train shape:", y_train.shape)
  print("y_test shape:", y_test.shape)

  # Initialize
  model = classifier()
  current_dir = os.getcwd()
  model_name = current_dir + '/HR_APP/weights/model.keras'

  # Checkpoint
  checkpoint_callback = ModelCheckpoint(
                            filepath=model_name,
                            monitor='val_accuracy',
                            save_best_only=True,
                            mode='max',
                            verbose=0
                        )

  early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

  # Train the classifier
  history = model.fit(X_train, y_train, epochs=50, batch_size=16,
            validation_data=(X_test, y_test), verbose=1,
            callbacks=[checkpoint_callback, early_stopping])

  model = tf.keras.models.load_model(model_name)

  # Predict on the test set
  test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0,)

  # Print classification report
  print(f"Model Validation accuracy : ", test_accuracy*100, "%")

def train(train_path, test_path, class_path, criteria_score):

  y = pd.read_csv(class_path)
  for i in range(len(y)):
    y['score'][i] = 1 if y['score'][i] > criteria_score else 0
  
  train_sheet, y_train = sheet_loader(train_path, y)
  train_sheet = scaler(train_sheet)
  
  test_sheet, y_test = sheet_loader(test_path, y)
  test_sheet = scaler(test_sheet)
  
  trainer(train_sheet, y_train, test_sheet, y_test)

if __name__ == "__main__":
  current_dir = os.getcwd()
  train_path = current_dir + '/HR_APP/Datasets/hr_dataset[1-10].xlsx'
  test_path = current_dir + '/HR_APP/Datasets/hr_dataset[11-20].xlsx'
  class_path = current_dir + "/HR_APP/Datasets/score.csv"
  criteria_score = 70
  train(train_path, test_path, class_path, criteria_score)

