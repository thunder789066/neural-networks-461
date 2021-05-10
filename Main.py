import csv
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from numpy.random import RandomState
import pandas as pd

#get data from raw url, read from csv file & store in dummy var

orig_data = tf.keras.utils.get_file('aug_train.csv', 'https://raw.githubusercontent.com/thunder789066/neural-networks-461/main/aug_train.csv')
orig_data = pd.read_csv(orig_data)
clean_data = pd.get_dummies(orig_data)

randState = RandomState()

#data divided 70% training, 15% test, 15% validation
training_dataset = clean_data.sample(frac = 0.7, random_state = randState)
other_dataset = clean_data.loc[~clean_data.index.isin(training_dataset.index)]

test_dataset = other_dataset.sample(frac = 0.5, random_state = randState)
validation_dataset = clean_data.loc[~clean_data.index.isin(test_dataset.index)]

training_target = training_dataset.pop('target')
test_target = test_dataset.pop('target')
validation_target = validation_dataset.pop('target')

train_data_subset = tf.data.Dataset.from_tensor_slices((training_dataset.values, training_target.values))
test_data_subset = tf.data.Dataset.from_tensor_slices((test_dataset.values, test_target.values))
validate_data_subset = tf.data.Dataset.from_tensor_slices((validation_dataset.values, validation_target.values))

#compile model
orig_model = tf.keras.Model
orig_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="tanh"),
        tf.keras.layers.Dense(128, activation="tanh"),
        tf.keras.layers.Dense(1, activation="sigmoid")])

orig_model.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics = ['accuracy'])
orig_model.fit(training_dataset, training_target, epochs = 20)
orig_model.evaluate(test_data_subset.batch(128))

