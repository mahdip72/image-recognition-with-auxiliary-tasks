import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import adabelief_tf
import numpy as np
import pandas as pd
import tensorflow_addons as tfa
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from supervised_only import SupervisedModel
from common import train_preprocessing, valid_preprocessing

tf.random.set_seed(111)
number_of_tiles = 4

autotune = tf.data.experimental.AUTOTUNE
tf.keras.backend.clear_session()
mixed_precision.set_global_policy('mixed_float16')
# dataset dir includes label and Manually_Annotated_Images folders
dataset_dir = os.path.abspath('/media/aryan/DATA/Datasets/AffectNet')
model_name = 'test'
# weights_path = f"./results/{model_name}/weights/weights.h5"
weights_path = ''
initial_epoch = 0

if not os.path.exists(f"./results/{model_name}"):
    os.makedirs(f"./results/{model_name}")
    os.makedirs(f"./results/{model_name}/weights")

train_csv_dir = os.path.join(dataset_dir, 'label', 'training.csv')
valid_csv_dir = os.path.join(dataset_dir, 'label', 'validation.csv')
train_csv_data = pd.read_csv(train_csv_dir)
train_csv_data = train_csv_data[~train_csv_data['subDirectory_filePath'].str.contains(".tif", case=False)]
train_csv_data = train_csv_data[~train_csv_data['subDirectory_filePath'].str.contains(".bmp", case=False)]
train_csv_data = train_csv_data[train_csv_data['expression'] <= 7]
train_csv_data['subDirectory_filePath'] = os.path.join(dataset_dir,
                                                       'Manually_Annotated_Images/Manually_Annotated_Images/') + \
                                          train_csv_data['subDirectory_filePath'].astype(str)
valid_csv_data = pd.read_csv(valid_csv_dir)
valid_csv_data = valid_csv_data[~valid_csv_data['subDirectory_filePath'].str.contains(".tif", case=False)]
valid_csv_data = valid_csv_data[valid_csv_data['expression'] <= 7]
valid_csv_data['subDirectory_filePath'] = os.path.join(dataset_dir,
                                                       'Manually_Annotated_Images/Manually_Annotated_Images/') + \
                                          valid_csv_data['subDirectory_filePath'].astype(str)
batch = 64
input_data = train_csv_data.iloc[:, 0]
train_labels = train_csv_data.iloc[:, 6]

train_ds = tf.data.Dataset.from_tensor_slices((np.array(input_data), np.array(train_labels)))
train_ds = train_ds.shuffle(int(len(train_csv_data))).map(train_preprocessing, num_parallel_calls=autotune)
train_ds = train_ds.batch(batch).prefetch(autotune)
valid_labels = valid_csv_data.iloc[:, 6]
input_data = valid_csv_data.iloc[:, 0]
valid_ds = tf.data.Dataset.from_tensor_slices((np.array(input_data), np.array(valid_labels)))
valid_ds = valid_ds.map(valid_preprocessing, num_parallel_calls=autotune).batch(batch).prefetch(autotune)


model = SupervisedModel(n_tiles=number_of_tiles)
op = adabelief_tf.AdaBeliefOptimizer(learning_rate=0.0001,
                                     print_change_log=False)

losses = {
    'emotion': tf.keras.losses.CategoricalCrossentropy(),
}
for i in range(number_of_tiles):
    losses[f'puzzle_{i + 1}'] = tf.keras.losses.CategoricalCrossentropy()
metrics = {
    'emotion': [
        tf.keras.metrics.CategoricalAccuracy('acc'),
        tfa.metrics.F1Score(8, name='f1', average='macro'),
    ]}
for i in range(number_of_tiles):
    metrics[f'puzzle_{i + 1}'] = [tf.keras.metrics.CategoricalAccuracy(name='acc')]
model.compile_(loss_fns=losses,
               optimizer=op,
               metrics=metrics,
               # loss_weights=loss_weights
               run_eagerly=True
               )

checkpoint = ModelCheckpoint(f"./results/{model_name}/weights/weights.h5",
                             monitor='val_loss', verbose=0,
                             save_best_only=True, save_weights_only=True,
                             mode='min', save_freq='epoch')
csv_callback = CSVLogger(f"./results/{model_name}/training_log.csv",
                         append=False)


def lr_scheduler(epoch, lr):
    if epoch == 20 or epoch == 40:
        lr = lr / 10
        return lr
    else:
        return lr


lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
callbacks_list = [
    checkpoint,
    lr_callback,
    csv_callback,
    tf.keras.callbacks.TensorBoard(log_dir=f"./results/{model_name}/log",
                                   update_freq='epoch')
]

if weights_path:
    model(tf.ones((1, 224, 224, 3)))
    model.load_weights(weights_path)

model.fit(train_ds,
          validation_data=valid_ds,
          callbacks=callbacks_list,
          verbose=1,
          epochs=80,
          initial_epoch=initial_epoch)
# model.evaluate(valid_ds, verbose=1)
print('finish')
