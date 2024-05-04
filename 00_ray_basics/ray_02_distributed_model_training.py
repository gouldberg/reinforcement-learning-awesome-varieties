# -*- coding: utf-8 -*-

import os, sys
import ray

import json
import numpy as np
import tensorflow as tf


# ----------
# REFERENCE
# https://docs.ray.io/en/latest/ray-overview/index.html#


# -----------------------------------------------------------------------------------------------------------
# data set and model
# -----------------------------------------------------------------------------------------------------------

def mnist_dataset(batch_size):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    # The `x` arrays are in uint8 and have values in the [0, 255] range.
    # You need to convert them to float32 with values in the [0, 1] range.
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
    return train_dataset


def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28)),
        tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    return model


# -----------------------------------------------------------------------------------------------------------
# single-worker TensorFlow training function.
# -----------------------------------------------------------------------------------------------------------

def train_func():
    batch_size = 64
    single_worker_dataset = mnist_dataset(batch_size)
    single_worker_model = build_and_compile_cnn_model()
    single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)

train_func()



# -----------------------------------------------------------------------------------------------------------
# distributed multi-worker training
#   - by TensorFlow distributed training strategy (MultiWorkerMirroredStrategy)
# -----------------------------------------------------------------------------------------------------------

def train_func_distributed():
    per_worker_batch_size = 64

    # This environment variable will be set by Ray Train.
    tf_config = json.loads(os.environ['TF_CONFIG'])
    num_workers = len(tf_config['cluster']['worker'])

    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    global_batch_size = per_worker_batch_size * num_workers
    multi_worker_dataset = mnist_dataset(global_batch_size)

    with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.
        multi_worker_model = build_and_compile_cnn_model()

    multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)


# ----------
from ray.train.tensorflow import TensorflowTrainer
from ray.air.config import ScalingConfig

# For GPU Training, set `use_gpu` to True.
use_gpu = False

trainer = TensorflowTrainer(
    train_func_distributed,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=use_gpu))

trainer.fit()



