import tensorflow as tf
import random
import adabelief_tf
import numpy as np
import pandas as pd
import os
import cv2
import tensorflow_addons as tfa
from collections import Counter
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger


def create_pairs(path):
    samples = []
    cats = os.listdir(path)
    for index, cat in enumerate(cats):
        images_dir = os.listdir(os.path.join(path, cat))
        for image_dir in images_dir:
            samples.append([os.path.join(path, cat, image_dir), int(index)])
    return np.array(samples)


def rand_crop(image, fmin, fmax):
    from tensorflow.python.ops import math_ops
    from tensorflow.python.ops import array_ops
    from tensorflow.python.framework import ops
    image = ops.convert_to_tensor(image, name='image')

    if fmin <= 0.0 or fmin > 1.0:
        raise ValueError('fmin must be within (0, 1]')
    if fmax <= 0.0 or fmax > 1.0:
        raise ValueError('fmin must be within (0, 1]')

    img_shape = array_ops.shape(image)
    depth = image.get_shape()[2]
    my_frac2 = tf.random.uniform([1], minval=fmin, maxval=fmax, dtype=tf.float32, seed=42, name="uniform_dist")
    fraction_offset = tf.cast(math_ops.div(1.0, math_ops.div(math_ops.sub(1.0, my_frac2[0]), 2.0)), tf.int32)
    bbox_h_start = math_ops.div(img_shape[0], fraction_offset)
    bbox_w_start = math_ops.div(img_shape[1], fraction_offset)
    bbox_h_size = img_shape[0] - bbox_h_start * 2
    bbox_w_size = img_shape[1] - bbox_w_start * 2

    bbox_begin = array_ops.pack([bbox_h_start, bbox_w_start, 0])
    bbox_size = array_ops.pack([bbox_h_size, bbox_w_size, -1])
    image = array_ops.slice(image, bbox_begin, bbox_size)

    # The first two dimensions are dynamic and unknown.
    image.set_shape([None, None, depth])
    return image


def barlow_twins_augmenting(augmented_img):
    augmented_img = tf.image.random_flip_left_right(augmented_img)
    rot_label = tf.random.uniform(shape=[1], minval=-15, maxval=15, dtype=tf.int32)
    rad = tf.divide((tf.cast(rot_label, tf.float32)) * np.pi, 180)
    augmented_img = tfa.image.rotate(augmented_img, rad)
    augmented_img = tf.squeeze(augmented_img)

    # augmented_img = rand_crop(augmented_img, fmin=0.75, fmax=0.99)
    size = tf.random.uniform(shape=[], minval=170, maxval=300, dtype=tf.int32)
    augmented_img = tf.image.random_crop(augmented_img, (size, size, 3))
    augmented_img = tf.image.resize(augmented_img, [112, 112])

    augmented_img = tf.image.random_hue(augmented_img, 0.05)
    augmented_img = tf.clip_by_value(augmented_img, 0.0, 1.0)

    # augmented_img = tf.image.random_contrast(augmented_img, lower=0.6, upper=1.4)
    # augmented_img = tf.image.random_brightness(augmented_img, max_delta=0.05)
    # augmented_img = tf.clip_by_value(augmented_img, 0.0, 1.0)

    kernel_size = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.int32)
    augmented_img = tfa.image.gaussian_filter2d(augmented_img, filter_shape=[kernel_size * 2 + 1, kernel_size * 2 + 1],
                                                sigma=[1.5, 1.5])

    # var = tf.random.uniform(shape=[], minval=0, maxval=0.04, dtype=tf.float32)
    # noise = tf.random.normal(shape=[224, 224, 3], mean=0.0,
    #                          stddev=tf.random.uniform(shape=[], minval=0, maxval=var, dtype=tf.float32),
    #                          dtype=tf.float32)
    # augmented_img = tf.add(augmented_img, noise)
    # augmented_img = tf.clip_by_value(augmented_img, 0.0, 1.0)
    # img, shuffle_label = jigsaw_puzzle(img, number_of_tiles)

    # augmented_img = tfa.image.random_cutout(tf.expand_dims(augmented_img, 0), mask_size=(40, 40),
    #                                         constant_values=0)
    # augmented_img = tf.squeeze(augmented_img)
    return augmented_img


def train_preprocessing(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32, expand_animations=False)
    # img = tf.image.resize(img, [350, 350])
    img = tf.image.resize_with_pad(img, 350, 350)
    # img = tf.image.resize_with_pad(img, 150, 150)
    # img = tf.image.resize(img, [150, 150])

    size = tf.random.uniform(shape=[], minval=200, maxval=300, dtype=tf.int32)
    # original_img = tf.image.central_crop(img, 0.8)
    original_img = tf.image.random_crop(img, (size, size, 3))

    # original_img = rand_crop(img, fmin=0.8, fmax=1.0)
    original_img = tf.image.resize(original_img, [112, 112])

    augmented_img = barlow_twins_augmenting(img)
    # augmented_img = rand_crop(img, fmin=0.7, fmax=1.0)
    augmented_img = tf.image.resize(augmented_img, [112, 112])

    all_labels = {
        'imagenet': tf.one_hot(label, depth=10),
    }

    return original_img, augmented_img, all_labels


def valid_preprocessing(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32, expand_animations=False)
    # img = tf.image.resize(img, [350, 350])
    # img = tf.image.resize_with_pad(img, 350, 350)
    img = tf.image.resize_with_pad(img, 150, 150)
    # img = tf.image.resize(img, [150, 150])

    # img = tf.image.random_crop(img, (size, size, 3))
    original_img = tf.image.central_crop(img, 0.85)
    # img = rand_crop(img, fmin=0.7, fmax=1.0)
    # img = tf.image.central_crop(img, 0.7)

    original_img = tf.image.resize(original_img, [112, 112])
    # img = tf.image.resize(img, [112, 112])

    all_labels = {
        'imagenet': tf.one_hot(label, depth=10),
    }

    return original_img, all_labels
    # return img, original_img


def off_diagonal(x):
    n = tf.shape(x)[0]
    flattened = tf.reshape(x, [-1])[:-1]
    off_diagonals = tf.reshape(flattened, (n - 1, n + 1))[:, 1:]
    return tf.reshape(off_diagonals, [-1])


def normalize_repr(z):
    z_norm = (z - tf.reduce_mean(z, axis=0)) / tf.math.reduce_std(z, axis=0)
    return z_norm


def compute_loss(z_a, z_b, lambd):
    # Get batch size and representation dimension.
    batch_size = tf.cast(tf.shape(z_a)[0], z_a.dtype)
    repr_dim = tf.shape(z_a)[1]

    # Normalize the representations along the batch dimension.
    z_a_norm = normalize_repr(z_a)
    z_b_norm = normalize_repr(z_b)

    # Cross-correlation matrix.
    c = tf.matmul(z_a_norm, z_b_norm, transpose_a=True) / batch_size

    # Loss.
    on_diag = tf.linalg.diag_part(c) + (-1)
    on_diag = tf.reduce_sum(tf.pow(on_diag, 2))
    off_diag = off_diagonal(c)
    off_diag = tf.reduce_sum(tf.pow(off_diag, 2))
    loss = on_diag + (lambd * off_diag)
    return loss


class BarlowTwins(tf.keras.Model):
    def __init__(self, encoder, lambd=5e-3):
        super(BarlowTwins, self).__init__()
        self.encoder = encoder
        self.lambd = lambd
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.all_loss_tracker = tf.keras.metrics.Mean(name="all_loss")
        self.imagenet_loss_tracker = tf.keras.metrics.Mean(name="imagenet_loss")
        self.imagenet_acc_tracker = tf.keras.metrics.Mean(name="imagenet_acc")

    @property
    def metrics(self):
        return [self.all_loss_tracker,
                self.loss_tracker,
                self.imagenet_loss_tracker,
                self.imagenet_acc_tracker
                ]

    def call(self, inputs):
        return self.encoder(inputs)

    def train_step(self, data):
        # Unpack the data.
        img1, img2, label = data
        with tf.GradientTape() as tape:
            imagenet_1, z_a = self(img1, training=True)
            imagenet_2, z_b = self(img2, training=True)

            imagenet_1_loss = tf.keras.losses.categorical_crossentropy(imagenet_1, label['imagenet'])
            imagenet_2_loss = tf.keras.losses.categorical_crossentropy(imagenet_2, label['imagenet'])
            imagenet_loss = imagenet_1_loss + imagenet_2_loss

            loss_barlowtwins = compute_loss(z_a, z_b, self.lambd)
            loss = loss_barlowtwins + imagenet_loss

        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # Monitor loss and acc
        self.all_loss_tracker.update_state(loss)
        self.loss_tracker.update_state(loss_barlowtwins)
        self.imagenet_loss_tracker.update_state(imagenet_loss)
        self.imagenet_acc_tracker.update_state(tf.keras.metrics.categorical_accuracy(label['imagenet'], imagenet_1))

        return {"all_loss": self.all_loss_tracker.result(),
                "ssl_loss": self.loss_tracker.result(),
                "imagenet_loss": self.imagenet_loss_tracker.result(),
                "imagenet_acc": self.imagenet_acc_tracker.result(),
                }

    def test_step(self, data):
        # Unpack the data.
        img1, label = data

        # Forward pass through the encoder and predictor.
        imagenet_1, z_a = self.encoder(img1, training=False)

        imagenet_loss = tf.keras.losses.categorical_crossentropy(imagenet_1, label['imagenet'])

        # Monitor loss.
        self.imagenet_loss_tracker.update_state(imagenet_loss)
        self.imagenet_acc_tracker.update_state(tf.keras.metrics.categorical_accuracy(label['imagenet'], imagenet_1))

        return {
            "imagenet_loss": self.imagenet_loss_tracker.result(),
            "imagenet_acc": self.imagenet_acc_tracker.result()
        }


def main():
    autotune = tf.data.experimental.AUTOTUNE
    tf.keras.backend.clear_session()
    # mixed_precision.set_global_policy('mixed_float16')
    batch = 128

    dataset_path = 'S:/Datasets/imagenette2'

    # model_name = 'sl-no_augment'
    # model_name = 'sl-weak_augment'
    # model_name = 'sl-strong_augment'
    model_name = 'test'

    if not os.path.exists(f"./results/{model_name}"):
        os.makedirs(f"./results/{model_name}")

    train_samples = create_pairs(os.path.join(dataset_path, 'train'))
    valid_samples = create_pairs(os.path.join(dataset_path, 'val'))

    train_ds = tf.data.Dataset.from_tensor_slices((train_samples[:, 0], train_samples[:, 1].astype(np.int32)))
    train_ds = train_ds.shuffle(int(len(train_samples))).map(train_preprocessing, num_parallel_calls=autotune)
    train_ds = train_ds.batch(batch).prefetch(autotune)

    valid_ds = tf.data.Dataset.from_tensor_slices((valid_samples[:, 0], valid_samples[:, 1].astype(np.int32)))
    valid_ds = valid_ds.map(valid_preprocessing, num_parallel_calls=autotune).batch(batch).prefetch(autotune)

    # for testing:
    # for i, j, l in train_ds:
        # for i, j in valid_ds:
        # print(i.numpy().shape)
        # original_imgs = i.numpy()
        # augmented_imgs = j.numpy()
        # original_imgs = (original_imgs * 255).astype(np.uint8)
        # for inx, b in enumerate(range(original_imgs.shape[0])):
        #     original_img = original_imgs[b, :, :, :]
        #     augmented_img = augmented_imgs[b, :, :, :]
        #     cv2.imshow('original', cv2.cvtColor(cv2.resize(original_img, (224, 224)), cv2.COLOR_RGB2BGR))
        #     cv2.imshow('augmented', cv2.cvtColor(cv2.resize(augmented_img, (224, 224)), cv2.COLOR_RGB2BGR))
        #     cv2.waitKey(0)

    tf.keras.backend.clear_session()
    backbone = MobileNetV2(include_top=False,
                           input_shape=(112, 112, 3),
                           weights=None,
                           )
    backbone.summary()
    embedding = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    projection_outputs = tf.keras.layers.Dense(1024, activation='linear')(embedding)
    projection_outputs = tf.keras.layers.BatchNormalization()(projection_outputs)
    projection_outputs = tf.keras.activations.relu(projection_outputs)
    projection_outputs = tf.keras.layers.Dense(512, activation='linear', dtype=tf.float32)(projection_outputs)
    projection_outputs = tf.keras.layers.BatchNormalization()(projection_outputs)

    imagenet = tf.keras.layers.Dense(10, activation='softmax', name="imagenet",
                                     dtype=tf.float32)(embedding)

    model = tf.keras.Model(inputs=backbone.input,
                           outputs=[imagenet, projection_outputs]
                           )

    model.summary()
    model = BarlowTwins(model)
    # op = tf.keras.optimizers.Adam(learning_rate=0.001)

    op = adabelief_tf.AdaBeliefOptimizer(learning_rate=0.001,
                                         print_change_log=False)

    # op = tf.keras.optimizers.SGD(learning_rate=0.0000001,
    #                              momentum=0.8, nesterov=False)

    model.compile(
        optimizer=op,
    )
    model.compute_output_shape(input_shape=(None, 112, 112, 3))
    # model.build(backbone.input)

    loss_checkpoint = ModelCheckpoint(f"./results/{model_name}/checkpoint",
                                      monitor='val_imagenet_loss', verbose=0,
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
        loss_checkpoint,
        lr_callback,
        csv_callback,
        tf.keras.callbacks.TensorBoard(log_dir=f"./results/{model_name}/log",
                                       update_freq='epoch')
    ]

    model.fit(
        train_ds,
        # valid_ds,
        validation_data=valid_ds,
        callbacks=callbacks_list,
        verbose=1,
        epochs=200)

    print('finish')


if __name__ == '__main__':
    main()
