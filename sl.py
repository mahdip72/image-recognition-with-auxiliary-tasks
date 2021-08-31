import os
import numpy as np
import cv2
import adabelief_tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow as tf
import tensorflow_addons as tfa

number_of_tiles = 4


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


def train_preprocessing(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32, expand_animations=False)
    img = tf.image.resize_with_pad(img, 350, 350)

    size = tf.random.uniform(shape=[], minval=224, maxval=330, dtype=tf.int32)
    img = tf.image.random_crop(img, (size, size, 3))

    # img = tf.image.random_hue(img, 0.2)
    # img = tf.image.random_contrast(img, lower=0.6, upper=1.4)
    # img = tf.image.random_brightness(img, max_delta=0.05)
    # img = tf.clip_by_value(img, 0.0, 1.0)

    # var = tf.random.uniform(shape=[], minval=0, maxval=0.05, dtype=tf.float16)
    # noise = tf.random.normal(shape=[224, 224, 3], mean=0.0,
    #                          stddev=tf.random.uniform(shape=[], minval=0, maxval=var, dtype=tf.float16),
    #                          dtype=tf.float16)
    # img = tf.add(img, noise)
    # img = tf.clip_by_value(img, 0.0, 1.0)

    # kernel_size = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
    # img = tfa.image.gaussian_filter2d(img, filter_shape=[kernel_size * 2 + 1, kernel_size * 2 + 1],
    #                                   sigma=[1.5, 1.5])

    # with tf.device("/gpu:0"):
    # size = tf.random.uniform(shape=[], minval=180, maxval=224, dtype=tf.int32)
    # img = tf.image.random_crop(img, (size, size, 3))
    img = rand_crop(img, fmin=0.8, fmax=1.0)
    # img = tf.image.central_crop(img, 0.69)

    img = tf.image.resize(img, [224, 224])

    # img = tfa.image.random_cutout(tf.expand_dims(img, 0), mask_size=(70, 70),
    #                               constant_values=0)
    # img = tf.squeeze(img)

    all_labels = {
        'sl': tf.one_hot(tf.cast(label, tf.int32), depth=10),
    }

    all_sample_weights = {
        'sl': 1,
    }

    return img, all_labels, all_sample_weights


def valid_preprocessing(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32, expand_animations=False)
    img = tf.image.resize(img, [300, 300])

    # img = tf.image.random_crop(img, (size, size, 3))
    # img = rand_crop(img, fmin=0.8, fmax=1.0)
    img = tf.image.central_crop(img, 0.8)

    img = tf.image.resize(img, [224, 224])

    all_labels = {
        'sl': tf.one_hot(tf.cast(label, tf.int32), depth=10),
    }

    return img, all_labels


def main():
    autotune = tf.data.experimental.AUTOTUNE
    # autotune = None
    tf.keras.backend.clear_session()
    mixed_precision.set_global_policy('mixed_float16')
    batch = 64

    dataset_path = 'S:/Datasets/imagenette2'

    model_name = 'sl-no_augment'
    # model_name = 'sl-weak_augment'
    # model_name = 'sl-strong_augment'
    # model_name = 'test'

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
    # for i, j, w in train_ds:
    # for i, j in valid_ds:
    #     print(i.numpy().shape)
    #     print(w)
    #     imgs = i.numpy()
    #     imgs = (imgs * 255).astype(np.uint8)
    #     for inx, b in enumerate(range(imgs.shape[0])):
    #         img = imgs[b, :, :, :]
    #         print('weights:', w)
    #         print('labels:', j)
    #         cv2.imshow('', cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_RGB2BGR))
    #         cv2.waitKey(0)

    tf.keras.backend.clear_session()
    # backbone = ResNet50(include_top=False,
    #                     input_shape=(224, 224, 3),
    #                     weights='imagenet',
    #                     weights=None,
    #                     )

    backbone = MobileNetV2(include_top=False,
                           input_shape=(224, 224, 3),
                           # weights='imagenet',
                           weights=None,
                           )
    backbone.summary()
    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    sl_dropout = tf.keras.layers.Dropout(0.3)(x)

    sl = tf.keras.layers.Dense(10, activation='softmax', name='sl', dtype=tf.float32)(sl_dropout)

    # supervised head
    heads = [
        sl
    ]

    model = tf.keras.Model(inputs=backbone.input,
                           outputs=heads
                           )
    model.summary()

    op = adabelief_tf.AdaBeliefOptimizer(learning_rate=0.001,
                                         print_change_log=False)

    losses = {
        'sl': tf.keras.losses.CategoricalCrossentropy(),
    }

    metrics = {
        'sl': [
            tf.keras.metrics.CategoricalAccuracy('acc'),
        ]}

    model.compile(loss=losses,
                  optimizer=op,
                  metrics=metrics,
                  # loss_weights=loss_weights,
                  )

    val_loss_checkpoint = ModelCheckpoint(f"./results/{model_name}/val_checkpoint.h5",
                                          monitor='val_loss', verbose=0,
                                          save_best_only=True, save_weights_only=False,
                                          mode='min', save_freq='epoch')

    loss_checkpoint = ModelCheckpoint(f"./results/{model_name}/checkpoint.h5",
                                      monitor='loss', verbose=0,
                                      save_best_only=True, save_weights_only=False,
                                      mode='min', save_freq='epoch')

    csv_callback = CSVLogger(f"./results/{model_name}/training_log.csv",
                             append=False)

    def lr_scheduler(epoch, lr):
        if epoch == 30 or epoch == 50:
            lr = lr / 10
            return lr
        else:
            return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    callbacks_list = [
        # loss_checkpoint,
        # val_loss_checkpoint,
        lr_callback,
        csv_callback,
        tf.keras.callbacks.TensorBoard(log_dir=f"./results/{model_name}/log",
                                       update_freq='epoch')
    ]

    model.fit(train_ds,
              validation_data=valid_ds,
              callbacks=callbacks_list,
              verbose=2,
              epochs=200)

    print('sl finished')


if __name__ == '__main__':
    main()
