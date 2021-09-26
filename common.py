import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from collections import Counter


def get_class_weights(y, inverse=False):
    counter = Counter(y)
    if not inverse:
        majority = max(counter.values())
        return {cls: round(float(majority) / float(count), 2) for cls, count in counter.items()}
    if inverse:
        minority = min(counter.values())
        return {cls: 1 / (float(count) / float(minority)) for cls, count in counter.items()}


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


def split_image(image3, tile_size, puzzle_size):
    image_shape = tf.shape(image3)
    tile_rows = tf.reshape(image3, [image_shape[0], -1, tile_size[1], image_shape[2]])
    serial_tiles = tf.transpose(tile_rows, [1, 0, 2, 3])
    split_img = tf.reshape(serial_tiles, [-1, tile_size[1], tile_size[0], image_shape[2]])

    if puzzle_size == 4:
        return tf.gather(split_img,
                         [0, 2, 1, 3],
                         axis=0)
    if puzzle_size == 9:
        return tf.gather(split_img,
                         [0, 3, 6, 1, 4, 7, 2, 5, 8],
                         axis=0)


def unsplit_image(tiles4, image_shape, puzzle_size):
    if puzzle_size == 4:
        tiles4 = tf.gather(tiles4,
                           [0, 2, 1, 3],
                           axis=0)
    if puzzle_size == 9:
        tiles4 = tf.gather(tiles4,
                           [0, 3, 6, 1, 4, 7, 2, 5, 8],
                           axis=0)

    tile_width = tf.shape(tiles4)[1]
    serialized_tiles = tf.reshape(tiles4, [-1, image_shape[0], tile_width, image_shape[2]])
    rowwise_tiles = tf.transpose(serialized_tiles, [1, 0, 2, 3])
    return tf.reshape(rowwise_tiles, [image_shape[0], image_shape[1], image_shape[2]])


def puzzle_piecess_rotation(split_img, puzzle_size):
    if puzzle_size == 9:
        # rotation puzzle pieces
        rot_labels = tf.convert_to_tensor(
            [0, 0, 0, 0, 0,
             tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32),
             tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32),
             tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32),
             tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)]
        )
        rot_labels = tf.random.shuffle(rot_labels)
        rot_idx = tf.cast(rot_labels, tf.float32) * 90
        rot_idx = tf.divide(tf.multiply(rot_idx, 3.1415), 180)
        split_img = tfa.image.rotate(split_img, rot_idx)
        return rot_labels, split_img

    elif puzzle_size == 4:
        # rotation puzzle pieces
        rot_labels = tf.convert_to_tensor(
            [0, 0,
             tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32),
             tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)]
        )
        rot_labels = tf.random.shuffle(rot_labels)
        rot_idx = tf.cast(rot_labels, tf.float32) * 90
        rot_idx = tf.divide(tf.multiply(rot_idx, 3.1415), 180)
        split_img = tfa.image.rotate(split_img, rot_idx)
        return rot_labels, split_img


def jigsaw_puzzle(img, puzzle_size):
    if puzzle_size == 9:
        img = tf.image.resize(img, [225, 225])
        split_img = split_image(img, [75, 75], puzzle_size=puzzle_size)
        idx = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8])
        idx = tf.random.shuffle(idx)
        split_img = tf.gather(split_img, idx, axis=0)

        rot_labels, split_img = puzzle_piecess_rotation(split_img, puzzle_size=puzzle_size)
        rec_img = unsplit_image(split_img, tf.shape(img), puzzle_size=puzzle_size)
        rec_img = tf.image.resize(rec_img, [224, 224])
        return rec_img, idx, rot_labels

    elif puzzle_size == 4:
        img = tf.image.resize(img, [224, 224])
        split_img = split_image(img, [112, 112], puzzle_size=puzzle_size)
        idx = tf.constant([0, 1, 2, 3])
        idx = tf.random.shuffle(idx)
        split_img = tf.gather(split_img, idx, axis=0)

        rot_labels, split_img = puzzle_piecess_rotation(split_img, puzzle_size=puzzle_size)
        rec_img = unsplit_image(split_img, tf.shape(img), puzzle_size=puzzle_size)
        rec_img = tf.image.resize(rec_img, [224, 224])
        return rec_img, idx, rot_labels

    else:
        return img


def assigning_weight(label):
    label_weights = {0: 1.8, 1: 1.0, 2: 5.28, 3: 9.54, 4: 21.08, 5: 35.34, 6: 5.4, 7: 35.85}
    return label_weights[label.numpy()]


def preprocess(paths, labels, number_of_tiles, mode='train'):
    paths, labels = np.array(paths), np.array(labels)
    outputs = []
    for path, label in zip(paths, labels):
        if mode == 'train':
            preprocessor_fn = train_preprocessing
        else:
            preprocessor_fn = valid_preprocessing
        img, labels_, weights = preprocessor_fn(path, label, number_of_tiles=number_of_tiles)
        outputs.append((img, labels_, weights))
    return outputs


def train_preprocessing(image_path, label, number_of_tiles=4):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32, expand_animations=False)
    img = tf.image.resize(img, [350, 350])

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

    img, shuffle_label, rot_labels = jigsaw_puzzle(img, number_of_tiles)

    # img = tfa.image.random_cutout(tf.expand_dims(img, 0), mask_size=(70, 70),
    #                               constant_values=0)
    # img = tf.squeeze(img)

    all_labels = {
        'emotion': tf.one_hot(label, depth=8),
    }

    shuffle_label = tf.one_hot(shuffle_label, depth=number_of_tiles)
    for i in range(number_of_tiles):
        all_labels[f'puzzle_{i + 1}'] = shuffle_label[i]

    # for i in range(number_of_tiles):
    #     all_labels[f'rotation_{i + 1}'] = tf.one_hot(rot_labels[i], depth=4)

    all_sample_weights = {
        'emotion': tf.py_function(func=assigning_weight, inp=[label], Tout=[tf.float32]),
    }

    for i in range(number_of_tiles):
        all_sample_weights[f'puzzle_{i + 1}'] = 1

    # not_zero = tf.not_equal(0, rot_labels)
    # indices = tf.cast(not_zero, tf.int32)

    # for i in range(number_of_tiles):
    #     all_sample_weights[f'rotation_{i + 1}'] = (indices[i] * 1) + 1

    return img, all_labels, all_sample_weights


def valid_preprocessing(image_path, label, number_of_tiles=4):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32, expand_animations=False)
    img = tf.image.resize(img, [350, 350])

    # img = tf.image.random_crop(img, (size, size, 3))
    # img = rand_crop(img, fmin=0.8, fmax=1.0)
    img = tf.image.central_crop(img, 0.8)

    img = tf.image.resize(img, [224, 224])

    all_labels = {
        'emotion': tf.one_hot(label, depth=8),
    }

    shuffle_label = np.array(list(range(number_of_tiles)))
    shuffle_label = tf.one_hot(shuffle_label, depth=number_of_tiles)
    for i in range(number_of_tiles):
        all_labels[f'puzzle_{i + 1}'] = shuffle_label[i]

    # rot_labels = np.array(list(range(number_of_tiles)))
    # for i in range(number_of_tiles):
    #     all_labels[f'rotation_{i + 1}'] = tf.one_hot(rot_labels[i], depth=4)

    return img, all_labels
