import tensorflow as tf
import numpy as np
import warnings


n_corrupted = 0


def _parse_function(example_proto):
    keys_to_features = {'image/encoded': tf.VarLenFeature(tf.string)}
    parsed_features = tf.parse_example(example_proto, keys_to_features)
    raw = tf.sparse_tensor_to_dense(parsed_features['image/encoded'], default_value="0", )

    return (tf.map_fn(decode_random_crop, tf.squeeze(raw), dtype=tf.uint8, back_prop=False))


def decode_random_crop(raw):
    warnings.filterwarnings("error")
    try:
        img = tf.image.decode_jpeg(raw, channels=3)

    except RuntimeWarning:
        global n_corrupted
        n_corrupted += 1
        print(n_corrupted)

        img = tf.image.decode_jpeg(raw, channels=3)
    warnings.filterwarnings("default")

    return tf.random_crop(img, [160, 160, 3])


def get_train_dataset():
    files = tf.data.Dataset.list_files("train/train-*")
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=1)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(16)).repeat(1)
    dataset = dataset.map(_parse_function)
    dataset = dataset.prefetch(30)

    return dataset


training_dataset = get_train_dataset()

iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)


training_init_op = iterator.make_initializer(training_dataset)

sess = tf.Session()

sess.run(training_init_op)

while True:
        try:
            value = sess.run(iterator.get_next())
        except tf.errors.OutOfRangeError:
            break

print("%d corrupted images", n_corrupted)