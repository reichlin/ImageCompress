import tensorflow as tf

for example in tf.python_io.tf_record_iterator("/media/luca/backup/out250/train-00000-of-00016"):
    result = tf.train.Example.FromString(example)

    print(result)