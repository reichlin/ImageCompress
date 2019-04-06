import tensorflow as tf
import numpy as np
import MSSSIM

# reset graph
tf.reset_default_graph()

# Input functions ------------------------------------------------------------------------------------------------------


def _parse_function(example_proto):
    keys_to_features = {'image/encoded': tf.VarLenFeature(tf.string)}
    parsed_features = tf.parse_example(example_proto, keys_to_features)
    raw = tf.sparse_tensor_to_dense(parsed_features['image/encoded'], default_value="0", )

    return (tf.map_fn(decode_random_crop, tf.squeeze(raw), dtype=tf.uint8, back_prop=False))


def decode_random_crop(raw):
    img = tf.image.decode_jpeg(raw, channels=3, try_recover_truncated=True, acceptable_fraction=0.5)

    return tf.random_crop(img, [160, 160, 3])


def get_train_dataset():
    files = tf.data.Dataset.list_files("train/train-*")
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=1)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(16))
    dataset = dataset.shuffle(3500).repeat()
    dataset = dataset.map(_parse_function)
    dataset = dataset.prefetch(30)

    return dataset


def get_test_dataset():
    files = tf.data.Dataset.list_files("/path/to/validation/validation-*")
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=1)
    dataset = dataset.batch(100).map(_parse_function)

    return dataset

# MS-SSIM functions ------------------------------------------------------------------------------------------------------------------

# dataset and iterator initialization

training_dataset = get_train_dataset()
test_dataset = get_test_dataset()

iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)

# variables initialization ------------------------------------------------------------------------------------------------------------------

# network hyper-parameter
batch_size = 16
n_update = 20000 * 6

#sigma = tf.constant(1.)
depth = 5 # Depth residual block for the AutoEncoder
learning_rate = 1e-4 # Learning rate
regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)  # Regularization term for all layers
regularizer2 = tf.contrib.layers.l2_regularizer(scale=0.1)  # Regularization term for layer that outputs y
initializer = None #tf.initializers.he_normal()
image_height = 160
image_width = 160

beta = 1000

K = 128
L = 64

# Graph definition ------------------------------------------------------------------------------------------------------------------

# Network Placeholders
lr = tf.placeholder(shape=(), dtype=tf.float32)
training = tf.placeholder(dtype=tf.bool, shape=(), name="isTraining")
file_path = tf.placeholder(tf.string, name="path")

# Read input from pipeline
with tf.device('/cpu:0'):
    x = tf.reshape(tf.image.convert_image_dtype(iterator.get_next(), dtype=tf.float32), [batch_size, 160, 160, 3])
    #filenames = iterator.get_next()[1]

# Encoder

# [mean, var] = tf.nn.moments(x, axes=[0, 1, 2])
# mean = tf.transpose(tf.expand_dims(tf.expand_dims(tf.expand_dims(mean, -1), -1), -1), [1, 2, 3, 0])
# var = tf.transpose(tf.expand_dims(tf.expand_dims(tf.expand_dims(var, -1), -1), -1), [1, 2, 3, 0])
#
# x_n = (x - mean) / tf.sqrt(var + 1e-10)

with tf.name_scope("Encoder"):

    conv1 = tf.layers.conv2d(inputs=x,
                             filters=64,
                             kernel_size=[5, 5],
                             strides=(2, 2),
                             padding="same",
                             kernel_regularizer=regularizer,
                             kernel_initializer=initializer,
                             name="conv1")

    conv1 = tf.layers.batch_normalization(inputs=conv1, training=training)
    conv1 = tf.nn.relu(conv1)
    conv2 = tf.layers.conv2d(inputs=conv1,
                             filters=128,
                             kernel_size=[5, 5],
                             strides=(2, 2),
                             padding="same",
                             kernel_regularizer=regularizer,
                             kernel_initializer=initializer,
                             name="conv2")

    conv2 = tf.layers.batch_normalization(inputs=conv2, training=training)
    conv2 = tf.nn.relu(conv2)

    E_residual_blocks = []
    tmp = conv2
    for i in range(depth):
        tmp3 = tmp
        for j in range(3):
            tmp2 = tmp
            E_residual_blocks.append(tf.layers.conv2d(inputs=tmp,
                                                      filters=128,
                                                      kernel_size=[3, 3],
                                                      strides=(1, 1),
                                                      padding="same",
                                                      kernel_regularizer=regularizer,
                                                      kernel_initializer=initializer,
                                                      name="conv"+str(6*i+2*j+3)))

            E_residual_blocks[-1] = tf.layers.batch_normalization(inputs=E_residual_blocks[-1], training=training)
            E_residual_blocks[-1] = tf.nn.relu(E_residual_blocks[-1])
            tmp = E_residual_blocks[-1]
            E_residual_blocks.append(tf.layers.conv2d(inputs=tmp,
                                                      filters=128,
                                                      kernel_size=[3, 3],
                                                      strides=(1, 1),
                                                      padding="same",
                                                      kernel_regularizer=regularizer,
                                                      kernel_initializer=initializer,
                                                      name="conv"+str(6*i+2*j+4)))

            tmp = E_residual_blocks[-1] + tmp2
        tmp = tmp3 + tmp

    tmp2 = tmp
    E_residual_blocks.append(tf.layers.conv2d(inputs=tmp,
                                              filters=128,
                                              kernel_size=[3, 3],
                                              strides=(1, 1),
                                              padding="same",
                                              kernel_regularizer=regularizer,
                                              kernel_initializer=initializer,
                                              name="conv"+str(depth*6+3)))

    E_residual_blocks[-1] = tf.layers.batch_normalization(inputs=E_residual_blocks[-1], training=training)
    E_residual_blocks[-1] = tf.nn.relu(E_residual_blocks[-1])
    tmp = E_residual_blocks[-1]
    E_residual_blocks.append(tf.layers.conv2d(inputs=tmp,
                                              filters=128,
                                              kernel_size=[3, 3],
                                              strides=(1, 1),
                                              padding="same",
                                              kernel_regularizer=regularizer,
                                              kernel_initializer=initializer,
                                              name="conv"+str(depth*6+4)))

    tmp = E_residual_blocks[-1] + tmp2 + conv2

    e_out = tf.layers.conv2d(inputs=tmp,
                             filters=K,
                             kernel_size=[5, 5],
                             strides=(2, 2),
                             padding="same",
                             kernel_regularizer=regularizer,
                             kernel_initializer=initializer,
                             name="conv"+str(depth*6+5))

    z = tf.layers.conv3d(inputs=tf.expand_dims(e_out, axis=-1),
                         filters=L,
                         kernel_size=[1, 1, 1],
                         strides=(1, 1, 1),
                         padding="same",
                         kernel_regularizer=regularizer,
                         kernel_initializer=initializer,
                         name="conv3d")

    z_soft = tf.nn.softmax(z, axis=-1) # probability for each value to be every symbol

    y_out = tf.layers.conv2d(inputs=tmp,
                             filters=1,
                             kernel_size=[5, 5],
                             strides=(2, 2),
                             padding="same",
                             kernel_regularizer=regularizer2,
                             kernel_initializer=initializer,
                             name="conv"+str(depth * 6 + 6))

    weights_y = tf.get_default_graph().get_tensor_by_name("conv"+str(depth * 6 + 6)+"/kernel:0")

    mean_w = tf.reduce_mean(tf.abs(weights_y), [0, 1, 2, 3])

    tf.summary.scalar("y_out_weights_mean", mean_w)

    #y_out = tf.layers.batch_normalization(inputs=y_out, training=training)
    #y_out = tf.nn.sigmoid(y_out)

with tf.name_scope("Mask"):

    shape = tf.shape(y_out)

    y_max = tf.reduce_max(tf.abs(y_out), axis=[1, 2])
    y_mean = tf.reduce_mean(tf.abs(y_out), axis=[1, 2])

    tf.summary.scalar("Max_value_mask", tf.reduce_mean(y_max))
    tf.summary.scalar("Mean_value_mask", tf.reduce_mean(y_mean))

    y_max_tile = tf.reshape(tf.tile(y_max, [1, shape[1] * shape[2]]), shape)

    y = tf.exp(y_out / y_max_tile)

    y_exp_mean = tf.reduce_mean(tf.abs(y), axis=[1, 2])

    tf.summary.scalar("Mean_value_y_exp", tf.reduce_mean(y_exp_mean))

    y_sum = tf.reshape(tf.reduce_sum(y, axis=[1, 2]), [-1, 1])
    tile = tf.tile(y_sum, [1, shape[1] * shape[2]])

    y = tf.div(y, tf.reshape(tile, shape))

    tf.summary.image("prob", y, 5)

    yy = tf.transpose(tf.reshape(tf.tile(tf.reshape(y,
                                                    [-1]),
                                         [K]),
                                 [K, tf.shape(y)[0], tf.shape(y)[1], tf.shape(y)[2]]),
                      [1, 2, 3, 0])

    kk = tf.transpose(tf.reshape(tf.tile(tf.linspace(0., K - 1, K),
                                         [np.prod(y.get_shape().as_list())]),
                                 [tf.shape(y)[0], tf.shape(y)[1], tf.shape(y)[2], K]),
                      [0, 1, 2, 3])

    m = yy * K - kk
    zero = tf.zeros(tf.shape(m))
    m = tf.maximum(x=m, y=zero)
    zero = tf.ones(tf.shape(m))
    m = tf.minimum(x=m, y=zero)

    # gradient trick
    m = m + tf.stop_gradient(tf.ceil(m) - m)

with tf.name_scope("Quantizer"):

    z_scaled = z * 1.

    z_hat = tf.cast(tf.argmax(z_soft, axis=-1) + 1, dtype=tf.float32)

    cs = tf.cumsum(tf.ones_like(z), axis=-1)

    z_soft_scaled = tf.nn.softmax((beta * z_scaled), axis=-1)
    z_tilde = tf.reduce_sum((z_soft_scaled * cs), axis=-1)

    quant_error = tf.reduce_mean(tf.abs(z_hat - z_tilde), axis=[0, 1, 2, 3])
    tf.summary.scalar('Quantization_error', quant_error)

    z_differentiable = tf.stop_gradient(z_hat - z_tilde) + z_tilde

    z_masked = tf.multiply(z_differentiable, m)


# Decoder
with tf.name_scope("Decoder"):

    D_residual_blocks = []
    D_residual_blocks.append(tf.layers.conv2d_transpose(inputs=z_masked,
                                                        filters=128,
                                                        kernel_size=[3, 3],
                                                        strides=(2, 2),
                                                        padding="same",
                                                        kernel_regularizer=regularizer,
                                                        kernel_initializer=initializer,
                                                        name="conv"+str(depth*6+7)))

    D_residual_blocks[-1] = tf.layers.batch_normalization(inputs=D_residual_blocks[-1], training=training)
    D_residual_blocks[-1] = tf.nn.relu(D_residual_blocks[-1])
    tmp = D_residual_blocks[-1]

    for i in range(depth):
        tmp3 = tmp
        for j in range(3):
            tmp2 = tmp
            D_residual_blocks.append(tf.layers.conv2d(inputs=tmp,
                                                      filters=128,
                                                      kernel_size=[3, 3],
                                                      strides=(1, 1),
                                                      padding="same",
                                                      kernel_regularizer=regularizer,
                                                      kernel_initializer=initializer,
                                                      name="conv"+str(6*i+2*j+depth*6+8)))

            D_residual_blocks[-1] = tf.layers.batch_normalization(inputs=D_residual_blocks[-1], training=training)
            D_residual_blocks[-1] = tf.nn.relu(D_residual_blocks[-1])
            tmp = D_residual_blocks[-1]
            D_residual_blocks.append(tf.layers.conv2d(inputs=tmp,
                                                      filters=128,
                                                      kernel_size=[3, 3],
                                                      strides=(1, 1),
                                                      padding="same",
                                                      kernel_regularizer=regularizer,
                                                      kernel_initializer=initializer,
                                                      name="conv"+str(6*i+2*j+depth*6+9)))

            tmp = D_residual_blocks[-1] + tmp2
        tmp = tmp3 + tmp

    tmp2 = tmp
    D_residual_blocks.append(tf.layers.conv2d(inputs=tmp,
                                              filters=128,
                                              kernel_size=[3, 3],
                                              strides=(1, 1),
                                              padding="same",
                                              kernel_regularizer=regularizer,
                                              kernel_initializer=initializer,
                                              name="conv"+str(depth*14+4)))

    D_residual_blocks[-1] = tf.layers.batch_normalization(inputs=D_residual_blocks[-1], training=training)
    D_residual_blocks[-1] = tf.nn.relu(D_residual_blocks[-1])
    tmp = D_residual_blocks[-1]
    D_residual_blocks.append(tf.layers.conv2d(inputs=tmp,
                                              filters=128,
                                              kernel_size=[3, 3],
                                              strides=(1, 1),
                                              padding="same",
                                              kernel_regularizer=regularizer,
                                              kernel_initializer=initializer,
                                              name="conv"+str(depth*14+5)))

    tmp = D_residual_blocks[-1] + tmp2 + D_residual_blocks[0]

    deconv1 = tf.layers.conv2d_transpose(inputs=tmp,
                                         filters=64,
                                         kernel_size=[5, 5],
                                         strides=(2, 2),
                                         padding="same",
                                         kernel_regularizer=regularizer,
                                         kernel_initializer=initializer,
                                         name="deconv1")

    deconv1 = tf.layers.batch_normalization(inputs=deconv1, training=training)
    deconv1 = tf.nn.relu(deconv1)
    deconv2 = tf.layers.conv2d_transpose(inputs=deconv1,
                                         filters=3,
                                         kernel_size=[5, 5],
                                         strides=(2, 2),
                                         padding="same",
                                         kernel_regularizer=regularizer,
                                         kernel_initializer=initializer,
                                         name="deconv2")

    #deconv2 = tf.nn.sigmoid(deconv2) # images must be between 0 and 1

    # Output Decoder

x_hat = tf.minimum(tf.maximum(deconv2, 0.), 1.) # bounded ReLu

relu_err = tf.reduce_mean(tf.abs(x_hat-deconv2), axis=[0, 1, 2, 3])

tf.summary.scalar("Bounded_RELU_error", relu_err)


# Denormalize Reconstructed Image

# x_hat_norm = x_hat * tf.sqrt(var + 1e-10) + mean
# x_hat_norm = tf.clip_by_value(x_hat_norm, 0, 1.0)

tf.summary.image("x", x, 5)
tf.summary.image("x_hat", x_hat, 5)

# Distortion rate index
msssim_indexR = MSSSIM.tf_ms_ssim(x[:, :, :, 0:1], x_hat[:, :, :, 0:1])
msssim_indexG = MSSSIM.tf_ms_ssim(x[:, :, :, 1:2], x_hat[:, :, :, 1:2])
msssim_indexB = MSSSIM.tf_ms_ssim(x[:, :, :, 2:3], x_hat[:, :, :, 2:3])

acc = (msssim_indexR + msssim_indexG + msssim_indexB) / 3.

mse = tf.reduce_mean(tf.squared_difference(x, x_hat))

loss = mse

#acc = (1. - mse)

tf.summary.scalar('accuracy', acc*100.)
tf.summary.scalar('loss', loss)


# Optimizer Context Model
optimizer = tf.train.AdamOptimizer(learning_rate=lr)


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = optimizer.minimize(loss)

# Graph initialization ------------------------------------------------------------------------------------------------------------------

training_init_op = iterator.make_initializer(training_dataset)

sess = tf.Session()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/train', sess.graph)

init1 = tf.global_variables_initializer()
init2 = tf.local_variables_initializer()
sess.run(init1)
sess.run(init2)

saver = tf.train.Saver()

# Model Training ------------------------------------------------------------------------------------------------------------------

update = 0
sess.run(training_init_op)
for update in range(n_update):
    _, summary = sess.run((train, merged), feed_dict={training: True, lr: learning_rate})

    train_writer.add_summary(summary, update)

    if update % 40000 == 39999:
        learning_rate *= 0.1

try:
    saver.save(sess, "model/model.ckpt")
    print("model saved successfully")
except Exception:
    pass


#
# for i in range(num_batch):
#     fn, batch_img_out, batch_img = sess.run((filenames, x_hat_norm, x), feed_dict={training: True})
# 
#     for j in range(len(fn)):
#         name = "/mnt/disks/disk2/ae_out/label/" + str(fn[j])[2:-1]
#         plt.imsave(name, batch_img[j])
#
#         name = "/mnt/disks/disk2/ae_out/in/" + str(fn[j])[2:-1]
#         plt.imsave(name, batch_img_out[j])
