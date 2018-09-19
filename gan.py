import tensorflow as tf
import numpy as np

# reset graph
tf.reset_default_graph()

'''
 x_ae = imagine blurry
x_hat = imagine ricostruita
    x = imagine originale
    G = generator
    D = discriminator
'''


# Input functions ------------------------------------------------------------------------------------------------------


def _parse_function(example_proto):
    keys_to_features = {'image/encoded': tf.VarLenFeature(tf.string),
                        'image/label': tf.VarLenFeature(tf.string)}
    parsed_features = tf.parse_example(example_proto, keys_to_features)
    raw1 = tf.sparse_tensor_to_dense(parsed_features['image/encoded'], default_value="0", )
    raw2 = tf.sparse_tensor_to_dense(parsed_features['image/label'], default_value="0", )

    return (tf.map_fn(decode, tf.squeeze(raw1), dtype=tf.uint8, back_prop=False),
            tf.map_fn(decode, tf.squeeze(raw2), dtype=tf.uint8, back_prop=False))


def decode(raw):
    return tf.image.decode_jpeg(raw, channels=3)


def get_train_dataset(path, batch_size):
    files = tf.data.Dataset.list_files(path)
    dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=4))
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(20))
    dataset = dataset.map(_parse_function, num_parallel_calls=4)
    dataset = dataset.prefetch(batch_size)

    return dataset


# MS-SSIM functions ------------------------------------------------------------------------------------------------------------------

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=9, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a different scale)
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)),
                 (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)

    return value


def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level - 1] ** weight[0:level - 1]) * (mssim[level - 1] ** weight[level - 1]))

    if mean_metric:
        value = tf.reduce_mean(value)

    return value


def getMSSSIM(x, x_hat):

    msssim_indexR = tf_ms_ssim(x[:, :, :, 0:1], x_hat[:, :, :, 0:1])
    msssim_indexG = tf_ms_ssim(x[:, :, :, 1:2], x_hat[:, :, :, 1:2])
    msssim_indexB = tf_ms_ssim(x[:, :, :, 2:3], x_hat[:, :, :, 2:3])
    acc = (msssim_indexR + msssim_indexG + msssim_indexB) / 3

    return acc


def getMSE(x, x_hat):

    mse = tf.reduce_mean(tf.square(x - x_hat), axis=[1, 2, 3, 0])

    return mse

def getAlpha(acc):

    return 2000*((acc - 1)**2) + 5


def generator(x_ae, Gregularizer, Ginitializer, batch_size):

    Gconv1 = tf.layers.conv2d(inputs=x_ae,
                              filters=64,
                              kernel_size=[5, 5],
                              strides=(1, 1),
                              padding="same",
                              kernel_regularizer=Gregularizer,
                              kernel_initializer=Ginitializer,
                              name="Gconv1",
                              reuse=tf.AUTO_REUSE)

    Gconv1 = tf.layers.batch_normalization(Gconv1)

    Gconv1 = tf.nn.relu(Gconv1)

    Gconv2 = tf.layers.conv2d(inputs=Gconv1,
                              filters=128,
                              kernel_size=[5, 5],
                              strides=(1, 1),
                              padding="same",
                              kernel_regularizer=Gregularizer,
                              kernel_initializer=Ginitializer,
                              name="Gconv2",
                              reuse=tf.AUTO_REUSE)

    Gconv2 = tf.layers.batch_normalization(Gconv2)

    Gconv2 = tf.nn.relu(Gconv2)

    Gconv3 = tf.layers.conv2d(inputs=Gconv2,
                              filters=32,
                              kernel_size=[5, 5],
                              strides=(1, 1),
                              padding="same",
                              kernel_regularizer=Gregularizer,
                              kernel_initializer=Ginitializer,
                              name="Gconv3",
                              reuse=tf.AUTO_REUSE)

    Gconv3 = tf.layers.batch_normalization(Gconv3)

    Gconv3 = tf.nn.relu(Gconv3)

    Gconv4 = tf.layers.conv2d(inputs=Gconv3,
                              filters=3,
                              kernel_size=[7, 7],
                              strides=(1, 1),
                              padding="same",
                              kernel_regularizer=Gregularizer,
                              kernel_initializer=Ginitializer,
                              name="Gconv4",
                              reuse=tf.AUTO_REUSE)
    Gconv4 = Gconv4 + x_ae

    x_hat = tf.clip_by_value(Gconv4, 0.0, 1.0)

    return x_hat


def discriminator(x, Dregularizer, DregularizerDense, batch_size, reuse_variables=None):

    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        Dconv1 = tf.layers.conv2d(inputs=x,
                                  filters=64,
                                  kernel_size=[3, 3],
                                  strides=(2, 2),
                                  padding="same",
                                  kernel_regularizer=Dregularizer,
                                  name="Dconv1")

        Dconv1 = tf.layers.batch_normalization(Dconv1)
        Dconv1 = tf.nn.relu(Dconv1)

        Dconv2 = tf.layers.conv2d(inputs=Dconv1,
                                  filters=128,
                                  kernel_size=[3, 3],
                                  strides=(2, 2),
                                  padding="same",
                                  kernel_regularizer=Dregularizer,
                                  name="Dconv2")

        Dconv2 = tf.layers.batch_normalization(Dconv2)
        Dconv2 = tf.nn.relu(Dconv2)

        Dconv3 = tf.layers.conv2d(inputs=Dconv2,
                                  filters=256,
                                  kernel_size=[3, 3],
                                  strides=(2, 2),
                                  padding="same",
                                  kernel_regularizer=Dregularizer,
                                  name="Dconv3")

        Dconv3 = tf.layers.batch_normalization(Dconv3)
        Dconv3 = tf.nn.relu(Dconv3)

        Dconv4 = tf.layers.conv2d(inputs=Dconv3,
                                  filters=128,
                                  kernel_size=[3, 3],
                                  strides=(2, 2),
                                  padding="same",
                                  kernel_regularizer=Dregularizer,
                                  name="Dconv4")

        Dconv4 = tf.layers.batch_normalization(Dconv4)
        Dconv4 = tf.nn.relu(Dconv4)

        Dconv5 = tf.layers.conv2d(inputs=Dconv4,
                                  filters=32,
                                  kernel_size=[3, 3],
                                  strides=(2, 2),
                                  padding="same",
                                  kernel_regularizer=Dregularizer,
                                  name="Dconv5")

        Dconv5 = tf.layers.batch_normalization(Dconv5)
        Dconv5 = tf.nn.relu(Dconv5)

        DconvOut = tf.reshape(Dconv5, [batch_size, -1])

        Ddense1 = tf.layers.dense(inputs=DconvOut,
                                  units=1024,
                                  activation=tf.nn.relu,
                                  # kernel_initializer=tf.contrib.layers.xavier_initializer,
                                  kernel_regularizer=DregularizerDense,
                                  name="Ddense1")

        Ddense2 = tf.layers.dense(inputs=Ddense1,
                                  units=512,
                                  activation=tf.nn.relu,
                                  # kernel_initializer=tf.contrib.layers.xavier_initializer,
                                  kernel_regularizer=DregularizerDense,
                                  name="Ddense2")

        Dout = tf.layers.dense(inputs=Ddense2,
                               units=1,
                               activation=None,
                               # kernel_initializer=tf.contrib.layers.xavier_initializer,
                               kernel_regularizer=DregularizerDense,
                               name="Dout")

        return Dout


# constanti
Dregularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
DregularizerDense = tf.contrib.layers.l2_regularizer(scale=0.001)
Gregularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
Ginitializer = tf.random_normal_initializer(stddev=0.00001)
lrD = tf.Variable(5e-4, dtype=tf.float32)
lrG = tf.Variable(5e-4, dtype=tf.float32)
epochsAE = 1
epochsD = 1
epochsGAN = 10
batch_size = 30

dataset = get_train_dataset("/mnt/disks/disk2/ae_out/records/train/train-*", batch_size)

iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                           dataset.output_shapes)

with tf.device('/cpu:0'):
    data = iterator.get_next()
    x_ae = tf.reshape(tf.image.convert_image_dtype(data[0], dtype=tf.float32), [batch_size, 160, 160, 3])
    x = tf.reshape(tf.image.convert_image_dtype(data[1], dtype=tf.float32), [batch_size, 160, 160, 3])

# Generator
Gz = generator(x_ae, Gregularizer, Ginitializer, batch_size)
s_gz = tf.summary.image("Gz", Gz, 1)
s_x_ae = tf.summary.image("x_ae", x_ae, 1)
s_x = tf.summary.image("x", x, 1)

g_acc = getMSSSIM(x, Gz)
ae_acc = getMSSSIM(x, x_ae)
mse = getMSE(x, Gz)
tf.summary.scalar('ms-ssim_G', g_acc)
tf.summary.scalar('delta_ms-ssim', (g_acc - ae_acc))

Dx = discriminator(x, Dregularizer, DregularizerDense, batch_size)

Dg = discriminator(Gz, Dregularizer, DregularizerDense, batch_size, reuse_variables=True)

# losses
ae_loss = tf.losses.mean_squared_error(x, Gz)  # getMSSSIM(x, Gz)
s_ae_loss = tf.summary.scalar('loss_autoencoder', ae_loss)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))
d_loss = d_loss_real + d_loss_fake
s_d_loss = tf.summary.scalar('loss_discriminator', d_loss)

image_error = tf.where(tf.is_nan(g_acc), mse, (1 - g_acc))

alpha = tf.stop_gradient(getAlpha(tf.where(tf.is_nan(g_acc), (1 - mse), g_acc)))

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg))) + image_error * alpha
tf.summary.scalar('loss_generator', g_loss)

# Optimizers
optimizer_D = tf.train.AdamOptimizer(learning_rate=lrD)
optimizer_G = tf.train.AdamOptimizer(learning_rate=lrG)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train1 = optimizer_D.minimize(ae_loss)
    traind = optimizer_D.minimize(d_loss)
    traing = optimizer_G.minimize(g_loss)

training_init_op1 = iterator.make_initializer(dataset)

sess = tf.Session()

merged = tf.summary.merge_all()
train_writer1 = tf.summary.FileWriter('log_gan/autoencoder', sess.graph)
train_writer2 = tf.summary.FileWriter('log_gan/discriminator', sess.graph)
train_writer3 = tf.summary.FileWriter('log_gan/adversarial', sess.graph)

init1 = tf.global_variables_initializer()
init2 = tf.local_variables_initializer()
sess.run(init1)
sess.run(init2)

saver = tf.train.Saver()

# Model Training ------------------------------------------------------------------------------------------------------------------

sess.run(training_init_op1)

# alleno AE
num_batch = 50
for e in range(epochsAE):
    print("epoch: " + str(e) + " of " + str(epochsAE))
    for i in range(num_batch):
        age = e * num_batch + i

        _, summary = sess.run((train1, tf.summary.merge([s_ae_loss, s_gz, s_x_ae, s_x])))

        train_writer1.add_summary(summary, age)

# alleno discriminator da solo
num_batch = 1000
for e in range(epochsD):
    print("epoch: " + str(e) + " of " + str(epochsD))
    for i in range(num_batch):
        age = e * num_batch + i

        _, summary = sess.run((traind, s_d_loss))

        train_writer2.add_summary(summary, age)

# alleno GAN
num_batch = 4968
for e in range(epochsGAN):
    print("epoch: " + str(e) + " of " + str(epochsGAN))
    for i in range(num_batch):
        age = e * num_batch + i

        _, _, summary = sess.run((traind, traing, merged))  # forse anche g_acc

        train_writer3.add_summary(summary, age)

    if e % 2 == 1:
        lrD = tf.assign(lrD, lrD * 0.5)
        lrG = tf.assign(lrG, lrG * 0.5)

try:
    saver.save(sess, "/home/luca.marson1994/model/gan/model.ckpt")
    print("model saved successfully")
except Exception:
    pass

