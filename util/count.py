from os import listdir
import tensorflow as tf

c = 0
for fn in listdir('/mnt/disks/disk2/ae_out/records/train'):
  for record in tf.python_io.tf_record_iterator('/mnt/disks/disk2/records/train/' + fn):
     c += 1

print(c)
