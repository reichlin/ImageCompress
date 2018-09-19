"""

"""
import itertools
import tensorflow as tf
from os import path
import os
import sys
import re
import random
import argparse
import glob
from PIL import Image
from fjcommon import tf_helpers
from fjcommon import printing
from fjcommon import iterable_ext
from fjcommon import functools_ext


_JOB_SUBDIR_PREFIX = 'job_'
_TF_RECORD_EXT = 'tfrecord'
_DEFAULT_FEATURE_KEY = 'M'  # TODO: let this be a parameter


# TODO: let this be a parameter
_FRAME_ID_REGEX = r'(.*?)(\d{3,})\.png'  # filename, at least 3 digits right before the extension


def create_images_records_distributed(image_glob, job_id, num_jobs, out_dir, num_per_shard, num_per_example, feature_key):
    assert 1 <= job_id <= num_jobs, 'Invalid job_id: {}'.format(job_id)
    assert num_jobs >= 1, 'Invalid num_jobs: {}'.format(num_jobs)
    image_paths = _get_image_paths(image_glob, shuffle=num_per_example == 1)
    image_paths_per_job = iterable_ext.chunks(image_paths, num_chunks=num_jobs)
    image_paths_current_job = iterable_ext.get_element_at(job_id - 1, image_paths_per_job)
    consecutive_frames_paths = list(iterate_in_consecutive_frame_tuples(
        image_paths_current_job, num_consecutive=num_per_example))
    _shuffle_in_place(consecutive_frames_paths)
    feature_dicts = wrap_frames_in_feature_dicts(consecutive_frames_paths, feature_key=feature_key)

    out_dir_job = out_dir if num_jobs == 1 else path.join(out_dir, '{}{}'.format(_JOB_SUBDIR_PREFIX, job_id))
    create_records_with_feature_dicts(feature_dicts, out_dir_job, num_per_shard)


def join_created_images_records(out_dir, num_jobs):
    jobs_dirs_glob = path.join(out_dir, '{}*'.format(_JOB_SUBDIR_PREFIX))
    jobs_dirs = glob.glob(jobs_dirs_glob)
    assert len(jobs_dirs) == num_jobs, 'Expected {} subdirs, got {}'.format(num_jobs, jobs_dirs)

    records = glob.glob(path.join(jobs_dirs_glob, '*.{}'.format(_TF_RECORD_EXT)))
    assert len(records) > 0, 'Did not find any records in {}/{}_*'.format(out_dir, _JOB_SUBDIR_PREFIX)

    base_records_file_name = path.basename(records[0]).split('_')[0]  # get SHARD from out_dir/job_x/SHARD_xxx.ext
    for shard_number, records_p in enumerate(printing.ProgressPrinter('Moving records...', iter_list=records)):
        target_p = path.join(out_dir, _records_file_name(base_records_file_name, shard_number))
        os.rename(records_p, target_p)

    print('Removing empty job dirs...')
    list(map(os.removedirs, jobs_dirs))  # remove all job dirs, which are now empty

    print('Counting...')
    all_records_glob = path.join(out_dir, '*.{}'.format(_TF_RECORD_EXT))
    printing.print_join('{}: {}'.format(path.basename(p), _number_of_examples_in_record(p))
                        for p in sorted(glob.glob(all_records_glob)))


def _number_of_examples_in_record(p):
    return sum(1 for _ in tf.python_io.tf_record_iterator(p))


def _get_image_paths(image_glob, shuffle):
    paths = sorted(glob.glob(image_glob))
    assert len(paths) > 0, 'No matches for glob {}'.format(image_glob)
    if shuffle:
        _shuffle_in_place(paths)
    return paths


def _shuffle_in_place(paths):
    random.Random(6).shuffle(paths)  # shuffle deterministically, so that the returned list is consistent between jobs


def iterate_in_consecutive_frame_tuples(frame_paths, num_consecutive, frame_id_regex=_FRAME_ID_REGEX):
    if num_consecutive == 1:
        yield from ([p] for p in frame_paths)
        return

    pat = re.compile(frame_id_regex)

    def _get_path_base_id(p):
        m = pat.search(p)
        if not m:
            raise ValueError('Regex did not match: {} not in {}'.format(frame_id_regex, p))
        p_base, p_id = m.group(1, 2)
        return p_base, p_id

    for image_paths_slice in iterable_ext.sliced_iter(
            frame_paths, slice_len=num_consecutive, allow_smaller_final_slice=False):
        # image_paths_slice is a list of paths
        image_paths_slice_it = iter(image_paths_slice)
        p0 = next(image_paths_slice_it)
        p_0_base, p_0_id = _get_path_base_id(p0)
        p_prev_id = p_0_id
        for p_cur in image_paths_slice_it:
            p_cur_base, p_cur_id = _get_path_base_id(p_cur)
            if p_0_base != p_cur_base or int(p_prev_id) + 1 != int(p_cur_id):
                tf.logging.info('Non consequtive paths found in {}'.format(image_paths_slice))
                break
            p_prev_id = p_cur_id
        else:  # no-break, i.e., all consequtive
            yield image_paths_slice


def wrap_frames_in_feature_dicts(frame_paths, feature_key):
    keys = None
    for frame_paths_slice in frame_paths:
        if not keys:
            keys = keys_for_num_frames_per_example(len(frame_paths_slice), feature_key)
        yield {key: bytes_feature(open(p, 'rb').read()) for key, p in zip(keys, frame_paths_slice)}


def keys_for_num_frames_per_example(num_per_ex, feature_key):
    if num_per_ex > 1:
        return [feature_key + '_' + str(i) for i in range(num_per_ex)]
    else:
        return [feature_key]


def features_dict_for_decoding(num_per_ex, feature_key):
    return {key: tf.FixedLenFeature([], tf.string)
            for key in keys_for_num_frames_per_example(num_per_ex, feature_key)}


def wrap_bytes_in_feature_dicts(feature_bytes_it, feature_key=_DEFAULT_FEATURE_KEY):
    return ({feature_key: bytes_feature(b)} for b in feature_bytes_it)


def bytes_feature(b):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))


def int64_feature(i):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))


def create_records_with_feature_dicts(feature_dicts, out_dir, num_per_shard, max_shards=None, file_name='shard'):
    """
    :param feature_dicts: iterator yielding dictionaries with tf.train.Feature as values, to encode as features
    :param out_dir:
    :param num_per_shard:
    :param file_name:
    :return:
    """
    os.makedirs(out_dir, exist_ok=True)
    writer = None
    with printing.ProgressPrinter() as progress_printer:
        for count, feature in enumerate(feature_dicts):
            if count % num_per_shard == 0:
                progress_printer.finish_line()
                if writer:
                    writer.close()
                shard_number = count // num_per_shard
                if max_shards is not None and shard_number == max_shards:
                    print('Created {} shards...'.format(max_shards))
                    return
                record_p = path.join(out_dir, _records_file_name(file_name, shard_number))
                assert not path.exists(record_p), 'Record already exists! {}'.format(record_p)
                print('Creating {}...'.format(record_p))
                writer = tf.python_io.TFRecordWriter(record_p)
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            progress_printer.update((count % num_per_shard) / num_per_shard)
    if writer:
        writer.close()
    else:
        print('Nothing written...')


def create_record(in_paths, out_record_path, key=_DEFAULT_FEATURE_KEY, assert_size=None):
    with tf.python_io.TFRecordWriter(out_record_path) as writer:
        print('Writing {} images to {}...'.format(len(in_paths), out_record_path))
        for p in in_paths:
            if assert_size:
                w, h = Image.open(p).size
                assert w >= assert_size and h >= assert_size
            feature_dict = {key: bytes_feature(open(p, 'rb').read()),
                            'image/label': bytes_feature(open("/mnt/disks/disk2/ae_out/label" + p[1:], 'rb').read()),
                            'image/filename': bytes_feature(tf.compat.as_bytes(os.path.basename(p)))}
            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            writer.write(example.SerializeToString())


def _records_file_name(base_filename, shard_number):
    return '{}_{:08d}.{}'.format(base_filename, shard_number, _TF_RECORD_EXT)


def feature_to_image(feature):
    """ Use case: feature_to_img(read_records(...)) """
    im = tf.image.decode_image(feature, channels=3)
    im.set_shape((None, None, 3))
    return im


def features_to_images(features):
    """ Use case: features_to_img(read_records(..., num_per_ex>1)) """
    assert isinstance(features, list)
    return tf.stack([feature_to_image(feature) for feature in features], axis=0)


def extract_images(records_glob, max_images, out_dir, feature_key=_DEFAULT_FEATURE_KEY):
    tf.logging.set_verbosity(tf.logging.INFO)
    image = feature_to_image(read_records(records_glob, num_epochs=1, shuffle=False, feature_key=feature_key))
    image = tf.expand_dims(image, axis=0)  # make 'batched'
    index_iterator = range(max_images) if max_images else itertools.count()
    img_names_iterator = map('img_{:010d}'.format, index_iterator)
    img_saver = tf_helpers.ImageSaver(out_dir)
    with tf_helpers.start_queues_in_sess() as (sess, coord):
        img_fetcher = sess.make_callable(img_saver.get_fetch_dict(image))
        for img_name in img_names_iterator:
            tf.logging.info('Saving {}...'.format(img_name))
            img_saver.save(img_fetcher(), img_names=[img_name])


def read_records(records_glob, num_epochs=None, shuffle=True, feature_key=_DEFAULT_FEATURE_KEY, num_per_ex=1):
    features_dict = features_dict_for_decoding(num_per_ex, feature_key)
    features = read_records_with_features_dict(records_glob, features_dict, num_epochs, shuffle)
    return features[feature_key] if num_per_ex == 1 else [features[k] for k in sorted(features.keys())]


def read_records_with_features_dict(records_glob, features_dict, num_epochs=None, shuffle=True):
    reader = tf.TFRecordReader()
    records_paths = glob.glob(records_glob)
    assert records_paths, 'Did not find any records matching {}'.format(records_glob)
    if not shuffle:
        records_paths = sorted(records_paths)

    filename_queue = tf.train.string_input_producer(records_paths, num_epochs=num_epochs, shuffle=shuffle)
    _, serialized_example = reader.read(filename_queue)
    return tf.parse_single_example(serialized_example, features=features_dict)


def check(records_glob, out_dir, num_imgs_to_save, num_per_ex):
    features = read_records(records_glob, num_per_ex=num_per_ex)
    imgs = features_to_images(features)
    imsaver = tf_helpers.ImageSaver(out_dir)
    imsaver_fetch = imsaver.get_fetch_dict(imgs)
    with tf_helpers.start_queues_in_sess() as (sess, _):
        for run in range(num_imgs_to_save):
            print('Run {}...'.format(run))
            imsaver.save(sess.run(imsaver_fetch),
                         img_names=['{:02d}_{:02d}.png'.format(run, ex) for ex in range(num_per_ex)])


@functools_ext.print_generator()
def inspect(records_glob):
    all_keys = set()
    num_examples = 0
    for rec in sorted(glob.glob(records_glob)):
        rec_keys = set()
        count = -1
        yield 'Iterating {}...'.format(rec)

        for count, example in enumerate(
                map(tf.train.Example.FromString, tf.python_io.tf_record_iterator(rec))):
            rec_keys.update(set(example.features.feature))
        count += 1

        num_examples += count
        all_keys.update(rec_keys)

        yield 'Found {} examples, keys:'.format(count)
        yield from rec_keys
        yield None

    yield 'Found {} examples in total'.format(num_examples)
    yield from all_keys


def main(args):
    parser = argparse.ArgumentParser()
    mode_subparsers = parser.add_subparsers(dest='mode', title='Mode')
    # Make single image record ---
    parser_make_single = mode_subparsers.add_parser('mk_img_rec', help='Make single TF record from images.')
    parser_make_single.add_argument('paths', type=str, nargs='+')
    parser_make_single.add_argument('--out_record_path', '-o', type=str, required=True)
    parser_make_single.add_argument('--feature_key', type=str, default=_DEFAULT_FEATURE_KEY)
    parser_make_single.add_argument('--assert_size', metavar='LENGTH', type=str,
                                    help='If given, assert that for each image, width >= LENGTH and height >= LENGTH.')
    # Make image records ---
    parser_make = mode_subparsers.add_parser('mk_img_recs', help='Make TF records from images.')
    parser_make.add_argument('out_dir', type=str)
    parser_make.add_argument('image_glob', type=str)
    parser_make.add_argument('--num_per_shard', type=int, required=True)
    parser_make.add_argument('--num_per_ex', type=int, default=1)
    parser_make.add_argument('--feature_key', type=str, default=_DEFAULT_FEATURE_KEY)
    # Make image records, distributed ---
    parser_make_dist = mode_subparsers.add_parser(
            'mk_img_recs_dist',
            help='Like mk_img_recs but use --job_id and --num_jobs to split list of images into disjoint subsets and '
                 'only work on a subset at a time. For this to work in parallel, you need to use something like SGE '
                 'with array jobs. Needs to be followed by one call to tf_records join after all jobs have finished.')
    parser_make_dist.add_argument('out_dir', type=str)
    parser_make_dist.add_argument('image_glob', type=str)
    parser_make_dist.add_argument('--job_id', type=int, required=True)
    parser_make_dist.add_argument('--num_jobs', type=int, required=True)
    parser_make_dist.add_argument('--num_per_shard', type=int, required=True)
    parser_make_dist.add_argument('--num_per_ex', type=int, default=1)
    parser_make_dist.add_argument('--feature_key', type=str, default=_DEFAULT_FEATURE_KEY)
    # Join image records ---
    parser_join = mode_subparsers.add_parser('join')
    parser_join.add_argument('out_dir', type=str)
    parser_join.add_argument('--num_jobs', type=int, required=True)
    # Extract image Records ---
    parser_extract = mode_subparsers.add_parser('extract')
    parser_extract.add_argument('records_glob', type=str)
    parser_extract.add_argument('out_dir', type=str)
    parser_extract.add_argument('max_imgs', type=int)
    parser_extract.add_argument('--feature_key', type=str, default=_DEFAULT_FEATURE_KEY)
    # Inspect ---
    parser_inspect = mode_subparsers.add_parser('inspect')
    parser_inspect.add_argument('records_glob', type=str)
    #
    parser_check = mode_subparsers.add_parser('check', help='Check records with multiple examples, save images')
    parser_check.add_argument('records_glob', type=str)
    parser_check.add_argument('out_dir', type=str)
    parser_check.add_argument('num_imgs', type=int)
    parser_check.add_argument('num_per_ex', type=int)
    # ---
    flags = parser.parse_args(args)
    if flags.mode == 'mk_img_rec':
        create_record(flags.paths, flags.out_record_path, flags.feature_key, flags.assert_size)
    elif flags.mode == 'mk_img_recs':
        create_images_records_distributed(flags.image_glob, job_id=1, num_jobs=1, out_dir=flags.out_dir,
                                          num_per_shard=flags.num_per_shard, num_per_example=flags.num_per_ex,
                                          feature_key=flags.feature_key)
    elif flags.mode == 'mk_img_recs_dist':
        create_images_records_distributed(flags.image_glob, flags.job_id, flags.num_jobs, flags.out_dir,
                                          flags.num_per_shard, flags.num_per_ex, feature_key=flags.feature_key)
    elif flags.mode == 'join':
        join_created_images_records(flags.out_dir, flags.num_jobs)
    elif flags.mode == 'extract':
        extract_images(flags.records_glob, flags.max_imgs, flags.out_dir, flags.feature_key)
    elif flags.mode == 'inspect':
        inspect(flags.records_glob)
    elif flags.mode == 'check':
        check(flags.records_glob, flags.out_dir, flags.num_imgs, flags.num_per_ex)
    else:
        parser.print_usage()


if __name__ == '__main__':
    main(sys.argv[1:])
