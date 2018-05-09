"""
    control neural network evalutation process
"""
import os
import sys
import importlib
import numpy as np
import h5py
import tensorflow as tf
from progressbar import ProgressBar

# PATH add './' './utils'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
from util import dict2str
from color import colors

def init(f):
    global LOG_FOUT
    global MODEL
    global MODEL_CONF
    global TEST_FILES
    global FLIST_FOUT

    # IMPORT network module
    sys.path.append(f['model_path'])
    module = importlib.import_module(f['model'])
    MODEL = getattr(module, f['model'])
    MODEL_CONF = getattr(module, f['model'] + '_conf')

    # MAKE log directory
    if not os.path.exists(f['dump_dir']):
        os.mkdir(f['dump_dir'])
    # CREATE log file
    LOG_FOUT = open(os.path.join(f['dump_dir'], 'log_evaluate.txt'), 'w')
    LOG_FOUT.write(dict2str(f))
    FLIST_FOUT = open(os.path.join(f['dump_dir'], 'list_evaluate.txt'), 'w')

    TEST_FILES = provider.getDataFiles( \
        os.path.join(BASE_DIR, os.path.join(f['dataset_path'], 'test_files.txt')))

# -----------------------------------------------------------------------------------

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(colors.info(out_str))

# -----------------------------------------------------------------------------------

def evaluate(f):
    is_training = False

    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(f['gpu'])):
            # FLAG is_training
            is_training_pl = tf.placeholder(tf.bool, shape=())

            conf = MODEL_CONF(is_training=is_training_pl, bn_decay=None, **f)

            # GET model
            model = MODEL(conf)
            pointclouds_pl, labels_pl, loss, predict, accuracy = model.build_graph()
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', accuracy)

        # CREATE session
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.InteractiveSession(
            config=tf.ConfigProto(
                log_device_placement=False,
                allow_soft_placement=True,
                gpu_options=gpu_options,
            )
        )

        # CREATE Saver to restore all the variables.
        log_string('create tf saver')
        saver = tf.train.Saver()

        # Restore variables from disk.
        checkpoint_file = tf.train.latest_checkpoint(f['model_path'])
        saver.restore(sess, checkpoint_file)
        log_string('Model restored.')

        ops = {
            # input
            'pointclouds_pl': pointclouds_pl,
            'labels_pl': labels_pl,
            'is_training_pl': is_training_pl,
            # output
            'predict': predict,
            'loss': loss,
        }

        def exists_and_is_not_none(obj, attribute):
            return hasattr(obj, attribute) and getattr(obj, attribute) is not None
        if exists_and_is_not_none(model, 'aux'):
            ops = {**ops, **model.aux}
            has_aux = True
        else:
            has_aux = False
        eval_one_epoch(f, sess, ops, has_aux)

# -----------------------------------------------------------------------------------

def eval_one_epoch(f, sess, ops, has_aux):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(f['num_class'])]
    total_correct_class = [0 for _ in range(f['num_class'])]

    for fn in range(len(TEST_FILES)):
        data_file = TEST_FILES[fn]
        log_string('E---- %d/%d -----' % (fn+1, len(TEST_FILES)))
        # load data and labels
        current_data, current_label = provider.loadDataFile(data_file, f['dataset_path'])
        current_data = current_data[:, 0:f['num_point'], :]
        current_label = np.squeeze(current_label)

        num_batches = current_data.shape[0] // f['batch_size']

        dump_file = h5py.File(os.path.join(f['dump_dir'], data_file), 'w')
        FLIST_FOUT.write(data_file + '\n')
        dump_file.create_dataset('pred_and_label', [num_batches * f['batch_size'], 2], dtype=int)
        if has_aux:
            dump_file.create_dataset('global_feature', [num_batches * f['batch_size'], 512])
            dump_file.create_dataset('max_idx', [num_batches * f['batch_size'], 512], dtype=int)

        bar = ProgressBar(maxval=num_batches).start()
        for batch_idx in range(num_batches):
            start_idx = batch_idx * f['batch_size']
            end_idx = (batch_idx+1) * f['batch_size']
            # feed data
            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            # fetched data & run
            if has_aux:
                fetches = [ops['predict'],
                           ops['loss'],
                           ops['max_idx'],
                           ops['global_feature']]
                pred_val, loss_val, max_idx, global_feature = sess.run(fetches=fetches, feed_dict=feed_dict)
                dump_file['max_idx'][start_idx:end_idx,:] = max_idx
                dump_file['global_feature'][start_idx:end_idx,:] = global_feature
            else:
                fetches = [ops['predict'],
                           ops['loss']]
                pred_val, loss_val = sess.run(fetches=fetches, feed_dict=feed_dict)
            # collect
            pred_val = np.argmax(pred_val, axis=1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += f['batch_size']
            loss_sum += (loss_val * f['batch_size'])
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)
            #
            pred_val = np.expand_dims(pred_val, axis=-1)
            true_val = np.expand_dims(current_label[start_idx:end_idx], axis=-1)
            dump_file['pred_and_label'][start_idx:end_idx,:] = np.concatenate((pred_val, true_val), axis=1)
            #
            bar.update(batch_idx)
        bar.finish()

        dump_file.close()

    class_accuracies = np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % np.mean(class_accuracies))
    for i in range(class_accuracies.shape[0]):
        log_string('[%d] %f' % (i, class_accuracies[i]))

# -----------------------------------------------------------------------------------

def run(flags):
    print(colors.info('pid: %s' % str(os.getpid())))
    init(flags)
    evaluate(flags)
    LOG_FOUT.close()
    FLIST_FOUT.close()
