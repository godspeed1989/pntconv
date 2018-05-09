"""
    control neural network train process
"""
import os
import sys
import importlib
import numpy as np
import tensorflow as tf
from progressbar import ProgressBar

# PATH add './'  './models'  './utils'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import pc_util
from util import dict2str
from color import colors

def init(f):
    global LOG_FOUT
    global MODEL
    global MODEL_CONF
    global TRAIN_FILES
    global TEST_FILES

    # IMPORT network module
    module = importlib.import_module(f['model'])
    MODEL = getattr(module, f['model'])
    MODEL_CONF = getattr(module, f['model'] + '_conf')
    MODEL_FILE = os.path.join(BASE_DIR, 'models', f['model'] + '.py')

    # MAKE log directory
    if not os.path.exists(f['log_dir']):
        os.mkdir(f['log_dir'])
    os.system('cp %s %s' % (MODEL_FILE, f['log_dir'])) # bkp of model def
    # CREATE log file
    LOG_FOUT = open(os.path.join(f['log_dir'], 'log_train.txt'), 'a')
    LOG_FOUT.write(dict2str(f))

    # GET dataset files' list
    TRAIN_FILES = provider.getDataFiles(os.path.join(f['dataset_path'], 'train_files.txt'))
    TEST_FILES = provider.getDataFiles(os.path.join(f['dataset_path'], 'test_files.txt'))

# -----------------------------------------------------------------------------------

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(colors.info(out_str))

def get_learning_rate(f, batch):
    learning_rate = tf.train.exponential_decay(
        f['learning_rate'],       # Base learning rate
        batch * f['batch_size'],  # Current index into the dataset
        f['decay_step'],          # Decay step
        f['decay_rate'],          # Decay rate
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(f, batch):
    BN_INIT_DECAY = 0.5
    BN_DECAY_CLIP = 0.99
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * f['batch_size'],
        f['decay_step'],          # Decay step
        f['decay_rate'],          # Decay rate
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

# -----------------------------------------------------------------------------------

def train(f):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(f['gpu'])):
            # FLAG is_training
            is_training_pl = tf.placeholder(tf.bool, shape=())
            # BATCH index, the `global_step == batch` parameter to minimize
            batch = tf.get_variable('batch', shape=[], dtype=tf.int64,
                                    initializer=tf.constant_initializer(0),
                                    trainable=False)

            bn_decay = get_bn_decay(f, batch)
            tf.summary.scalar('bn_decay', bn_decay)

            conf = MODEL_CONF(is_training=is_training_pl,  bn_decay=bn_decay, **f)

            # GET model
            model = MODEL(conf)
            pointclouds_pl, labels_pl, loss, predict, accuracy = model.build_graph()
            if f['regularization_loss']:
                reg_losses = tf.get_default_graph().get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                regularization_loss = tf.reduce_sum(reg_losses)
                loss = loss + f['reg_weight_alpha'] * regularization_loss
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', accuracy)

            # CREATE training optimizer
            learning_rate = get_learning_rate(f, batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if f['optimizer'] == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=f['momentum'])
            elif f['optimizer'] == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_optimizer = optimizer.minimize(loss, global_step=batch)

        parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
        log_string('Parameter number: {}'.format(parameter_num))

        # CREATE session
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.InteractiveSession(
            config=tf.ConfigProto(
                log_device_placement=False,
                allow_soft_placement=True,
                gpu_options=gpu_options,
            )
        )

        # CREATE Saver to save and restore all the variables.
        log_string('create tf saver')
        saver = tf.train.Saver(max_to_keep=3)

        # CREATE summary writers
        log_string('create summary writer')
        merged_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(f['log_dir'], 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(f['log_dir'], 'test'), sess.graph)

        # INIT variables to start a new training
        log_string('initialize variables')
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        ops = {
            # input
            'pointclouds_pl': pointclouds_pl,
            'labels_pl': labels_pl,
            'is_training_pl': is_training_pl,
            # output
            'predict': predict,
            'loss': loss,
            'merged_summary': merged_summary,
            # inout
            'train_optimizer': train_optimizer,
            'batch': batch
        }

        # START to train
        log_string('start training')
        best_accuracy = 0
        for epoch in range(f['max_epoch']):
            log_string('************** EPOCH %03d **************' % (epoch))
            sys.stdout.flush()

            train_one_epoch(f, sess, ops, train_writer)
            accuracy = eval_one_epoch(f, sess, ops, test_writer)

            # SAVE check point
            if accuracy >= best_accuracy:
                save_path = saver.save(sess, os.path.join(f['log_dir'], 'model.ckpt'), global_step=epoch)
                log_string('Model saved in file: %s' % save_path)
                best_accuracy = accuracy

# -----------------------------------------------------------------------------------

def train_one_epoch(f, sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train files order
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)

    for fn in range(len(TRAIN_FILES)):
        data_file = TRAIN_FILES[train_file_idxs[fn]]
        log_string('T---- %d/%d -----' % (fn+1, len(TRAIN_FILES)))
        # load data and labels
        current_data, current_label = provider.loadDataFile(data_file, f['dataset_path'])
        current_data = current_data[:, 0:f['num_point'], :]
        current_label = np.squeeze(current_label)
        # shuffle
        current_data, current_label, _ = provider.shuffle_data(current_data, current_label)
        current_label = np.squeeze(current_label)

        num_batches = current_data.shape[0] // f['batch_size']

        total_correct = 0
        total_seen = 0
        loss_sum = 0

        bar = ProgressBar(maxval=num_batches).start()
        for batch_idx in range(num_batches):
            start_idx = batch_idx * f['batch_size']
            end_idx = (batch_idx+1) * f['batch_size']
            # Augment batched point clouds by rotation and jittering
            rotated_data = pc_util.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            jittered_data = pc_util.jitter_point_cloud(rotated_data)
            shuffled_data = pc_util.shuffle_point_cloud(jittered_data)
            # feed data
            feed_dict = {ops['pointclouds_pl']: shuffled_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            # fetched data
            fetches = [ops['predict'],
                       ops['loss'],
                       ops['train_optimizer'],
                       ops['merged_summary'],
                       ops['batch']]
            # run
            pred_val, loss_val, _, summary, step = sess.run(
                fetches=fetches, feed_dict=feed_dict)
            # collect
            train_writer.add_summary(summary, step)
            #
            pred_val = np.argmax(pred_val, axis=1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += f['batch_size']
            loss_sum += loss_val
            #
            bar.update(batch_idx)
        bar.finish()

        log_string('train mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('train accuracy: %f' % (total_correct / float(total_seen)))

# -----------------------------------------------------------------------------------

def eval_one_epoch(f, sess, ops, test_writer):
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

        bar = ProgressBar(maxval=num_batches).start()
        for batch_idx in range(num_batches):
            start_idx = batch_idx * f['batch_size']
            end_idx = (batch_idx+1) * f['batch_size']
            # feed data
            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            # fetched data
            fetches = [ops['predict'],
                       ops['loss'],
                       ops['merged_summary'],
                       ops['batch']]
            # run
            pred_val, loss_val, summary, step = sess.run(fetches=fetches, feed_dict=feed_dict)
            # collect
            test_writer.add_summary(summary, step)
            #
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += f['batch_size']
            loss_sum += (loss_val*f['batch_size'])
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)
            #
            bar.update(batch_idx)
        bar.finish()

    loss = loss_sum / float(total_seen)
    accuracy = total_correct / float(total_seen)
    log_string('eval mean loss: %f' % loss)
    log_string('eval accuracy: %f'% accuracy)
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class) /
                                                   np.array(total_seen_class, dtype=np.float))))
    return accuracy

# -----------------------------------------------------------------------------------

def run(flags):
    print(colors.info('pid: %s' % str(os.getpid())))
    init(flags)
    train(flags)
    LOG_FOUT.close()
