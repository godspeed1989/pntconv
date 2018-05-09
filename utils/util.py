import numpy as np
import tensorflow as tf
blue = lambda x:'\033[94m' + x + '\033[0m'

def dict2str(dc):
    res = ''
    for key, val in dc.items():
        if callable(val):
            v = val.__name__
        else:
            v = str(val)
        res += '%20s: %s\n' % (str(key), v)
    return res

def run_tetris(is_training_pl, pointclouds_pl, labels_pl, loss, predict, accuracy):
    # 数据集
    tetris = [[(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
              [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)], # chiral_shape_2
              [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
              [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # T
              [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # zigzag
              [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]]  # L
    dataset = [np.array(points_) for points_ in tetris]

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print(blue('Parameter number: {}'.format(parameter_num)))
    # 优化配置
    learning_rate_base = 0.001
    decay_steps = 8000
    learning_rate_min = 0.00001
    decay_rate = 0.7
    epsilon = 1e-4
    global_step = tf.Variable(0, trainable=False, name='global_step')
    lr_exp_op = tf.train.exponential_decay(learning_rate_base, global_step,
                                           decay_steps, decay_rate, staircase=True)
    lr_clip_op = tf.maximum(lr_exp_op, learning_rate_min)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_clip_op, epsilon=epsilon)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=global_step)

    # 会话执行
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    gpu_options = tf.GPUOptions(allow_growth=True)
    cfg = tf.ConfigProto(log_device_placement=False,
                         allow_soft_placement=True,
                         gpu_options=gpu_options)
    max_epochs = 2001
    print_freq = 100
    with tf.Session(config=cfg) as sess:
        sess.run(init_op)
        # training
        for epoch in range(max_epochs):
            loss_sum = 0.
            accuracy_sum = 0.
            for label, shape in enumerate(dataset):
                label = np.expand_dims(label, axis=0) # (1,)
                shape = np.expand_dims(shape, axis=0) # (1, 4, 3)
                loss_value, accuracy_value, _ = sess.run(fetches=[loss, accuracy, train_op],
                                                         feed_dict={pointclouds_pl: shape,
                                                                    labels_pl: label,
                                                                    is_training_pl: True})
                loss_sum += loss_value
                accuracy_sum += accuracy_value
            if epoch % print_freq == 0:
                print(blue('Epoch %d: loss = %.3f accuracy = %.3f' %
                      (epoch, loss_sum / len(dataset), accuracy_sum / len(dataset))))


if __name__ == '__main__':
    a = {
        'item1': 1,
        'item2': 2,
        'function': dict2str
    }
    print(dict2str(a))
