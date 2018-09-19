import os
import sys
import h5py
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# PATH add './' './utils'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
from util import dict2str
from color import colors
from tsne import tsne

def init(f):
    global LOG_FOUT
    global DUMP_FILES
    global TEST_FILES

    # MAKE log directory
    if not os.path.exists(f['visual_dir']):
        os.mkdir(f['visual_dir'])
    # CREATE log file
    LOG_FOUT = open(os.path.join(f['visual_dir'], 'log_visualization.txt'), 'w')
    LOG_FOUT.write(dict2str(f))

    DUMP_FILES = provider.getDataFiles( \
        os.path.join(BASE_DIR, os.path.join(f['dump_dir'], 'list_evaluate.txt')))
    TEST_FILES = provider.getDataFiles( \
        os.path.join(BASE_DIR, os.path.join(f['dataset_path'], 'test_files.txt')))

# -----------------------------------------------------------------------------------

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(colors.info(out_str))

# -----------------------------------------------------------------------------------

def visualization(f, cfmtrx, maxpnt, tsne, weights_hist):
    pred_and_label = np.empty([0, 2], dtype=np.int32)
    max_idx = np.empty([0, 512], dtype=np.int32)
    global_feature = np.empty([0, 512])

    # CONCAT data from all files
    for fn in range(len(DUMP_FILES)):
        dump_file = DUMP_FILES[fn]
        log_string('V---- %d/%d -----' % (fn+1, len(DUMP_FILES)))
        # load dump file
        fin = h5py.File(os.path.join(f['dump_dir'], dump_file))
        # concatenate
        pred_and_label = np.concatenate((pred_and_label, fin['pred_and_label'][:]), axis=0)
        max_idx = np.concatenate((max_idx, fin['max_idx'][:]), axis=0)
        global_feature = np.concatenate((global_feature, fin['global_feature'][:]), axis=0)
        fin.close()
    log_string('pred_and_label {}'.format(pred_and_label.shape)) # (N, 2)
    log_string('max_idx {}'.format(max_idx.shape))               # (N, C)
    log_string('global_feature {}'.format(global_feature.shape)) # (N, 512)

    # PLOT confusion matrix
    if cfmtrx:
        log_string('PLOT confusion matrix')
        cmat = np.zeros([f['num_class'], f['num_class']])
        for i in range(pred_and_label.shape[0]):
            pred_val = pred_and_label[i][0]
            true_val = pred_and_label[i][1]
            cmat[true_val, pred_val] += 1
        plot_confusion_matrix(cmat, class_names=label_modelnet.keys(), normalize=True, title='')

    # PLOT max point
    if maxpnt:
        log_string('PLOT maximum point')
        fdump = h5py.File(os.path.join(f['dump_dir'], DUMP_FILES[0]))
        ftest = h5py.File(os.path.join(f['dataset_path'], TEST_FILES[0]))
        max_idx = fdump['max_idx'][:]
        points = ftest['data'][:, 0:f['num_point'], :]
        assert max_idx.shape[0] == points.shape[0]
        # random choose to show
        shows = np.random.random_integers(0, max_idx.shape[0], 20)
        pc_list = []
        for s in range(shows.shape[0]):
            i = shows[s]
            pc = points[i,:,:]
            pidx = np.unique(max_idx[i, :])
            color_tab = np.full((f['num_point']), 35)
            color_tab[pidx] = 99
            plot_point_cloud(pc, color_tab)
        fdump.close()
        ftest.close()

    # PLOT T-SNE
    if tsne:
        log_string('PLOT T-SNE')
        tlabel = []
        tfeature = []
        for i in range(pred_and_label.shape[0]):
            if pred_and_label[i][0] == pred_and_label[i][1]:
                tlabel.append(pred_and_label[i][0])
                tfeature.append(global_feature[i])
        tlabel = np.array(tlabel)
        tfeature = np.array(tfeature)
        log_string('tlabel {}'.format(tlabel.shape))     # (N,)
        log_string('tfeature {}'.format(tfeature.shape)) # (N, C)
        plot_TSNE(tlabel, tfeature, f['num_class'])

    # PLOT
    if weights_hist:
        import tensorflow as tf
        from tensorflow import pywrap_tensorflow
        checkpoint_file = tf.train.latest_checkpoint(f['model_path'])
        # Read data from checkpoint file
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_file)
        var_to_shape_map = reader.get_variable_to_shape_map()
        def plot_weights_hist(ax, data):
            data = np.abs(data)
            data = np.squeeze(data)
            data = np.sum(data, axis=1, keepdims=False)
            data = data / np.sum(data)
            print(data.shape)
            s = ['$x$', '$y$', '$z$', '$x^2$', '$y^2$', '$z^2$', '$x^3$', '$y^3$', '$z^3$',
                 '$\overline{X}$', '$\overline{Y}$', '$\overline{Z}$',
                 '$\overline{X^2}$', '$\overline{Y^2}$', '$\overline{Z^2}$',
                 '$\overline{X^3}$', '$\overline{Y^3}$', '$\overline{Z^3}$',
                 '$xy$', '$yz$', '$zx$',
                 '$x^2y$', '$y^2z$', '$z^2x$',
                 '$x^2z$', '$y^2x$', '$z^2y$',
                 '$l2$', '$d_x$', '$d_y$', '$d_z$', '$\\theta_x$', '$\\theta_y$', '$\\theta_z$']
            ax.bar(s, data)
        # Print tensor name and values
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.set_xlabel('(a)')
        ax2 = fig.add_subplot(212)
        ax2.set_xlabel('(b)')
        for key in var_to_shape_map:
            if key == 'LinearCombLayer/conv2d_128_pc/weights':
                print(key, reader.get_tensor(key).shape)
                plot_weights_hist(ax1, reader.get_tensor(key))
            if key == 'LinearCombLayer/conv2d_128_nn/weights':
                print(key, reader.get_tensor(key).shape)
                plot_weights_hist(ax2, reader.get_tensor(key))
        #plt.tight_layout()
        plt.show()

# -----------------------------------------------------------------------------------
def discrete_cmap(N, base_cmap='cubehelix'):
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def plot_TSNE(labels, features, num_class):
    Y = tsne(features, no_dims=2, initial_dims=512, perplexity=20.0, max_iter=1000)
    #
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sct = ax.scatter(Y[:, 0], Y[:, 1], s=20, c=labels, cmap=discrete_cmap(num_class))
    cbar = plt.colorbar(sct, ticks=range(num_class))
    labels = []
    i = 0
    for k in label_modelnet.keys():
        i += 1
        labels.append(k)
        if i >= num_class:
            break
    cbar.set_ticklabels(labels)
    cbar.set_clim(-0.5, num_class - 0.5)
    plt.tight_layout()
    plt.show()

def plot_point_cloud(points, color_table):
    """
        points is a Nx3 numpy array
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=15, alpha=1,
                c=color_table, cmap=plt.cm.rainbow, vmin=0, vmax=100)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_names=None,
                          normalize=False, plt_text=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks([])
        plt.yticks(tick_marks, class_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    if plt_text:
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------------

def run(flags):
    print(colors.info('pid: %s' % str(os.getpid())))
    init(flags)
    visualization(flags, cfmtrx=True, maxpnt=True, tsne=False, weights_hist=True)
    LOG_FOUT.close()

# -----------------------------------------------------------------------------------

label_modelnet = {
    # ModelNet 10/40
    'bathtub':      0,
    'bed':          1,
    'chair':        2,
    'desk':         3,
    'dresser':      4,
    'monitor':      5,
    'night_stand':  6,
    'sofa':         7,
    'table':        8,
    'toilet':       9,
    # ModelNet 40
    'airplane':     10,
    'bench':        11,
    'bookshelf':    12,
    'bottle':       13,
    'bowl':         14,
    'car':          15,
    'cone':         16,
    'cup':          17,
    'curtain':      18,
    'door':         19,
    'flower_pot':   20,
    'glass_box':    21,
    'guitar':       22,
    'keyboard':     23,
    'lamp':         24,
    'laptop':       25,
    'mantel':       26,
    'person':       27,
    'piano':        28,
    'plant':        29,
    'radio':        30,
    'range_hood':   31,
    'sink':         32,
    'stairs':       33,
    'stool':        34,
    'tent':         35,
    'tv_stand':     36,
    'vase':         37,
    'wardrobe':     38,
    'xbox':         39
}
