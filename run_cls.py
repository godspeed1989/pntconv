import os
import shutil
import readline

MODEL_dict = {
    'pnt_full': 'pointnet_cls',
    'pnt_basic': 'pointnet_cls_basic',
    'pnt_basic_rnn': 'pointnet_cls_basic_rnn',
    'pcnn_cls': 'pointcnn_cls',
    'dgcnn': 'dgcnn_cls',
    'dgxconv': 'dgxconv_cls',
    'pntconv': 'pntconv_cls'
}

DATASET_PATH_dict = {
    'm10': 'data/modelnet10_ply_n_2048',
    'mnist': '/mine/mnist_to_h5/h5',
    'm40': '/mine/modelnet40_ply_hdf5_2048'
}

model = 'pntconv'
dataset = 'm40'
# set number of points
if dataset in ['m10', 'm40']:
    num_point = 1024
elif dataset in ['mnist']:
    num_point = 200
else:
    assert('invalid dataset')
# set number of classes
if dataset in ['m40']:
    num_class = 40
elif dataset in ['m10', 'mnist']:
    num_class = 10
else:
    assert('invalid dataset')

num_neighbor = 32

flags = {
    'gpu': 0,                       # GPU to use [default: GPU 0]
    'max_epoch':  51,               # Epoch to run [default: 201]
    'batch_size': 64,               # Batch Size during training
    #
    'learning_rate': 0.001,         # Initial learning rate [default: 0.001]
    'momentum': 0.9,                # Initial learning rate [default: 0.9]
    'optimizer': 'adam',            # adam or momentum [default: adam]
    'decay_step': 200000,           # Decay step for lr decay [default: 200000]
    'decay_rate': 0.7,              # Decay rate for lr decay [default: 0.7]
    'regularization_loss': True,    # With regularization loss
    'reg_weight_alpha': 0.01,       # Weight of regularization loss
    # model and dataset
    'model': MODEL_dict[model],
    'dataset_path': DATASET_PATH_dict[dataset],
    'num_point': num_point,
    'num_class': num_class,
    'num_neighbor': num_neighbor,
    'with_LC': True,
    'with_mean_rep': True,
    'with_dist_rep': True,
    'with_angle_rep': True,
    'with_shortcut': False,
    'with_pc_feature': False,
    'with_nn_feature': False,
    # for evaluation
    'model_path': './weights',
    'dump_dir': 'dump_'+model+'_'+dataset+'_n'+str(num_point),
    # for visualization
    'visual_dir': 'visual_'+model+'_'+dataset+'_n'+str(num_point),

    # data augment
    'aug_rotation': '1',  # 1 from web, 2 pointnet
    # load weights to continue training
    'contiune_training': False,
    'weights_dir': './log_before',
    # Rotation Transform Net
    'enable_rtn': False,
    'rtn_weights_dir': './RTN/log',

    # for PointCNN
    'xconv_sel': 'mnist',
    # for dgcnn
    'dgcnn_k': 20,
    # for dgxconv
    'dgx_k': 20,
    'dgx_transform_input': True,
    'dgx_params_sel': 'modelnet',
    'dgx_x_transform': True,
    'dgx_rnn_n_hidden': 128,
    'dgx_prev_layer_feat': True
}

if dataset == 'mnist':
    flags['arg_rotation'] = 0

def input_with_default(prompt, prefill=''):
    readline.set_startup_hook(lambda: readline.insert_text(prefill))
    try:
        return input(prompt)
    finally:
        readline.set_startup_hook()

def run():
    import train_cls
    flags['log_dir'] = 'log_'+model+'_'+dataset+'_n'+str(flags['num_point'])+'_nn'+str(flags['num_neighbor'])
    if flags['with_LC'] == False:
        flags['log_dir'] += '_nonLC'
    flags['log_dir'] += '_1' if flags['with_mean_rep'] else '_0'
    flags['log_dir'] += '_1' if flags['with_dist_rep'] else '_0'
    flags['log_dir'] += '_1' if flags['with_angle_rep'] else '_0'
    #
    flags['log_dir'] += '_sc' if flags['with_shortcut'] else ''
    flags['log_dir'] += '_pc' if flags['with_pc_feature'] else ''
    flags['log_dir'] += '_nn' if flags['with_nn_feature'] else ''
    train_cls.run(flags)

if False:
    user_input = input('Train, Evaluate or Visual?: [T/e/v]')
    if user_input in ['Train', 'train', 'T', 't', '']:
        flags['log_dir'] = input_with_default('log_dir? ', flags['log_dir'])
        if os.path.exists(flags['log_dir']) and flags['contiune_training'] is not True:
            shutil.rmtree(flags['log_dir'])
        run()
    elif user_input in ['Evaluate', 'evaluate', 'E', 'e']:
        import eval_cls
        eval_cls.run(flags)
    elif user_input in ['Visual', 'visual', 'V', 'v']:
        import vis_cls
        vis_cls.run(flags)
    else:
        print('invalid input')
elif False:
    # (num points, neighbors, )
    test_cases = [
        [1024, 32, False, False, False],
        [1024, 32, True, False, False],
        [1024, 32, False, True, False],
        [1024, 32, False, False, True],
        [1024, 32, False, True, True],
        [1024, 32, True, False, True],
        [1024, 32, True, True, False],
    ]
    for p in test_cases:
        print(p)
        flags['num_point'] = p[0]
        flags['num_neighbor'] = p[1]
        flags['with_mean_rep'] = p[2]
        flags['with_dist_rep'] = p[3]
        flags['with_angle_rep'] = p[4]
        run()
else:
    # (shortcut, pc, nn)
    test_cases = [
        [True, False, False],
        [True, False, True],
        [True, True, False],
        [False, False, True],
        [False, True, False],
        [False, True, True],
        [True, True, True]
    ]
    for p in test_cases:
        flags['with_shortcut'] = p[0]
        flags['with_pc_feature'] = p[1]
        flags['with_nn_feature'] = p[2]
        run()
