import os
import shutil
import readline

DATASET_PATH_dict = {
    'm10': 'data/modelnet10_ply_n_2048',
    'mnist': '/mine/mnist_to_h5/h5',
    'm40': '/mine/modelnet40_ply_hdf5_2048'
}

model = 'pntconv_cls'
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

flags = {
    'gpu': 0,                       # GPU to use [default: GPU 0]
    'max_epoch': 401,               # Epoch to run [default: 401]
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
    'model': 'pntconv_cls',
    'dataset_path': DATASET_PATH_dict[dataset],
    'num_point': num_point,
    'num_class': num_class,
    # for train
    'log_dir': 'log_'+model+'_'+dataset+'_n'+str(num_point),
    # for evaluation
    'model_path': 'log_'+model+'_'+dataset+'_n'+str(num_point),
    'dump_dir': 'dump_'+model+'_'+dataset+'_n'+str(num_point),
    # for visualization
    'visual_dir': 'visual_'+model+'_'+dataset+'_n'+str(num_point),
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

def input_with_default(prompt, prefill=''):
    readline.set_startup_hook(lambda: readline.insert_text(prefill))
    try:
        return input(prompt)
    finally:
        readline.set_startup_hook()

user_input = input('Train, Evaluate or Visual?: [T/e/v]')
if user_input in ['Train', 'train', 'T', 't', '']:
    flags['log_dir'] = input_with_default('log_dir? ', flags['log_dir'])
    if os.path.exists(flags['log_dir']):
        shutil.rmtree(flags['log_dir'])
    import train_cls
    train_cls.run(flags)
elif user_input in ['Evaluate', 'evaluate', 'E', 'e']:
    import eval_cls
    eval_cls.run(flags)
elif user_input in ['Visual', 'visual', 'V', 'v']:
    import vis_cls
    vis_cls.run(flags)
else:
    print('invalid input')
