import os
import sys
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../fps'))
import tf_util
from util import run_tetris
from color import colors

''' Compute pairwise distance of a point cloud
input
    A shape is (N, P_A, C), B shape is (N, P_B, C)
return
    D shape is (N, P_A, P_B)
'''
def batch_distance_matrix_general(A, B=None):
    with tf.variable_scope('batch_distance_matrix_general'):
        if B is None:
            B = A
        r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
        r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
        m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
        D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
    return D

""" Get KNN based on the pairwise distance
Args:
    pairwise distance: (batch_size, num_points, num_points)
    k: int
Returns:
    nearest neighbors: (batch_size, num_points, k)
"""
def knn(adj_matrix, k, with_first=False):
    with tf.variable_scope('knn'):
        neg_adj = -adj_matrix
        if with_first:
            _, nn_idx = tf.nn.top_k(neg_adj, k=k, sorted=True)
        else:
            _, nn_idx = tf.nn.top_k(neg_adj, k=k+1, sorted=True)
            nn_idx = nn_idx[:, :, 1:]
    return nn_idx

def perm_xyz(data, permutation):
    with tf.variable_scope('perm_xyz'):
        # *** batch indices
        # [[0],[1],...,[n]]
        batch_size = data.get_shape()[0]
        batch_idx = tf.reshape(tf.range(batch_size, dtype=tf.int32), shape=[batch_size, 1])
        # [[0,0,0],[1,1,1],...,[n,n,n]]
        batch_idx = tf.tile(batch_idx, [1, 3])
        # [[0],[0],[0],[1],[1],[1],...,[n],[n],[n]] (N*3, 1)
        batch_idx = tf.reshape(batch_idx, shape=[-1, 1])
        # *** perm indices (3,)
        perm_idx = tf.Variable(permutation, dtype=tf.int32, trainable=False, name='perm')
        # (3*N, 1)
        perm_idx = tf.tile(perm_idx, [batch_size])
        perm_idx = tf.expand_dims(perm_idx, axis=-1)
        # *** concat (3*N, 2)
        indices = tf.concat([batch_idx, perm_idx], axis=1)
        # *** gather
        # (N, P, 3) -> (N, 3, P)
        tdat = tf.transpose(data, perm=[0, 2, 1])
        tdat = tf.gather_nd(tdat, indices)
        tdat = tf.reshape(tdat, [batch_size, 3, -1])
        # (N, 3, P) -> (N, P, 3)
        tdat = tf.transpose(tdat, perm=[0, 2, 1])
    return tdat

""" Construct nearest neighbor for each point
Args:
    pts: (N, P, 3)
    nn_idx: (N, P, k) 每个点临近k个点的索引
Returns:
    (N, P, C), (N, P, K, C)
"""
def get_nn(pts, nn_idx):
    with tf.variable_scope('get_nn'):
        batch_size = pts.get_shape()[0].value
        num_points = pts.get_shape()[1].value
        num_dims = pts.get_shape()[2].value
        k = nn_idx.get_shape()[2].value

        pts_xyz = pts
        pts_yzx = perm_xyz(pts_xyz, [1,2,0])
        pts_zxy = perm_xyz(pts_xyz, [2,0,1])

        # 收集临近的点 (N, P, K, 3)
        # [0*P, 1*P,..., (N-1)*P]
        batch_idx_ = tf.range(batch_size) * num_points
        # [[[0]], [[P]], ..., [[(N-1)*P]]] = (N, 1, 1)
        batch_idx_ = tf.reshape(batch_idx_, [batch_size, 1, 1])
        # 因为flatten了point_cloud，所以需要加上batch_idx
        # indices = (N, 1, 1) + (N, p, K) = (N, p, K)
        pts_xyz_flat = tf.reshape(pts_xyz, [-1, num_dims]) # flatten
        pts_yzx_flat = tf.reshape(pts_yzx, [-1, num_dims])
        pts_zxy_flat = tf.reshape(pts_zxy, [-1, num_dims])
        neighbors_indices = nn_idx + batch_idx_
        pts_xyz_neighbors = tf.gather(pts_xyz_flat, neighbors_indices)
        pts_yzx_neighbors = tf.gather(pts_yzx_flat, neighbors_indices)
        pts_zxy_neighbors = tf.gather(pts_zxy_flat, neighbors_indices)
        # 复制原始的点作为中心点 (N, p, K, 3)
        # (N, p, 3) -> (N, p, 1, 3)
        pts_xyz_central = tf.tile(tf.expand_dims(pts_xyz, axis=-2), [1, 1, k, 1])
        pts_yzx_central = tf.tile(tf.expand_dims(pts_yzx, axis=-2), [1, 1, k, 1])
        pts_zxy_central = tf.tile(tf.expand_dims(pts_zxy, axis=-2), [1, 1, k, 1])

        # Feature Organization
        # (N, P, C)
        # pow
        pts_xyz2 = tf.pow(pts_xyz, 2)
        pts_xyz3 = tf.pow(pts_xyz, 3)
        # mean
        pts_xyz_mean = tf.reduce_mean(pts_xyz, axis=1, keepdims=True)
        pts_xyz_mean = tf.tile(pts_xyz_mean, [1,num_points,1])
        pts_xyz2_mean = tf.reduce_mean(pts_xyz2, axis=1, keepdims=True)
        pts_xyz2_mean = tf.tile(pts_xyz2_mean, [1,num_points,1])
        pts_xyz3_mean = tf.reduce_mean(pts_xyz3, axis=1, keepdims=True)
        pts_xyz3_mean = tf.tile(pts_xyz3_mean, [1,num_points,1])
        # xy, yz, zx
        pts_p2 = tf.multiply(pts_xyz, pts_yzx)
        # xxy, yyz, zzx
        pts_p31 = tf.multiply(pts_xyz2, pts_yzx)
        # xxz, yyx, zzy
        pts_p32 = tf.multiply(pts_xyz2, pts_zxy)
        # L2
        pts_l2 = tf.reduce_sum(pts_xyz2, axis=-1, keepdims=True)
        pts_l2 = tf.sqrt(pts_l2)
        # distance
        pts_yzx2 = tf.pow(pts_yzx, 2)
        pts_dist = pts_xyz2 + pts_yzx2
        pts_dist = tf.sqrt(pts_dist)
        # angle
        pts_l2_ext = tf.add(pts_l2, 0.00001) # prevent div zero
        pts_l2_ext = tf.tile(pts_l2_ext, [1,1,3])
        pts_angle = tf.div(pts_dist, pts_l2_ext)
        pts_angle = tf.acos(pts_angle)
        # concat
        pc_feature = tf.concat([pts_xyz, pts_xyz2, pts_xyz3,
                                pts_xyz_mean, pts_xyz2_mean, pts_xyz3_mean,
                                pts_p2, pts_p31, pts_p32,
                                pts_l2, pts_dist, pts_angle],
                                axis=-1)

        # (N, P, K, C)
        edge_xyz_feature = pts_xyz_neighbors - pts_xyz_central # centroid
        edge_yzx_feature = pts_yzx_neighbors - pts_yzx_central
        edge_zxy_feature = pts_zxy_neighbors - pts_zxy_central
        # pow
        edge_xyz_feature2 = tf.pow(edge_xyz_feature, 2)
        edge_xyz_feature3 = tf.pow(edge_xyz_feature, 3)
        # mean
        edge_xyz_feature_mean = tf.reduce_mean(edge_xyz_feature, axis=2, keepdims=True)
        edge_xyz_feature_mean = tf.tile(edge_xyz_feature_mean, [1,1,k,1])
        edge_xyz_feature2_mean = tf.reduce_mean(edge_xyz_feature2, axis=2, keepdims=True)
        edge_xyz_feature2_mean = tf.tile(edge_xyz_feature2_mean, [1,1,k,1])
        edge_xyz_feature3_mean = tf.reduce_mean(edge_xyz_feature3, axis=2, keepdims=True)
        edge_xyz_feature3_mean = tf.tile(edge_xyz_feature3_mean, [1,1,k,1])
        # xy, yz, zx
        edge_feature_p2 = tf.multiply(edge_xyz_feature, edge_yzx_feature)
        # xxy, yyz, zzx
        edge_feature_p31 = tf.multiply(edge_xyz_feature2, edge_yzx_feature)
        # xxz, yyx, zzy
        edge_feature_p32 = tf.multiply(edge_xyz_feature2, edge_zxy_feature)
        # L2
        edge_feature_l2 = tf.reduce_sum(edge_xyz_feature2, axis=-1, keepdims=True)
        edge_feature_l2 = tf.sqrt(edge_feature_l2)
        # distance
        edge_yzx_feature2 = tf.pow(edge_yzx_feature, 2)
        edge_feature_dist = edge_xyz_feature2 + edge_yzx_feature2
        edge_feature_dist = tf.sqrt(edge_feature_dist)
        # angle
        edge_feature_l2_ext = tf.add(edge_feature_l2, 0.00001) # prevent div zero
        edge_feature_l2_ext = tf.tile(edge_feature_l2_ext, [1,1,1,3])
        edge_feature_angle = tf.div(edge_feature_dist, edge_feature_l2_ext)
        edge_feature_angle = tf.acos(edge_feature_angle)

        # concat
        nn_feature = tf.concat([edge_xyz_feature, edge_xyz_feature2, edge_xyz_feature3,
                                edge_xyz_feature_mean, edge_xyz_feature2_mean, edge_xyz_feature3_mean,
                                edge_feature_p2, edge_feature_p31, edge_feature_p32,
                                edge_feature_l2, edge_feature_dist, edge_feature_angle],
                                axis=-1)

        return pc_feature, nn_feature

def LinearCombLayer(pc_feat, nn_feat, channels, conf):
    with tf.variable_scope('LinearCombLayer', reuse=False):
        # (N, P, C) -> (N, P, 1, C)
        pc_feat = tf.expand_dims(pc_feat, axis=-2)
        # (N, P, 1\K, C) -> (N, P, 1\K, channels)
        conv2d_scope = 'conv2d_' + str(channels)
        pc_feature = tf_util.conv2d(pc_feat, channels, [1,1],
                                    padding='VALID', stride=[1,1],
                                    bn=True, is_training=conf.is_training,
                                    scope=conv2d_scope + '_pc', bn_decay=conf.bn_decay,
                                    activation_fn=None)
        nn_feature = tf_util.conv2d(nn_feat, channels, [1,1],
                                    padding='VALID', stride=[1,1],
                                    bn=True, is_training=conf.is_training,
                                    scope=conv2d_scope + '_nn', bn_decay=conf.bn_decay,
                                    activation_fn=None)
        # (N, P, 1, channels) -> (N, P, channels)
        pc_feature = tf.squeeze(pc_feature, axis=2)
        # (N, P, K, channels) -> (N, P, channels)
        nn_feature = tf.reduce_max(nn_feature, axis=2, keepdims=False)
    print(colors.warning('LinearCombLayerPC'), colors.info(pc_feat.shape), colors.info(pc_feature.shape))
    print(colors.warning('LinearCombLayerNN'), colors.info(nn_feat.shape), colors.info(nn_feature.shape))
    return pc_feature, nn_feature

def FeatureMapLayer(pc_feat, nn_feat, channels, conf):
    with tf.variable_scope('FeatureMapLayer', reuse=False):
        # (N, P, C) -> (N, P, 1, C)
        pc_feat = tf.expand_dims(pc_feat, axis=-2)
        nn_feat = tf.expand_dims(nn_feat, axis=-2)
        # (N, P, 1, C) -> (N, P, 1, channels)
        conv2d_scope = 'conv2d_' + str(channels)
        pc_feature = tf_util.conv2d(pc_feat, channels, [1,1],
                                    padding='VALID', stride=[1,1],
                                    bn=True, is_training=conf.is_training,
                                    scope=conv2d_scope + '_pc', bn_decay=conf.bn_decay,
                                    activation_fn=tf.nn.elu)
        nn_feature = tf_util.conv2d(nn_feat, channels, [1,1],
                                    padding='VALID', stride=[1,1],
                                    bn=True, is_training=conf.is_training,
                                    scope=conv2d_scope + '_nn', bn_decay=conf.bn_decay,
                                    activation_fn=tf.nn.elu)
        # (N, P, 1, channels) -> (N, P, channels)
        pc_feature = tf.squeeze(pc_feature, axis=2)
        nn_feature = tf.squeeze(nn_feature, axis=2)
    print(colors.warning('FeatureMapLayerPC'), colors.info(pc_feat.shape), colors.info(pc_feature.shape))
    print(colors.warning('FeatureMapLayerNN'), colors.info(nn_feat.shape), colors.info(nn_feature.shape))
    return pc_feature, nn_feature

def SetConvLayer(pt_feat, channels, tag, conf, bnorm=True, activation=tf.nn.elu):
    with tf.variable_scope(tag):
        # (N, P, C) -> (N, P, 1, C)
        pt_feat = tf.expand_dims(pt_feat, axis=-2)
        # (N, P, 1, C) -> (N, P, 1, channels)
        conv2d_scope = 'conv2d_' + str(channels)
        pt_feature = tf_util.conv2d(pt_feat, channels, [1,1],
                                    padding='VALID', stride=[1,1],
                                    bn=bnorm, is_training=conf.is_training,
                                    scope=conv2d_scope, bn_decay=conf.bn_decay,
                                    activation_fn=activation)
        # (N, P, 1, channels) -> (N, P, channels)
        pt_feature = tf.squeeze(pt_feature, axis=2)
        print(colors.warning(tag), colors.info(pt_feat.shape), colors.info(pt_feature.shape))
    return pt_feature

def GetNNFeature(pts, K, tag):
    with tf.variable_scope(tag):
        # get nearest neighbors index (N, K)
        adj_matrix = batch_distance_matrix_general(pts)
        nn_idx = knn(adj_matrix, k=K, with_first=False)
        # extract neighbor (N, P, C) -> (N, P, C+C*K)
        nn_feat = get_nn(pts, nn_idx=nn_idx)
    return nn_feat

"""
    pass configuration between train and model
"""
class pntconv_cls_conf():
    def __init__(self, **args):
        self.num_point = args['num_point']
        self.is_training = args['is_training']
        self.bn_decay = args['bn_decay']
        self.num_class = args['num_class']
        self.batch_size = args['batch_size']

    def exists_and_is_not_none(self, attribute):
        return hasattr(self, attribute) and getattr(self, attribute) is not None

"""
    Classification model
"""
class pntconv_cls():
    def __init__(self, conf):
        print(colors.info('init %s model' % self.__class__.__name__))
        self.configuration = conf

    def build_graph(self):
        c = self.configuration
        # GET input placeholder
        pointclouds_pl, labels_pl = self.get_input()
        # GET model
        predict = self.get_model(pointclouds_pl)
        # GET loss
        loss = self.get_loss(predict, labels_pl)
        # GET accuracy
        correct = tf.equal(tf.argmax(predict, 1), tf.to_int64(labels_pl))
        accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / tf.cast(c.batch_size, tf.float32)
        #
        return pointclouds_pl, labels_pl, loss, predict, accuracy

    def get_input(self):
        c = self.configuration
        pointclouds_pl = tf.placeholder(tf.float32, shape=(c.batch_size, c.num_point, 3))
        labels_pl = tf.placeholder(tf.int32, shape=(c.batch_size,))
        return pointclouds_pl, labels_pl

    def get_model(self, point_cloud):
        """ Classification PointNet, input is BxNx3, output Bxnum_class """
        c = self.configuration
        print(colors.warning('Input'), colors.info(point_cloud.shape))

        pc_feat, nn_feat = GetNNFeature(point_cloud, 32, 'GetNNFeature')
        print(colors.warning('PC_Feature'), colors.info(pc_feat.shape))
        print(colors.warning('NN_Feature'), colors.info(nn_feat.shape))
        # linear combination
        pc_feat, nn_feat = LinearCombLayer(pc_feat, nn_feat, 128, c)
        # nonlinear feature mapping
        pc_feat, nn_feat = FeatureMapLayer(pc_feat, nn_feat, 128, c)
        #
        pt_feat = tf.concat([point_cloud, pc_feat, nn_feat], axis=2)
        #
        pt_feat = SetConvLayer(pt_feat, 256, 'SetConv_1', c)
        #
        pt_feat = SetConvLayer(pt_feat, 384, 'SetConv_2', c)
        #
        pt_feat = SetConvLayer(pt_feat, 512, 'SetConv_3', c)

        # (N, P, C) -> (N, C)
        net = tf.reduce_max(pt_feat, axis=1, keepdims=False)
        self.aux = {}
        self.aux['max_idx'] = tf.argmax(pt_feat, axis=1)
        self.aux['global_feature'] = net
        # MLP on global point cloud vector
        with tf.variable_scope('MLP_classify'):
            net = tf_util.fully_connected(net, 512, bn=True, is_training=c.is_training,
                                        scope='fc1', bn_decay=c.bn_decay)
            net = tf_util.dropout(net, keep_prob=0.5, is_training=c.is_training,
                                scope='dp1')
            net = tf_util.fully_connected(net, 256, bn=True, is_training=c.is_training,
                                        scope='fc2', bn_decay=c.bn_decay)
            net = tf_util.dropout(net, keep_prob=0.5, is_training=c.is_training,
                                scope='dp2')
            net = tf_util.fully_connected(net, c.num_class, activation_fn=None, scope='fc3')
        return net

    def get_loss(self, pred, label):
        """ pred: B*NUM_CLASSES, label: B, """
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
        classify_loss = tf.reduce_mean(loss)
        return classify_loss

if __name__=='__main__':
    # 模型配置信息
    is_training_pl = tf.placeholder(tf.bool, shape=())
    conf = pntconv_cls_conf(num_point=4, bn_decay=None, batch_size=1,
                                is_training=is_training_pl, num_class=8)
    # 构建模型
    dgc = pntconv_cls(conf)
    pointclouds_pl, labels_pl, loss, predict, accuracy = dgc.build_graph()

    run_tetris(is_training_pl, pointclouds_pl, labels_pl, loss, predict, accuracy)
