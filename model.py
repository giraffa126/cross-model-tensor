# coding: utf8
import sys
import os
import tensorflow as tf

from tensorflow.contrib.slim import nets

slim = tf.contrib.slim

class CrossModel(object):
    """ CrossModel
    """
    def __init__(self, vocab_size=None, embedding_size=256, hidden_size=256, loss_type="rank_loss"):
        self.loss_type = loss_type
        # inputs
        self.text = tf.placeholder(tf.int32, [None, None], name="text")
        self.img_pos = tf.placeholder(tf.float32, [None, 224, 224, 3], name="img_pos")
        self.img_neg = tf.placeholder(tf.float32, [None, 224, 224, 3], name="img_neg")

        # text
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            # pretrain emb
            self.embedding = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), 
                    trainable=True, name="emb_mat")
            self.embedding_in = tf.placeholder(tf.float32, [vocab_size, embedding_size])
            self.embedding_init = self.embedding.assign(self.embedding_in)

            # text
            self.text_emb = tf.nn.embedding_lookup(self.embedding, self.text)
            self.text_pool = tf.reduce_sum(self.text_emb, axis=1)

        # img
        with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()) :
            pos_net, pos_endpoints = nets.resnet_v2.resnet_v2_152(self.img_pos, 
                    num_classes=None, is_training=True, reuse=tf.AUTO_REUSE)
            self.pos_net_pool = tf.squeeze(pos_net, axis=[1, 2])

            neg_net, neg_endpoints = nets.resnet_v2.resnet_v2_152(self.img_neg, 
                    num_classes=None, is_training=True, reuse=tf.AUTO_REUSE)
            self.neg_net_pool = tf.squeeze(neg_net, axis=[1, 2])

        with tf.variable_scope("forword", reuse=tf.AUTO_REUSE):
            self.text_vec = self.fc_layer(self.text_pool, shape=[embedding_size, hidden_size], 
                    name="text", activation_function=tf.tanh)
            self.img_pos_vec = self.fc_layer(self.pos_net_pool, shape=[2048, hidden_size], 
                    name="img", activation_function=tf.tanh)
            self.img_neg_vec = self.fc_layer(self.neg_net_pool, shape=[2048, hidden_size], 
                    name="img", activation_function=tf.tanh)

        with tf.variable_scope("consine", reuse=tf.AUTO_REUSE):
            self.left = self.cosine_similarity(self.text_vec, self.img_pos_vec)
            self.right = self.cosine_similarity(self.text_vec, self.img_neg_vec)

        with tf.variable_scope("loss"): 
            if self.loss_type == "rank_loss":
                self.loss = self.rank_loss(self.left, self.right)
            elif self.loss_type == "log_loss":
                self.loss = self.log_loss(self.left, self.right)
            elif self.loss_type == "hinge_loss":
                self.loss = self.hinge_loss(self.left, self.right)

    def fc_layer(self, inputs, shape, name, activation_function=None):
        """ fc layer
        """
        with tf.variable_scope("%s" % name, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable(name="%s_w" % name, shape=shape, 
                    initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
            biases = tf.get_variable(name="%s_b" % name, shape=[shape[1]], 
                    initializer=tf.constant_initializer(1e-5))
            wx_plus_b = tf.add(tf.matmul(inputs, weights), biases)
            if activation_function is None:
                outputs = wx_plus_b
            else:
                outputs = activation_function(wx_plus_b)
            return outputs 

    def rank_loss(self, score_pos, score_neg):
        """ rank loss
        """
        pred_diff = score_pos - score_neg
        loss = tf.log1p(tf.exp(pred_diff)) - pred_diff
        return tf.reduce_mean(loss)
    
    def log_loss(self, score_pos, score_neg):
        """ log_loss
        """
        return tf.reduce_mean(tf.nn.sigmoid(score_neg - score_pos))
    
    def hinge_loss(self, score_pos, score_neg, margin=0.1):
        """ hinge loss
        """
        return tf.reduce_mean(tf.maximum(0., score_neg + 
                    margin - score_pos))

    def cosine_similarity(self, x1, x2, eps=1e-12):
        """ calc sim
        """
        w1 = tf.sqrt(tf.reduce_sum(x1 ** 2, axis=1))
        w2 = tf.sqrt(tf.reduce_sum(x2 ** 2, axis=1))
        w12 = tf.reduce_sum(x1 * x2, axis=1)
        return (w12 / (w1 * w2 + eps)) * 5
    
