# coding: utf8
import sys
import os
import logging
import numpy as np
import tensorflow as tf
import argparse
from model import CrossModel 
from reader import DataReader
from reader import load_embedding

slim = tf.contrib.slim

def get_variables_available_in_checkpoint(variables,
                                          checkpoint_path,
                                          include_global_step=True):
    if not isinstance(variables, dict):
        raise ValueError('`variables` is expected to be a dict.')
    
    # Available variables
    ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
    ckpt_vars_to_shape_map = ckpt_reader.get_variable_to_shape_map()
    if not include_global_step:
        ckpt_vars_to_shape_map.pop(tf.GraphKeys.GLOBAL_STEP, None)
    vars_in_ckpt = {}
    for variable_name, variable in sorted(variables.items()):
        if variable_name in ckpt_vars_to_shape_map:
            if ckpt_vars_to_shape_map[variable_name] == variable.shape.as_list():
                vars_in_ckpt[variable_name] = variable
            else:
                logging.warning('Variable [%s] is avaible in checkpoint, but '
                                'has an incompatible shape with model '
                                'variable. Checkpoint shape: [%s], model '
                                'variable shape: [%s]. This variable will not '
                                'be initialized from the checkpoint.',
                                variable_name, 
                                ckpt_vars_to_shape_map[variable_name],
                                variable.shape.as_list())
        else:
            logging.warning('Variable [%s] is not available in checkpoint',
                            variable_name)
    return vars_in_ckpt


def init_variables_from_checkpoint(pretrain_path, checkpoint_exclude_scopes=None):
    exclude_patterns = None
    if checkpoint_exclude_scopes:
        exclude_patterns = [scope.strip() for scope in 
                            checkpoint_exclude_scopes.split(',')]
    variables_to_restore = tf.global_variables()
    variables_to_restore.append(slim.get_or_create_global_step())
    variables_to_init = tf.contrib.framework.filter_variables(
        variables_to_restore, exclude_patterns=exclude_patterns)
    variables_to_init_dict = {var.op.name: var for var in variables_to_init}
    
    available_var_map = get_variables_available_in_checkpoint(
        variables_to_init_dict, pretrain_path, 
        include_global_step=False)
    tf.train.init_from_checkpoint(pretrain_path, available_var_map)
    

def train(args):
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    #tf.reset_default_graph()
    model = CrossModel(vocab_size=args.vocab_size)
    # optimizer
    train_step = tf.contrib.opt.LazyAdamOptimizer(learning_rate=args.learning_rate).minimize(model.loss)
    saver = tf.train.Saver()
    loss_summary = tf.summary.scalar("train_loss", model.loss)
    init = tf.group(tf.global_variables_initializer(), 
            tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)
        #variables_to_restore = slim.get_variables_to_restore()
        #restore_fn = slim.assign_from_checkpoint_fn(args.pretrain_path, variables_to_restore)
        #restore_fn(sess)
        #sess.run(tf.global_variables_initializer())
        init_variables_from_checkpoint(args.pretrain_path)

        _writer = tf.summary.FileWriter(args.logdir, sess.graph)
        # init embedding
        embedding = load_embedding(args.emb_path, args.vocab_size, 256)
        _ = sess.run(model.embedding_init, feed_dict={model.embedding_in: embedding})
        print("loading pretrain emb succ.")

        # summary
        summary_op = tf.summary.merge([loss_summary])
        step = 0
        for epoch in range(args.epochs):
            train_reader = DataReader(args.vocab_path, args.train_data_path, args.image_data_path, 
                    args.vocab_size, args.batch_size, is_shuffle=True)
            print("train reader load succ.")
            for train_batch in train_reader.batch_generator():
                query, pos, neg = train_batch

                _, _loss, _summary = sess.run([train_step, model.loss, summary_op],
                        feed_dict={model.text: query, model.img_pos: pos, model.img_neg: neg})
                _writer.add_summary(_summary, step)
                step += 1

                # test
                sum_loss = 0.0
                iters = 0
                summary = tf.Summary()
                if step % args.eval_interval == 0:
                    print("Epochs: {}, Step: {}, Train Loss: {:.4}".format(epoch, step, _loss))

                    test_reader = DataReader(args.vocab_path, args.test_data_path, args.image_data_path, 
                            args.vocab_size, args.batch_size)
                    for test_batch in test_reader.batch_generator():
                        query, pos, neg = test_batch
                        _loss = sess.run(model.loss,
                                feed_dict={model.text: query, model.img_pos: pos, model.img_neg: neg})
                        sum_loss += _loss
                        iters += 1
                    avg_loss = sum_loss / iters
                    summary.value.add(tag="test_loss", simple_value=avg_loss)
                    _writer.add_summary(summary, step)
                    print("Epochs: {}, Step: {}, Test Loss: {:.4}".format(epoch, step, sum_loss / iters))
                if step % args.save_interval == 0:
                    save_path = saver.save(sess, "{}/model.ckpt".format(args.model_path), global_step=step)
                    print("Model save to path: {}/model.ckpt".format(args.model_path))


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--vocab_size", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_path", type=str, default="./model")
    parser.add_argument("--pretrain_path", type=str, default="./pretrain_model/resnet_v2_152.ckpt")
    parser.add_argument("--train_data_path", type=str, default="./data/train.txt")
    parser.add_argument("--test_data_path", type=str, default="./data/test.txt")
    parser.add_argument("--image_data_path", type=str, default="./data/images")
    parser.add_argument("--vocab_path", type=str, default="./data/vocab.txt")
    parser.add_argument("--emb_path", type=str, default="./data/embedding.txt")
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--save_interval", type=int, default="1000")
    parser.add_argument("--eval_interval", type=int, default="100")
    args = parser.parse_args()
    train(args)
    
