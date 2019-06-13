# coding: utf8
import sys
import os
import numpy as np
import tensorflow as tf
import argparse
from model import CrossModel 
from reader import DataReader

def train(args):
    model = CrossModel(vocab_size=args.vocab_size)
    saver = tf.train.Saver()
    init = tf.group(tf.global_variables_initializer(), 
            tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, tf.train.latest_checkpoint(args.model_path))
        test_reader = DataReader(args.vocab_path, args.test_data_path, args.image_path, 
                args.vocab_size, args.batch_size)
        for query in test_reader.extract_img_emb_generator():
            _vec_list = sess.run(model.img_pos_vec,
                    feed_dict={model.img_pos: query})
            for vec in _vec_list:
                print(" ".join([str(x) for x in vec]))


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_path", type=str, default="./model")
    parser.add_argument("--train_data_path", type=str, default="./data/train.txt")
    parser.add_argument("--test_data_path", type=str, default="./data/query1.txt")
    parser.add_argument("--image_path", type=str, default="./data/images")
    parser.add_argument("--vocab_path", type=str, default="./data/vocab.txt")
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--save_interval", type=int, default="10000")
    parser.add_argument("--eval_interval", type=int, default="1000")
    args = parser.parse_args()
    train(args)
    
