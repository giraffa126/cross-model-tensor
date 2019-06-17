# coding: utf8
import sys
import os
import collections
import random
from PIL import Image
import numpy as np

from image_process import process_image

def load_embedding(filename, vocab_size, embedding_size):
    embedding = np.zeros([vocab_size, embedding_size])
    with open(filename) as ifs:
        idx = 0
        for line in ifs:
            cols = line.strip().split(" ")
            vec = list(map(float, cols[1:]))
            embedding[idx] = vec
            idx += 1
    return np.asarray(embedding)

class DataReader(object):
    def __init__(self, vocab_path, data_path, image_path, vocab_size=500000,
            batch_size=512, max_seq_len=32, is_shuffle=False):
        """ init
        """
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._max_seq_len = max_seq_len
        self._is_shuffle = is_shuffle
        self._build_vocab(vocab_path)
        self._data = self._build_data(data_path)
        self._image_path = image_path

    def _build_vocab(self, filename):
        self._word_to_id = {}
        with open(filename, "r") as ifs:
            idx = 0
            for line in ifs:
                cols = line.strip().split()
                word = cols[0]
                self._word_to_id[word] = idx
                idx += 1

    def _build_data(self, filename):
        with open(filename, "r") as ifs:
            lines = ifs.readlines()
            data = list(map(lambda x: x.strip().split("\t"), lines))
            if self._is_shuffle:
                random.shuffle(data)
        return data

    def _padding_batch(self, batch):
        # neg sample
        for idx, line in enumerate(batch[1]):
            neg_idx = random.randint(0, len(batch[1]) -1)
            while neg_idx == idx:
                neg_idx = random.randint(0, len(batch[1]) -1)
            batch[2].append(batch[1][neg_idx])
        return batch

    def batch_generator(self):
        curr_size = 0
        batch = [[], [], []]
        for line in self._data:
            if len(line) != 2:
                continue
            curr_size += 1
            query, imageid = line
            query_list = query.split("\1\2")
            random.shuffle(query_list)
            query = query_list[0]
            query_ids = [self._word_to_id.get(x, self._word_to_id["<unk>"]) for x in query.split()]
            if len(query_ids) > self._max_seq_len:
                query_ids = query_ids[:self._max_seq_len]
            else:
                query_ids = query_ids + [self._word_to_id["<pad>"]] * (self._max_seq_len - len(query_ids))

            img_array = process_image(self._image_path + "/%s.jpg" % imageid,
                    mode="train", 
                    color_jitter=False, 
                    rotate=False)
            if img_array.shape != (224, 224, 3):
                continue
            batch[0].append(query_ids)
            batch[1].append(img_array)
            if curr_size >= self._batch_size:
                yield self._padding_batch(batch)
                batch = [[], [], []]
                curr_size = 0
        if curr_size > 0:
            yield self._padding_batch(batch)


    def extract_img_emb_generator(self):
        curr_size = 0
        batch = []
        for line in self._data:
            if len(line) != 1:
                continue
            imageid = line[0]
            img_array = process_image(self._image_path + "/%s.jpg" % imageid,
                    mode="train", 
                    color_jitter=False, 
                    rotate=False)
            if img_array.shape != (224, 224, 3):
                continue
            curr_size += 1
            batch.append(img_array)
            if curr_size >= self._batch_size:
                yield batch
                batch = []
                curr_size = 0
        if curr_size > 0:
            yield batch

    def extract_emb_generator(self):
        curr_size = 0
        batch = []
        for line in self._data:
            if len(line) != 1:
                continue
            curr_size += 1
            query = line[0]
            query_ids = [self._word_to_id.get(x, self._word_to_id["<unk>"]) for x in query.split()]
            query_ids = query_ids + [self._word_to_id["<pad>"]] * (self._max_seq_len - len(query_ids))
            batch.append(query_ids)
            if curr_size >= self._batch_size:
                yield batch
                batch = []
                curr_size = 0
        if curr_size > 0:
            yield batch

if __name__ == "__main__":
    reader = DataReader("data/vocab.txt", "data/test.txt", "data/images")
    for batch in reader.batch_generator():
        for idx, line in enumerate(batch[0]):
            print(batch[0][idx], batch[1][idx], batch[2][idx])
           
