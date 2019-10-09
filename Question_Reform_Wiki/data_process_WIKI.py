#!/usr/bin/python
# -*- coding: UTF-8 -*-
import re
import csv
import os
import numpy as np
# from nltk import *
import pickle
# from openpyxl import load_workbook
import xlrd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import jieba
import jieba.posseg as pseg
import sys
reload(sys)
sys.setdefaultencoding('utf8')


class DataManager(object):
    '''
  read data and return json object
  '''

    def load_data(self, qfile, afile):
        noisy_question,clean_question,answer = self.__read_data(qfile,afile)
        question_num = len(noisy_question)
        print'data loaded'
        print 'num question and answer:', question_num
        # return noisy_question,clean_question,answer


    def process_data(self, noisy_q, clean_q, answer, productfile):
        vocab = []
        vocab.insert(0, 'PAD')
        vocab.insert(1, 'EOS')
        print '***building for noisy sentence...'
        vocab, noisy_index = self.__build_vocab_index(noisy_q, productfile, vocab)
        print '***building for clean sentence...'
        vocab, clean_index = self.__build_vocab_index(clean_q, productfile, vocab)
        print '***building for answer sentence...'
        vocab, answer_index = self.__build_vocab_index(answer, productfile, vocab)
        print '***processing done'
        return noisy_index, clean_index, answer_index, vocab


    def clean_sen(self, sen):
        sen = re.sub('<[^<]+?>', '', sen)  # remove html tags
        sen = re.sub('[a-zA-z]+://[^\s]*', '', sen)  # remove link
        sen = re.sub('[a-zA-z]', '', sen)  # []
        sen = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。?、~@#￥%……&*（）:; ：；？《）① ② ③ < > = →／ ※《》“”()»〔〕-]+".decode("utf8"), "".decode("utf8"),
                      sen)
        return sen

    def __read_data(self, question_txt):
        ##check whether q and a file exist
        with open(question_txt, 'r') as f:
            while True:
                line = f.readline()  # 逐行读取
                if not line:
                    break
                print line,


    def __build_vocab_index(self, data,productfile,vocab):
        Ids = []
        Ids_len = []
        d_num = len(data)
        for i in range(d_num):
            text = data[i]
            sentence_id = []
            # jieba.load_userdict(productfile)
            words = jieba.cut(text)
            for n_word, w in enumerate(words):
                if w not in vocab:
                    vocab.append(w)
                sentence_id.append(vocab.index(w))
            # sentence_length = n_word + 2
            Ids.append(sentence_id)
            # Ids_len.append(sentence_length)
        return vocab, Ids


def main():
    if 1: ##orignal dataset
        qfile = '../data/Wiki/questions.txt'
        wfile='../data/Wiki/word_alignments.txt'
        with open(wfile, 'r') as f:
            while True:
                line = f.readline()  # 逐行读取
                # print re.split('\t', line)[1]
                print line



if __name__ == '__main__':
    main()