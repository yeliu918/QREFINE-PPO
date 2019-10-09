# -*- coding: UTF-8 -*-
import json
import jieba

jieba.dt.tmp_dir = '/home/ye/cache'
import sys
import pickle
import numpy as np
import re

import unicodedata

reload(sys)
sys.setdefaultencoding('utf8')

a_q = {}
cq_nq_pair = {}

f = open('../data/Wiki/wiki_ids_Q10', 'rb')
data = pickle.load(f)
noisy_Id = data['noisy_Id']
clean_Id = data['clean_Id']
answer_Id = data['answer_Id']
vocab = data['vocab']
embd = data['embd']


example_noisy=noisy_Id[0:5]
example_clean = clean_Id[0:5]
example_answer = answer_Id[0:5]

# cosine_result = []
# for i in range(len(noisy_Id)):
#     if clean_Id[i] != noisy_Id[i]:
#         cosine_each = [clean_Id[i], noisy_Id[i], answer_Id[i]]
#         cosine_result.append(cosine_each)
#
# print("finish cosine similiarity")
# print("pair number", len(cosine_result))
#
# question1 = []
# question2 = []
# answer = []
# cosine_sim = []
# for pair in cosine_result:
#     if pair not in cosine_sim:
#         cosine_sim.append(pair)
#
# for pair in cosine_sim:
#     question1.append(pair[0])
#     question2.append(pair[1])
#     answer.append(pair[2])
#     # cosine_sim.append(pair[3])
#
# print("the number of question:", len(question1))

output_path = '../data/Wiki/Wiki_example'
result = {'noisy_Id': example_noisy, 'clean_Id': example_clean,
          'answer_Id': example_answer, 'vocab': vocab, 'embd': embd}
output = open(output_path, 'wb')
pickle.dump(result, output)
output.close()
