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

f = open('../result/Huawei/large_unique0.95_with', 'rb')
data = pickle.load(f)
noisy_Id = data['noisy_Id']
clean_Id = data['clean_Id']
answer_Id = data['answer_Id']
vocab = data['vocab']
embd = data['embd']

cosine_result = []
for i in range(len(noisy_Id)):
    if clean_Id[i] != noisy_Id[i]:
        cosine_each = [clean_Id[i], noisy_Id[i], answer_Id[i]]
        cosine_result.append(cosine_each)

print("finish cosine similiarity")
print("pair number", len(cosine_result))

question1 = []
question2 = []
answer = []
cosine_sim = []
for pair in cosine_result:
    if pair not in cosine_sim:
        cosine_sim.append(pair)

for pair in cosine_sim:
    question1.append(pair[0])
    question2.append(pair[1])
    answer.append(pair[2])
    # cosine_sim.append(pair[3])

print("the number of question:", len(question1))



output_path = '../result/Huawei/que_ans_simi'
result = {'question1': question1, 'question2': question2,
          'answer': answer, 'vocab': vocab, 'embd': embd}
output = open(output_path, 'wb')
pickle.dump(result, output)
output.close()
