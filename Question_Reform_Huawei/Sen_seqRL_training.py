import tensorflow as tf


import Data_loading
import Sen_seqRL

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random

FLAGS = tf.app.flags.FLAGS
# import os
#
print os.getcwd()
import pickle
import numpy as np

# reward_rate_list=[0.1, 0.5, 1, 2]
# reward_rate = random.sample(reward_rate_list, 1)[0]
# discount_factor_list=[0.01, 0.1, 1]
# discount_factor = random.sample(discount_factor_list, 1)[0]
# dec_L_list=[1,2,3]
# dec_L = random.sample(dec_L_list, 1)[0]
# epoch = 34/dec_L

# batch_size = random.sample(batch_size_list, 1)[0]

Dataset_name = "HUAWEI"
Dataset_name = "YAHOO"

Dataset_name1 = "threeopt"

tf.app.flags.DEFINE_integer("dec_L", 2, "the decade length")
tf.app.flags.DEFINE_integer("batch_size", 60, "the batch size")
tf.app.flags.DEFINE_integer("num_units", 200, "the number of hidden state units")
tf.app.flags.DEFINE_float("discount_factor", 0.01, "the discount factor of the reward")
tf.app.flags.DEFINE_float("sen_reward_rate", 0.5, "the rate of sentence reward")
tf.app.flags.DEFINE_float("dropout", 0.9, "the dropout rate")
tf.app.flags.DEFINE_integer("beam_width", 5, "the size of beam search")
tf.app.flags.DEFINE_bool("Bidirection", False, "bidirection")
tf.app.flags.DEFINE_bool("Embd_train", False, "Embedding trainable")
tf.app.flags.DEFINE_bool("Attention", False, "Attention")
tf.app.flags.DEFINE_string("Data_FileName", '/mnt/WDRed4T/ye/DataR/' + Dataset_name + '/' + Dataset_name1 + "_final", "The processing data file name")
tf.app.flags.DEFINE_string("Vocab_FileName", '/mnt/WDRed4T/ye/DataR/' + Dataset_name + '/' + Dataset_name1 + '_vocab_embd', "The vocab data file name")
# huawei: result/Huawei/large_unique0.95_with
# wiki: data/Wiki/wiki_ids_16

# train_data_com, val_data_com, test_data_com, max_data, all_data = Data_loading.load_data(FLAGS.Data_FileName)


# data_comb = [train_data_com, val_data_com,test_data_com, max_data]

# FileName = '/mnt/WDRed4T/ye/DataR/'+ Dataset_name + '/wrongword1_final'
FileName = FLAGS.Data_FileName
input = open(FileName, 'rb')
data_com = pickle.load(input)
train_noisy_Id = data_com["train_noisy_Id"]
test_noisy_Id = data_com["test_noisy_Id"]
eval_noisy_Id = data_com["eval_noisy_Id"]

train_noisy_len = data_com["train_noisy_len"]
test_noisy_len = data_com["test_noisy_len"]
eval_noisy_len = data_com["eval_noisy_len"]

train_noisy_char_Id = data_com["train_noisy_char_Id"]
test_noisy_char_Id = data_com["test_noisy_char_Id"]
eval_noisy_char_Id = data_com["eval_noisy_char_Id"]

train_noisy_char_len = data_com["train_noisy_char_len"]
test_noisy_char_len = data_com["test_noisy_char_len"]
eval_noisy_char_len = data_com["eval_noisy_char_len"]

train_target_Id = data_com["train_ground_truth"]
test_target_Id = data_com["test_ground_truth"]
eval_target_Id = data_com["eval_ground_truth"]

train_input_Id = data_com["train_input_Id"]
test_input_Id = data_com["test_input_Id"]
eval_input_Id = data_com["eval_input_Id"]

train_clean_len = data_com["train_clean_len"]
test_clean_len = data_com["test_clean_len"]
eval_clean_len = data_com["eval_clean_len"]

train_answer_Id = data_com["train_answer_Id"]
test_answer_Id = data_com["test_answer_Id"]
eval_answer_Id = data_com["eval_answer_Id"]

train_answer_len = data_com["train_answer_len"]
test_answer_len = data_com["test_answer_len"]
eval_answer_len = data_com["eval_answer_len"]



length = len(train_input_Id[0])
print "the epoch time is", length

tf.app.flags.DEFINE_integer("epoch", length, "the epoch number")

max_char = data_com['max_char']
char_num = 44
max_word = data_com['max_word']

# output = '/mnt/WDRed4T/ye/DataR/' + Dataset_name/wrongword1_vocab_embd'
output = FLAGS.Vocab_FileName
input = open(output, 'rb')
vocab_embd = pickle.load(input)
vocab = vocab_embd['vocab']
embd = vocab_embd['embd']
embd = np.asarray(embd)

print "\n"+"reward_rate "+str(FLAGS.sen_reward_rate)+" discount_factor "+str(FLAGS.discount_factor)+" dec_L "+str(FLAGS.dec_L)+" batch_size "+str(FLAGS.batch_size)+"epoch"+str(FLAGS.epoch)+"\n"

train_data = {"noisy_Id": train_noisy_Id, "noisy_len":train_noisy_len, "ground_truth": train_target_Id, "train_Id": train_input_Id, "clean_len": train_clean_len,
              "char_Id": train_noisy_char_Id, "char_len":train_noisy_char_len,"answer_len": train_answer_len,"answer_Id": train_answer_Id}
test_data = {"noisy_Id": test_noisy_Id, "noisy_len":test_noisy_len, "ground_truth": test_target_Id, "train_Id": test_input_Id, "clean_len": test_clean_len,
              "char_Id": test_noisy_char_Id, "char_len":test_noisy_char_len,"answer_len": test_answer_len,"answer_Id": test_answer_Id}
eval_data = {"noisy_Id": eval_noisy_Id, "noisy_len":eval_noisy_len, "ground_truth": eval_target_Id, "train_Id": eval_input_Id, "clean_len": eval_clean_len,
              "char_Id": eval_noisy_char_Id, "char_len":eval_noisy_char_len,"answer_len": eval_answer_len,"answer_Id": eval_answer_Id}
# max_data = {"max_noisy_len":, "max_clean_len":, "max_answer_len":}

data_comb = [train_data, test_data, eval_data]

for reward_rate in range(1, 5):
    print "the reward rate is:", reward_rate
    FLAGS.sen_reward_rate = reward_rate
    Sen_seqRL.RL_tuning_model(
        data_comb=data_comb,
        epoch=FLAGS.epoch,
        batch_size=FLAGS.batch_size,
        num_units=FLAGS.num_units,
        beam_width=FLAGS.beam_width,
        discount_factor=FLAGS.discount_factor,
        sen_reward_rate=FLAGS.sen_reward_rate,
        Bidirection=FLAGS.Bidirection,
        Embd_train=FLAGS.Embd_train,
        Attention=FLAGS.Attention,
        dropout = FLAGS.dropout,
    dec_L=FLAGS.dec_L)
