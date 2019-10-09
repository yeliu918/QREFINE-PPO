import tensorflow as tf


import Data_loading
import simple_seqRL

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import random

FLAGS = tf.app.flags.FLAGS
# import os
#
print os.getcwd()

# reward_rate_list=[0.1, 0.5, 1, 2]
# reward_rate = random.sample(reward_rate_list, 1)[0]
# discount_factor_list=[0.01, 0.1, 1]
# discount_factor = random.sample(discount_factor_list, 1)[0]
# dec_L_list=[1,2,3]
# dec_L = random.sample(dec_L_list, 1)[0]
# epoch = 34/dec_L
# v
# batch_size = random.sample(batch_size_list, 1)[0]


tf.app.flags.DEFINE_integer("epoch", 34, "the epoch number")
tf.app.flags.DEFINE_integer("dec_L", 2, "the decade length")
tf.app.flags.DEFINE_integer("batch_size", 100, "the batch size")
tf.app.flags.DEFINE_integer("num_units", 200, "the number of hidden state units")
tf.app.flags.DEFINE_float("discount_factor", 0.1, "the discount factor of the reward")
tf.app.flags.DEFINE_float("sen_reward_rate", 1, "the rate of sentence reward")
tf.app.flags.DEFINE_float("dropout", 0.9, "the dropout rate")
tf.app.flags.DEFINE_integer("beam_width", 5, "the size of beam search")
tf.app.flags.DEFINE_bool("Bidirection", False, "bidirection")
tf.app.flags.DEFINE_bool("Embd_train", False, "Embedding trainable")
tf.app.flags.DEFINE_bool("Attention", False, "Attention")

tf.app.flags.DEFINE_string("Data_FileName", "../result/Huawei/large_unique0.95_with", "The processing data file name")
# huawei: result/Huawei/large_unique0.95_with
# wiki: data/Wiki/wiki_ids_16

train_data_com, val_data_com, test_data_com, max_data, all_data = Data_loading.load_data(FLAGS.Data_FileName)


data_comb = [train_data_com, val_data_com,test_data_com, max_data]

print "\n"+"reward_rate "+str(FLAGS.sen_reward_rate)+" discount_factor "+str(FLAGS.discount_factor)+" dec_L "+str(FLAGS.dec_L)+" batch_size "+str(FLAGS.batch_size)+"epoch"+str(FLAGS.epoch)+"\n"


for reward_rate in range(1, 5):
    print "the reward rate is:", reward_rate
    FLAGS.sen_reward_rate = reward_rate
    simple_seqRL.RL_tuning_model(
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
