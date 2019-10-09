import tensorflow as tf
import CharacterBA
import QA_similiarity
import parameter as prm
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True

import sys
import h5py
import numpy as np
import pickle
import os
from Agent import *
sys.path.append('/home/ye/PycharmProjects/Bert/bert-as-service')
from service.client import BertClient

# def discounted_rewards_cal(reward_buffer, discount_factor):
#     discounted_rewards = []
#     for i in range(len(reward_buffer)):
#         N = len(reward_buffer[i])
#         r = 0  # use discounted reward to approximate Q value
#         discounted_reward = np.zeros(N)
#         for t in reversed(range(N)):
#             # future discounted reward from now on
#             r = reward_buffer[i][t] + discount_factor * r
#             discounted_reward[t] = r
#         discounted_rewards.append(list(discounted_reward))
#     return discounted_rewards

def discounted_rewards_cal(reward_buffer, discount_factor):
    N = len(reward_buffer)
    r = 0  # use discounted reward to approximate Q value
    discounted_reward = np.zeros(N)
    for t in reversed(range(N)):
        # future discounted reward from now on
        r = reward_buffer[t] + discount_factor * r
        discounted_reward[t] = r
    return list(discounted_reward)

def Reward(config, Character_data, QA_data , vocab, embd):
    batch_size = config.batch_size
    vocab_size = len(vocab)
    num_units = config.num_units
    beam_width = config.beam_width
    Bidirection = config. Bidirection
    Embd_train = config.Embd_train
    Attention = config.Attention
    discount_factor = config.discount_factor
    dropout = config.dropout
    char_num = config.char_num
    char_dim = config.char_dim
    char_hidden_units = config.char_hidden_units
    Seq2Seq_pretain_ckp = config.seq2seq_ckp_dir
    QA_pretain_ckp = config.QA_ckp_dir
    model_cond ="training"
    G_Seq2Seq = tf.Graph()
    sess_word_rw = tf.Session(graph=G_Seq2Seq)
    with G_Seq2Seq.as_default():
        Seq2Seq_model = CharacterBA.Char_Seq2Seq(batch_size, vocab_size, num_units, embd, model_cond, Bidirection, Embd_train, char_hidden_units,
                 Attention, char_num, char_dim)
        saver_word_rw = tf.train.Saver()
        saver_word_rw.restore(sess_word_rw,
                              Seq2Seq_pretain_ckp)

    model_type = "testing"
    G_QA_similiarity = tf.Graph()
    sess_QA_rw = tf.Session(graph=G_QA_similiarity)
    with G_QA_similiarity.as_default():
        QA_simi_model = QA_similiarity.QA_similiarity(batch_size, num_units, embd, model_type)
        saver_sen_rw = tf.train.Saver()
        saver_sen_rw.restore(sess_QA_rw, QA_pretain_ckp)


    source_shuffle, source_len, source_emd, whole_noisy_char_Id, char_len, generated_sen, sen_len_batch_beam, generated_target_len, bert_emd = Character_data
    logits_reward_beam = []
    # print "the generated_sen shape: ", generated_sen.shape
    # print "sen_len_batch_beam: ", sen_len_batch_beam.shape
    # print "bert_emd: ", bert_emd.shape
    for i in range(beam_width):
        fd_seq = {Seq2Seq_model.encoder_inputs: source_shuffle, Seq2Seq_model.encoder_inputs_length: source_len,
                  Seq2Seq_model.encoder_emb: source_emd,
                  Seq2Seq_model.encoder_char_ids: whole_noisy_char_Id, Seq2Seq_model.encoder_char_len: char_len,
                  Seq2Seq_model.decoder_inputs: generated_sen[:,:,i], Seq2Seq_model.decoder_length: sen_len_batch_beam[:,i],
                  # Seq2Seq_model.target_emb: bert_emd,
                  Seq2Seq_model.dropout_rate: dropout}
        logits_reward_beam.append(sess_word_rw.run(Seq2Seq_model.softmax_logits, fd_seq))


    answer_shuffle, answer_len, source_shuffle, source_len, source_emd, generated_sen, sen_len_batch_beam, bert_emd, vocab = QA_data
    EOS = 1
    answer_shuffle = tf.keras.preprocessing.sequence.pad_sequences(answer_shuffle, maxlen=None,
                                                                   padding='post', value=EOS)
    QA_reward_beam = []
    for i in range(beam_width):
        fd_qa = {QA_simi_model.answer_inputs: answer_shuffle, QA_simi_model.answer_inputs_length: answer_len,
                 QA_simi_model.question1_inputs: source_shuffle,
                 QA_simi_model.question1_inputs_length: source_len,
                 QA_simi_model.question2_inputs: generated_sen[:,:,i],
                 QA_simi_model.question2_inputs_length: sen_len_batch_beam[:,i],
                 QA_simi_model.question1_emd : source_emd,
                 QA_simi_model.question2_emd: bert_emd[:,i,:,:]} #[]
        QA_reward_beam.append(sess_QA_rw.run([QA_simi_model.two_distance], fd_qa))

    # print "the shape logits_reward_beam", np.array(logits_reward_beam).shape

    logits_reward_beam = np.array(logits_reward_beam).reshape(batch_size * beam_width, generated_target_len, vocab_size)
    QA_reward_beam = np.array(QA_reward_beam).reshape(batch_size * beam_width)
    generated_sen = np.array(generated_sen).reshape(batch_size*beam_width, generated_target_len)
    # print "the shape of generated_sen ", generated_sen.shape
    logits_batch_beam = []
    for b in range(batch_size * beam_width):
        logits_batch = []
        for l in range(generated_target_len):
            id = generated_sen[b][l]
            logits_v = logits_reward_beam[b][l][id]
            logits_batch.append(logits_v)
        logits_batch_beam.append(logits_batch)

    bc = BertClient(port=4445, port_out=4446)
    gen_sen = []
    for idx in range(batch_size*beam_width):
        sen = []
        for l in range(generated_target_len):
            sen.append(vocab[int(generated_sen[idx][l])])
        gen_sen.append(" ".join(sen))
    bert_pro_beam = bc.encode(gen_sen)
    bert_pro_beam = bert_pro_beam[:,1:generated_target_len + 1,:]

    bert_weight_path = config.bert_weight_path
    if os.path.isfile(bert_weight_path):
        f = open(bert_weight_path, "rb")
        data = pickle.load(f)
        output_weights = data["bert_W"]
        output_bias = data["bert_b"]
        output_weights = np.transpose(output_weights)
        logits = np.matmul(bert_pro_beam, output_weights)
        logits = np.add(logits, output_bias)
        log_probs = np.divide(np.exp(logits),
                              np.tile(np.sum(np.exp(logits), axis=2), output_bias.shape[0]).reshape(logits.shape[0],
                                                                                                    logits.shape[1],
                                                                                                    logits.shape[2]))
        # print "the size of log_probs", log_probs.shape
        # tf.nn.log_softmax(logits, axis=-1)

        bert_Id, bert_vocab_size= bert_vocab_map_Id(generated_sen, config, vocab)

        bert_reward_beam = []
        for i in range(batch_size * beam_width):
            bert_reward = []
            for j in range(generated_target_len):
                bert_reward.append(log_probs[i][j][bert_Id[i][j]])
            bert_reward_beam.append(bert_reward)

    # bert_reward_beam =[]
    # for i in range(beam_width):
    #     gen_sen =[]
    #     for b in range(batch_size):
    #         sen = []
    #         for l in range(generated_sen.shape[1]):
    #             sen.append(vocab[int(generated_sen[b][l][i])])
    #         gen_sen.append(" ".join(sen))
    #     bert_pro = bc.encode(gen_sen)
    #     bert_pro = bert_pro[:,:generated_target_len,:]
    #
    #     bert_weight_path = config.bert_weight_path
    #     if os.path.isfile(bert_weight_path):
    #         f=open(bert_weight_path,"rb")
    #         data = pickle.load(f)
    #         output_weights = data["bert_W"]
    #         output_bias = data["bert_b"]
    #         output_weights = np.transpose(output_weights)
    #         logits = np.matmul(bert_pro, output_weights)
    #         logits = np.add(logits, output_bias)
    #         log_probs = np.divide(np.exp(logits),np.tile(np.sum(np.exp(logits), axis=2),output_bias.shape[0]).reshape(logits.shape[0],logits.shape[1],logits.shape[2]))
    #             # tf.nn.log_softmax(logits, axis=-1)
    #
    #         bert_Id = bert_vocab_map_Id(generated_sen, config, vocab)
    #
    #         bert_reward_beam = []
    #         for i in range(batch_size * beam_width):
    #             bert_reward = []
    #             for j in range(generated_target_len):
    #                 bert_reward.append(log_probs[i][j][bert_Id[i][j]])
    #             bert_reward_beam.append(bert_reward)


    RL_reward_beam_dis = []
    for beam in range(beam_width * batch_size):
        RL_reward = []
        for idx in range(generated_target_len):
            if idx != len(logits_batch_beam[beam]) - 1:
                RL_reward.append(vocab_size * logits_batch_beam[beam][idx] + bert_vocab_size * bert_reward_beam[beam][idx])
            elif idx == len(logits_batch_beam[beam]) - 1:
                RL_reward.append(vocab_size * logits_batch_beam[beam][idx] + bert_vocab_size * bert_reward_beam[beam][idx] + 10 * QA_reward_beam[beam])
        RL_reward_batch_dis = discounted_rewards_cal(RL_reward, discount_factor)
        RL_reward_beam_dis.append(RL_reward_batch_dis)

    RL_reward_beam_dis = np.array(RL_reward_beam_dis).reshape(batch_size, beam_width, generated_target_len)
    RL_reward_beam_dis_mm = RL_reward_beam_dis - np.tile(np.mean(RL_reward_beam_dis, axis = 1), (beam_width,1)).reshape(batch_size,beam_width,generated_target_len)
    # print "the RL_reward_dis: ", np.mean(RL_reward_beam_dis_mm)
    RL_reward_beam_dis_mm = np.transpose(RL_reward_beam_dis_mm,(1,0,2))

    return RL_reward_beam_dis_mm

