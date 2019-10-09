# -*- coding: UTF-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import gc
from tensorflow.python import pywrap_tensorflow

import numpy as np
import tensorflow as tf
# import tflearn

import seq_last
import QA_similiarity
import BA_Seq2Seq
from nltk.translate import bleu

PAD = 0
SOS = 1
EOS = 2
batches_in_epoch = 200
epoch_print = 2


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# import os
# print os.getcwd()

def next_batch(noisy_Id, noisy_len, train_Id, ground_truth, clean_len, answer_Id, answer_len, batch_size, batch_num,
               idx):
    if (batch_num + 1) * batch_size > len(noisy_Id):
        batch_num = batch_num % (len(noisy_Id) / batch_size)
    idx_n = idx[batch_num * batch_size: (batch_num + 1) * batch_size]
    data_shuffle = [noisy_Id[i] for i in idx_n]
    data_len = [noisy_len[i] for i in idx_n]
    target_len = [clean_len[i] for i in idx_n]
    train_shuffle = [train_Id[i] for i in idx_n]
    target_shuffle = [ground_truth[i] for i in idx_n]
    answer_shuffle = [answer_Id[i] for i in idx_n]
    ans_len = [answer_len[i] for i in idx_n]
    return data_shuffle, data_len, train_shuffle, target_shuffle, target_len, answer_shuffle, ans_len


def discounted_rewards_cal(reward_buffer, discount_factor):
    discounted_rewards = []
    for i in range(reward_buffer.shape[0]):
        N = len(reward_buffer[i])
        r = 0  # use discounted reward to approximate Q value
        discounted_reward = np.zeros(N)
        for t in reversed(range(N)):
            # future discounted reward from now on
            r = reward_buffer[i][t] + discount_factor * r
            discounted_reward[t] = r
        discounted_rewards.append(list(discounted_reward))
    return discounted_rewards


def RL_tuning_model(data_comb, epoch,
                    batch_size=64,
                    num_units=200,
                    beam_width=5,
                    discount_factor=0.1,
                    sen_reward_rate=2,
                    dropout = 1,
                    dec_L=1,
                    Bidirection=False,
                    Embd_train=True,
                    Attention=False,):
    max_noisy_len = data_comb[3]["max_noisy_len"]
    max_clean_len = data_comb[3]["max_clean_len"]
    max_answer_len = data_comb[3]["max_answer_len"]

    L = max_clean_len

    noisy_Id = data_comb[0]['noisy_Id']
    noisy_len = data_comb[0]['noisy_len']
    ground_truth = data_comb[0]['ground_truth']
    clean_len = data_comb[0]['clean_len']
    answer_Id = data_comb[0]['answer_Id']
    answer_len = data_comb[0]['answer_len']
    train_Id = data_comb[0]['train_Id']
    vocab = data_comb[0]['vocab']
    embd = data_comb[0]['embd']
    embd = np.array(embd)


    val_noisy_Id = data_comb[1]['noisy_Id']
    val_noisy_len = data_comb[1]['noisy_len']
    val_ground_truth = data_comb[1]['ground_truth']
    val_clean_len = data_comb[1]['clean_len']
    val_answer_Id = data_comb[1]['answer_Id']
    val_answer_len = data_comb[1]['answer_len']
    val_train_Id = data_comb[1]['train_Id']

    test_noisy_Id = data_comb[2]['noisy_Id']
    test_noisy_len = data_comb[2]['noisy_len']
    test_ground_truth = data_comb[2]['ground_truth']
    test_clean_len = data_comb[2]['clean_len']
    test_answer_Id = data_comb[2]['answer_Id']
    test_answer_len = data_comb[2]['answer_len']
    test_train_Id = data_comb[2]['train_Id']

    vocab_size = len(vocab)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    model_name = "BS_"
    values = {}
    checkpoint_path = "/home/ye/PycharmProjects/Qrefine/Seq_ckpt/Wiki/ALL_seq2seq_Bi_" + str(Bidirection) + "_Att_" + str(
                                  Attention) + "_Emb_" + str(
                                  Embd_train)+"_hiddenunits_" + str(num_units)
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        if 'loss_fun' not in key:
            values[model_name + key + ':0'] = reader.get_tensor(key)

    model_cond = "training"
    G_Seq2Seq = tf.Graph()


    sess_word_rw = tf.Session(config=config, graph=G_Seq2Seq)
    with G_Seq2Seq.as_default():
        Seq2Seq_model = BA_Seq2Seq.Bi_Att_Seq2Seq(batch_size, vocab_size, num_units, embd, model_cond,
                                                  Bidirection, Embd_train, Attention)
        saver_word_rw = tf.train.Saver()
        # saver_word_rw.restore(sess_word_rw,
        #                       "Seq_ckpt/pretrainALL-seq2seq_Bi_" + str(Bidirection) + "_Att_" + str(
        #                           Attention) + "_Emb_" + str(
        #                           Embd_train))
        saver_word_rw.restore(sess_word_rw,
                              "/home/ye/PycharmProjects/Qrefine/Seq_ckpt/Wiki/ALLdata_seq2seq_Bi_" + str(Bidirection) + "_Att_" + str(
                                  Attention) + "_Emb_" + str(
                                  Embd_train)+"_hiddenunits_" + str(num_units))

    # model_type = "testing"
    # G_QA_similiarity = tf.Graph()
    # sess_QA_rw = tf.Session(config=config, graph=G_QA_similiarity)
    # with G_QA_similiarity.as_default():
    #     QA_simi_model = QA_similiarity.QA_similiarity(batch_size, num_units, embd, model_type)
    #     saver_sen_rw = tf.train.Saver()
    #     # saver_sen_rw.restore(sess_QA_rw, "Seq_ckpt/Wiki_qa_similiarity")
    #     saver_sen_rw.restore(sess_QA_rw, "/home/ye/PycharmProjects/Qrefine/Seq_ckpt/Huawei/qa_similiarity")

    G_BeamSearch = tf.Graph()
    with G_BeamSearch.as_default():
        BeamSearch_seq2seq = seq_last.BeamSearch_Seq2seq(vocab_size = vocab_size,
                                                         num_units = num_units,
                                                         beam_width = beam_width,
                                                         model_name = model_name, embd=embd,
                                                         Bidirection = Bidirection,
                                                         Embd_train = Embd_train, Attention = Attention,
                                                         max_target_length = L)

    seq2seq_len = L
    RL_len = 0

    with tf.Session(config=config, graph=G_BeamSearch) as sess_beamsearch:
        sess_beamsearch.run(tf.global_variables_initializer())

        for v in tf.trainable_variables():
            if v.name in values.keys():
                v.load(values[v.name], sess_beamsearch)

        val_loss_epo = []
        patience_cnt = 0
        for epo in range(epoch):
            print(" epoch: {}".format(epo))
            # RL_len = epo
            RL_len = L - seq2seq_len

            idx = np.arange(0, len(noisy_Id))
            idx = list(np.random.permutation(idx))

            Bleu_score1 = []

            for batch in range(len(noisy_Id) / batch_size):

                if seq2seq_len < 0:
                    seq2seq_len = 0
                    source_shuffle, source_len, train_shuffle, target_shuffle, target_len, answer_shuffle, ans_len = next_batch(
                        noisy_Id, noisy_len, train_Id, ground_truth, clean_len, answer_Id, answer_len, batch_size,
                        batch,
                        idx)

                    source_shuffle = tf.keras.preprocessing.sequence.pad_sequences(source_shuffle, maxlen=None,
                                                                                   padding='post', truncating="post",
                                                                                   value=EOS)

                    target_shuffle = tf.keras.preprocessing.sequence.pad_sequences(target_shuffle, maxlen=seq2seq_len,
                                                                                   padding='post', truncating="post",
                                                                                   value=EOS)

                    train_shuffle = tf.keras.preprocessing.sequence.pad_sequences(train_shuffle, maxlen=seq2seq_len,
                                                                              padding='post', truncating="post",
                                                                              value=EOS)

                    initial_input_in = [[EOS] for i in range(batch_size)]

                    target_len = np.tile(0, batch_size)

                    fd = {BeamSearch_seq2seq.encoder_inputs: source_shuffle,
                          BeamSearch_seq2seq.encoder_inputs_length: source_len,
                          BeamSearch_seq2seq.decoder_length: target_len,
                          BeamSearch_seq2seq.decoder_inputs: train_shuffle,
                          BeamSearch_seq2seq.decoder_targets: target_shuffle,
                          BeamSearch_seq2seq.initial_input: initial_input_in}

                    generated_que, policy = sess_beamsearch.run(
                        [BeamSearch_seq2seq.RL_ids, BeamSearch_seq2seq.max_policy],fd)

                    generated_que_input = np.insert(generated_que, 0, SOS, axis=1)[:, 0:-1]
                    generated_target_len = np.tile(generated_que_input.shape[1], batch_size)

                    fd_seq = {Seq2Seq_model.encoder_inputs: source_shuffle,
                              Seq2Seq_model.encoder_inputs_length: source_len,
                              Seq2Seq_model.decoder_inputs: generated_que_input,
                              Seq2Seq_model.decoder_length: generated_target_len,
                              Seq2Seq_model.ground_truth: generated_que,
                              Seq2Seq_model.dropout_rate: dropout}
                    logits_pro = sess_word_rw.run(Seq2Seq_model.softmax_logits, fd_seq)

                    answer_shuffle = tf.keras.preprocessing.sequence.pad_sequences(answer_shuffle, maxlen=None,
                                                                                   padding='post', truncating="post",
                                                                                   value=EOS)

                    generated_len = np.tile(generated_que.shape[1], batch_size)

                    # print QA_similiarity_rd

                    reward = np.zeros((batch_size, RL_len))
                    for i in range(generated_que.shape[0]):
                        for j in range(seq2seq_len, generated_que.shape[1]):
                            max_index = generated_que[i][j]
                            reward[i][L - 1 - j] = logits_pro[i, j, max_index]
                            if j == generated_que.shape[1] - 1:
                                reward[i][L - 1 - j] = reward[i][L - 1 - j]

                    discounted_rewards = discounted_rewards_cal(reward, discount_factor)

                    RL_rewards = discounted_rewards

                    fd = {BeamSearch_seq2seq.encoder_inputs: source_shuffle,
                          BeamSearch_seq2seq.encoder_inputs_length: source_len,
                          BeamSearch_seq2seq.decoder_length: target_len,
                          BeamSearch_seq2seq.decoder_inputs: train_shuffle,
                          BeamSearch_seq2seq.decoder_targets: target_shuffle,
                          BeamSearch_seq2seq.initial_input: initial_input_in,
                          BeamSearch_seq2seq.dis_rewards: RL_rewards}

                    _, rl_reward = sess_beamsearch.run(
                        [BeamSearch_seq2seq.RL_train_op,
                         BeamSearch_seq2seq.rl_reward], fd)
                    for t in range(batch_size):
                        ref = []
                        hyp = []
                        bleu_s = []
                        for tar_id in target_shuffle[t]:
                            if tar_id != 2:
                                ref.append(vocab[tar_id])
                        for pre_id in generated_que[t, :]:
                            if pre_id !=0 and pre_id !=-1 and pre_id !=2:
                                hyp.append(vocab[pre_id])
                        sen_bleu1 = bleu([ref], hyp, weights=(1, 0, 0, 0))
                        bleu_s.append(sen_bleu1)
                        Bleu_score1.append(bleu_s)

                    top1_bleu = []
                    for i in range(len(Bleu_score1)):
                        top1_bleu.append(Bleu_score1[i][0])

                    if batch == 0 or batch % 100 == 0:
                        print(' batch {}'.format(batch))
                        print('   minibatch reward of training: {}'.format(- rl_reward))
                        print("the current whole bleu_score1 is:", sum(top1_bleu) / float(len(top1_bleu)))

                beam_QA_similar = []
                # beam_cnQA_similar = []
                source_shuffle, source_len, train_shuffle, target_shuffle, target_len, answer_shuffle, ans_len = next_batch(
                    noisy_Id, noisy_len, train_Id, ground_truth, clean_len, answer_Id, answer_len, batch_size, batch,
                    idx)

                source_shuffle = tf.keras.preprocessing.sequence.pad_sequences(source_shuffle, maxlen=None,
                                                                               padding='post', truncating="post",
                                                                               value=EOS)

                target_shuffle_in = tf.keras.preprocessing.sequence.pad_sequences(target_shuffle, maxlen=seq2seq_len,
                                                                               padding='post', truncating="post",
                                                                               value=EOS)

                train_shuffle = tf.keras.preprocessing.sequence.pad_sequences(train_shuffle, maxlen=seq2seq_len,
                                                                              padding='post', truncating="post",
                                                                              value=EOS)

                initial_input_in = [target_shuffle_in[i][-1] for i in range(batch_size)]

                target_len = np.tile(seq2seq_len, batch_size)

                fd = {BeamSearch_seq2seq.encoder_inputs: source_shuffle,
                      BeamSearch_seq2seq.encoder_inputs_length: source_len,
                      BeamSearch_seq2seq.decoder_length: target_len,
                      BeamSearch_seq2seq.decoder_inputs: train_shuffle,
                      BeamSearch_seq2seq.decoder_targets: target_shuffle_in,
                      BeamSearch_seq2seq.initial_input: initial_input_in}

                cl_loss, _, S2S_ids = sess_beamsearch.run(
                    [BeamSearch_seq2seq.loss_seq2seq, BeamSearch_seq2seq.train_op, BeamSearch_seq2seq.ids], fd)

                # tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')


                if batch == 0 or batch % batches_in_epoch == 0:
                    print(' batch {}'.format(batch))
                    print('   minibatch loss of training: {}'.format(cl_loss))
                    val_loss = []
                    for batch_val in range(len(val_noisy_Id) / batch_size):
                        idx_v = np.arange(0, len(val_noisy_Id))
                        idx_v = list(np.random.permutation(idx_v))

                        val_source_shuffle, val_source_len, val_train_shuffle, val_target_shuffle, val_target_len, val_answer_shuffle, val_ans_len = next_batch(
                            val_noisy_Id, val_noisy_len, val_train_Id, val_ground_truth, val_clean_len, val_answer_Id,
                            val_answer_len,
                            batch_size, batch_val, idx_v)
                        val_source_shuffle_in = tf.keras.preprocessing.sequence.pad_sequences(val_source_shuffle,
                                                                                           maxlen=None,
                                                                                           padding='post', value=EOS)
                        val_target_shuffle_in = tf.keras.preprocessing.sequence.pad_sequences(val_target_shuffle,
                                                                                           maxlen=seq2seq_len,
                                                                                           padding='post', value=EOS)
                        val_train_shuffle_in = tf.keras.preprocessing.sequence.pad_sequences(val_train_shuffle,
                                                                                          maxlen=seq2seq_len,
                                                                                          padding='post', value=EOS)

                        val_initial_input = [val_target_shuffle_in[i][-1] for i in range(batch_size)]

                        val_target_len = np.tile(seq2seq_len, batch_size)

                        fd_val = {BeamSearch_seq2seq.encoder_inputs: val_source_shuffle_in,
                                  BeamSearch_seq2seq.encoder_inputs_length: val_source_len,
                                  BeamSearch_seq2seq.decoder_length: val_target_len,
                                  BeamSearch_seq2seq.decoder_inputs: val_train_shuffle_in,
                                  BeamSearch_seq2seq.decoder_targets: val_target_shuffle_in,
                                  BeamSearch_seq2seq.initial_input: val_initial_input}
                        val_loss.append(sess_beamsearch.run(BeamSearch_seq2seq.loss_seq2seq, fd_val))
                        avg_val_loss = sum(val_loss) / float(len(val_loss))

                    print('   minibatch loss of validation: {}'.format(avg_val_loss))

            # val_loss_epo.append(avg_val_loss)

                gc.collect()

                if L - seq2seq_len == 0: continue

                RL_logits, policy = sess_beamsearch.run([BeamSearch_seq2seq.RL_ids, BeamSearch_seq2seq.max_policy], fd)

                # print "The size of Rl:", RL_logits.shape[0], RL_logits.shape[1]
                # print "The size of Policy:", policy.shape[0], policy.shape[1]

                max_target_length = RL_logits.shape[1]

                # for batch in range(batch_size):
                #     label = 0
                #     for i in range(RL_logits.shape[1]):
                #         if RL_logits[batch][i]==2 and label ==0:
                #             label = 1
                #             for id in range(i+1, max_target_length):
                #                 RL_logits[batch][id]=2
                #             continue


                Sequnce_len = seq2seq_len + max_target_length
                # print "the max_target_length is:", max_target_length

                RL_rewards = np.zeros((batch_size, max_target_length))

                generated_que = []

                for i in range(batch_size):
                    generated_que.append(list(np.append(S2S_ids[i], RL_logits[i])))

                generated_que = np.asarray(generated_que)

                logits_batch = []
                # for i in range(generated_que.shape[0]):
                #     for j in range(generated_que.shape[1]):
                #         if generated_que[i][j] == 2 or j == generated_que.shape[1] - 1:
                #             logits_batch.append(j)
                #             continue
                #
                # reward = np.zeros((batch_size, max_target_length))
                # # generated_que = SOS + generated_que - PAD
                generated_que_input = np.insert(generated_que, 0, SOS, axis=1)[:, 0:-1]
                # target_shuffle = tf.keras.preprocessing.sequence.pad_sequences(target_shuffle, maxlen=None,
                #                                                                padding='post',truncating="post",  value=EOS)

                generated_target_len = np.tile(generated_que_input.shape[1], batch_size)

                fd_seq = {Seq2Seq_model.encoder_inputs: source_shuffle,
                          Seq2Seq_model.encoder_inputs_length: source_len,
                          Seq2Seq_model.decoder_inputs: generated_que_input,
                          Seq2Seq_model.decoder_length: generated_target_len,
                          Seq2Seq_model.ground_truth: generated_que,
                          Seq2Seq_model.dropout_rate: dropout}
                logits_pro = sess_word_rw.run(Seq2Seq_model.softmax_logits, fd_seq)

                # logits_pro = logits_pro[:, :, :]
                # print "the logits_pro is:", logits_pro.shape[1]

                # print QA_similiarity_rd

                # reward = np.zeros((batch_size, RL_len))

                for i in range(generated_que.shape[0]):
                    label = 0
                    for j in range(seq2seq_len, generated_que.shape[1]):
                        max_index = generated_que[i][j]
                        RL_rewards[i][j - seq2seq_len] = logits_pro[i, j, max_index]
                        if max_index == 2 and label ==0:
                            label = 1
                            RL_rewards[i][j - seq2seq_len] = logits_pro[i, j, max_index]
                            break
                        # if j == generated_que.shape[1] - 1:
                        #     reward[i][j  - seq2seq_len] = reward[i][j - seq2seq_len] + sen_reward_rate * (QA_similiarity_rd[0][i])

                # fd = {an_Lstm.answer_inputs: answer_shuffle, an_Lstm.answer_inputs_length: ans_len}
                # reward_similiarity = sess_sen_rw.run([an_Lstm.answer_state], fd)

                discounted_rewards = discounted_rewards_cal(RL_rewards, discount_factor)

                RL_rewards = discounted_rewards

                fd = {BeamSearch_seq2seq.encoder_inputs: source_shuffle,
                      BeamSearch_seq2seq.encoder_inputs_length: source_len,
                      BeamSearch_seq2seq.decoder_length: target_len,
                      BeamSearch_seq2seq.decoder_inputs: train_shuffle,
                      BeamSearch_seq2seq.decoder_targets: target_shuffle_in,
                      BeamSearch_seq2seq.initial_input: initial_input_in,
                      BeamSearch_seq2seq.dis_rewards: RL_rewards}


                _, rl_reward, word_policy, policy= sess_beamsearch.run(
                    [BeamSearch_seq2seq.RL_train_op,
                     BeamSearch_seq2seq.rl_reward, BeamSearch_seq2seq.word_log_prob, BeamSearch_seq2seq.softmax_policy], fd)

                for t in range(batch_size):
                    ref = []
                    hyp = []
                    bleu_s = []
                    for tar_id in target_shuffle[t]:
                        if tar_id !=2:
                            ref.append(vocab[tar_id])
                    for pre_id in generated_que[t, :]:
                        if pre_id !=0 and pre_id !=-1 and pre_id !=2:
                            hyp.append(vocab[pre_id])
                    sen_bleu1 = bleu([ref], hyp, weights=(1, 0, 0, 0))
                    bleu_s.append(sen_bleu1)
                    Bleu_score1.append(bleu_s)

                top1_bleu = []
                for i in range(len(Bleu_score1)):
                    top1_bleu.append(Bleu_score1[i][0])

                if batch == 0 or batch % 100 == 0:
                    print(' batch {}'.format(batch))
                    print('   minibatch reward of training: {}'.format(- rl_reward))
                    #print the result
                    for t in range(5):  ## five sentences
                        print('Question {}'.format(t))
                        print("noisy question:")
                        print(" ".join(map(lambda i: vocab[i], list(source_shuffle[t]))).strip().replace("<EOS>"," "))
                        print("clean question:")
                        print(" ".join(map(lambda i: vocab[i], list(target_shuffle[t]))).strip().replace("<EOS>"," "))
                        print("the generated question:")
                        pre_sen = []
                        for log_id in generated_que[t, :]:
                            if log_id != -1 and log_id!=2 and log_id!=0:
                                pre_sen.append(log_id)
                        print(" ".join(map(lambda i: vocab[i], pre_sen)).strip())
                        print("\n")
                    print("the current whole bleu_score1 is:", sum(top1_bleu) / float(len(top1_bleu)))
                gc.collect()

            seq2seq_len = seq2seq_len - dec_L

            ##testing set result:
            val_bleu_s1 = []
            val_bleu_s2 = []
            val_bleu_s3 = []
            val_bleu_s4 = []


            generated_test_sen = []
            test_answer_sen = []
            test_noisy_sen = []
            test_clean_sen = []

            for batch_test in range(len(test_noisy_Id) / batch_size):
                idx_t = np.arange(0, len(test_noisy_Id))
                idx_t = list(np.random.permutation(idx_t))

                val_source_shuffle, val_source_len, val_train_shuffle, val_target_shuffle, val_target_len, val_answer_shuffle, val_ans_len = next_batch(
                    test_noisy_Id, test_noisy_len, test_train_Id, test_ground_truth, test_clean_len, test_answer_Id,
                test_answer_len, batch_size, batch_test, idx_t)

                for an in val_answer_shuffle:
                    test_answer_sen.append(vocab[anum] for anum in an)
                for no in val_source_shuffle:
                    test_noisy_sen.append(vocab[si] for si in no)
                for cl in val_target_shuffle:
                    test_clean_sen.append(vocab[ci] for ci in cl)

                val_source_shuffle_in = tf.keras.preprocessing.sequence.pad_sequences(val_source_shuffle, maxlen=None,
                                                                               padding='post', truncating="post",value=EOS)

                val_target_shuffle_in=[]
                for val in val_target_shuffle:
                    val_target_shuffle_in.append([val[0]])

                val_train_shuffle_in = []
                for tra in val_train_shuffle:
                    val_train_shuffle_in.append([tra[0]])

                initial_input_in = [SOS for i in range(batch_size)]
                # [SOS for i in range(batch_size)]

                val_target_len_in = np.tile(1, batch_size)

                fd = {BeamSearch_seq2seq.encoder_inputs: val_source_shuffle_in,
                      BeamSearch_seq2seq.encoder_inputs_length: val_source_len,
                      BeamSearch_seq2seq.decoder_length: val_target_len_in,
                      BeamSearch_seq2seq.decoder_inputs: val_train_shuffle_in,
                      BeamSearch_seq2seq.decoder_targets: val_target_shuffle_in,
                      BeamSearch_seq2seq.initial_input: initial_input_in}

                val_id = sess_beamsearch.run([BeamSearch_seq2seq.RL_ids], fd)

                final_id = val_id[0]

                for t in range(batch_size):
                    ref = []
                    hyp = []
                    for tar_id in val_target_shuffle[t]:
                        if tar_id != 2:
                          ref.append(vocab[tar_id])
                    for pre_id in final_id[t, :]:
                        if pre_id !=2 and pre_id !=0 and pre_id != -1:
                            hyp.append(vocab[pre_id])
                            generated_test_sen.append(hyp)
                            sen_bleu1 = bleu([ref], hyp, weights=(1, 0, 0, 0))
                            sen_bleu2 = bleu([ref], hyp, weights=(0.5, 0.5, 0, 0))
                            sen_bleu3 = bleu([ref], hyp, weights=(0.333, 0.333, 0.333, 0))
                            sen_bleu4 = bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25))
                            val_bleu_s1.append(sen_bleu1)
                            val_bleu_s2.append(sen_bleu2)
                            val_bleu_s3.append(sen_bleu3)
                            val_bleu_s4.append(sen_bleu4)

                        bleu_score1 = sum(val_bleu_s1) / float(len(val_bleu_s1))
                        bleu_score2 = sum(val_bleu_s2) / float(len(val_bleu_s2))
                        bleu_score3 = sum(val_bleu_s3) / float(len(val_bleu_s3))
                        bleu_score4 = sum(val_bleu_s4) / float(len(val_bleu_s4))
                        print "the bleu score on test is:", bleu_score1

                        if bleu_score1 > 0.538:
                            fname = "Wiki_result_word/wiki_bleu1_" + str(bleu_score1) + "bleu2_" + str(
                                bleu_score2) + "bleu3_" + str(bleu_score3) + "bleu4_" + str(bleu_score4)
                            f = open(fname, "wb")
                            print "the length test set is:", len(generated_test_sen), len(test_answer_sen), len(
                                test_noisy_sen), len(test_clean_sen)
                            f.write("the bleu score is " + str(bleu_score1) + "\n")
                            for i in range(len(generated_test_sen)):
                                f.write("question" + str(i) + "\n")
                                f.write("answer: " + "".join(test_answer_sen[i]) + "\n")
                                f.write("noisy question: " + "".join(test_noisy_sen[i]) + "\n")
                                f.write("clean question: " + "".join(test_clean_sen[i]) + "\n")
                                f.write("generated question: " + "".join(generated_test_sen[i]) + "\n")

            #
            # saver_word_rw.save(sess_beamsearch,
            #                    "/home/ye/PycharmProjects/Qrefine/Seq_ckpt/Huawei_result/BS_Bi_" + str(Bidirection) + "_Att_" + str(
            #                        Attention) + "_Emb_" + str(
            #                        Embd_train) + "_SenRewRate_" + str(sen_reward_rate))

            # if epo > 0:
            #     for loss in val_loss_epo:
            #         print "the val_loss_epo:", loss
            #     print "the loss difference:", val_loss_epo[-2] - val_loss_epo[-1]
            #
            #     if val_loss_epo[-2] - val_loss_epo[-1] > 0.1:
            #         patience_cnt = 0
            #     else:
            #         patience_cnt += 1
            #
            #     print patience_cnt
            #
            #     if patience_cnt > 2:
            #         print("early stopping...")
            #         saver_word_rw.save(sess_beamsearch,
            #                            "/home/ye/PycharmProjects/Qrefine/Seq_ckpt/Huawei/BS_Bi_" + str(Bidirection) + "_Att_" + str(
            #                                Attention) + "_Emb_" + str(
            #                                Embd_train) + "_SenRewRate_" + str(sen_reward_rate))
            #         break



if __name__ == '__main__':
    RL_tuning_model()
