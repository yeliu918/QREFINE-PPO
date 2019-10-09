import gc

import numpy as np
import tensorflow as tf
from nltk.translate import bleu

import BeamSearch_Seq2seq
import QA_similiarity
import BA_Seq2Seq

#  Seq2Seq_model. logits

PAD = 0
SOS = 1
EOS = 2
batches_in_epoch = 200
epoch_print = 2


def next_batch(noisy_Id, noisy_len, train_Id, ground_truth, clean_len, answer_Id, answer_len, batch_size, batch_num, idx):
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
                    Bidirection=False,
                    Embd_train=True,
                    Attention=False):
    if len(data_comb) > 1:
        noisy_Id = data_comb[0]['noisy_Id']
        noisy_len = data_comb[0]['noisy_len']
        ground_truth = data_comb[0]['ground_truth']
        clean_len = data_comb[0]['clean_len']
        answer_Id = data_comb[0]['answer_Id']
        answer_len = data_comb[0]['answer_len']
        train_Id = data_comb[0]['train_Id']
        vocab = data_comb[0]['vocab']
        embd = data_comb[0]['embd']

        max_batches = len(noisy_Id) / batch_size

        val_noisy_Id = data_comb[1]['noisy_Id']
        val_noisy_len = data_comb[1]['noisy_len']
        val_ground_truth = data_comb[1]['ground_truth']
        val_clean_len = data_comb[1]['clean_len']
        val_answer_Id = data_comb[1]['answer_Id']
        val_answer_len = data_comb[1]['answer_len']
        val_train_Id = data_comb[1]['train_Id']
        max_val_batches = len(val_noisy_Id) / batch_size

    elif len(data_comb) == 1:
        noisy_Id = data_comb['noisy_Id']
        noisy_len = data_comb['noisy_len']
        ground_truth = data_comb['ground_truth']
        clean_len = data_comb['clean_len']
        answer_Id = data_comb['answer_Id']
        answer_len = data_comb['answer_len']
        train_Id= data_comb['train_Id']
        vocab = data_comb['vocab']
        embd = data_comb['embd']
        embd = np.array(embd)
        max_batches = len(noisy_Id) / batch_size

    vocab_size = len(vocab)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    model_name = "BS_"

    # values = {}
    # checkpoint_path = "../Seq_ckpt/pretrain-seq2seq_Bi_"+str(Bidirection) + "_Att_"+ str(Attention) + "_Emb_"+str(Embd_train)
    # reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    # var_to_shape_map = reader.get_variable_to_shape_map()
    # for key in var_to_shape_map:
    #     if 'loss_fun' not in key:
    #         values[model_name + key + ':0'] = reader.get_tensor(key)

    model_cond = "training"
    G_Seq2Seq = tf.Graph()
    sess_word_rw = tf.Session(config=config, graph=G_Seq2Seq)
    with G_Seq2Seq.as_default():
        Seq2Seq_model = BA_Seq2Seq.Bi_Att_Seq2Seq(batch_size, vocab_size, num_units, embd, model_cond,
                                                  Bidirection, Embd_train, Attention)
        saver_word_rw = tf.train.Saver()
        saver_word_rw.restore(sess_word_rw,
                              "../Seq_ckpt/pretrainALL-seq2seq_Bi_" + str(Bidirection) + "_Att_" + str(
                                  Attention) + "_Emb_" + str(
                                  Embd_train))

    model_type = "testing"
    G_QA_similiarity = tf.Graph()
    sess_QA_rw = tf.Session(config=config, graph=G_QA_similiarity)
    with G_QA_similiarity.as_default():
        QA_simi_model = QA_similiarity.QA_similiarity(batch_size, num_units, embd, model_type)
        saver_sen_rw = tf.train.Saver()
        saver_sen_rw.restore(sess_QA_rw, "../Seq_ckpt/qa_similiarity")

    G_BeamSearch = tf.Graph()
    with G_BeamSearch.as_default():
        BeamSearch_seq2seq = BeamSearch_Seq2seq.BeamSearch_Seq2seq(vocab_size=vocab_size,
                                                                   num_units=num_units,
                                                                   beam_width=beam_width,
                                                                   model_name=model_name, embd=embd,
                                                                   Bidirection=Bidirection,
                                                                   Embd_train=Embd_train, Attention=Attention)

    with tf.Session(config=config, graph=G_BeamSearch) as sess_beamsearch:
        # if os.path.isfile("../Seq_ckpt/BS_Bi_" + str(Bidirection) + "_Att_" + str(
        #         Attention) + "_Emb_" + str(Embd_train) + "_SenRewRate_" + str(sen_reward_rate)):
        #     saver_word_rw = tf.train.Saver()
        #     saver_word_rw.restore(sess_beamsearch, "../Seq_ckpt/BS_Bi_" + str(Bidirection) + "_Att_" + str(
        #         Attention) + "_Emb_" + str(Embd_train) + "_SenRewRate_" + str(sen_reward_rate))
        sess_beamsearch.run(tf.global_variables_initializer())

        for RL_loop in range(epoch):
            Seq2seq_loop = epoch - RL_loop

            for epo in range(Seq2seq_loop):
                if epo == 0:
                    print ("Staring Seq2Seq:")
                print('Epoch {}'.format(epo))

                # print [v for v in tf.trainable_variables()]

                idx = np.arange(0, len(noisy_Id))
                idx = list(np.random.permutation(idx))

                for batch in range(len(noisy_Id) / batch_size):
                    source_shuffle, source_len, train_shuffle, target_shuffle, target_len, answer_shuffle, ans_len = next_batch(
                        noisy_Id, noisy_len, train_Id, ground_truth, clean_len, answer_Id, answer_len, batch_size, batch, idx)

                    source_shuffle = tf.keras.preprocessing.sequence.pad_sequences(source_shuffle, maxlen=None,
                                                                                   padding='post', value=EOS)

                    target_shuffle = tf.keras.preprocessing.sequence.pad_sequences(target_shuffle, maxlen=None,
                                                                                   padding='post', value=EOS)

                    train_shuffle = tf.keras.preprocessing.sequence.pad_sequences(train_shuffle, maxlen=None,
                                                                                  padding='post', value=EOS)

                    target_len = np.tile(max(target_len) + 1, batch_size)

                    fd = {BeamSearch_seq2seq.encoder_inputs: source_shuffle,
                          BeamSearch_seq2seq.encoder_inputs_length: source_len,
                          BeamSearch_seq2seq.decoder_length: target_len,
                          BeamSearch_seq2seq.decoder_inputs: train_shuffle,
                          BeamSearch_seq2seq.decoder_targets: target_shuffle}

                    cl_loss, _ = sess_beamsearch.run(
                        [BeamSearch_seq2seq.loss_seq2seq, BeamSearch_seq2seq.train_op], fd)

                    if batch == 0 or batch % batches_in_epoch == 0:
                        print(' batch {}'.format(batch))
                        print('   minibatch loss of training: {}'.format(cl_loss))
                        val_loss=[]
                        for batch_val in range(len(val_noisy_Id)/batch_size):
                            idx_v = np.arange(0, len(val_noisy_Id))
                            idx_v = list(np.random.permutation(idx_v))

                            val_source_shuffle, val_source_len, val_train_shuffle, val_target_shuffle, val_target_len, val_answer_shuffle, val_ans_len = next_batch(
                                val_noisy_Id, val_noisy_len,val_train_Id,val_ground_truth, val_clean_len, val_answer_Id, val_answer_len,
                                batch_size, batch_val, idx_v)
                            val_source_shuffle = tf.keras.preprocessing.sequence.pad_sequences(val_source_shuffle, maxlen=None,
                                                                                               padding='post', value=EOS)
                            val_target_shuffle = tf.keras.preprocessing.sequence.pad_sequences(val_target_shuffle, maxlen=None,
                                                                                               padding='post', value=EOS)
                            val_train_shuffle = tf.keras.preprocessing.sequence.pad_sequences(val_target_shuffle, maxlen=None,
                                                                                               padding='post', value=EOS)

                            val_target_len = np.tile(max(val_target_len) + 1, batch_size)

                            fd = {BeamSearch_seq2seq.encoder_inputs: val_source_shuffle,
                                  BeamSearch_seq2seq.encoder_inputs_length: val_source_len,
                                  BeamSearch_seq2seq.decoder_length: val_target_len,
                                  BeamSearch_seq2seq.decoder_inputs:val_train_shuffle,
                                  BeamSearch_seq2seq.decoder_targets: val_target_shuffle}
                            val_loss.append(sess_beamsearch.run(BeamSearch_seq2seq.loss_seq2seq, fd))

                        print('   minibatch loss of training on validation: {}'.format(sum(val_loss)/float(len(val_loss))
                            ))

                    gc.collect()
        # saver_word_rw.save(sess_beamsearch,  "../Seq_ckpt/BS_Bi_" + str(Bidirection) + "_Att_" + str(Attention) + "_Emb_" + str(
        #                Embd_train))
        #
        # with tf.Session(config=config, graph=G_BeamSearch) as sess_beamsearch:
        #     # saver = tf.train.Saver()
        #     # saver= tf.train.Saver(var_list=['encoder_model/rnn/basic_lstm_cell/bias','encoder_model/rnn/basic_lstm_cell/kernel'])
        #     # saver.restore(sess_beamsearch,"../Seq_ckpt/pretrain-seq2seq")
        #
        #     if os.path.isfile("../Seq_ckpt/BS_Bi_" + str(Bidirection) + "_Att_" + str(
        #             Attention) + "_Emb_" + str(Embd_train)):
        #         saver = tf.train.Saver()
        #     saver.restore(sess_beamsearch, "../Seq_ckpt/BS_Bi_" + str(Bidirection) + "_Att_" + str(
        #         Attention) + "_Emb_" + str(Embd_train))
        #
        #     sess_beamsearch.run(tf.global_variables_initializer())

            # print [v for v in tf.trainable_variables()]
            #
            # for v in tf.trainable_variables():
            #     if v.name in values.keys():
            #         v.load(values[v.name], sess_beamsearch)

            for epo in range(RL_loop):
                if epo == 0:
                    print ("staring RL tuning:")
                print('Epoch {}'.format(epo))
                idx = np.arange(0, len(noisy_Id))
                idx = list(np.random.permutation(idx))

                # print [v for v in tf.trainable_variables()]

                Bleu_score1 = []

                for batch in range(len(noisy_Id) /batch_size):
                    beam_QA_similar = []
                    beam_cnQA_similar =[]
                    source_shuffle, source_len, train_shuffle, target_shuffle, target_len, answer_shuffle, ans_len = next_batch(
                        noisy_Id, noisy_len, train_Id, ground_truth, clean_len, answer_Id, answer_len, batch_size, batch, idx)

                    source_shuffle = tf.keras.preprocessing.sequence.pad_sequences(source_shuffle, maxlen=None,
                                                                                   padding='post', value=EOS)
                    dis_rewards = np.zeros((batch_size, max(source_len), beam_width))

                    fd = {BeamSearch_seq2seq.encoder_inputs: source_shuffle,
                          BeamSearch_seq2seq.encoder_inputs_length: source_len,
                          BeamSearch_seq2seq.dis_rewards: dis_rewards}
                    predicting_scores1, logits, log_pro = sess_beamsearch.run(
                        [BeamSearch_seq2seq.predicting_scores, BeamSearch_seq2seq.predicting_logits,
                         BeamSearch_seq2seq.predicting_scores], fd)

                    max_target_length = logits.shape[1]
                    # print "the max_target_length is:", max_target_length

                    beam_rewards = np.zeros((batch_size, max_target_length, beam_width))

                    for bw in range(beam_width):
                        generated_que = logits[:, :, bw]

                        logits_batch = []
                        for i in range(generated_que.shape[0]):
                            for j in range(generated_que.shape[1]):
                                if generated_que[i][j] == 2 or j == generated_que.shape[1] - 1:
                                    logits_batch.append(j)
                                    continue

                        reward = np.zeros((batch_size, max_target_length))
                        # generated_que = SOS + generated_que - PAD
                        generated_que_input = np.insert(generated_que, 0, SOS, axis=1)
                        target_shuffle = tf.keras.preprocessing.sequence.pad_sequences(target_shuffle, maxlen=None,
                                                                                       padding='post', value=EOS)

                        generated_target_len = np.tile(generated_que_input.shape[1], batch_size)

                        fd = {Seq2Seq_model.encoder_inputs: source_shuffle,
                              Seq2Seq_model.encoder_inputs_length: source_len,
                              Seq2Seq_model.decoder_inputs: generated_que_input,
                              Seq2Seq_model.decoder_length: generated_target_len,
                              Seq2Seq_model.ground_truth: target_shuffle}
                        logits_pro = sess_word_rw.run(Seq2Seq_model.softmax_logits, fd)

                        logits_pro = logits_pro[:, :-1, :]
                        # print "the logits_pro is:", logits_pro.shape[1]

                        answer_shuffle = tf.keras.preprocessing.sequence.pad_sequences(answer_shuffle, maxlen=None,
                                                                                       padding='post', value=EOS)

                        generated_len = np.tile(generated_que.shape[1], batch_size)

                        fd = {QA_simi_model.answer_inputs: answer_shuffle, QA_simi_model.answer_inputs_length: ans_len,
                              QA_simi_model.question1_inputs: source_shuffle,
                              QA_simi_model.question1_inputs_length: source_len,
                              QA_simi_model.question2_inputs: generated_que,
                              QA_simi_model.question2_inputs_length: generated_len}
                        QA_similiarity_rd = sess_QA_rw.run([QA_simi_model.two_distance], fd)

                        beam_QA_similar.append(QA_similiarity_rd[0])
                        # print QA_similiarity_rd

                        for i in range(generated_que.shape[0]):
                            for j in range(generated_que.shape[1]):
                                if j < logits_batch[i]:
                                    max_index = generated_que[i][j]
                                    reward[i][j] = logits_pro[i, j, max_index]
                                if j == logits_batch[i]:
                                    reward[i][j] = reward[i][j] + sen_reward_rate * (QA_similiarity_rd[0][i])
                                    continue

                        # fd = {an_Lstm.answer_inputs: answer_shuffle, an_Lstm.answer_inputs_length: ans_len}
                        # reward_similiarity = sess_sen_rw.run([an_Lstm.answer_state], fd)

                        discounted_rewards = discounted_rewards_cal(reward, discount_factor)

                        for i in range(batch_size):
                            for j in range(max_target_length):
                                beam_rewards[i][j][bw] = discounted_rewards[i][j]
                                # print np.asarray(beam_rewards).shape

                    fd = {QA_simi_model.answer_inputs: answer_shuffle, QA_simi_model.answer_inputs_length: ans_len,
                          QA_simi_model.question1_inputs: source_shuffle,
                          QA_simi_model.question1_inputs_length: source_len,
                          QA_simi_model.question2_inputs: target_shuffle,
                          QA_simi_model.question2_inputs_length: target_len}
                    cnQA_similiarity_rd = sess_QA_rw.run([QA_simi_model.two_distance], fd)

                    beam_cnQA_similar.append(cnQA_similiarity_rd[0])

                    for i in range(batch_size):
                        discounted_rewards = beam_rewards[i]
                        discounted_rewards = np.float32(discounted_rewards)
                        discounted_rewards -= np.mean(discounted_rewards, axis=1).reshape(max_target_length, 1)
                        beam_rewards[i] = discounted_rewards

                    # print "the size of beam_rewards:", beam_rewards.shape[1]

                    fd = {BeamSearch_seq2seq.encoder_inputs: source_shuffle,
                          BeamSearch_seq2seq.encoder_inputs_length: source_len,
                          BeamSearch_seq2seq.dis_rewards: beam_rewards}
                    predicting_scores2, _, rl_reward = sess_beamsearch.run(
                        [BeamSearch_seq2seq.predicting_scores, BeamSearch_seq2seq.RL_train_op,
                         BeamSearch_seq2seq.rl_reward], fd)

                    for t in range(batch_size):
                        ref = []
                        hyp = []
                        bleu_s = []
                        for tar_id in target_shuffle[t]:
                            if tar_id != 2:
                                ref.append(vocab[tar_id])
                        for i in range(beam_width):
                            for pre_id in logits[t, :, i]:
                                if pre_id != 2 and pre_id != -1:
                                    hyp.append(vocab[pre_id])
                            sen_bleu1 = bleu([ref], hyp, weights=(1, 0, 0, 0))
                            bleu_s.append(sen_bleu1)
                        Bleu_score1.append(bleu_s)

                    top1_bleu = []
                    top2_bleu = []
                    top3_bleu = []
                    top4_bleu = []
                    top5_bleu = []
                    for i in range(len(Bleu_score1)):
                        top1_bleu.append(Bleu_score1[i][0])
                        top2_bleu.append(Bleu_score1[i][1])
                        top3_bleu.append(Bleu_score1[i][2])
                        top4_bleu.append(Bleu_score1[i][3])
                        top5_bleu.append(Bleu_score1[i][4])

                    if batch == 0 or batch % 20 == 0:
                        print(' batch {}'.format(batch))
                        print('   minibatch reward of training: {}'.format(- rl_reward))
                        for t in range(5):  ## five sentences
                            print('Question {}'.format(t))
                            print("noisy question:")
                            print(" ".join(map(lambda i: vocab[i], list(source_shuffle[t]))).strip())
                            print("clean question:")
                            print(" ".join(map(lambda i: vocab[i], list(target_shuffle[t]))).strip())
                            # print " QA similiarity of the noisy and clean question:", beam_cnQA_similar[t]
                            # text_file.write(
                            #     " ".join(map(lambda i: vocab[i], list(target_shuffle[t]))).encode('utf-8').strip() + "\n")
                            print("the beam search producted:\n")
                            for i in range(beam_width):  ##beam size
                                pre_sen = []
                                for log_id in logits[t, :, i]:
                                    if log_id != -1:
                                        pre_sen.append(log_id)
                                print(" ".join(map(lambda i: vocab[i], pre_sen)).strip())
                                print "    QA similiarity:", beam_QA_similar[t][i]

                                # logP = []
                                # for j in range(predicting_scores1[t].shape[0]):t
                                #     logP.append(predicting_scores1[t][j][i])
                                # print "log P", logP

                                # disrewards=[]
                                # for j in range(beam_rewards[t].shape[0]):
                                #     disrewards.append(beam_rewards[t][j][i])
                                # print "Discounted Reward", disrewards

                                # print " ".join(map(lambda i: vocab[i], list(logits[t, :, i]))).strip()
                                # text_file.write(
                                #     " ".join(map(lambda i: vocab[i], list(logits[t, :, i]))).encode('utf-8').strip() + "\n")
                        print("the current whole bleu_score1 is:", sum(top1_bleu) / float(len(top1_bleu)))
                        print("the current whole bleu_score2 is:", sum(top2_bleu) / float(len(top2_bleu)))
                        print("the current whole bleu_score3 is:", sum(top3_bleu) / float(len(top3_bleu)))
                        print("the current whole bleu_score4 is:", sum(top4_bleu) / float(len(top4_bleu)))
                        print("the current whole bleu_score4 is:", sum(top5_bleu) / float(len(top5_bleu)))

                    gc.collect()
        saver_word_rw.save(sess_beamsearch,
                           "../Seq_ckpt/BS_Bi_" + str(Bidirection) + "_Att_" + str(Attention) + "_Emb_" + str(
                               Embd_train)+"_SenRewRate_"+ str(sen_reward_rate))


if __name__ == '__main__':
    RL_tuning_model()
