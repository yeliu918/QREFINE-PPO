# -*- coding: UTF-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import tensorflow as tf
import pickle
from nltk.translate import bleu
from tensorflow.contrib import keras
import Data_loading
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple
import gc
from Evaluation_Package.rouge.rouge import Rouge
from Evaluation_Package.meteor.meteor import Meteor
from Evaluation_Package.cider.cider import Cider
from Evaluation_Package.bleu.bleu import Bleu

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class Bi_Att_Seq2Seq(object):
    def __init__(self, batch_size, vocab_size, num_units, embd, model_cond, Bidirection, Embd_train, Attention
                 ):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.embedding = embd
        self.model_cond = model_cond
        self.Bidirection = Bidirection
        self.embd_train = Embd_train
        self.Attention =  Attention
        self.build_graph()

    def build_graph(self):
        # input placehodlers
        with tf.variable_scope("model_inputs"):
            self.encoder_inputs = tf.placeholder(shape=(self.batch_size, None),
                                                 dtype=tf.int32, name='encoder_inputs')
            self.encoder_inputs_length = tf.placeholder(shape=(self.batch_size,), dtype=tf.int32,
                                                        name='encoder_inputs_length')
            if self.model_cond == "training":
                self.decoder_inputs = tf.placeholder(shape=(self.batch_size, None),
                                                     dtype=tf.int32, name='decoder_targets')
                self.decoder_length = tf.placeholder(shape=(self.batch_size,), dtype=tf.int32, name='decoder_length')
                self.ground_truth = tf.placeholder(shape=(self.batch_size, None),
                                                   dtype=tf.int32, name='ground_truth')
            self.dropout_rate = tf.placeholder(dtype=tf.float32,name="dropout_rate")

        with tf.variable_scope("embeddings"):
            self.embeddings = tf.get_variable(name="embedding_W", shape=self.embedding.shape,
                                              initializer=tf.constant_initializer(self.embedding), trainable= self.embd_train)

            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)

            if self.model_cond == "training":
                self.decoder_outputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)

        self.inference()
        self.loss()
        # if self.model_cond == "training":
        #     tf.summary.scalar("loss", self.loss_seq2seq)
        #

    def inference(self):
        ####Encoder
        with tf.variable_scope("encoder_model"):
            if self.Bidirection == False:
                encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units)
                encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell,input_keep_prob=self.dropout_rate, output_keep_prob=self.dropout_rate, state_keep_prob=self.dropout_rate)
                encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                                              inputs=self.encoder_inputs_embedded,
                                                                              sequence_length=self.encoder_inputs_length,
                                                                              time_major=False,
                                                                              dtype=tf.float32)
                self.hidden_units = self.num_units

            elif self.Bidirection == True:
                encoder_cell_fw = LSTMCell(self.num_units)
                encoder_cell_bw = LSTMCell(self.num_units)
                ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)) = (
                    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_fw, cell_bw=encoder_cell_bw,
                                                    inputs=self.encoder_inputs_embedded,
                                                    sequence_length=self.encoder_inputs_length,
                                                    dtype=tf.float32, time_major=False))
                # Concatenates tensors along one dimension.
                encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

                encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
                encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

                # TF Tuple used by LSTM Cells for state_size, zero_state, and output state.
                self.encoder_final_state = LSTMStateTuple(c=encoder_final_state_c, h=encoder_final_state_h)
                self.hidden_units = 2 * self.num_units

        ###Decoder
        with tf.variable_scope("decoder_model"):
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_units)

            decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell ,input_keep_prob=self.dropout_rate, output_keep_prob=self.dropout_rate, state_keep_prob=self.dropout_rate)
            # attention_states: [batch_size, max_time, num_units]
            # attention_states = tf.transpose(encoder_outputs, [0, 1, 2])
            self.initial_state = self.encoder_final_state

            if self.Attention == True:
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    self.hidden_units, encoder_outputs,
                    memory_sequence_length=self.encoder_inputs_length)

                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    decoder_cell, attention_mechanism,
                    attention_layer_size=self.hidden_units)

                zero_state = decoder_cell.zero_state(self.batch_size, tf.float32)

                self.initial_state = zero_state.clone(cell_state=self.encoder_final_state)

            if self.model_cond == 'training':
                helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_outputs_embedded, self.decoder_length,
                                                           time_major=False)
            elif self.model_cond == 'testing':
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embeddings,
                                                                  start_tokens=tf.fill([self.batch_size], 1),
                                                                  end_token=2)

            decoder = tf.contrib.seq2seq.BasicDecoder(cell = decoder_cell, helper=helper,
                                                      initial_state=self.initial_state,
                                                      output_layer=tf.layers.Dense(self.vocab_size))
            # Dynamic decoding
            if self.model_cond == "training":
                outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False,
                                                                            impute_finished=True,
                                                                            maximum_iterations=tf.reduce_max(self.decoder_length, axis=0))
            elif self.model_cond == "testing":
                outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False,
                                                                            impute_finished=True,
                                                                            maximum_iterations=2 * tf.reduce_max(
                                                                                self.encoder_inputs_length, axis=0))

            self.logits = tf.identity(outputs.rnn_output)
            self.ids = tf.identity(outputs.sample_id)
            self.softmax_logits = tf.nn.softmax(self.logits, dim=2)
            self.final_state = final_state

    def loss(self):
        # loss function
        if self.model_cond == "training":
            with tf.variable_scope("loss_fun"):
                weights=["encoder_model/rnn/basic_lstm_cell/kernel:0",'decoder_model/decoder/basic_lstm_cell/kernel:0','decoder_model/decoder/dense/kernel:0']
                # self.regularizer = 0.01 * tf.nn.l2_loss(weights)
                padding = 2 * tf.ones(
                    [tf.shape(self.logits)[0], tf.shape(self.ground_truth)[1] - tf.shape(self.logits)[1],
                     tf.shape(self.logits)[2]])
                self.logits_padded = tf.concat([self.logits, padding], 1)

                # self.mask = tf.sequence_mask(tf.fill([self.batch_size], tf.shape(self.logits)[1]),tf.shape(self.logits)[1])
                self.mask = tf.sequence_mask(self.decoder_length, tf.shape(self.ground_truth)[1])

                self.loss_seq2seq = tf.contrib.seq2seq.sequence_loss(logits=self.logits_padded,
                                                                     targets=self.ground_truth, weights=tf.cast(self.mask, tf.float32))
                self.loss_seq2seq = tf.reduce_mean(self.loss_seq2seq)
                self.train_op = tf.train.AdamOptimizer().minimize(self.loss_seq2seq)

                # if self.optimize =="Adam_dec":
                #     hparams = tf.contrib.training.HParams(
                #         learning_rate=0.001,
                #         decay_rate=0.96,
                #         decay_steps=1,
                #     )
                #     global_step = tf.get_variable(
                #         'global_step', [],
                #         initializer=tf.constant_initializer(0),
                #         trainable=False)
                #
                #     learning_rate = tf.train.exponential_decay(
                #         learning_rate=hparams.learning_rate,
                #         global_step=global_step,
                #         decay_steps=hparams.decay_steps,
                #         decay_rate=hparams.decay_rate)
                #     learning_rate = tf.maximum(learning_rate, 1e-6)
                #     self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_seq2seq)



def next_batch(noisy_Id, noisy_len, input_Id,  target_Id, clean_len, batch_size, batch_num, idx):
    if (batch_num + 1) * batch_size > len(noisy_Id):
        batch_num = batch_num % (len(noisy_Id) / batch_size)
    idx_n = idx[batch_num * batch_size: (batch_num + 1) * batch_size]
    data_shuffle = [noisy_Id[i] for i in idx_n]
    data_len = [noisy_len[i] for i in idx_n]
    train_shuffle = [input_Id[i] for i in idx_n]
    target_len = [clean_len[i] for i in idx_n]
    target_shuffle = [target_Id[i] for i in idx_n]
    return data_shuffle, data_len, train_shuffle, target_shuffle, target_len



def next_batch_answer(noisy_Id, noisy_len, input_Id,  ground_truth, clean_len, answer_Id, batch_size, batch_num,
               idx):
    if (batch_num + 1) * batch_size > len(noisy_Id):
        batch_num = batch_num % (len(noisy_Id) / batch_size)
    idx_n = idx[batch_num * batch_size: (batch_num + 1) * batch_size]
    data_shuffle = [noisy_Id[i] for i in idx_n]
    data_len = [noisy_len[i] for i in idx_n]
    target_len = [clean_len[i] for i in idx_n]
    train_shuffle = [input_Id[i] for i in idx_n]
    target_shuffle = [ground_truth[i] for i in idx_n]
    answer_shuffle = [answer_Id[i] for i in idx_n]
    return data_shuffle, data_len, train_shuffle, target_shuffle, target_len, answer_shuffle


def main():
    PAD = 0
    SOS = 1
    EOS = 2
    epoch = 20
    batches_in_epoch = 100
    batch_size = 64
    num_units = 300
    epoch_print = 2
    Bidirection = False
    Attention = True
    Embd_train = False
    model_type = 'testing'

    # logs_path = "../Seq_ckpt/seq2seq_BIA_board"

    # FileName = '../result/Huawei/large_unique0.95_with'

    # FileName = "../data/Wiki/wiki_ids_16"
    FileName = "../data/Wiki/wiki_ids_Q10"
    train_data_com, val_data_com, test_data_com, max_data, data_com = Data_loading.load_data(FileName)

    # data = pickle.load(f)
    # noisy_Id = data['noisy_Id']
    # clean_Id = data['clean_Id']
    # gt_Id = []
    # train_Id = []
    # # answer_Id = data['answer_Id']
    # vocab = data['vocab']
    # embd = data['embd']
    # vocab_size = len(vocab)
    #
    # noisy_len = []
    # for n in noisy_Id:
    #     noisy_len.append(len(n))
    # max_source_length = max(noisy_len)
    # print "the noisy length:", max_source_length
    #
    # clean_len = []
    # for c in clean_Id:
    #     clean_len.append(len(c) + 1)
    # max_target_length = max(clean_len)
    # max_target_length = max_target_length
    # print "the clean length:", max_target_length
    #
    # # answer_len = []
    # # for a in answer_Id:
    # #     answer_len.append(len(a))
    # # max_answer_length = max(answer_len)
    #
    # num_data = len(noisy_Id)
    # for i in range(num_data):
    #     train_Id.append([SOS] + clean_Id[i])
    #     gt_Id.append(clean_Id[i] + [EOS])
    #
    # for i in range(len(clean_len)):
    #     clean_len[i] = clean_len[i] - 1
    #
    # # train_noisy_Id = noisy_Id[0:int(0.8 * num_data)]
    # # train_noisy_len = noisy_len[0:int(0.8 * num_data)]
    # # train_input_Id = train_Id[0:int(0.8 * num_data)]
    # # train_clean_len = clean_len[0:int(0.8 * num_data)]
    # # train_target_Id = gt_Id[0:int(0.8 * num_data)]
    #
    # train_noisy_Id = noisy_Id
    # train_noisy_len = noisy_len
    # train_input_Id = train_Id
    # train_clean_len = clean_len
    # train_target_Id = gt_Id
    #
    # test_noisy_Id = noisy_Id[int(0.8 * num_data):]
    # test_noisy_len = noisy_len[int(0.8 * num_data):]
    # test_input_Id = train_Id[int(0.8 * num_data):]
    # test_clean_len = clean_len[int(0.8 * num_data):]
    # test_target_Id = gt_Id[int(0.8 * num_data):]
    #
    #
    #
    # # eval_noisy_Id = noisy_Id[int(0.6 * num_data) + 1:int(0.8 * num_data)]
    # # eval_noisy_len = noisy_len[int(0.6 * num_data) + 1:int(0.8 * num_data)]
    # # eval_clean_Id = clean_Id[int(0.6 * num_data) + 1:int(0.8 * num_data)]
    # # eval_clean_len = clean_len[int(0.6 * num_data) + 1:int(0.8 * num_data)]

    # train_noisy_Id = train_data_com['noisy_Id']
    # train_noisy_len = train_data_com['noisy_len']
    # train_target_Id = train_data_com['ground_truth']
    # train_clean_len = train_data_com['clean_len']
    # train_input_Id = train_data_com['train_Id']
    # vocab = train_data_com['vocab']
    # embd = train_data_com['embd']
    #
    val_noisy_Id = val_data_com['noisy_Id']
    val_noisy_len = val_data_com['noisy_len']
    val_ground_truth = val_data_com['ground_truth']
    val_clean_len = val_data_com['clean_len']
    val_train_Id = val_data_com['train_Id']

    test_noisy_Id = test_data_com['noisy_Id']
    test_noisy_len = test_data_com['noisy_len']
    test_target_Id = test_data_com['ground_truth']
    test_clean_len = test_data_com['clean_len']
    test_input_Id = test_data_com['train_Id']
    test_answer_Id = test_data_com['answer_Id']

    ## use the whole data to train the model
    # train_noisy_Id = data_com['noisy_Id']
    # train_noisy_len = data_com['noisy_len']
    # train_target_Id = data_com['ground_truth']
    # train_clean_len = data_com['clean_len']
    # train_input_Id = data_com['train_Id']

    train_noisy_Id = train_data_com['noisy_Id']
    train_noisy_len = train_data_com['noisy_len']
    train_target_Id = train_data_com['ground_truth']
    train_clean_len = train_data_com['clean_len']
    train_input_Id = train_data_com['train_Id']


    vocab = train_data_com['vocab']
    embd = train_data_com['embd']
    embd = np.array(embd)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    vocab_size = len(vocab)

    Bleu_obj = Bleu()
    Rouge_obj = Rouge()
    Meteor_obj = Meteor()
    cIDER_obj = Cider()


    if model_type == 'training':
        Seq2Seq_model = Bi_Att_Seq2Seq(batch_size, vocab_size, num_units, embd, model_type, Bidirection, Embd_train, Attention)

        print "the number of training question is:", len(train_noisy_Id)
        # print "the number of evaluate question is:", len(eval_noisy_Id)

        patience_cnt = 0

        # summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        saver = tf.train.Saver(sharded=False)
        merge = tf.summary.merge_all()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            # saver.restore(sess, "../Seq_ckpt/pretrain-model")
            val_loss_epo = []
            val_Bleu_epo = []
            # print [v for v in tf.trainable_variables()]
            for epo in range(epoch):

                idx = np.arange(0, len(train_noisy_Id))
                idx = list(np.random.permutation(idx))

                print " Epoch {}".format(epo)

                Bleu_score1 = []
                Bleu_score2 = []
                Bleu_score3 = []
                Bleu_score4 = []
                Rouge_score = []
                Meteor_score = []
                Cider_score = []

                for batch in range(len(train_noisy_Id) / batch_size):
                    source_shuffle, source_len, train_shuffle, target_shuffle, target_len = next_batch(train_noisy_Id,
                                                                                                       train_noisy_len,
                                                                                                       train_input_Id,
                                                                                                       train_target_Id,
                                                                                                       train_clean_len,
                                                                                                       batch_size,
                                                                                                       batch,
                                                                                                       idx)
                    source_shuffle = tf.keras.preprocessing.sequence.pad_sequences(source_shuffle, maxlen=None,
                                                                                   padding='post', value=EOS)
                    train_shuffle = tf.keras.preprocessing.sequence.pad_sequences(train_shuffle, maxlen=None,
                                                                                  padding='post', value=EOS)

                    # train_shuffle = tf.keras.preprocessing.sequence.pad_sequences(target_shuffle, maxlen=None,
                    #                                                               padding='post', value=EOS)
                    target_shuffle = tf.keras.preprocessing.sequence.pad_sequences(target_shuffle, maxlen=None,
                                                                                   padding='post', value=EOS)

                    # target_len = np.tile(max(target_len) + 1, batch_size)

                    fd = {Seq2Seq_model.encoder_inputs: source_shuffle, Seq2Seq_model.encoder_inputs_length: source_len,
                          Seq2Seq_model.decoder_inputs: train_shuffle, Seq2Seq_model.decoder_length: target_len,
                          Seq2Seq_model.ground_truth: target_shuffle, Seq2Seq_model.dropout_rate : 1}

                    mask, logit_id, logit_id_pad, l_fun, pro_op = sess.run(
                        [Seq2Seq_model.mask, Seq2Seq_model.ids, Seq2Seq_model.logits_padded, Seq2Seq_model.loss_seq2seq, Seq2Seq_model.train_op], fd)

                    # mask, logit_id, logit_id_pad= sess.run(
                    #     [Seq2Seq_model.mask, Seq2Seq_model.ids, Seq2Seq_model.logits_padded], fd)
                    # l_fun =0

                    for t in range(batch_size):
                        ref = []
                        hyp = []
                        for tar_id in target_shuffle[t]:
                            if tar_id != EOS and tar_id!=PAD:
                                ref.append(vocab[tar_id])
                                # print vocab[tar_id]
                        for pre_id in logit_id[t]:
                            if pre_id !=EOS and pre_id!=PAD:
                                hyp.append(vocab[pre_id])
                                # print vocab[pre_id]
                        # sen_bleu1 = bleu([ref], hyp, weights=(1, 0, 0, 0))
                        # sen_bleu2 = bleu([ref], hyp, weights=(0.5, 0.5, 0, 0))
                        # sen_bleu3 = bleu([ref], hyp, weights=(0.333, 0.333, 0.333, 0))
                        # sen_bleu4 = bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25))

                        hyp_sen = " ".join(hyp)
                        ref_sen = " ".join(ref)
                        dic_hyp = {}
                        dic_hyp[0] = [hyp_sen]
                        dic_ref = {}
                        dic_ref[0] = [ref_sen]
                        sen_bleu,_ = Bleu_obj.compute_score(dic_ref,dic_hyp)
                        sen_rouge = Rouge_obj.compute_score(dic_ref,dic_hyp)
                        sen_meteor,_ = Meteor_obj.compute_score(dic_ref,dic_hyp)
                        sen_cider,_ = cIDER_obj.compute_score(dic_ref,dic_hyp)
                        Bleu_score1.append(sen_bleu[0])
                        Bleu_score2.append(sen_bleu[1])
                        Bleu_score3.append(sen_bleu[2])
                        Bleu_score4.append(sen_bleu[3])
                        Rouge_score.append(sen_rouge[0])
                        Meteor_score.append(sen_meteor)
                        Cider_score.append(sen_cider)

                    # summary_writer.add_summary(summary, batch)

                    if batch == 0 or batch % batches_in_epoch == 0:
                        ##print the training
                        print('batch {}'.format(batch))
                        print('   minibatch loss: {}'.format(l_fun))

                        # for t in xrange(3):
                        #     print('Training Question {}'.format(t))
                        #     print("NQ: " + " ".join(map(lambda i: vocab[i], list(source_shuffle[t]))).strip().replace(
                        #         "<EOS>", " "))
                        #     print("CQ: " + " ".join(map(lambda i: vocab[i], list(target_shuffle[t]))).strip().replace(
                        #         "<EOS>", " "))
                        #     print("GQ: " + " ".join(map(lambda i: vocab[i], list(logit_id[t]))).strip().replace("<EOS>", " "))

                        print("\n Bleu_score1, Bleu_score2, Bleu_score3, Bleu_score4, Rouge, Meteor Cider\n")
                        print("   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f}".format(sum(Bleu_score1) / float(len(Bleu_score1)),
                                                                                 sum(Bleu_score2) / float(len(Bleu_score2)),
                                                                                 sum(Bleu_score3) / float(len(Bleu_score3)),
                                                                                 sum(Bleu_score4) / float(len(Bleu_score4)),
                                                                                 sum(Rouge_score) / float(len(Rouge_score)),
                                                                                 sum(Meteor_score) / float(len(Meteor_score)),
                                                                                 sum(Cider_score) / float(len(Cider_score))
                                                                                 ))

                        val_loss = []
                        val_Bleu_score1 = []
                        val_Bleu_score2 = []
                        val_Bleu_score3 = []
                        val_Bleu_score4 = []
                        val_Rouge_score = []
                        val_Meteor_score = []
                        val_Cider_score = []

                        for batch_val in range(len(val_noisy_Id) / batch_size):
                            idx_v = np.arange(0, len(val_noisy_Id))
                            idx_v = list(np.random.permutation(idx_v))

                            val_source_shuffle, val_source_len, val_train_shuffle, val_target_shuffle, val_target_len = next_batch(
                                val_noisy_Id, val_noisy_len, val_train_Id, val_ground_truth, val_clean_len, batch_size, batch_val, idx_v)
                            val_source_shuffle = tf.keras.preprocessing.sequence.pad_sequences(val_source_shuffle, maxlen=None,
                                                                                               padding='post', value=EOS)
                            val_target_shuffle = tf.keras.preprocessing.sequence.pad_sequences(val_target_shuffle, maxlen=None,
                                                                                               padding='post', value=EOS)
                            val_train_shuffle = tf.keras.preprocessing.sequence.pad_sequences(val_train_shuffle, maxlen=None,
                                                                                              padding='post', value=EOS)

                            # val_target_len = np.tile(max(val_target_len) + 1, batch_size)

                            fd_val = {Seq2Seq_model.encoder_inputs: val_source_shuffle, Seq2Seq_model.encoder_inputs_length: val_source_len,
                                      Seq2Seq_model.decoder_length: val_target_len, Seq2Seq_model.decoder_inputs: val_train_shuffle,
                                      Seq2Seq_model.ground_truth: val_target_shuffle,  Seq2Seq_model.dropout_rate:1}
                            val_ids, val_loss_seq = sess.run([Seq2Seq_model.ids, Seq2Seq_model.loss_seq2seq], fd_val)
                            val_loss.append(val_loss_seq)

                            for t in range(batch_size):
                                ref = []
                                hyp = []
                                for tar_id in val_target_shuffle[t]:
                                    if tar_id != EOS and tar_id !=PAD:
                                        ref.append(vocab[tar_id])
                                for pre_id in val_ids[t]:
                                    if pre_id != EOS and pre_id !=PAD:
                                        hyp.append(vocab[pre_id])
                                        # print vocab[pre_id]
                                # sen_bleu1 = bleu([ref], hyp, weights=(1, 0, 0, 0))
                                # sen_bleu2 = bleu([ref], hyp, weights=(0.5, 0.5, 0, 0))
                                # val_Bleu_score1.append(sen_bleu1)
                                # val_Bleu_score2.append(sen_bleu2)

                                hyp_sen = " ".join(hyp)
                                ref_sen = " ".join(ref)
                                dic_hyp = {}
                                dic_hyp[0] = [hyp_sen]
                                dic_ref = {}
                                dic_ref[0] = [ref_sen]
                                sen_bleu, _ = Bleu_obj.compute_score(dic_ref, dic_hyp)
                                sen_rouge = Rouge_obj.compute_score(dic_ref, dic_hyp)
                                sen_meteor, _ = Meteor_obj.compute_score(dic_ref, dic_hyp)
                                sen_cider, _ = cIDER_obj.compute_score(dic_ref, dic_hyp)
                                val_Bleu_score1.append(sen_bleu[0])
                                val_Bleu_score2.append(sen_bleu[1])
                                val_Bleu_score3.append(sen_bleu[2])
                                val_Bleu_score4.append(sen_bleu[3])
                                val_Rouge_score.append(sen_rouge[0])
                                val_Meteor_score.append(sen_meteor)
                                val_Cider_score.append(sen_cider)

                        # for t in xrange(3):
                        #     print 'Validation Question {}'.format(t)
                        #     print "NQ: "+" ".join(map(lambda i: vocab[i], list(val_source_shuffle[t]))).strip().replace("<EOS>"," ")
                        #     print "CQ: "+" ".join(map(lambda i: vocab[i], list(val_target_shuffle[t]))).strip().replace("<EOS>"," ")
                        #     print "GQ: "+" ".join(map(lambda i: vocab[i], list(val_ids[t]))).strip().replace("<EOS>"," ")

                        avg_val_loss = sum(val_loss) / float(len(val_loss))
                        print("\n")
                        print('   minibatch loss of validation: {}'.format(avg_val_loss))

                        print("\n Bleu_score1, Bleu_score2, Bleu_score3, Bleu_score4, Rouge, Meteor Cider\n")
                        print("   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f}".format(
                            sum(val_Bleu_score1) / float(len(val_Bleu_score1)),
                            sum(val_Bleu_score2) / float(len(val_Bleu_score2)),
                            sum(val_Bleu_score3) / float(len(val_Bleu_score3)),
                            sum(val_Bleu_score4) / float(len(val_Bleu_score4)),
                            sum(val_Rouge_score) / float(len(val_Rouge_score)),
                            sum(val_Meteor_score) / float(len(val_Meteor_score)),
                            sum(val_Cider_score) / float(len(val_Cider_score))
                            ))

                val_loss_epo.append(avg_val_loss)

                if epo > 0:
                    # for loss in val_loss_epo:
                    #     print "the val_loss_epo:", loss
                    # for Bleu in val_Bleu_epo:
                    #     print "the val_Bleu_epo:", Bleu
                    # print "the loss difference:", val_loss_epo[-2] - val_loss_epo[-1]
                    # print "the Bleu score difference:", val_Bleu_epo[-1]- val_Bleu_epo[-2]

                    if val_loss_epo[-2] - val_loss_epo[-1] > 0:
                        patience_cnt = 0
                    else:
                        patience_cnt += 1

                    print "patience time:", patience_cnt

                    if patience_cnt > 5:
                        print("early stopping...")
                        saver.save(sess,
                                   "../Seq_ckpt/Wiki/ALL_seq2seq_Bi_" + str(Bidirection) + "_Att_" + str(
                                       Attention) + "_Emb_" + str(Embd_train) + "_hiddenunits_" + str(num_units))

                        break

                # if epo % epoch_print == 0:
                #     for t in range(10):
                #         print 'Question {}'.format(t)
                #         print " ".join(map(lambda i: vocab[i], list(target_shuffle[t]))).strip()
                #         print " ".join(map(lambda i: vocab[i], list(logit_id[t, :]))).strip()
                saver.save(sess, "../Seq_ckpt/Wiki/ALL_seq2seq_Bi_"+str(Bidirection) + "_Att_"+ str(Attention) + "_Emb_"+str(Embd_train)+"_hiddenunits_"+str(num_units))

    elif model_type == 'testing':
        Seq2Seq_model = Bi_Att_Seq2Seq(batch_size, vocab_size, num_units, embd, model_type, Bidirection, Embd_train, Attention)
        with tf.Session(config=config) as sess:
            saver_word_rw = tf.train.Saver()
            saver_word_rw.restore(sess, "/home/ye/PycharmProjects/Qrefine/Seq_ckpt/Wiki/ALL_seq2seq_Bi_" + str(Bidirection) + "_Att_" + str(
                                  Attention) + "_Emb_" + str(Embd_train)+"_hiddenunits_" + str(num_units))
            max_batches = len(test_noisy_Id) / batch_size

            idx = np.arange(0, len(test_noisy_Id))
            # idx = list(np.random.permutation(idx))

            test_Bleu_score1 = []
            test_Bleu_score2 = []
            test_Bleu_score3 = []
            test_Bleu_score4 = []
            test_Rouge_score = []
            test_Meteor_score = []
            test_Cider_score = []

            generated_test_sen = []
            test_answer_sen = []
            test_noisy_sen = []
            test_clean_sen = []
            for batch in range(max_batches):
                source_shuffle, source_len, train_shuffle, target_shuffle, target_len, answer_shuffle = next_batch_answer(
                    test_noisy_Id,
                    test_noisy_len,
                    test_input_Id,
                    test_target_Id,
                    test_clean_len,
                    test_answer_Id,
                    batch_size, batch,
                    idx)

                for an in answer_shuffle:
                    test_answer_sen.append(vocab[anum] for anum in an)
                for no in source_shuffle:
                    test_noisy_sen.append(vocab[si] for si in no)
                for cl in target_shuffle:
                    test_clean_sen.append(vocab[ci] for ci in cl)

                source_shuffle = tf.keras.preprocessing.sequence.pad_sequences(source_shuffle, maxlen=None,
                                                                               padding='post', value=EOS)

                fd = {Seq2Seq_model.encoder_inputs: source_shuffle, Seq2Seq_model.encoder_inputs_length: source_len, Seq2Seq_model.dropout_rate:1}
                ids = sess.run([Seq2Seq_model.ids], fd)
                for t in range(batch_size):
                    ref = []
                    hyp = []
                    for tar_id in target_shuffle[t]:
                        if tar_id != 2:
                            ref.append(vocab[tar_id])
                    for pre_id in ids[0][t]:
                        if pre_id != 2 and pre_id != 0:
                            hyp.append(vocab[pre_id])
                    generated_test_sen.append(hyp)
                    # sen_bleu1 = bleu([ref], hyp, weights=(1, 0, 0, 0))
                    # sen_bleu2 = bleu([ref], hyp, weights=(0.5, 0.5, 0, 0))
                    # sen_bleu3 = bleu([ref], hyp, weights=(0.333, 0.333, 0.333, 0))
                    # sen_bleu4 = bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25))
                    # Bleu_score1.append(sen_bleu1)
                    # Bleu_score2.append(sen_bleu2)
                    # Bleu_score3.append(sen_bleu3)
                    # Bleu_score4.append(sen_bleu4)
                    hyp_sen = " ".join(hyp)
                    ref_sen = " ".join(ref)
                    dic_hyp = {}
                    dic_hyp[0] = [hyp_sen]
                    dic_ref = {}
                    dic_ref[0] = [ref_sen]
                    sen_bleu, _ = Bleu_obj.compute_score(dic_ref, dic_hyp)
                    sen_rouge = Rouge_obj.compute_score(dic_ref, dic_hyp)
                    sen_meteor, _ = Meteor_obj.compute_score(dic_ref, dic_hyp)
                    sen_cider, _ = cIDER_obj.compute_score(dic_ref, dic_hyp)
                    test_Bleu_score1.append(sen_bleu[0])
                    test_Bleu_score2.append(sen_bleu[1])
                    test_Bleu_score3.append(sen_bleu[2])
                    test_Bleu_score4.append(sen_bleu[3])
                    test_Rouge_score.append(sen_rouge[0])
                    test_Meteor_score.append(sen_meteor)
                    test_Cider_score.append(sen_cider)

                # for t in xrange(5):
                #     print('Training Question {}'.format(t))
                #     print("NQ: " + " ".join(map(lambda i: vocab[i], list(source_shuffle[t]))).strip().replace(
                #         "<EOS>", "  "))
                #     print("CQ: " + " ".join(map(lambda i: vocab[i], list(target_shuffle[t]))).strip().replace(
                #         "<EOS>", "  "))
                #     print("GQ: " + " ".join(map(lambda i: vocab[i], list(ids[0][t]))).strip().replace("<EOS>",
                #                                                                                         "  "))

                bleu_score1 = sum(test_Bleu_score1) / float(len(test_Bleu_score1))
                bleu_score2 = sum(test_Bleu_score2) / float(len(test_Bleu_score2))
                bleu_score3 = sum(test_Bleu_score3) / float(len(test_Bleu_score3))
                bleu_score4 = sum(test_Bleu_score4) / float(len(test_Bleu_score4))

                print("\n Bleu_score1, Bleu_score2, Bleu_score3, Bleu_score4, Rouge, Meteor  Cider\n")
                print("   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f}".format(
                    bleu_score1, bleu_score2, bleu_score3, bleu_score4,
                    sum(test_Rouge_score) / float(len(test_Rouge_score)),
                    sum(test_Meteor_score) / float(len(test_Meteor_score)),
                    sum(test_Cider_score) / float(len(test_Cider_score))
                ))

        fname = "Wiki_result_Seq2Seq/Seq2Seq_Att_" + str(Attention)+"Bi_" + str(Bidirection) + "Wiki_bleu1_" + str(bleu_score1)
        f = open(fname, "wb")
        f.write("\n Bleu_score1, Bleu_score2, Bleu_score3, Bleu_score4, Rouge, Meteor  Cider\n")
        f.write("   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f} \n".format(
            bleu_score1, bleu_score2, bleu_score3, bleu_score4,
            sum(test_Rouge_score) / float(len(test_Rouge_score)),
            sum(test_Meteor_score) / float(len(test_Meteor_score)),
            sum(test_Cider_score) / float(len(test_Cider_score))
        ))
        for i in range(len(generated_test_sen)):
            f.write("question " + str(i) + "\n")
            f.write("answer: " + " ".join(test_answer_sen[i]) + "\n")
            f.write("noisy question: " + " ".join(test_noisy_sen[i]) + "\n")
            f.write("clean question: " + " ".join(test_clean_sen[i]) + "\n")
            f.write("generated question: " + " ".join(generated_test_sen[i]) + "\n")

        # output_path = "../data/Wiki/Wiki_Gen_Q10"
        # result={"answer_sen":test_answer_sen,"clean_sen":generated_test_sen}
        # output = open(output_path, 'wb')
        # pickle.dump(result, output)
        # output.close()

if __name__ == '__main__':
    main()
