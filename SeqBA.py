# -*- coding: utf-8 -*
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
from Data_loading import load_data
import gc

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from Evaluation_Package.rouge.rouge import Rouge
from Evaluation_Package.meteor.meteor import Meteor
from Evaluation_Package.bleu.bleu import Bleu
from Evaluation_Package.cider.cider import Cider


class Seq2Seq(object):
    def __init__(self, batch_size, vocab_size, num_units, embd, model_cond, Bidirection, Embd_train, char_hidden_units,
                 Attention, char_num, char_dim
                 ):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.embedding = embd
        self.model_cond = model_cond
        self.Bidirection = Bidirection
        self.embd_train = Embd_train
        self.Attention = Attention
        self.char_hidden_units = char_hidden_units
        self.char_num = char_num
        self.char_dim = char_dim
        self.build_graph()

    def build_graph(self):
        # input placehodlers
        with tf.variable_scope("model_inputs"):
            self.encoder_inputs = tf.placeholder(shape=(self.batch_size, None),
                                                 dtype=tf.int32, name='encoder_inputs')
            self.encoder_inputs_length = tf.placeholder(shape=(self.batch_size,), dtype=tf.int32,
                                                        name='encoder_inputs_length')
            # self.encoder_char_ids = tf.placeholder(shape=(self.batch_size, None, None), dtype=tf.int32,
            #                                        name="encoder_char_ids")
            # self.encoder_char_len = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32,
            #                                        name="encoder_char_length")  ##batch * word num * char num
            self.encoder_emb = tf.placeholder(shape=(self.batch_size, None, 768), dtype=tf.float32,
                                                   name="encoder_char_length")  ##batch * word num * char num
            # self.target_emb = tf.placeholder(shape=(self.batch_size, None, 768), dtype=tf.float32,
            #                                        name="encoder_char_length")  ##batch * word num * char num

            if self.model_cond == "training":
                self.decoder_inputs = tf.placeholder(shape=(self.batch_size, None),
                                                     dtype=tf.int32, name='decoder_targets')

                self.decoder_length = tf.placeholder(shape=(self.batch_size,), dtype=tf.int32, name='decoder_length')
                self.ground_truth = tf.placeholder(shape=(self.batch_size, None),
                                                   dtype=tf.int32, name='ground_truth')

            self.dropout_rate = tf.placeholder(dtype=tf.float32, name="dropout_rate")

        with tf.variable_scope("embeddings"):
            self.embeddings = tf.get_variable(name="embedding_W", shape=self.embedding.shape,
                                              initializer=tf.constant_initializer(self.embedding),
                                              trainable=self.embd_train)
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)

            # _char_embeddings = tf.get_variable(name="_char_embeddings", dtype=tf.float32,
            #                                    shape=[self.char_num, self.char_dim])  ## char num 44, char dim? 50
            #
            # self.char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.encoder_char_ids, name="char_embeddings")

            if self.model_cond == "training":
                self.decoder_outputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)
                # self.decoder_embd = tf.concat([self.decoder_outputs_embedded, self.target_emb], axis=-1)

        self.inference()
        self.loss()

    def inference(self):
        ####Encoder
        with tf.variable_scope("encoder_model"):
            # put the time dimension on axis=1
            # s = tf.shape(self.char_embeddings) #batch * word * char * embedding
            # char_embeddings = tf.reshape(self.char_embeddings,
            #                              shape=[s[0] * s[1], s[-2], self.char_dim])
            # word_lengths = tf.reshape(self.encoder_char_len, shape=[s[0] * s[1]])
            #
            # # bi lstm on chars
            # cell_fw = tf.contrib.rnn.LSTMCell(self.char_hidden_units,
            #                                   state_is_tuple=True)
            # cell_bw = tf.contrib.rnn.LSTMCell(self.char_hidden_units,
            #                                   state_is_tuple=True)
            # _output = tf.nn.bidirectional_dynamic_rnn(
            #     cell_fw, cell_bw, char_embeddings,
            #     sequence_length = word_lengths, dtype=tf.float32)
            #
            # # read and concat output
            # _, ((_, output_fw), (_, output_bw)) = _output
            # output = tf.concat([output_fw, output_bw], axis=-1)
            #
            # # shape = (batch size, max sentence length, char hidden size)
            # output = tf.reshape(output,
            #                     shape=[s[0], s[1], 2 * self.char_hidden_units])
            # self.word_embeddings_cat = tf.concat([self.encoder_inputs_embedded, output], axis=-1)

            ## the whole emb of encoder
            self.encoder_emb_whole = tf.concat([self.encoder_inputs_embedded, self.encoder_emb], axis = -1)

            if self.Bidirection == False:
                encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units)
                encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell, input_keep_prob=self.dropout_rate,
                                                             output_keep_prob=self.dropout_rate,
                                                             state_keep_prob=self.dropout_rate)
                encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                                              inputs=self.encoder_emb_whole,
                                                                              sequence_length=self.encoder_inputs_length,
                                                                              time_major=False,
                                                                              dtype=tf.float32)
                self.hidden_units = self.num_units

            elif self.Bidirection == True:
                encoder_cell_fw = LSTMCell(self.num_units)
                encoder_cell_bw = LSTMCell(self.num_units)
                ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)) = (
                    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_fw, cell_bw=encoder_cell_bw,
                                                    inputs=self.encoder_emb_whole,
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

            decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell, input_keep_prob=self.dropout_rate,
                                                         output_keep_prob=self.dropout_rate,
                                                         state_keep_prob=self.dropout_rate)
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
                                                                  start_tokens=tf.fill([self.batch_size], 0),
                                                                  end_token = 1)

            decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper,
                                                      initial_state=self.initial_state,
                                                      output_layer=tf.layers.Dense(self.vocab_size))
            # Dynamic decoding
            if self.model_cond == "training":
                outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False,
                                                                            impute_finished=True,
                                                                            maximum_iterations=tf.reduce_max(
                                                                                self.decoder_length, axis=0))
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
                weights = ["encoder_model/rnn/basic_lstm_cell/kernel:0",
                           'decoder_model/decoder/basic_lstm_cell/kernel:0', 'decoder_model/decoder/dense/kernel:0']
                # self.regularizer = 0.01 * tf.nn.l2_loss(weights)
                padding = 2 * tf.ones(
                    [tf.shape(self.logits)[0], tf.shape(self.ground_truth)[1] - tf.shape(self.logits)[1],
                     tf.shape(self.logits)[2]])
                self.logits_padded = tf.concat([self.logits, padding], 1)
                # self.mask = tf.sequence_mask(tf.fill([self.batch_size], tf.shape(self.logits)[1]),tf.shape(self.logits)[1])
                self.mask = tf.sequence_mask(self.decoder_length, tf.shape(self.ground_truth)[1])

                self.loss_seq2seq = tf.contrib.seq2seq.sequence_loss(logits=self.logits_padded,
                                                                     targets=self.ground_truth,
                                                                     weights=tf.cast(self.mask, tf.float32))
                self.loss_seq2seq = tf.reduce_mean(self.loss_seq2seq)
                self.train_op = tf.train.AdamOptimizer().minimize(self.loss_seq2seq)


def next_batch(noisy_Id, noisy_len, n_emd, input_Id,  target_Id, clean_len, batch_size,
               batch_num, idx):
    if (batch_num + 1) * batch_size > len(noisy_Id):
        batch_num = batch_num % (len(noisy_Id) / batch_size)
    idx_n = idx[batch_num * batch_size: (batch_num + 1) * batch_size]

    data_shuffle = [noisy_Id[i] for i in idx_n]
    data_len = [noisy_len[i] for i in idx_n]
    data_nemd = [n_emd[i] for i in idx_n]
    train_shuffle = [input_Id[i] for i in idx_n]
    target_len = [clean_len[i] for i in idx_n]
    target_shuffle = [target_Id[i] for i in idx_n]
    return data_shuffle, data_len, data_nemd, train_shuffle, target_shuffle,  target_len

def next_batch_answer(noisy_Id, noisy_len, input_Id, ground_truth, clean_len, answer_Id, batch_size, batch_num,
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


def seq_run_BA(model_type,config):

    SOS = 0
    EOS = 1
    PAD = 2
    UNK = 3
    data_file = config.data_dir
    emd_file = [config.nemd_dir, config.cemd_dir]
    train_data, test_data, eval_data, vocab, embd= load_data(data_file, emd_file, "BA")
    batch_size = config.batch_size
    num_units = config.num_units
    Bidirection = config.Bidirection
    Attention = config.Attention
    Embd_train = config.Embd_train
    char_hidden_units = config.char_hidden_units
    char_num = config.char_num
    char_dim = config.char_dim
    batches_in_epoch = config.batches_in_epoch
    epoch = config.epoch
    S2S_ckp_dir = config.S2S_ckp_dir

    train_noisy_Id, train_noisy_len, train_noisy_char_Id, train_noisy_char_len, train_nemd,train_target_Id, train_input_Id, train_clean_Id, train_clean_len, train_answer_Id, train_answer_len, max_char, max_word = train_data
    test_noisy_Id, test_noisy_len, test_noisy_char_Id, test_noisy_char_len, test_nemd, test_target_Id, test_input_Id, test_clean_Id, test_clean_len, test_answer_Id, test_answer_len = test_data
    eval_noisy_Id, eval_noisy_len, eval_noisy_char_Id, eval_noisy_char_len, eval_nemd, eval_target_Id, eval_input_Id, eval_clean_Id, eval_clean_len, eval_answer_Id, eval_answer_len = eval_data

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True

    print "the number of training question is:", len(train_noisy_Id), len(train_noisy_char_Id), len(
        train_noisy_char_len)

    print "the number of eval question is:", len(eval_noisy_Id), len(eval_noisy_char_Id), len(
        eval_noisy_char_len)

    vocab_size = len(vocab)

    Bleu_obj = Bleu()
    Rouge_obj = Rouge()
    Meteor_obj = Meteor()
    cIDER_obj = Cider()

    if model_type == 'training':
        Seq2Seq_model = Seq2Seq(batch_size, vocab_size, num_units, embd, model_type, Bidirection, Embd_train,
                                     char_hidden_units, Attention, char_num, char_dim)

        # print "the number of evaluate question is:", len(eval_noisy_Id)

        patience_cnt = 0

        # summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        # merge = tf.summary.merge_all()
        saver = tf.train.Saver(sharded=False)

        f = open(config.perf_path,"w")

        # config_gpu = tf.ConfigProto()
        # config_gpu.gpu_options.allow_growth = True
        # config=config_gpu
        with tf.Session() as sess:
            if os.path.isfile(S2S_ckp_dir):
                saver.restore(sess, S2S_ckp_dir)
            else:
                sess.run(tf.global_variables_initializer())
            # sess.run(tf.global_variables_initializer())
            # saver.restore(sess, config.S2S_ckp_dir)

            val_loss_epo = []
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
                    source_shuffle, source_len, source_nemd, train_shuffle, target_shuffle, target_len = next_batch(
                        train_noisy_Id,
                        train_noisy_len,
                        train_nemd,
                        train_input_Id,
                        train_target_Id,
                        train_clean_len,
                        batch_size,
                        batch,
                        idx)

                    source_shuffle = tf.keras.preprocessing.sequence.pad_sequences(source_shuffle, maxlen = max_word,
                                                                                   padding='post', value=EOS)
                    source_emd = []
                    for n in source_nemd:
                        source_emd.append(n[0:source_shuffle.shape[1]])
                    source_emd = np.asarray(source_emd)


                    train_shuffle = tf.keras.preprocessing.sequence.pad_sequences(train_shuffle, maxlen= max_word+1,
                                                                                  padding='post', value=EOS)

                    # target_emd = []
                    # for c in train_cemd:
                    #     target_emd.append(c[0:train_shuffle.shape[1]])
                    # target_emd = np.asarray(target_emd)

                    target_shuffle = tf.keras.preprocessing.sequence.pad_sequences(target_shuffle, maxlen = max_word+1,
                                                                                   padding='post', value=EOS)

                    # whole_noisy_char_Id =[]
                    # for sen_char in char_Id:
                    #     sen_len = len(sen_char)
                    #     for i in range(len(source_shuffle[0]) - sen_len):
                    #         sen_char.append([0] * max_char)  ## fix the char with the length of max_word
                    #     whole_noisy_char_Id.append(sen_char)
                    #
                    # whole_noisy_char_Id = np.asarray(whole_noisy_char_Id)
                    # print whole_noisy_char_Id.shape
                    #
                    # print len(char_len[0])

                    # target_len = np.tile(max(target_len) + 1, batch_size)


                    fd = {Seq2Seq_model.encoder_inputs: source_shuffle, Seq2Seq_model.encoder_inputs_length: source_len,
                          Seq2Seq_model.encoder_emb: source_emd,
                          Seq2Seq_model.decoder_inputs: train_shuffle, Seq2Seq_model.decoder_length: target_len,
                          # Seq2Seq_model.target_emb: target_emd,
                          Seq2Seq_model.ground_truth: target_shuffle, Seq2Seq_model.dropout_rate: 1}

                    mask, logit_id, logit_id_pad, l_fun, pro_op = sess.run(
                        [Seq2Seq_model.mask, Seq2Seq_model.ids, Seq2Seq_model.logits_padded, Seq2Seq_model.loss_seq2seq,
                         Seq2Seq_model.train_op], fd)

                    # mask, logit_id, logit_id_pad= sess.run(
                    #     [Seq2Seq_model.mask, Seq2Seq_model.ids, Seq2Seq_model.logits_padded], fd)
                    # l_fun =0

                    for t in range(batch_size):
                        ref = []
                        hyp = []
                        for tar_id in target_shuffle[t]:
                            if tar_id != EOS and tar_id != PAD:
                                ref.append(vocab[tar_id])
                                # print vocab[tar_id]
                        for pre_id in logit_id[t]:
                            if pre_id != EOS and pre_id != PAD:
                                hyp.append(vocab[pre_id])

                        hyp_sen = u" ".join(hyp).encode('utf-8')
                        ref_sen = u" ".join(ref).encode('utf-8')
                        dic_hyp = {}
                        dic_hyp[0] = [hyp_sen]
                        dic_ref = {}
                        dic_ref[0] = [ref_sen]
                        sen_bleu, _ = Bleu_obj.compute_score(dic_ref, dic_hyp)
                        sen_rouge = Rouge_obj.compute_score(dic_ref, dic_hyp)
                        sen_meteor, _ = Meteor_obj.compute_score(dic_ref, dic_hyp)
                        sen_cider, _ = cIDER_obj.compute_score(dic_ref, dic_hyp)
                        Bleu_score1.append(sen_bleu[0])
                        Bleu_score2.append(sen_bleu[1])
                        Bleu_score3.append(sen_bleu[2])
                        Bleu_score4.append(sen_bleu[3])
                        Rouge_score.append(sen_rouge[0])
                        Meteor_score.append(sen_meteor)
                        Cider_score.append(sen_cider)

                    if batch == 0 or batch % batches_in_epoch == 0:
                        ##print the training
                        print('batch {}'.format(batch))
                        print('   minibatch loss: {}'.format(l_fun))
                        print(" the loss_epo:", epo)

                        for t in xrange(3):
                            print 'Training Question {}'.format(t)
                            print "NQ: " + " ".join(map(lambda i: vocab[i], list(source_shuffle[t]))).strip().replace(
                                "<EOS>", " ")
                            print "CQ: " + " ".join(map(lambda i: vocab[i], list(target_shuffle[t]))).strip().replace(
                                "<EOS>", " ")
                            print "GQ: " + " ".join(map(lambda i: vocab[i], list(logit_id[t]))).strip().replace("<EOS>",
                                                                                                                " ")
                        print("\n Bleu_score1, Bleu_score2, Bleu_score3, Bleu_score4, Rouge,  Cider\n")
                        print("   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f}".format(
                            sum(Bleu_score1) / float(len(Bleu_score1)),
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

                        for batch_val in range(len(eval_noisy_Id) / batch_size):
                            idx_v = np.arange(0, len(eval_noisy_Id))
                            idx_v = list(np.random.permutation(idx_v))

                            val_source_shuffle, val_source_len, val_nemd, val_train_shuffle, val_target_shuffle, val_target_len = next_batch(
                                eval_noisy_Id, eval_noisy_len,eval_nemd, eval_input_Id, eval_target_Id, eval_clean_len,
                                batch_size, batch_val, idx_v)
                            val_source_shuffle = tf.keras.preprocessing.sequence.pad_sequences(val_source_shuffle,
                                                                                               maxlen= max_word,
                                                                                               padding='post',
                                                                                               value=EOS)
                            val_target_shuffle = tf.keras.preprocessing.sequence.pad_sequences(val_target_shuffle,
                                                                                               maxlen= max_word +1,
                                                                                               padding='post',
                                                                                               value= EOS)
                            val_train_shuffle = tf.keras.preprocessing.sequence.pad_sequences(val_train_shuffle,
                                                                                              maxlen= max_word +1,
                                                                                              padding='post', value=EOS)
                            val_source_emd = []
                            for n in val_nemd:
                                val_source_emd.append(n[0:val_source_shuffle.shape[1]])
                            val_source_emd = np.asarray(val_source_emd)

                            # val_target_emd = []
                            # for c in val_train_cemd:
                            #     val_target_emd.append(c[0:val_train_shuffle.shape[1]])
                            # val_target_emd = np.asarray(val_target_emd)

                            # val_batch_noisy_char_Id = []
                            # for sen_char in val_char_Id:
                            #     sen_len = len(sen_char)
                            #     for i in range(len(val_source_shuffle[0]) - sen_len):
                            #         sen_char.append([0] * max_char) ## fix the char with the length of max_word
                            #     val_batch_noisy_char_Id.append(sen_char)
                            #
                            # val_batch_noisy_char_Id = np.asarray(val_batch_noisy_char_Id)

                            # val_target_len = np.tile(max(val_target_len) + 1, batch_size)

                            fd_val = {Seq2Seq_model.encoder_inputs: val_source_shuffle, Seq2Seq_model.encoder_inputs_length: val_source_len,
                                      Seq2Seq_model.encoder_emb: val_source_emd,
                                      # Seq2Seq_model.target_emb: val_target_emd,
                                      Seq2Seq_model.decoder_length: val_target_len,
                                      Seq2Seq_model.decoder_inputs: val_train_shuffle, Seq2Seq_model.ground_truth: val_target_shuffle,
                                      Seq2Seq_model.dropout_rate: 1}
                            val_ids, val_loss_seq = sess.run([Seq2Seq_model.ids, Seq2Seq_model.loss_seq2seq], fd_val)
                            val_loss.append(val_loss_seq)

                            for t in range(batch_size):
                                ref = []
                                hyp = []
                                for tar_id in val_target_shuffle[t]:
                                    if tar_id != EOS and tar_id != PAD:
                                        ref.append(vocab[tar_id])
                                for pre_id in val_ids[t]:
                                    if pre_id != EOS and pre_id != PAD:
                                        hyp.append(vocab[pre_id])
                                        # print vocab[pre_id]
                                # sen_bleu1 = bleu([ref], hyp, weights=(1, 0, 0, 0))
                                # sen_bleu2 = bleu([ref], hyp, weights=(0.5, 0.5, 0, 0))
                                # val_Bleu_score1.append(sen_bleu1)
                                # val_Bleu_score2.append(sen_bleu2)
                                hyp_sen = u" ".join(hyp).encode('utf-8')
                                ref_sen = u" ".join(ref).encode('utf-8')
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

                        for t in xrange(3):
                            print 'Validation Question {}'.format(t)
                            print "NQ: " + " ".join(
                                map(lambda i: vocab[i], list(val_source_shuffle[t]))).strip().replace("<EOS>", " ")
                            print "CQ: " + " ".join(
                                map(lambda i: vocab[i], list(val_target_shuffle[t]))).strip().replace("<EOS>", " ")
                            print "GQ: " + " ".join(map(lambda i: vocab[i], list(val_ids[t]))).strip().replace("<EOS>", " ")

                        avg_val_loss = sum(val_loss) / float(len(val_loss))

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
                    for loss in val_loss_epo:
                        print "the val_loss_epo:", loss
                    print "the loss difference:", val_loss_epo[-2] - val_loss_epo[-1]

                    if val_loss_epo[-2] - val_loss_epo[-1]:
                        patience_cnt = 0
                    else:
                        patience_cnt += 1

                    print patience_cnt

                    if patience_cnt > 5:
                        print("early stopping...")
                        saver.save(sess, S2S_ckp_dir)

                        break

                # if epo % epoch_print == 0:
                #     for t in range(10):
                #         print 'Question {}'.format(t)
                #         print " ".join(map(lambda i: vocab[i], list(target_shuffle[t]))).strip()
                #         print " ".join(map(lambda i: vocab[i], list(logit_id[t, :]))).strip()
                avg_bleu_val = sum(val_Bleu_score1) / float(len(val_Bleu_score1))
                if epo % 5 == 0:
                    saver.save(sess, S2S_ckp_dir)

                f.write("the performance: " + S2S_ckp_dir + "_epoch_" + str(epo) + "avg_eval_bleu_" + str(avg_bleu_val))
            f.close()
            print "save file"



    elif model_type == 'testing':

        Seq2Seq_model = Seq2Seq(batch_size, vocab_size, num_units, embd, model_type, Bidirection, Embd_train,
                                     char_hidden_units, Attention, char_num, char_dim)

        with tf.Session() as sess:
            saver_word_rw = tf.train.Saver()
            saver_word_rw.restore(sess, S2S_ckp_dir)

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

                test_source_shuffle, test_source_len, test_source_nemd, test_train_shuffle, test_target_shuffle, test_target_len = next_batch(
                    test_noisy_Id,
                    test_noisy_len,
                    test_nemd,
                    test_input_Id,
                    test_target_Id,
                    test_clean_len,
                    batch_size,
                    batch, idx)

                for no in test_source_shuffle:
                    test_noisy_sen.append(vocab[si] for si in no)
                for cl in test_target_shuffle:
                    test_clean_sen.append(vocab[ci] for ci in cl)

                test_source_shuffle = tf.keras.preprocessing.sequence.pad_sequences(test_source_shuffle, maxlen = max_word,
                                                                               padding='post', value=EOS)

                test_source_emd = []
                for n in test_source_nemd:
                    test_source_emd.append(n[0:test_source_shuffle.shape[1]])
                test_source_emd = np.asarray(test_source_emd)


                # test_batch_noisy_char_Id = []
                # for sen_char in test_char_Id:
                #     sen_len = len(sen_char)
                #     for i in range(len(test_source_shuffle[0]) - sen_len):
                #         sen_char.append([0] * max_char)  ## fix the char with the length of max_word
                #     test_batch_noisy_char_Id.append(sen_char)
                # test_batch_noisy_char_Id = np.asarray(test_batch_noisy_char_Id)

                fd = {Seq2Seq_model.encoder_inputs: test_source_shuffle, Seq2Seq_model.encoder_inputs_length: test_source_len, Seq2Seq_model.encoder_emb: test_source_emd,
                      Seq2Seq_model.dropout_rate: 1}
                ids = sess.run([Seq2Seq_model.ids], fd)
                for t in range(batch_size):
                    ref = []
                    hyp = []
                    for tar_id in test_target_shuffle[t]:
                        if tar_id != 2:
                            ref.append(vocab[tar_id])
                    for pre_id in ids[0][t]:
                        if pre_id != 2 and pre_id != 0:
                            hyp.append(vocab[pre_id])
                    generated_test_sen.append(hyp)

                    hyp_sen = u" ".join(hyp).encode('utf-8')
                    ref_sen = u" ".join(ref).encode('utf-8')
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

                for t in xrange(5):
                    print 'Training Question {}'.format(t)
                    print "NQ: " + " ".join(map(lambda i: vocab[i], list(test_source_shuffle[t]))).strip().replace(
                        "<EOS>", "  ")
                    print "CQ: " + " ".join(map(lambda i: vocab[i], list(test_target_shuffle[t]))).strip().replace(
                        "<EOS>", "  ")
                    print "GQ: " + " ".join(map(lambda i: vocab[i], list(ids[0][t]))).strip().replace("<EOS>",
                                                                                                      "  ")

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

            #  len(test_answer_sen)
            fname = S2S_ckp_dir + "_bleu1_" + str(bleu_score1)
            f = open(fname, "wb")
            print "the length test set is:", len(generated_test_sen), len(
                test_noisy_sen), len(test_clean_sen)
            f.write("\n Bleu_score1, Bleu_score2, Bleu_score3, Bleu_score4, Rouge, Meteor  Cider\n")
            f.write("   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f} \n".format(
                bleu_score1, bleu_score2, bleu_score3, bleu_score4,
                sum(test_Rouge_score) / float(len(test_Rouge_score)),
                sum(test_Meteor_score) / float(len(test_Meteor_score)),
                sum(test_Cider_score) / float(len(test_Cider_score))
            ))
            for i in range(len(generated_test_sen)):
                f.write("question" + str(i) + "\n")
                # f.write("answer: " + " ".join(test_answer_sen[i]) + "\n")
                f.write("noisy question: " + " ".join(test_noisy_sen[i]) + "\n")
                f.write("clean question: " + " ".join(test_clean_sen[i]) + "\n")
                f.write("generated question: " + " ".join(generated_test_sen[i]) + "\n")

            f.close()

            # for t in xrange(10):
            #     print 'Question {}'.format(t)
            #     print " ".join(map(lambda i: vocab[i], list(source_shuffle[t]))).strip()
            #     print " ".join(map(lambda i: vocab[i], list(target_shuffle[t]))).strip()
            #     print " ".join(map(lambda i: vocab[i], list(ids[0][t, :]))).strip()

            # BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0.5, 0.5))
            # print BLEUscore


# if __name__ == '__main__':
#     main()
