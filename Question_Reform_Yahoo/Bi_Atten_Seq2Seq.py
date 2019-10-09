import numpy as np
import tensorflow as tf
import pickle
from nltk.translate import bleu
from tensorflow.contrib import keras
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple
import os

PAD = 0
SOS = 1
EOS = 2
batches_in_epoch = 200
epoch_print = 2

class Bi_Att_Seq2Seq(object):
    def __init__(self, batch_size, vocab_size, num_units, embd, model_cond, Bidirection, Embd_train, Attention
                 ):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.embedding = embd
        self.model_cond = model_cond
        self.Bidirection = Bidirection
        self.Embd_train = Embd_train
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

        with tf.variable_scope("embeddings"):
            self.embeddings = tf.get_variable(name="embedding_W", shape=self.embedding.shape,
                                              initializer=tf.constant_initializer(self.embedding), trainable= self.Embd_train)

            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)

            if self.model_cond == "training":
                self.decoder_outputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)

        self.inference()
        self.loss()
        # if self.model_cond == "training":
        #     tf.summary.scalar("loss", self.loss_seq2seq)

    def inference(self):
        ####Encoder
        with tf.variable_scope("encoder_model"):
            if self.Bidirection == False:
                encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units)
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
            # attention_states: [self.batch_size, max_time, self.num_units]
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

            decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper,
                                                      initial_state=self.initial_state,
                                                      output_layer=tf.layers.Dense(self.vocab_size))
            # Dynamic decoding
            if self.model_cond == "training":
                outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False,
                                                                            impute_finished=True,
                                                                            maximum_iterations=self.decoder_length[0])
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
                self.loss_seq2seq = tf.contrib.seq2seq.sequence_loss(logits=self.logits,
                                                                     targets=self.ground_truth, weights=tf.cast(
                        tf.sequence_mask(tf.fill([self.batch_size], tf.shape(self.logits)[1]),
                                         tf.shape(self.logits)[1]), tf.float32))
                self.loss_seq2seq = tf.reduce_mean(self.loss_seq2seq)
                self.train_op = tf.train.AdamOptimizer().minimize(self.loss_seq2seq)

def next_batch(noisy_Id, noisy_len, input_Id, clean_len, target_Id, batch_size, batch_num, idx):
    if (batch_num + 1) * batch_size > len(noisy_Id):
        batch_num = batch_num % (len(noisy_Id) / batch_size)
    idx_n = idx[batch_num * batch_size: (batch_num + 1) * batch_size]
    data_shuffle = [noisy_Id[i] for i in idx_n]
    data_len = [noisy_len[i] for i in idx_n]
    train_shuffle = [input_Id[i] for i in idx_n]
    target_len = [clean_len[i] for i in idx_n]
    target_shuffle = [target_Id[i] for i in idx_n]
    return data_shuffle, data_len, train_shuffle, target_shuffle, target_len

def run_graph(data_comb, batch_size, num_units, model_cond, epoch, Bidirection, Embd_train, Attention):
    noisy_Id = data_comb['noisy_Id']
    noisy_len = data_comb['noisy_len']
    ground_truth = data_comb['ground_truth']
    clean_len = data_comb['clean_len']
    train_Id=data_comb["train_Id"]
    vocab = data_comb['vocab']
    embd = data_comb['embd']
    vocab_size = len(vocab)
    model_type=model_cond

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if model_type == 'training':
        Seq2Seq_model = Bi_Att_Seq2Seq(batch_size, vocab_size, num_units, embd, model_type, Bidirection, Embd_train, Attention)
        max_batches = len(noisy_Id) / batch_size

        for epo in range(epoch):
            print('Epoch {}'.format(epo))
            with tf.Session(config=config) as sess:
                if os.path.isfile("../Seq_ckpt/pretrain-seq2seq_Bi_" + str(Bidirection) + "_Att_" + str(
                        Attention) + "_Emb_" + str(Embd_train)):
                    saver = tf.train.Saver()
                saver.restore(sess, "../Seq_ckpt/pretrain-seq2seq_Bi_" + str(Bidirection) + "_Att_" + str(
                    Attention) + "_Emb_" + str(Embd_train))
                sess.run(tf.global_variables_initializer())
                # saver.restore(sess, "../Seq_ckpt/pretrain-model")
                idx = np.arange(0, len(noisy_Id))
                idx = list(np.random.permutation(idx))

                for batch in range(max_batches):
                    source_shuffle, source_len, train_shuffle, target_shuffle, target_len = next_batch(noisy_Id,
                                                                                                       noisy_len,
                                                                                                       train_Id,
                                                                                                       clean_len,
                                                                                                       ground_truth,
                                                                                                       batch_size,
                                                                                                       batch,
                                                                                                       idx)
                    source_shuffle = tf.keras.preprocessing.sequence.pad_sequences(source_shuffle, maxlen=None,
                                                                                   padding='post', value=EOS)
                    train_shuffle = tf.keras.preprocessing.sequence.pad_sequences(train_shuffle, maxlen=None,
                                                                                  padding='post', value=EOS)
                    target_shuffle = tf.keras.preprocessing.sequence.pad_sequences(target_shuffle, maxlen=None,
                                                                                   padding='post', value=EOS)

                    target_len = np.tile(max(target_len) + 1, batch_size)

                    fd = {Seq2Seq_model.encoder_inputs: source_shuffle, Seq2Seq_model.encoder_inputs_length: source_len,
                          Seq2Seq_model.decoder_inputs: train_shuffle, Seq2Seq_model.decoder_length: target_len,
                          Seq2Seq_model.ground_truth: target_shuffle}

                    if model_type == "training":
                        logit_id, l_fun, pro_op = sess.run(
                            [Seq2Seq_model.ids, Seq2Seq_model.loss_seq2seq, Seq2Seq_model.train_op], fd)
                    elif model_type == "validation":
                        logit_id, l_fun = sess.run( [Seq2Seq_model.ids, Seq2Seq_model.loss_seq2seq], fd)
                    # summary_writer.add_summary(summary, batch)

                    if batch == 0 or batch % batches_in_epoch == 0:
                        ##print the training
                        print('batch {}'.format(batch))
                        print('  minibatch loss: {}'.format(l_fun))
                        for t in range(5):
                            print 'Question {}'.format(t)
                            print " ".join(map(lambda i: vocab[i], list(target_shuffle[t]))).strip()
                            print " ".join(map(lambda i: vocab[i], list(logit_id[t, :]))).strip()
                    # if batch == max_batches -1:
                    #     print("validation dataset\n")
                    #     print('  minibatch loss: {}'.format(logit_id, l_fun = sess.run([Seq2Seq_model.ids, Seq2Seq_model.loss_seq2seq], fd)))

                saver.save(sess, "../Seq_ckpt/pretrain-seq2seq_Bi_"+str(Bidirection) + "_Att_"+ str(Attention) + "_Emb_"+str(Embd_train))

    elif model_type == 'testing':
        print "starting Seq2Seq testing"
        Seq2Seq_model = Bi_Att_Seq2Seq(batch_size, vocab_size, num_units, embd, model_type, Bidirection, Embd_train, Attention)
        with tf.Session(config=config) as sess:
            saver_word_rw = tf.train.Saver()
            saver_word_rw.restore(sess, "../Seq_ckpt/pretrain-seq2seq_Bi_"+str(Bidirection) + "_Att_"+ str(Attention) + "_Emb_"+str(Embd_train))
            max_batches = len(noisy_Id) / batch_size

            idx = np.arange(0, len(noisy_Id))
            idx = list(np.random.permutation(idx))
            Bleu_score1 = []
            Bleu_score2 = []
            ex_match = []
            for batch in range(max_batches):
                source_shuffle, source_len, train_shuffle, target_shuffle, target_len = next_batch(noisy_Id,
                                                                                                   noisy_len,
                                                                                                   train_Id,
                                                                                                   clean_len,
                                                                                                   ground_truth,
                                                                                                   batch_size,
                                                                                                   batch,
                                                                                                   idx)
                source_shuffle = tf.keras.preprocessing.sequence.pad_sequences(source_shuffle, maxlen=None,
                                                                               padding='post', value=EOS)

                fd = {Seq2Seq_model.encoder_inputs: source_shuffle, Seq2Seq_model.encoder_inputs_length: source_len}
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
                    sen_bleu1 = bleu([ref], hyp, weights=(1, 0, 0, 0))
                    sen_bleu2 = bleu([ref], hyp, weights=(0.5, 0.5, 0, 0))
                    Bleu_score1.append(sen_bleu1)
                    Bleu_score2.append(sen_bleu2)
                    if ref == hyp:
                        ex_match.append(1)
                    else:
                        ex_match.append(0)

                print "the current whole bleu_score1 is:", sum(Bleu_score1) / float(len(Bleu_score1))
                
                print "the current whole bleu_score2 is:", sum(Bleu_score2) / float(len(Bleu_score2))
                print "the whole accuracy is:", sum(ex_match) / float(len(ex_match))
