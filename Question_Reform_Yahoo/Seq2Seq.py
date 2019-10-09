import numpy as np
import tensorflow as tf
import pickle
from nltk.translate import bleu
from tensorflow.contrib import keras
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple

embedding_trainable=True


class Seq2Seq(object):
    def __init__(self, batch_size, vocab_size, num_units, embd, model_cond):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.embedding = embd
        self.model_cond = model_cond
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
                                              initializer=tf.constant_initializer(self.embedding), trainable=True)
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)

            if self.model_cond == "training":
                self.decoder_outputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)

        self.inference()
        self.loss()
        if self.model_cond == "training":
            tf.summary.scalar("loss", self.loss_seq2seq)

    def inference(self):
        ####Encoder
        with tf.variable_scope("encoder_model"):
            encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units)
            encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                                    inputs=self.encoder_inputs_embedded,
                                                                    sequence_length=self.encoder_inputs_length,
                                                                    time_major=False,
                                                                    dtype=tf.float32)
        ###Decoder
        with tf.variable_scope("decoder_model"):
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units)
            if self.model_cond == 'training':
                helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_outputs_embedded, self.decoder_length,
                                                             time_major=False)
            elif self.model_cond == 'testing':
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embeddings,
                                                                   start_tokens=tf.fill([self.batch_size], 1),
                                                                   end_token=2)

            decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper,
                                                      initial_state=self.encoder_state,
                                                      output_layer=tf.layers.Dense(self.vocab_size))
            # Dynamic decoding
            if self.model_cond == "training":
                outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False,
                                                                        impute_finished=True,
                                                                        maximum_iterations=self.decoder_length[0])
            elif self.model_cond == "testing":
                outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False,
                                                                            impute_finished=True,
                                                                            maximum_iterations=2 * tf.reduce_max(self.encoder_inputs_length,axis=0))
            self.logits = tf.identity(outputs.rnn_output)
            # padding = 2 * tf.ones(
            #     [tf.shape(self.logits)[0], self.max_target_sequence_length - tf.shape(self.logits)[1],
            #      tf.shape(self.logits)[2]])
            # self.logits_padded = tf.concat([self.logits, padding], 1)
            self.logits_id = tf.argmax(self.logits, axis=2)
            self.softmax_logits = tf.nn.softmax(self.logits, dim=2)
            self.ids = tf.identity(outputs.sample_id)
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

            # self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            #     labels = self.ground_truth, logits=self.logits_padded)
            # self.loss_seq2seq = (tf.reduce_sum(self.crossent) / self.batch_size)
            # self.train_op = tf.train.AdamOptimizer().minimize(self.loss_seq2seq)


# class Seq2Seq_Test(object):





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


def main():
    SOS = 0
    EOS = 1
    PAD = 2
    UNK = 3
    epoch = 100
    batches_in_epoch = 100
    batch_size = 32
    num_units = 200
    char_dim = 50  ## the size of char dimention
    char_hidden_units = 100  ##the length of the char_lstm (word embedding:2 * char_hidden)
    epoch_print = 2
    Bidirection = False
    Attention = False
    Embd_train = False
    model_type = 'training'

    Dataset_name = "HUAWEI"
    Dataset_name = "YAHOO"

    logs_path = "/mnt/WDRed4T/ye/Qrefine/ckpt/" + Dataset_name + "/seq2seq_BIA_board"
    # FileName = "/mnt/WDRed4T/ye/DataR/" + Dataset_name + "/wrongword_Id_1"
    FileName = '/mnt/WDRed4T/ye/DataR/YAHOO/wrongword1_final'
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

    max_char = data_com['max_char']
    char_num = 44
    max_word = data_com['max_word']

    output = '/mnt/WDRed4T/ye/DataR/YAHOO/wrongword1_vocab_embd'
    input = open(output, 'rb')
    vocab_embd = pickle.load(input)
    vocab = vocab_embd['vocab']
    embd = vocab_embd['embd']
    embd = np.asarray(embd)
    vocdic = zip(vocab, range(len(vocab)))
    index_voc = dict((index, vocab) for vocab, index in vocdic)
    voc_index = dict((vocab, index) for vocab, index in vocdic)  ##dic[char]= index

    SOS = voc_index[u"<SOS>"]
    EOS = voc_index[u"<EOS>"]
    PAD = voc_index[u"<PAD>"]
    vocab_size = len(vocab)

    model_type = 'testing'

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    if model_type == 'training':
        Seq2Seq_model = Seq2Seq(batch_size, vocab_size, num_units, embd, model_type)

        print "the number of training question is:", len(train_noisy_Id)
        # print "the number of evaluate question is:", len(eval_noisy_Id)

        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        saver = tf.train.Saver(sharded=False)
        merge = tf.summary.merge_all()
        max_batches = len(train_noisy_Id) / batch_size
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            # saver.restore(sess, "../Seq_ckpt/pretrain-model")
            for epo in range(epoch):
                idx = np.arange(0, len(train_noisy_Id))
                idx = list(np.random.permutation(idx))
                # idx_e = np.arange(0, len(eval_noisy_Id))
                # idx_e = list(np.random.permutation(idx_e))

                print "Epoch {}".format(epo)
                for batch in range(max_batches):
                    source_shuffle, source_len, train_shuffle, target_shuffle, target_len = next_batch(train_noisy_Id,
                                                                                                       train_noisy_len,
                                                                                                       train_input_Id,
                                                                                                       train_clean_len,
                                                                                                       train_target_Id,
                                                                                                       batch_size, batch,
                                                                                                       idx)
                    source_shuffle = tf.keras.preprocessing.sequence.pad_sequences(source_shuffle, maxlen=None, padding = 'post',value=EOS)
                    train_shuffle = tf.keras.preprocessing.sequence.pad_sequences(train_shuffle, maxlen=None, padding = 'post',value=EOS)
                    target_shuffle = tf.keras.preprocessing.sequence.pad_sequences(target_shuffle, maxlen=None, padding = 'post',value=EOS)

                    target_len = np.tile(max(target_len) + 1, batch_size)


                    fd = {Seq2Seq_model.encoder_inputs: source_shuffle, Seq2Seq_model.encoder_inputs_length: source_len,
                          Seq2Seq_model.decoder_inputs: train_shuffle, Seq2Seq_model.decoder_length: target_len,
                          Seq2Seq_model.ground_truth: target_shuffle}


                    summary,logits_padded, logits, ids, l, _ = sess.run(
                        [merge, Seq2Seq_model.logits, Seq2Seq_model.logits_id, Seq2Seq_model.ids, Seq2Seq_model.loss_seq2seq, Seq2Seq_model.train_op], fd)
                    summary_writer.add_summary(summary, batch)

                    if batch == 0 or batch % batches_in_epoch == 0:
                        ##print the training
                        print('batch {}'.format(batch))
                        print('  minibatch loss: {}'.format(sess.run(Seq2Seq_model.loss_seq2seq, fd)))

                        # ##print the validation
                        # source_shuffle, source_len, target_shuffle, target_len = next_batch(eval_noisy_Id, eval_noisy_len,
                        #                                                                     eval_clean_Id, eval_clean_len, batch_size, batch, idx_e)
                        # fd_eval = {Seq2Seq_model.encoder_inputs: source_shuffle, Seq2Seq_model.encoder_inputs_length: source_len,
                        #            Seq2Seq_model.decoder_targets: target_shuffle, Seq2Seq_model.decoder_length: target_len}
                        # print('  minibatch loss: {}'.format(sess.run(Seq2Seq_model.loss_seq2seq, fd_eval)))
                if epo % epoch_print == 0:
                    for t in range(10):
                        print 'Question {}'.format(t)
                        print " ".join(map(lambda i: vocab[i], list(target_shuffle[t]))).strip()
                        print " ".join(map(lambda i: vocab[i], list(ids[t, :]))).strip()

            saver.save(sess, "../Seq_ckpt/pretrain-seq2seq_unique_embedding_trainable")

    elif model_type == 'testing':
        Seq2Seq_model = Seq2Seq(batch_size, vocab_size, num_units, embd, model_type)
        with tf.Session(config=config) as sess:
            saver_word_rw = tf.train.Saver()
            saver_word_rw.restore(sess, "../Seq_ckpt/pretrain-seq2seq")
            max_batches = len(test_noisy_Id) / batch_size

            idx = np.arange(0, len(test_noisy_Id))
            idx = list(np.random.permutation(idx))
            Bleu_score1=[]
            Bleu_score2=[]
            Bleu_score3=[]
            ex_match=[]
            for batch in range(max_batches):
                source_shuffle, source_len, train_shuffle, target_shuffle, target_len = next_batch(test_noisy_Id,
                                                                                                   test_noisy_len,
                                                                                                   test_input_Id,
                                                                                                   test_clean_len,
                                                                                                   test_target_Id,
                                                                                                   batch_size, batch,
                                                                                                   idx)
                source_shuffle = tf.keras.preprocessing.sequence.pad_sequences(source_shuffle, maxlen=None,
                                                                               padding='post', value=EOS)

                fd = {Seq2Seq_model.encoder_inputs: source_shuffle, Seq2Seq_model.encoder_inputs_length: source_len}
                ids = sess.run([Seq2Seq_model.ids], fd)

                for t in range(batch_size):
                    ref=[]
                    hyp=[]
                    for tar_id in target_shuffle[t]:
                        if tar_id !=2:
                            ref.append(vocab[tar_id])
                    for pre_id in ids[0][t]:
                        if pre_id != 2 and pre_id != 0:
                            hyp.append(vocab[pre_id])
                    sen_bleu1 = bleu([ref], hyp, weights=(1,0,0,0))
                    sen_bleu2 = bleu([ref], hyp, weights=(0.5, 0.5, 0, 0))
                    Bleu_score1.append(sen_bleu1)
                    Bleu_score2.append(sen_bleu2)
                    if ref == hyp:
                        ex_match.append(1)
                    else:
                        ex_match.append(0)

                print "the current whole bleu_score1 is:", sum(Bleu_score1)/float(len(Bleu_score1))
                print "the current whole bleu_score2 is:", sum(Bleu_score2)/float(len(Bleu_score2))
                print "the whole accuracy is:", sum(ex_match)/float(len(ex_match))

                # for t in xrange(10):
                #     print 'Question {}'.format(t)
                #     print " ".join(map(lambda i: vocab[i], list(source_shuffle[t]))).strip()
                #     print " ".join(map(lambda i: vocab[i], list(target_shuffle[t]))).strip()
                #     print " ".join(map(lambda i: vocab[i], list(ids[0][t, :]))).strip()

                    # BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0.5, 0.5))
                    # print BLEUscore






if __name__ == '__main__':
    main()
