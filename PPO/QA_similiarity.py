import numpy as np
import tensorflow as tf
import pickle
from sklearn.metrics import mean_squared_error
from Data_loading import load_data
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class QA_similiarity(object):
    def __init__(self, batch_size, num_units, embd, model_type):
        self.batch_size = batch_size
        self.num_units = num_units
        self.embedding = embd
        self.model_type=model_type
        self.build_graph()

    def build_graph(self):
        with tf.variable_scope("answer_input"):
            self.answer_inputs = tf.placeholder(shape=(self.batch_size, None),
                                                dtype=tf.int32, name='answer_inputs')
            self.answer_inputs_length = tf.placeholder(shape=(self.batch_size,), dtype=tf.int32,
                                                       name='answer_length_inputs')
            self.embeddings = tf.get_variable(name="W", shape=self.embedding.shape,
                                              initializer=tf.constant_initializer(self.embedding), trainable=False)
            self.answer_embedded = tf.nn.embedding_lookup(self.embeddings, self.answer_inputs)
            # self.answer_emd = tf.placeholder(shape=(self.batch_size, None),
            #                                        dtype=tf.float32, name='answer_bert_emd')
            # self.answer_embd = tf.add(self.answer_embedded, self.answer_emd)

        with tf.variable_scope("question_input"):
            self.question1_inputs = tf.placeholder(shape=(self.batch_size, None),
                                                   dtype=tf.int32, name='question_inputs')
            self.question1_inputs_length = tf.placeholder(shape=(self.batch_size,), dtype=tf.int32,
                                                          name='question_length_inputs')
            self.question1_embedded = tf.nn.embedding_lookup(self.embeddings, self.question1_inputs)
            self.question1_emd = tf.placeholder(shape=(self.batch_size, None, 768),
                                                   dtype=tf.float32, name='clean_bert_emd')
            self.clean_embd =tf.concat([self.question1_embedded, self.question1_emd], axis=-1)

            self.question2_inputs = tf.placeholder(shape=(self.batch_size, None),
                                                   dtype=tf.int32, name='question2_inputs')
            self.question2_inputs_length = tf.placeholder(shape=(self.batch_size,), dtype=tf.int32,
                                                          name='question2_length_inputs')
            self.question2_embedded = tf.nn.embedding_lookup(self.embeddings, self.question2_inputs)
            self.question2_emd = tf.placeholder(shape=(self.batch_size, None, 768),
                                                 dtype=tf.float32, name='noisy_bert_emd')
            self.noisy_embd = tf.concat([self.question2_embedded, self.question2_emd], axis=-1)
            self.W = tf.Variable(np.eye(self.num_units, dtype=float), name="weight", dtype=tf.float32)

        # if self.model_type == "training":
        #     self.target_label = tf.placeholder(shape=(self.batch_size,), dtype=tf.int32, name="target_label")

        self.inference()
        self.loss()
        # Create a summary to monitor cost tensor
        # tf.summary.scalar("loss", self.loss_answer)

    def inference(self):
        with tf.variable_scope("answer_lstm"):
            answer_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units)
            self.answer_outputs, self.answer_state = tf.nn.dynamic_rnn(cell=answer_cell, inputs=self.answer_embedded,
                                                                       sequence_length=self.answer_inputs_length,
                                                                       time_major=False,
                                                                       dtype=tf.float32)

            # self.answer_logits = tf.layers.dense(answer_outputs, units=self.vocab_size)
            # padding = tf.ones(
            #     [tf.shape(self.answer_logits)[0], self.max_sequence_length - tf.shape(self.answer_logits)[1],
            #      tf.shape(self.answer_logits)[2]])
            # self.answer_logits = tf.concat([padding, self.answer_logits], 1)
            # # final prediction
            # self.answer_prediction = tf.argmax(self.answer_logits, 2)
        with tf.variable_scope("question_lstm"):
            question_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units)
            self.question1_outputs, self.question1_state = tf.nn.dynamic_rnn(cell=question_cell,
                                                                             inputs=self.clean_embd,
                                                                             sequence_length=self.question1_inputs_length,
                                                                             time_major=False,
                                                                             dtype=tf.float32)

            self.question2_outputs, self.question2_state = tf.nn.dynamic_rnn(cell=question_cell,
                                                                             inputs=self.noisy_embd,
                                                                             sequence_length=self.question2_inputs_length,
                                                                             time_major=False,
                                                                             dtype=tf.float32)

    def loss(self):
        with tf.variable_scope("cosin_similarity"):
            norm_question1 = tf.nn.l2_normalize(self.question1_state.h, 0)
            norm_question2 = tf.nn.l2_normalize(self.question2_state.h, 0)
            norm_answer = tf.nn.l2_normalize(self.answer_state.h, 0)
            # norm_w = tf.nn.l2_normalize(self.W, 0)
            # self.cosine_simi1 = tf.reduce_sum(tf.matmul(tf.matmul(norm_question1, norm_w),tf.transpose(norm_answer)), 1)
            # self.cosine_simi2 = tf.reduce_sum(tf.matmul(tf.matmul(norm_question2, norm_w),tf.transpose(norm_answer)), 1)
            self.cosine_simi1 = tf.reduce_sum(tf.matmul(tf.matmul(norm_question1, self.W),tf.transpose(norm_answer)), 1)
            self.cosine_simi2 = tf.reduce_sum(tf.matmul(tf.matmul(norm_question2, self.W),tf.transpose(norm_answer)), 1)
            self.norm_cosine1 = tf.nn.l2_normalize(self.cosine_simi1,0)
            self.norm_cosine2 = tf.nn.l2_normalize(self.cosine_simi2,0)
            self.two_distance = self.norm_cosine2 - self.norm_cosine1 ##noisy-clean
            # self.loss_distance = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(two_distance, tf.cast(self.target_label,dtype = tf.float32)))
            if self.model_type == "training":
                self.loss_distance = tf.maximum(tf.cast(0, tf.float32), 0.5 + self.two_distance)
                self.loss_score_cm = tf.reduce_mean(self.loss_distance)
                # self.loss_distance = tf.losses.mean_squared_error(labels=tf.cast(self.target_label,dtype = tf.float32), predictions=self.two_distance)
                self.train_lstm = tf.train.AdamOptimizer().minimize(self.loss_distance)

            # with tf.variable_scope("answer_lstm"):
            # stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            #     labels=tf.one_hot(self.answer_inputs, depth=self.vocab_size, dtype=tf.float32),
            #     logits=self.answer_logits)
            # # loss function
            # self.loss_answer = tf.reduce_mean(stepwise_cross_entropy)
            # # train it
            # self.train_lstm = tf.train.AdamOptimizer().minimize(self.loss_answer)

def next_batch(clean_Id, clean_len, c_emd, noisy_Id, noisy_len, n_emd, answer_Id, answer_len, batch_size, batch_num,
               idx):
    if (batch_num + 1) * batch_size > len(noisy_Id):
        batch_num = batch_num % (len(noisy_Id) / batch_size)
    idx_n = idx[batch_num * batch_size: (batch_num + 1) * batch_size]
    n_shuffle = [noisy_Id[i] for i in idx_n]
    nemd_shuffle = [n_emd[i][1:]for i in idx_n]
    n_len = [noisy_len[i] for i in idx_n]
    c_shuffle = [clean_Id[i] for i in idx_n]
    cemd_shuffle = [c_emd[i][1:] for i in idx_n]
    c_len = [clean_len[i] for i in idx_n]
    a_shuffle = [answer_Id[i] for i in idx_n]
    a_len = [answer_len[i] for i in idx_n]
    return n_shuffle, n_len, nemd_shuffle, c_shuffle, cemd_shuffle, c_len, a_shuffle, a_len


def run_QA(data_file,emd_file, ckp_dir):
    PAD = 0
    SOS = 1
    EOS = 2
    epoch = 30
    batches_in_epoch = 500
    batch_size = 64
    num_units = 500
    epoch_print = 5
    # path = "/mnt/WDRed4T/ye/DataR/YAHOO/"
    # dataset = "wrongordersmall_"
    # ckp_path ='/mnt/WDRed4T/ye/Qrefine/ckpt/QA_sm/'+ dataset
    # # dataset = "wrongorder_"
    # # # dataset = "back_"
    # # # dataset = "threeopt_"
    # IdFile = path + dataset + "Id"
    # f = open(IdFile , 'rb')
    # data = pickle.load(f)
    # n_emdPath = path + dataset + "bert_noisyemb.npy"
    # c_emdPath = path + dataset + "bert_cleanemb.npy"
    # # a_emdPath = path + dataset + "bert_answeremb.npy"
    # n_emd = np.load(n_emdPath)
    # c_emd = np.load(c_emdPath)
    # # a_emd = np.load(a_emdPath)
    # # text_file = open("../result/Huawei/QA_s_result.txt", "w")
    #
    # question_c = data['clean_Id']
    # question_n = data['noisy_Id']
    # answer = data['answer_Id']
    # embd = data['embd']
    # embd = np.array(embd)
    # vocab = data['vocab']
    # vocab_size = len(vocab)

    qa_data = load_data(data_file, emd_file, "QA")
    question_c, c_emd, question_n, n_emd, answer, embd, vocab = qa_data

    num_data = len(question_c)
    print("the num of data:", num_data)

    cq_len = []
    for c in question_c:
        cq_len.append(len(c))

    print("max clean len", max(cq_len))

    nq_len = []
    for n in question_n:
        nq_len.append(len(n))
    print("max noisy len: ", max(nq_len))

    answer_len = []
    for a in answer:
        answer_len.append(len(a))
    print("max answer len: ", max(answer_len))

    train_answer_Id = answer
    train_answer_len = answer_len
    train_qc_Id = question_c
    train_qc_len = cq_len
    train_qn_Id = question_n
    train_qn_len = nq_len
    # train_cosine_sim = cosine_sim[1:int(0.8 * num_data)]

    model_type="training"
    an_Lstm = QA_similiarity(batch_size, num_units, embd, model_type)

    # summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    saver = tf.train.Saver(sharded=False)
    # merge = tf.summary.merge_all()

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config=config

    max_batches = len(train_answer_Id) / batch_size
    with tf.Session() as sess:
        if os.path.isfile(ckp_dir):
            saver.restore(sess, ckp_dir)
        else:
            sess.run(tf.global_variables_initializer())

        for epo in range(epoch):
            idx = np.arange(0, len(train_answer_Id))
            idx = list(np.random.permutation(idx))
            print('epoch {}'.format(epo))

            for batch in range(max_batches):
                n_shuffle, n_len, nemd_shuffle, c_shuffle, cemd_shuffle, c_len, a_shuffle, a_len = next_batch(train_qc_Id, train_qc_len, list(c_emd),
                                                                                              train_qn_Id, train_qn_len, list(n_emd),
                                                                                              train_answer_Id, train_answer_len,
                                                                                              batch_size, batch, idx)
                a_shuffle = tf.keras.preprocessing.sequence.pad_sequences(a_shuffle, maxlen=None,
                                                                          padding='post', value=EOS)
                # tile_aemd = []
                # for a in aemd_shuffle:
                #     tile_aemd.append(a[0:a_shuffle.shape[1]])
                # tile_aemd =np.asarray(tile_aemd)

                c_shuffle = tf.keras.preprocessing.sequence.pad_sequences(c_shuffle, maxlen=None,
                                                                           padding='post', value=EOS)

                tile_cemd = []
                for c in cemd_shuffle:
                    tile_cemd.append(c[0:c_shuffle.shape[1]])
                tile_cemd = np.asarray(tile_cemd)

                n_shuffle = tf.keras.preprocessing.sequence.pad_sequences(n_shuffle, maxlen=None,
                                                                           padding='post', value=EOS)

                tile_nemd = []
                for n in nemd_shuffle:
                    tile_nemd.append(n[0:n_shuffle.shape[1]])
                tile_nemd = np.asarray(tile_nemd)

                # an_Lstm.answer_emd: tile_aemd

                fd = {an_Lstm.answer_inputs: a_shuffle, an_Lstm.answer_inputs_length: a_len,
                      an_Lstm.question1_inputs: c_shuffle, an_Lstm.question1_inputs_length: c_len, an_Lstm.question1_emd: tile_cemd,
                      an_Lstm.question2_inputs: n_shuffle, an_Lstm.question2_inputs_length: n_len, an_Lstm.question2_emd: tile_nemd}

                mean_dis, W, cosine_si1, cosine_si2, que_1, que_2, a, two_dis, l, _ = sess.run([an_Lstm.loss_score_cm, an_Lstm.W, an_Lstm.cosine_simi1, an_Lstm.cosine_simi2, an_Lstm.question1_state.h, an_Lstm.question2_state.h, an_Lstm.answer_state.h,an_Lstm.two_distance, an_Lstm.loss_distance, an_Lstm.train_lstm], fd)

                # summary, l, _ = sess.run([merge, an_Lstm.loss_answer, an_Lstm.train_lstm], fd)
                # summary_writer.add_summary(summary, batch)

                if batch == 0 or batch % batches_in_epoch == 0:
                    ## print the training result
                    print('batch {}'.format(batch))
                    print("     The mean distance between nq and cq is", mean_dis)

            if epo % 10 == 0:
                saver.save(sess, ckp_dir)
                    # print('  minibatch loss of trianing: {}'.format(l))
                    # print(cosine_si1)
                    # print(cosine_si2)
                    # print(que_1)
                    # print(que_2)
                    # print(a)

                    # text_file.write('batch {}'.format(batch))
                    # text_file.write('\n')
                    # text_file.write('  minibatch loss of trianing: {}'.format(l))
                    # text_file.write('\n')

                    # ## print the validation result
                    # answer_shuffle, ans_len = next_batch(eval_answer_Id, eval_answer_len, batch_size, batch, idx_e)
                    # fd_eval = {an_Lstm.answer_inputs: answer_shuffle, an_Lstm.answer_inputs_length: ans_len}
                    # print('  minibatch loss of evaluation: {}'.format(sess.run(an_Lstm.loss_answer, fd_eval)))
            # if epo % epoch_print == 0:
            #     for t in xrange(10):
            #         print 'Question {}'.format(t)
            #         print " ".join(map(lambda i: vocab[i], list(a_shuffle[t]))).strip()
            #         print " ".join(map(lambda i: vocab[i], list(a_out[t, :]))).strip()
            #         print " ".join(map(lambda i: vocab[i], list(q1_shuffle[t]))).strip()
            #         print " ".join(map(lambda i: vocab[i], list(q1_out[t, :]))).strip()
            #         print " ".join(map(lambda i: vocab[i], list(q2_shuffle[t]))).strip()
            #         print " ".join(map(lambda i: vocab[i], list(q2_out[t, :]))).strip()

        # text_file.close()


# if __name__ == '__main__':
#     main()

    # ##matplotlib inline
    # import matplotlib.pyplot as plt
    #
    # plt.plot(loss_track)
    # print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track) * batch_size, batch_size))
