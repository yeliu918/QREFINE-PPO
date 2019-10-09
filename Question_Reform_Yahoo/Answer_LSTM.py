import numpy as np
import tensorflow as tf
import pickle


###Answer lstm
class answer_lstm(object):
    def __init__(self, batch_size, max_answer_sequence_length,
                 vocab_size, num_units, embd):
        self.batch_size = batch_size
        self.max_sequence_length = max_answer_sequence_length
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.embedding=embd
        self.build_graph()

    def build_graph(self):
        with tf.variable_scope("answer_lstm"):
            self.answer_inputs = tf.placeholder(shape=(self.batch_size, self.max_sequence_length),
                                                dtype=tf.int32, name='answer_inputs')
            self.answer_inputs_length = tf.placeholder(shape=(self.batch_size,), dtype=tf.int32,
                                                       name='answer_length_inputs')
            self.embeddings = tf.get_variable(name="W", shape=self.embedding.shape, initializer=tf.constant_initializer(self.embedding),trainable=False)
            self.answer_embedded = tf.nn.embedding_lookup(self.embeddings, self.answer_inputs)
        with tf.variable_scope("question_lstm"):
            self.question_inputs = tf.placeholder(shape=(self.batch_size, self.max_sequence_length),
                                                dtype=tf.int32, name='question_inputs')
            self.question_inputs_length = tf.placeholder(shape=(self.batch_size,), dtype=tf.int32,
                                                       name='question_length_inputs')
            self.question_embedded = tf.nn.embedding_lookup(self.embeddings, self.answer_inputs)
        self.target_label = tf.placeholder(shape=(self.batch_size,), dtype=tf.int32, name="target_label")
        self.inference()
        self.loss()
        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", self.loss_answer)

    def inference(self):
        with tf.variable_scope("answer_lstm"):
            answer_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units)
            answer_outputs, self.answer_state = tf.nn.dynamic_rnn(cell=answer_cell, inputs=self.answer_embedded,
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

            question_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units)
            question_outputs, self.question_state = tf.nn.dynamic_rnn(cell=question_cell, inputs=self.question_embedded,
                                                                  sequence_length=self.question_inputs_length,
                                                                  time_major=False,
                                                                  dtype=tf.float32)

    def loss(self):
        with tf.variable_scope("cosin_similarity"):
            cosine_simi = tf.losses.cosine_distance(labels=self.question_state, predictions = self.answer_state, dim=None)
            self.loss = tf.metrics.mean_squared_error(labels=cosine_simi, predictions=self.target_label)
            self.train_lstm = tf.train.AdamOptimizer().minimize(self.loss)

        # with tf.variable_scope("answer_lstm"):
            # stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            #     labels=tf.one_hot(self.answer_inputs, depth=self.vocab_size, dtype=tf.float32),
            #     logits=self.answer_logits)
            # # loss function
            # self.loss_answer = tf.reduce_mean(stepwise_cross_entropy)
            # # train it
            # self.train_lstm = tf.train.AdamOptimizer().minimize(self.loss_answer)





def next_batch(answer_Id, answer_len, batch_size, batch_num, idx):
    if (batch_num + 1) * batch_size > len(answer_Id):
        batch_num = batch_num % (len(answer_Id) / batch_size)
    idx_n = idx[batch_num * batch_size: (batch_num + 1) * batch_size]
    answer_shuffle = [answer_Id[i] for i in idx_n]
    ans_len = [answer_len[i] for i in idx_n]
    return answer_shuffle, ans_len


def main():
    PAD = 0
    SOS = 1
    EOS = 2
    epoch = 20
    batches_in_epoch = 500
    batch_size = 64
    num_units = 200
    epoch_print = 5
    logs_path = "../Seq_ckpt/answer-lstm_board"
    f = open('../result/Huawei/large_pretrained', 'rb')
    text_file = open("../result/Huawei/Output_lstm.txt", "w")
    data = pickle.load(f)
    noisy_Id = data['noisy_Id']
    clean_Id = data['clean_Id']
    answer_Id = data['answer_Id']
    vocab = data['vocab']
    embd = data['embd']
    vocab_size = len(vocab)

    num_data = len(noisy_Id)

    noisy_len = []
    for n in noisy_Id:
        noisy_len.append(len(n))
    max_source_length = max(noisy_len)

    clean_len = []
    for c in clean_Id:
        clean_len.append(len(c))
    max_target_length = max(clean_len)
    max_target_length = max_target_length + 1

    answer_len = []
    for a in answer_Id:
        answer_len.append(len(a))
    max_answer_length = max(answer_len)

    num_data = len(noisy_Id)
    for i in range(num_data):
        answer_Id[i] = answer_Id[i] + ([PAD] * (max_answer_length - len(answer_Id[i])))

    train_answer_Id = answer_Id[1:int(0.8 * num_data)]
    train_answer_len = answer_len[1:int(0.8 * num_data)]
    # eval_answer_Id = answer_Id[int(0.3 * num_data) + 1:int(0.4 * num_data)]
    # eval_answer_len = answer_len[int(0.3 * num_data) + 1:int(0.4 * num_data)]

    print "the number of training question is:", len(train_answer_Id)
    # print "the number of evaluated question is:", len(eval_answer_Id)


    an_Lstm = answer_lstm(batch_size, max_answer_length,
                          vocab_size, num_units, embd)


    # summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    saver = tf.train.Saver(sharded=False)
    # merge = tf.summary.merge_all()

    max_batches = len(train_answer_Id) / batch_size
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, "../Seq_ckpt/pretrain-model")
        for epo in range(epoch):
            idx = np.arange(0, len(train_answer_Id))
            idx = list(np.random.permutation(idx))
            for batch in range(max_batches):
                answer_shuffle, ans_len = next_batch(train_answer_Id, train_answer_len, batch_size, batch, idx)
                fd = {an_Lstm.answer_inputs: answer_shuffle, an_Lstm.answer_inputs_length: ans_len}

                a_logtis, l, _ = sess.run([an_Lstm.answer_prediction, an_Lstm.loss_answer, an_Lstm.train_lstm], fd)

                # summary, l, _ = sess.run([merge, an_Lstm.loss_answer, an_Lstm.train_lstm], fd)
                # summary_writer.add_summary(summary, batch)

                if batch == 0 or batch % batches_in_epoch == 0:
                    ## print the training result
                    print('batch {}'.format(batch))
                    print('  minibatch loss of trianing: {}'.format(sess.run(an_Lstm.loss_answer, fd)))
                    text_file.write('batch {}'.format(batch))
                    text_file.write('\n')
                    text_file.write('  minibatch loss of trianing: {}'.format(sess.run(an_Lstm.loss_answer, fd)))
                    text_file.write('\n')

                    # ## print the validation result
                    # answer_shuffle, ans_len = next_batch(eval_answer_Id, eval_answer_len, batch_size, batch, idx_e)
                    # fd_eval = {an_Lstm.answer_inputs: answer_shuffle, an_Lstm.answer_inputs_length: ans_len}
                    # print('  minibatch loss of evaluation: {}'.format(sess.run(an_Lstm.loss_answer, fd_eval)))
            if epo % epoch_print == 0:
                for t in xrange(10):
                    print 'Question {}'.format(t)
                    print " ".join(map(lambda i: vocab[i], list(answer_shuffle[t]))).replace('<PAD>', '').strip()
                    print " ".join(map(lambda i: vocab[i], list(a_logtis[t, :]))).replace('<PAD>', '').strip()
                    text_file.write('Question {}'.format(t))
                    text_file.write('\n')
                    text_file.write(" ".join(map(lambda i: vocab[i], list(answer_shuffle[t]))).replace('<PAD>', '').encode('utf-8').strip())
                    text_file.write('\n')
                    text_file.write(" ".join(map(lambda i: vocab[i], list(a_logtis[t, :]))).replace('<PAD>', '').encode('utf-8').strip())
                    text_file.write('\n')
        text_file.close()
        saver.save(sess, "../Seq_ckpt/pretrain-lstm")


if __name__ == '__main__':
    main()

    # ##matplotlib inline
    # import matplotlib.pyplot as plt
    #
    # plt.plot(loss_track)
    # print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track) * batch_size, batch_size))
