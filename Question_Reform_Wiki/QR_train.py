import numpy as np
import tensorflow as tf
import pickle
from tensorflow.contrib import keras
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple
from sklearn.model_selection import train_test_split
import Answer_LSTM
import BeamSearch_Seq2seq
import Seq2Seq

PAD = 0
EOS = 1
max_batches = 3001
batches_in_epoch = 100
batch_size = 128
encoder_hidden_units = 500  # num neurons
decoder_hidden_units = 1000
num_units = 500
embedding_size = 100
beam_width = 10
discount_factor = 0.01
learning_rate = 0.01


def split_data(noisy_Id, noisy_len, clean_Id, clean_len, answer_Id, answer_len):
    num_data = len(noisy_Id)
    train_noisy_Id = noisy_Id[1:int(0.3 * num_data)]
    train_noisy_len = noisy_len[1:int(0.3 * num_data)]
    train_clean_Id = clean_Id[1:int(0.3 * num_data)]
    train_clean_len = clean_len[1:int(0.3 * num_data)]
    train_answer_Id = answer_Id[1:int(0.3 * num_data)]
    train_answer_len = answer_len[1:int(0.3 * num_data)]

    eval_noisy_Id = noisy_Id[int(0.3 * num_data) + 1:int(0.4 * num_data)]
    eval_noisy_len = noisy_len[int(0.3 * num_data) + 1:int(0.4 * num_data)]
    eval_clean_Id = clean_Id[int(0.3 * num_data) + 1:int(0.4 * num_data)]
    eval_clean_len = clean_len[int(0.3 * num_data) + 1:int(0.4 * num_data)]
    eval_answer_Id = answer_Id[int(0.3 * num_data) + 1:int(0.4 * num_data)]
    eval_answer_len = answer_len[int(0.3 * num_data) + 1:int(0.4 * num_data)]

    test_noisy_Id = noisy_Id[int(0.4 * num_data):]
    test_noisy_len = noisy_len[int(0.4 * num_data):]
    test_clean_Id = clean_Id[int(0.4 * num_data):]
    test_clean_len = clean_len[int(0.4 * num_data):]
    test_answer_Id = answer_Id[int(0.4 * num_data):]
    test_answer_len = answer_len[int(0.4 * num_data):]

    return train_noisy_Id, train_noisy_len, train_clean_Id, train_clean_len, train_answer_Id, train_answer_len, eval_noisy_Id, eval_noisy_len, eval_clean_Id, eval_clean_len, eval_answer_Id, eval_answer_len, test_noisy_Id, test_noisy_len, test_clean_Id, test_clean_len, test_answer_Id, test_answer_len


def get_length(seq):
    seq_len = []
    for s in seq:
        seq_len.append(len(s))
    return seq_len


def load_data():
    f = open('../result/Huawei/large', 'rb')
    data = pickle.load(f)
    noisy_Id = data['noisy_Id']
    clean_Id = data['clean_Id']
    answer_Id = data['answer_Id']
    vocab = data['vocab']
    vocab_size = len(vocab)
    print "the number of processing noisy_clean_answer:", len(noisy_Id)
    print "the number of processing vocabulary:", vocab_size

    noisy_len = get_length(noisy_Id)
    clean_len = get_length(clean_Id)
    answer_len = get_length(answer_Id)

    max_noisy_len = max(noisy_len)
    max_clean_len = max(clean_len)
    max_answer_len = max(answer_len)

    for i in range(len(noisy_Id)):
        noisy_Id[i] = noisy_Id[i] + ([PAD] * (max_noisy_len - len(noisy_Id[i])))
        clean_Id[i] = [EOS] + clean_Id[i] + ([PAD] * (max_clean_len - len(clean_Id[i])))
        answer_Id[i] = answer_Id[i] + ([PAD] * (max_answer_len - len(answer_Id[i])))

    train_noisy_Id, train_noisy_len, train_clean_Id, train_clean_len, train_answer_Id, train_answer_len, eval_noisy_Id, eval_noisy_len, eval_clean_Id, eval_clean_len, eval_answer_Id, eval_answer_len, test_noisy_Id, test_noisy_len, test_clean_Id, test_clean_len, test_answer_Id, test_answer_len = split_data(
        noisy_Id, noisy_len, clean_Id, clean_len, answer_Id, answer_len)

    # train_noisy_len = get_length(train_noisy_Id)
    # train_clean_len = get_length(train_clean_Id)
    # train_answer_len = get_length(train_answer_Id)
    # max_train_noisy = max(train_noisy_len)
    # max_trian_clean = max(train_clean_len)
    # max_train_answer = max(train_answer_len)
    # num_train_data = len(train_noisy_Id)
    # for i in range(num_train_data):
    #     train_noisy_Id[i] = train_noisy_Id[i] + ([PAD] * (max_train_noisy - len(train_noisy_Id[i])))
    #     train_clean_Id[i] = [EOS] + train_clean_Id[i] + ([PAD] * (max_trian_clean - len(train_clean_Id[i]) - 1))
    #     train_answer_Id[i] = train_answer_Id[i] + ([PAD] * (max_train_answer - len(train_answer_Id[i])))
    #
    # test_noisy_len = get_length(test_noisy_Id)
    # test_clean_len = get_length(test_clean_Id)
    # test_answer_len = get_length(test_answer_Id)
    # max_test_noisy = max(test_noisy_len)
    # max_test_clean = max(test_clean_len)
    # max_test_answer = max(test_answer_len)
    # num_test_data = len(test_noisy_Id)
    # for i in range(num_test_data):
    #     test_noisy_Id[i] = test_noisy_Id[i] + ([PAD] * (max_test_noisy - len(test_noisy_Id[i])))
    #     test_clean_Id[i] = [EOS] + test_clean_Id[i] + ([PAD] * (max_test_clean - len(test_clean_Id[i]) - 1))
    #     test_answer_Id[i] = test_answer_Id[i] + ([PAD] * (max_test_answer - len(test_answer_Id[i])))
    #
    # eval_noisy_len = get_length(eval_noisy_Id)
    # eval_clean_len = get_length(eval_clean_Id)
    # eval_answer_len = get_length(eval_answer_Id)
    # max_eval_noisy = max(eval_noisy_len)
    # max_eval_clean = max(eval_clean_len)
    # max_eval_answer = max(eval_answer_len)
    # num_eval_data = len(eval_noisy_Id)
    # for i in range(num_eval_data):
    #     eval_noisy_Id[i] = eval_noisy_Id[i] + ([PAD] * (max_eval_noisy - len(eval_noisy_Id[i])))
    #     eval_clean_Id[i] = [EOS] + eval_clean_Id[i] + ([PAD] * (max_eval_clean - len(eval_clean_Id[i]) - 1))
    #     eval_answer_Id[i] = eval_answer_Id[i] + ([PAD] * (max_eval_answer - len(eval_answer_Id[i])))

    return train_noisy_Id, train_noisy_len, train_clean_Id, train_clean_len, train_answer_Id, train_answer_len, eval_noisy_Id, eval_noisy_len, eval_clean_Id, eval_clean_len, eval_answer_Id, eval_answer_len, test_noisy_Id, test_noisy_len, test_clean_Id, test_clean_len, test_answer_Id, test_answer_len, vocab_size



def optimizer(discounted_rewards):
    action = tf.placeholder(shape=[None, 5])
    discounted_rewards = tf.placeholder(shape=[None, ])

    # Calculate cross entropy error function
    action_prob = tf.sum(action * self.model.output, axis=1)
    cross_entropy = tf.log(action_prob) * discounted_rewards
    loss = -tf.sum(cross_entropy)

    # create training function
    optimizer = tf.Adam(lr=learning_rate)
    updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
    train = tf.function([self.model.input, action, discounted_rewards], [], updates=updates)
    return train


def next_batch_Seq2seq(noisy_Id, noisy_len, clean_Id, clean_len, num):
    idx = np.arange(0, len(noisy_Id))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [noisy_Id[i] for i in idx]
    data_len = [noisy_len[i] for i in idx]
    target_shuffle = [clean_Id[i] for i in idx]
    target_len = [clean_len[i] for i in idx]
    return data_shuffle, data_len, target_shuffle, target_len


def next_batch_Lstm(answer_Id, answer_len, num):
    idx = np.arange(0, len(answer_Id))
    np.random.shuffle(idx)
    idx = idx[:num]
    answer_shuffle = [answer_Id[i] for i in idx]
    ans_len = [answer_len[i] for i in idx]
    return answer_shuffle, ans_len


def train_model():
    train_noisy_Id, train_noisy_len, train_clean_Id, train_clean_len, train_answer_Id, train_answer_len, test_noisy_Id, test_noisy_len, test_clean_Id, test_clean_len, test_answer_Id, test_answer_len, eval_noisy_Id, eval_noisy_len, eval_clean_Id, eval_clean_len, eval_answer_Id, eval_answer_len, vocab_size = load_data()

    max_answer_length = np.asarray(train_answer_Id).shape[1]
    max_target_length = np.asarray(train_clean_Id).shape[1]
    max_source_length = np.asarray(train_noisy_Id).shape[1]

    print "trian answer Lstm model"
    an_Lstm = Answer_LSTM.answer_lstm(batch_size, max_answer_length,
                                      vocab_size, embedding_size,
                                      num_units, None, None, None, None)
    an_Lstm.build_graph()

    saver = tf.train.Saver(sharded=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, "../Seq_ckpt/pretrain-model")
        for batch in range(max_batches):
            answer_shuffle, ans_len = next_batch_Lstm(train_answer_Id, train_answer_len, batch_size)
            fd = {an_Lstm.answer_inputs: answer_shuffle, an_Lstm.answer_inputs_length: ans_len}
            l, _ = sess.run([an_Lstm.loss_answer, an_Lstm.train_lstm], fd)
            if batch == 0 or batch % batches_in_epoch == 0:
                print('batch {}'.format(batch))
                answer_shuffle, ans_len = next_batch_Lstm(eval_answer_Id, eval_answer_len, batch_size)
                fd_eval = {an_Lstm.answer_inputs: answer_shuffle, an_Lstm.answer_inputs_length: ans_len}
                print('  minibatch loss: {}'.format(sess.run(an_Lstm.loss_answer, fd_eval)))
        saver.save(sess, "../Seq_ckpt/pretrain-lstm")

    print "trian Seq2seq model"
    Seq2Seq_model = Seq2Seq.Seq2Seq(batch_size, max_source_length, max_target_length,
                                    vocab_size, embedding_size,
                                    num_units, None, None, None, None, None, None, None)
    Seq2Seq_model.build_graph()

    saver = tf.train.Saver(sharded=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, "../Seq_ckpt/pretrain-model")
        for batch in range(max_batches):
            source_shuffle, source_len, target_shuffle, target_len = next_batch_Seq2seq(train_noisy_Id, train_noisy_len,
                                                                                        train_clean_Id, train_clean_len,
                                                                                        batch_size)
            fd = {Seq2Seq_model.encoder_inputs: source_shuffle, Seq2Seq_model.encoder_inputs_length: source_len,
                  Seq2Seq_model.decoder_targets: target_shuffle, Seq2Seq_model.decoder_length: target_len}
            l, _ = sess.run([Seq2Seq_model.loss_seq2seq, Seq2Seq_model.train_op], fd)
            if batch == 0 or batch % batches_in_epoch == 0:
                print('batch {}'.format(batch))
                source_shuffle, source_len, target_shuffle, target_len = next_batch_Seq2seq(eval_noisy_Id,
                                                                                            eval_noisy_len,
                                                                                            eval_clean_Id,
                                                                                            eval_clean_len, batch_size)
                fd_eval = {Seq2Seq_model.encoder_inputs: source_shuffle,
                           Seq2Seq_model.encoder_inputs_length: source_len,
                           Seq2Seq_model.decoder_targets: target_shuffle, Seq2Seq_model.decoder_length: target_len}
                print('  minibatch loss: {}'.format(sess.run(Seq2Seq_model.loss_seq2seq, fd_eval)))
        saver.save(sess, "../Seq_ckpt/pretrain-seq2seq")



if __name__ == '__main__':
    RL_train_model()
