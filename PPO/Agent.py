import numpy as np
import os

def get_training_data(rewards, values, next_values, lam, gamma):
    deltas = []
    returns = []
    V = 0
    for i in reversed(range(len(rewards))):
        reward = rewards[i]
        value = values[i]
        next_value = next_values[i]
        delta = reward + gamma * next_value - value
        V = delta + lam * gamma * V
        deltas.append(V)
        returns.append(V + value)
    deltas = np.array(list(reversed(deltas)), dtype=np.float32)
    returns = np.array(list(reversed(returns)), dtype=np.float32)
    # standardize advantages
    deltas = (deltas - deltas.mean()) / (deltas.std() + 1e-5)
    return list(returns), list(deltas)

def bert_vocab_map_Id(sen_Id, config, vocab_cur):
    bert_data_path = config.bert_data_path
    data_file = bert_data_path
    if os.path.isfile(data_file):
        f = open(data_file, 'rb')
        vocab = []
        for v in f.readlines():
            vocab.append(v.replace("\n", ""))
        vocab_idx = dict(zip(vocab, range(len(vocab))))
    cur_idx_vocab = dict(zip(range(len(vocab_cur)), vocab_cur))

    vocab_size = len(vocab)

    bert_Id = []
    for list_id in sen_Id:
        list_id_ber = []
        for w in list_id:
            word = cur_idx_vocab[w].encode("utf8")
            try:
                list_id_ber.append(vocab_idx[word])
            except:
                list_id_ber.append(0)
        bert_Id.append(list_id_ber)
    return bert_Id, vocab_size

def next_batch(noisy_Id, noisy_len, n_emd, noisy_char_Id, noisy_char_len, input_Id, target_Id, clean_len, answer_Id, answer_len, batch_size,
               batch_num, idx):
    if (batch_num + 1) * batch_size > len(noisy_Id):
        batch_num = batch_num % (len(noisy_Id) / batch_size)
    idx_n = idx[batch_num * batch_size: (batch_num + 1) * batch_size]

    data_shuffle = [noisy_Id[i] for i in idx_n]
    data_len = [noisy_len[i] for i in idx_n]
    data_nemd = [n_emd[i] for i in idx_n]
    char_Id = [noisy_char_Id[i] for i in idx_n]
    char_len = [noisy_char_len[i] for i in idx_n]
    train_shuffle = [input_Id[i] for i in idx_n]
    # train_cemd =[cemd_i[i] for i in idx_n]
    target_len = [clean_len[i] for i in idx_n]
    target_shuffle = [target_Id[i] for i in idx_n]
    answer_shuffle = [answer_Id[i] for i in idx_n]
    answer_lenth = [answer_len[i] for i in idx_n]
    return data_shuffle, data_len, data_nemd, char_Id, char_len, train_shuffle, target_shuffle,  target_len, answer_shuffle, answer_lenth