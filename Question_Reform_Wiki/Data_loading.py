import pickle
from sklearn.model_selection import train_test_split
import numpy as np

PAD = 0
SOS = 1
EOS = 2

def split_data(noisy_Id, noisy_len, ground_truth, train_Id, clean_len, answer_Id, answer_len):
    # tv_noisy_Id, test_noisy_Id, tv_noisy_len, test_noisy_len, tv_ground_truth, test_ground_truth, tv_train_Id, test_train_Id, tv_clean_len, test_clean_len,tv_answer_Id, test_answer_Id, tv_answer_len, test_answer_len =\
    #     train_test_split(noisy_Id, noisy_len, ground_truth, train_Id, clean_len, answer_Id, answer_len, test_size = 0.1, random_state = 42)
    #
    # train_noisy_Id, val_noisy_Id, train_noisy_len, val_noisy_len, train_ground_truth, val_ground_truth, train_train_Id, val_train_Id, train_clean_len, val_clean_len, train_answer_Id, val_answer_Id, train_answer_len, val_answer_len =\
    #     train_test_split(tv_noisy_Id, tv_noisy_len, tv_ground_truth, tv_train_Id, tv_clean_len, tv_answer_Id, tv_answer_len, test_size = 0.1, random_state = 42)
    num_data = len(noisy_Id)
    train_rate = 0.85
    train_noisy_Id = noisy_Id[1:int(train_rate * num_data)]
    train_noisy_len = noisy_len[1:int(train_rate * num_data)]
    train_ground_truth = ground_truth[1:int(train_rate * num_data)]
    train_train_Id = train_Id[1:int(train_rate * num_data)]
    train_clean_len = clean_len[1:int(train_rate * num_data)]
    train_answer_Id = answer_Id[1:int(train_rate * num_data)]
    train_answer_len = answer_len[1:int(train_rate * num_data)]

    eval_rate = 0.9
    val_noisy_Id = noisy_Id[int(train_rate * num_data) : int(eval_rate * num_data)]
    val_noisy_len = noisy_len[int(train_rate * num_data) : int(eval_rate * num_data)]
    val_ground_truth = ground_truth[int(train_rate * num_data) : int(eval_rate * num_data)]
    val_train_Id = train_Id[int(train_rate * num_data) : int(eval_rate * num_data)]
    val_clean_len = clean_len[int(train_rate * num_data) : int(eval_rate * num_data)]
    val_answer_Id = answer_Id[int(train_rate * num_data) : int(eval_rate * num_data)]
    val_answer_len = answer_len[int(train_rate * num_data) : int(eval_rate * num_data)]

    test_noisy_Id = noisy_Id[int(eval_rate * num_data):]
    test_noisy_len = noisy_len[int(eval_rate * num_data):]
    test_ground_truth = ground_truth[int(eval_rate * num_data):]
    test_train_Id = train_Id[int(eval_rate * num_data):]
    test_clean_len = clean_len[int(eval_rate * num_data):]
    test_answer_Id = answer_Id[int(eval_rate * num_data):]
    test_answer_len = answer_len[int(eval_rate * num_data):]

    return train_noisy_Id, train_noisy_len, train_ground_truth, train_train_Id,train_clean_len, train_answer_Id, train_answer_len, \
           val_noisy_Id, val_noisy_len, val_ground_truth, val_train_Id, val_clean_len, val_answer_Id, val_answer_len, \
           test_noisy_Id, test_noisy_len, test_ground_truth, test_train_Id, test_clean_len, test_answer_Id, test_answer_len


def get_length(seq):
    seq_len = []
    for s in seq:
        seq_len.append(len(s))
    return seq_len

def load_data(process_file_name):
    f = open(process_file_name, 'rb')
    data = pickle.load(f)
    old_noisy_Id = data['noisy_Id']
    old_clean_Id = data['clean_Id']
    old_answer_Id = data['answer_Id']
    vocab = data['vocab']
    embd = data['embd']
    vocab_size = len(vocab)

    print "the number of processing vocabulary:", vocab_size

    old_noisy_len = get_length(old_noisy_Id)
    old_clean_len = get_length(old_clean_Id)
    old_answer_len = get_length(old_answer_Id)


    noisy_Id = []
    clean_Id = []
    answer_Id = []
    noisy_len = []
    clean_len = []
    answer_len = []
    for idx in range(len(old_noisy_len)):
        if old_noisy_len[idx] > 0 and old_clean_len[idx] > 0  and old_answer_len[idx] > 0 :
            noisy_Id.append(old_noisy_Id[idx])
            noisy_len.append(old_noisy_len[idx])
            clean_Id.append(old_clean_Id[idx])
            clean_len.append(old_clean_len[idx])
            answer_Id.append(old_answer_Id[idx])
            answer_len.append(old_answer_len[idx])

    print "the number of tripe is:", len(noisy_Id)

    max_noisy_len = max(noisy_len)
    max_clean_len = max(clean_len)
    max_answer_len = max(answer_len)
    print "the max noisy length:", max_noisy_len
    print "the average length:", float(sum(noisy_len))/float(len(noisy_len))
    print "the max clean length:", max_clean_len
    print "the average length:", float(sum(clean_len))/float(len(clean_len))
    print "the max answer length:", max_answer_len
    print "the average answer length is:", sum(answer_len) / float(len(answer_len))


    ground_truth = []
    train_Id=[]
    for i in range(len(noisy_Id)):
        train_Id.append([SOS] + clean_Id[i])
        ground_truth.append(clean_Id[i] + [EOS])

    clean_len=get_length(ground_truth)


    train_noisy_Id, train_noisy_len, train_ground_truth, train_train_Id, train_clean_len, train_answer_Id, train_answer_len, \
    val_noisy_Id, val_noisy_len, val_ground_truth, val_train_Id, val_clean_len, val_answer_Id, val_answer_len, \
    test_noisy_Id, test_noisy_len, test_ground_truth, test_train_Id, test_clean_len, test_answer_Id, test_answer_len = split_data(noisy_Id, noisy_len, ground_truth, train_Id, clean_len, answer_Id, answer_len)

    train_data_com={"noisy_Id":train_noisy_Id, "noisy_len":train_noisy_len,"ground_truth":train_ground_truth,"train_Id":train_train_Id,"clean_len":train_clean_len,"answer_Id":train_answer_Id,"answer_len":train_answer_len, "vocab":vocab, "embd":embd}
    val_data_com={"noisy_Id":val_noisy_Id, "noisy_len":val_noisy_len, "ground_truth":val_ground_truth, "train_Id":val_train_Id, "clean_len":val_clean_len, "answer_Id":val_answer_Id, "answer_len":val_answer_len,"vocab":vocab, "embd":embd}
    test_data_com={ "noisy_Id":test_noisy_Id, "noisy_len":test_noisy_len, "ground_truth":test_ground_truth,"train_Id":test_train_Id,"clean_len":test_clean_len, "answer_Id":test_answer_Id,"answer_len":test_answer_len,
              "vocab":vocab, "embd":embd}
    data_com ={"noisy_Id": noisy_Id, "noisy_len": noisy_len, "ground_truth": ground_truth, "train_Id": train_Id, "clean_len": clean_len, "answer_Id": answer_Id, "answer_len": answer_len}
    max_data={"max_noisy_len":max_noisy_len,"max_clean_len":max_clean_len + 1,"max_answer_len":max_answer_len}
    return train_data_com, val_data_com, test_data_com, max_data, data_com

if __name__ == '__main__':
    load_data()