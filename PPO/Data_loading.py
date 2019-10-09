import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import random as rd
import os

def split_data(tripe_num):
    list_num = tripe_num
    # rd.seed(1)
    random_list = range(list_num)
    # rd.shuffle(random_list)
    train_fold = [[i] for i in range(5)]
    eval_fold = [[i] for i in range(5)]
    test_fold = [[i] for i in range(5)]
    one_fold = list_num / 5
    for iter in range(5): ##train 80%, test and eval 10%
        eval_fold[iter] = random_list[iter * one_fold : (iter+1) * (one_fold / 2)]
        test_fold[iter] = random_list[(iter+1) * (one_fold / 2): (iter+1) * one_fold]
        if iter == 0:
            train_fold[iter] = random_list[(iter + 1) * one_fold :]
        else:
            train_fold[iter] = random_list[0:(iter - 1) * one_fold] + random_list[(iter+1)*one_fold:]
    return train_fold, eval_fold, test_fold

def get_length(seq):
    seq_len = []
    for s in seq:
        seq_len.append(len(s))
    return seq_len


def load_data(data_file, emd_file, model_type):
    if os.path.isfile(data_file):
        f = open(data_file, 'rb')
        data = pickle.load(f)
    noisy_Id = data['noisy_Id']
    clean_Id = data['clean_Id']
    answer_Id = data['answer_Id']
    char_noisy_Id = data['noisy_char_Id']
    char_noisy_len = data['noisy_char_len']
    max_char = data["max_char"]
    max_word = data['max_word']
    vocab = data['vocab']
    embd = data['embd']
    vocab_size = len(vocab)
    embd = np.array(embd)


    n_emdPath, c_emdPath= emd_file
    n_emd = np.load(n_emdPath)
    c_emd = np.load(c_emdPath)


    # vocdic = zip(vocab, range(len(vocab)))
    voc_index = dict(zip(vocab, range(len(vocab))))  ##dic[char]= index

    SOS = voc_index[u"<SOS>"]
    EOS = voc_index[u"<EOS>"]

    print "the number of processing vocabulary:", vocab_size

    print len(noisy_Id)
    print len(clean_Id)
    print len(answer_Id)

    noisy_len = get_length(noisy_Id)
    answer_len = get_length(answer_Id)

    tripe_num = len(noisy_Id)
    print "the number of tripe is:", tripe_num

    ground_truth = []
    train_Id=[]
    for i in range(len(noisy_Id)):
        train_Id.append([SOS] + clean_Id[i]) ## decoder input   [SOS] + sentence
        ground_truth.append(clean_Id[i] + [EOS])  ## decoder output   sentence+[EOS]

    clean_len=get_length(ground_truth)

    max_noisy_len = max(noisy_len)
    max_clean_len = max(clean_len)
    max_answer_len = max(answer_len)

    print "the max noisy length:", max_noisy_len
    print "the max clean length:", max_clean_len
    print "the max answer length:", max_answer_len
    print "the average answer length is:", sum(answer_len) / float(len(answer_len))
    print "the average noisy length is:", sum(noisy_len)/float(len(noisy_len))
    print "the average clean length is:", sum(clean_len)/float(len(clean_len))

    if model_type == "QA":
        qa_data = [clean_Id, c_emd, noisy_Id, n_emd, answer_Id, embd, vocab]
        return qa_data

    if model_type == "BA":
        train_fold, eval_fold, test_fold = split_data(tripe_num)

        iter = 0
        train_noisy_Id = [noisy_Id[i] for i in train_fold[iter]]
        train_noisy_len = [noisy_len[i] for i in train_fold[iter]]
        train_char_noisy_Id = [char_noisy_Id[i] for i in train_fold[iter]]
        train_char_noisy_len = [char_noisy_len[i] for i in train_fold[iter]]
        train_target_Id = [ground_truth[i] for i in train_fold[iter]]
        train_input_Id = [train_Id[i] for i in train_fold[iter]]
        train_clean_Id = [clean_Id[i] for i in train_fold[iter]]
        train_clean_len = [clean_len[i] for i in train_fold[iter]]
        train_answer_Id = [answer_Id[i] for i in train_fold[iter]]
        train_answer_len = [answer_len[i] for i in train_fold[iter]]

        train_nemd = [n_emd[i] for i in train_fold[iter]]
        # train_aemd = [a_emd[i] for i in train_fold[iter]]

        train_data = [train_noisy_Id, train_noisy_len,train_char_noisy_Id, train_char_noisy_len, train_nemd,
                      train_target_Id,  train_input_Id,  train_clean_Id, train_clean_len, train_answer_Id, train_answer_len, max_char, max_word]

        test_noisy_Id = [noisy_Id[i] for i in test_fold[iter]]
        test_noisy_len = [noisy_len[i] for i in test_fold[iter]]
        test_char_noisy_Id = [char_noisy_Id[i] for i in test_fold[iter]]
        test_char_noisy_len = [char_noisy_len[i] for i in test_fold[iter]]
        test_target_Id = [ground_truth[i] for i in test_fold[iter]]
        test_input_Id = [train_Id[i] for i in test_fold[iter]]
        test_clean_Id =[clean_Id[i] for i in test_fold[iter]]
        test_clean_len = [clean_len[i] for i in test_fold[iter]]
        test_answer_Id = [answer_Id[i] for i in test_fold[iter]]
        test_answer_len = [answer_len[i] for i in test_fold[iter]]

        test_nemd = [n_emd[i] for i in test_fold[iter]]
        # test_aemd = [a_emd[i] for i in test_fold[iter]]
        test_data = [test_noisy_Id, test_noisy_len, test_char_noisy_Id, test_char_noisy_len, test_nemd,
                     test_target_Id, test_input_Id, test_clean_Id, test_clean_len, test_answer_Id, test_answer_len]

        eval_noisy_Id = [noisy_Id[i] for i in eval_fold[iter]]
        eval_noisy_len = [noisy_len[i] for i in eval_fold[iter]]
        eval_char_noisy_Id = [char_noisy_Id[i] for i in eval_fold[iter]]
        eval_char_noisy_len = [char_noisy_len[i] for i in eval_fold[iter]]
        eval_target_Id = [ground_truth[i] for i in eval_fold[iter]]
        eval_input_Id = [train_Id[i] for i in eval_fold[iter]]
        eval_clean_Id = [clean_Id[i] for i in eval_fold[iter]]
        eval_clean_len = [clean_len[i] for i in eval_fold[iter]]
        eval_answer_Id = [answer_Id[i] for i in eval_fold[iter]]
        eval_answer_len = [answer_len[i] for i in eval_fold[iter]]

        eval_nemd = [n_emd[i] for i in test_fold[iter]]
        # eval_aemd = [a_emd[i] for i in test_fold[iter]]
        eval_data = [eval_noisy_Id, eval_noisy_len, eval_char_noisy_Id, eval_char_noisy_len, eval_nemd,
                     eval_target_Id,  eval_input_Id, eval_clean_Id, eval_clean_len, eval_answer_Id, eval_answer_len]

        print "the length of training and eval and testing", len(train_noisy_Id), len(test_noisy_Id), len(eval_noisy_Id)

        return train_data, test_data, eval_data, vocab, embd



    #
    # if 0:
    #     # output_path = '/mnt/WDRed4T/ye/DataR/YAHOO/wrongword1_final'
    #     # output_path = '/mnt/WDRed4T/ye/DataR/YAHOO/wrongorder1_final'
    #     output_path = "/mnt/WDRed4T/ye/DataR/YAHOO/threeopt_final"
    #     # output_path = '/mnt/WDRed4T/ye/DataR/YAHOO/back_final'
    #     # output_path = "/mnt/WDRed4T/ye/DataR/HUAWEI/huawei_final"
    #
    #     output = open(output_path, 'wb')
    #     data_com = {"train_data": train_data, "test_data":test_data, "eval_data": eval_data,
    #                 "max_char": max_char, "max_word": max_word}
    #     pickle.dump(data_com, output)
    #     output.close()
    #     print "finished the final dataset"

    # if 1:
    #     # output_path = '/mnt/WDRed4T/ye/DataR/YAHOO/wrongword1_vocab_embd'
    #     # output_path = '/mnt/WDRed4T/ye/DataR/YAHOO/wrongorder1_vocab_embd'
    #     output_path = "/mnt/WDRed4T/ye/DataR/YAHOO/threeopt_vocab_embd"
    #     # output_path = '/mnt/WDRed4T/ye/DataR/YAHOO/back1_vocab_embd'
    #     # output_path = "/mnt/WDRed4T/ye/DataR/HUAWEI/huawei_vocab_embd"
    #     output = open(output_path, 'wb')
    #     print "the vocab size is", len(vocab)
    #     result ={'vocab': vocab, 'embd':embd}
    #     pickle.dump(result, output)
    #     output.close()
    #     print "finished the vocab_embd"


# if __name__ == '__main__':
#     load_data()