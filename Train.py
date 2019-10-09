# -*- coding: UTF-8 -*-
import Network
from reward import *
from Agent import *
import parameter as prm
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import gc
from tensorflow.python import pywrap_tensorflow
from Data_loading import load_data
import Build_model
import numpy as np
import tensorflow as tf
import sys
sys.path.append('/home/ye/PycharmProjects/Bert/bert-as-service')
from service.client import BertClient
import QA_similiarity
import CharacterBA
from nltk.translate import bleu
from Evaluation_Package.rouge.rouge import Rouge
from Evaluation_Package.meteor.meteor import Meteor
from Evaluation_Package.bleu.bleu import Bleu
from Evaluation_Package.cider.cider import Cider

SOS = 0
EOS = 1
PAD = 2
batches_in_epoch = 200
epoch_print = 2

Bleu_obj = Bleu()
Rouge_obj = Rouge()
Meteor_obj = Meteor()
cIDER_obj = Cider()

# import os
# print os.getcwd()

def RL_tuning_model(config):
    emd_file = config.nemd_dir, config.cemd_dir
    train_data, test_data, eval_data, vocab, embd = load_data(config.data_dir, emd_file, "BA")
    ##train_Id = [SOS] + clean_Id, ground_truth = clean_Id + [EOS], clean_len = get_length(ground_truth)

    train_noisy_Id, train_noisy_len, train_char_noisy_Id, train_char_noisy_len, train_nemd, train_target_Id, train_input_Id, train_clean_Id, train_clean_len, train_answer_Id, train_answer_len, max_char, max_word = train_data
    # noisy_Id, noisy_len, char_noisy_Id, char_noisy_len, ground_truth, train_Id, clean_Id, clean_len, answer_Id, answer_len, max_char, max_word = train_data
    test_noisy_Id, test_noisy_len, test_char_noisy_Id, test_char_noisy_len,test_nemd, test_ground_truth, test_train_Id, test_clean_Id, test_clean_len, test_answer_Id, test_answer_len = test_data
    # eval_noisy_Id, eval_noisy_len, eval_char_noisy_Id, eval_char_noisy_len,eval_target_Id, eval_input_Id, eval_clean_Id, eval_clean_len, eval_answer_Id, eval_answer_len =eval_data

    # config.max_target_length = max_word

    L = max_word
    embd = np.array(embd)
    batch_size =config.batch_size
    discount_factor = config.discount_factor
    sen_reward_rate = config.sen_reward_rate

    lam = config.lam
    gamma = config.gamma
    beam_width = config.beam_width
    update_N = config.update_N


    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True

    model_name = "BS_"
    values = {}
    checkpoint_path = config.seq2seq_ckp_dir
    BS_ckp_dir = config.BS_ckp_dir
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        if 'loss_fun' not in key:
            values[model_name + key + ':0'] = reader.get_tensor(key)

    vocab_size =len(vocab)

    G_BeamSearch = tf.Graph()
    with G_BeamSearch.as_default():
        BeamSearch_seq2seq = Network.BeamSearch_Seq2seq(config, embd, vocab_size, model_name)

    seq2seq_len = L

    with tf.Session(graph=G_BeamSearch) as sess_beamsearch:
        sess_beamsearch.run(tf.global_variables_initializer())

        for v in tf.trainable_variables():
            if v.name in values.keys():
                v.load(values[v.name], sess_beamsearch)

        for epo in range(config.epoch):
            print(" epoch: {}".format(epo))

            idx = np.arange(0, len(train_noisy_Id))
            idx = list(np.random.permutation(idx))

            Bleu_score1 = []
            Bleu_score2 = []
            Bleu_score3 = []
            Bleu_score4 = []
            Rouge_score = []
            Meteor_score = []
            Cider_score = []


            for batch in range(len(train_noisy_Id) / config.batch_size):

                source_shuffle, source_len, source_nemd, char_Id, char_len, train_shuffle, target_shuffle, target_len, answer_shuffle, answer_len = next_batch(
                    train_noisy_Id,
                    train_noisy_len,
                    train_nemd,
                    train_char_noisy_Id,
                    train_char_noisy_len,
                    train_input_Id,
                    train_target_Id,
                    train_clean_len,
                    train_answer_Id,
                    train_answer_len,
                    batch_size,
                    batch,
                    idx)

                source_shuffle = tf.keras.preprocessing.sequence.pad_sequences(source_shuffle, maxlen=max_word,
                                                                               padding='post', truncating="post",
                                                                               value=EOS)
                source_emd = []
                for n in source_nemd:
                    source_emd.append(n[0:source_shuffle.shape[1]])
                source_emd = np.asarray(source_emd)

                target_shuffle_in = tf.keras.preprocessing.sequence.pad_sequences(target_shuffle, maxlen=seq2seq_len,
                                                                               padding='post', truncating="post",
                                                                               value=EOS)
                # target_emd = []
                # for c in train_cemd:
                #     target_emd.append(c[0:train_shuffle.shape[1]])
                # target_emd = np.asarray(target_emd)

                train_shuffle = tf.keras.preprocessing.sequence.pad_sequences(train_shuffle, maxlen=seq2seq_len,
                                                                              padding='post', truncating="post",
                                                                              value=EOS)

                whole_noisy_char_Id = []
                for sen_char in char_Id:
                    sen_len = len(sen_char)
                    for i in range(len(source_shuffle[0]) - sen_len):
                        sen_char.append([0] * max_char)  ## fix the char with the length of max_word
                    whole_noisy_char_Id.append(sen_char)

                whole_noisy_char_Id = np.asarray(whole_noisy_char_Id)

                whole_char_len = []
                for char_List in char_len:
                    whole_char_len.append(char_List[:len(source_shuffle[0])])

                # initial_input_in = [target_shuffle_in[i][-1] for i in range(config.batch_size)]

                target_len = np.tile(seq2seq_len, config.batch_size)

                # print "the shape of source_shuffle ", np.array(source_shuffle).shape
                # print "the shape of source_len ", np.array(source_len).shape
                # print "the shape of source_emd ", np.array(source_emd).shape
                # print "the shape of whole_noisy_char_Id ", np.array(whole_noisy_char_Id).shape
                # print "the shape of whole_char_len ", np.array(whole_char_len).shape
                # print "the shape of target_len ", np.array(target_len).shape
                # print "the shape of train_shuffle ", np.array(train_shuffle).shape
                # print "the shape of target_shuffle_in ", np.array(target_shuffle_in).shape

                fd = {BeamSearch_seq2seq.encoder_inputs: source_shuffle,
                      BeamSearch_seq2seq.encoder_inputs_length: source_len,
                      BeamSearch_seq2seq.encoder_emb: source_emd,
                      BeamSearch_seq2seq.encoder_char_ids: whole_noisy_char_Id, BeamSearch_seq2seq.encoder_char_len: whole_char_len,
                      BeamSearch_seq2seq.decoder_length: target_len,
                      BeamSearch_seq2seq.decoder_inputs: train_shuffle,
                      # BeamSearch_seq2seq.target_emb: target_emd,
                      BeamSearch_seq2seq.decoder_targets: target_shuffle_in,
                      }
                if batch <= 100:
                    ids, _, loss_seq2seq = sess_beamsearch.run(
                        [BeamSearch_seq2seq.seq_ids, BeamSearch_seq2seq.train_op,BeamSearch_seq2seq.loss_seq2seq], fd)
                    if batch % 100 == 0:
                        print ('Batch {} Seq2seq_loss: '.format(batch),loss_seq2seq)

                    # noisy_sen = []
                    # for pre_id in source_shuffle[0]:
                    #     if pre_id != EOS and pre_id != PAD and pre_id != SOS:
                    #         noisy_sen.append(vocab[pre_id])
                    #
                    # clean_sen = []
                    # for clean_id in target_shuffle[0]:
                    #     if clean_id != EOS and clean_id != PAD and clean_id != SOS:
                    #         clean_sen.append(vocab[clean_id])
                    #
                    # gene_sen = []
                    # for gen_id in ids[0]:
                    #     if gen_id != EOS and gen_id != PAD and gen_id != SOS:
                    #         gene_sen.append(vocab[gen_id])
                    #
                    # print("question" + str(1) + "\n");
                    # print("noisy question: " + " ".join(noisy_sen) + "\n")
                    # print("clean question: " + " ".join(clean_sen) + "\n")
                    # print("generated question: " + " ".join(gene_sen) + "\n")

                elif batch > 100:

                    RL_logits, policy, values = sess_beamsearch.run([BeamSearch_seq2seq.predicting_logits, BeamSearch_seq2seq.predicting_scores, BeamSearch_seq2seq.value], fd)

                    # generated_que_input = np.insert(generated_que, 0, SOS, axis=1)[:, 0:-1]
                    # target_shuffle = tf.keras.preprocessing.sequence.pad_sequences(target_shuffle, maxlen=None,
                    #                                                                padding='post',truncating="post",  value=EOS)

                    generated_target_len = RL_logits.shape[1]

                    bc = BertClient(port=5555, port_out=5556)

                    generated_sen_beam = []
                    for beam in range(beam_width):
                        for b in range(batch_size):
                            sen = []
                            for l in range(RL_logits.shape[1]):
                                sen.append(vocab[int(RL_logits[b][l][beam])])
                            generated_sen_beam.append(" ".join(sen))

                    bert_emd_beam = bc.encode(generated_sen_beam)
                    # bert_emd =
                    # bert_emd_beam.append(bert_emd[:,:generated_target_len,:])
                    bert_emd_beam = bert_emd_beam[:,1:generated_target_len+1,:]
                    bert_emd_beam = bert_emd_beam.reshape(batch_size, beam_width, bert_emd_beam.shape[1], bert_emd_beam.shape[2])
                    sen_len_batch_beam = generated_target_len * np.ones((batch_size,beam_width))
                    # print "the shape of bert emd: ", bert_emd_beam.shape
                    #
                    #
                    # source_shuffle_beam = np.tile(source_shuffle, (beam_width, 1))
                    # source_len_beam = np.tile(source_len, (beam_width))
                    # source_emd_beam = np.tile(source_emd, (beam_width,1,1))
                    # print "the shape of source_emd_beam", source_emd_beam.shape
                    # whole_noisy_char_Id_beam = np.tile(whole_noisy_char_Id,(beam_width,1,1))
                    # whole_char_len_beam = np.tile(whole_char_len,(beam_width,1))
                    # print "the char_len: ", whole_char_len_beam.shape
                    # RL_logits_beam = np.transpose(RL_logits,(0,2,1)).reshape(batch_size*beam_width, generated_target_len)
                    #
                    # answer_shuffle_beam = np.tile(answer_shuffle,(beam_width,1))
                    # answer_len_beam = np.tile(answer_len, (beam_width))
                    #
                    # print "the shape of source_shuffle: ", source_shuffle_beam.shape
                    # print "the input of target: ", RL_logits_beam.shape
                    #
                    # # for beam in range(beam_width):
                    # #     sen_len_batch_beam = []
                    # #     for b in range(batch_size):
                    # #         sen_len_batch =[]
                    # #         l=0
                    # #         while (l < generated_target_len and RL_logits[b][l][beam] != 1):
                    # #             l = l + 1
                    # #         sen_len_batch.append(l)
                    # #     sen_len_batch_beam.append(sen_len_batch)
                    #
                    # Character_data= source_shuffle_beam,source_len_beam,source_emd_beam, whole_noisy_char_Id_beam, whole_char_len_beam, RL_logits_beam, sen_len_batch_beam, generated_target_len, bert_emd_beam
                    # QA_data = answer_shuffle_beam, answer_len_beam, source_shuffle_beam, source_len_beam, source_emd_beam, RL_logits_beam, sen_len_batch_beam, bert_emd_beam, vocab

                    Character_data = source_shuffle, source_len, source_emd, whole_noisy_char_Id, whole_char_len, RL_logits, sen_len_batch_beam, generated_target_len, bert_emd_beam
                    QA_data = answer_shuffle, answer_len, source_shuffle, source_len, source_emd, RL_logits, sen_len_batch_beam, bert_emd_beam, vocab

                    RL_rewards = Reward(config, Character_data, QA_data, vocab, embd)

                    values_t = np.transpose(values,(2,0,1))
                    values_t1 = np.transpose(values[:,1:,:],(2,0,1))
                    values_t1 = np.insert(values_t1, values_t1.shape[2], values=np.zeros(batch_size), axis=2)

                    Returns_beam =[]
                    Deltas_beam = []
                    for beam in range(len(RL_rewards)):
                        Returns_batch =[]
                        Deltas_batch = []
                        for batch_len in range(len(RL_rewards[1])):
                            returns, deltas = get_training_data(RL_rewards[beam][batch_len], values_t[beam][batch_len], values_t1[beam][batch_len], lam, gamma)
                            Returns_batch.append(returns)
                            Deltas_batch.append(deltas)
                        Returns_beam.append(Returns_batch)
                        Deltas_beam.append(Deltas_batch)

                    Returns_beam = np.transpose(Returns_beam,(1,2,0))
                    Deltas_beam = np.transpose(Deltas_beam,(1,2,0))

                    old_policy = policy
                    for N in range(update_N):
                        fd = {BeamSearch_seq2seq.returns_ph: Returns_beam, BeamSearch_seq2seq.advantages_ph: Deltas_beam,
                              BeamSearch_seq2seq.old_log_probs_ph: old_policy,
                              BeamSearch_seq2seq.encoder_inputs: source_shuffle,
                              BeamSearch_seq2seq.encoder_inputs_length: source_len,
                              BeamSearch_seq2seq.encoder_emb: source_emd,
                              BeamSearch_seq2seq.encoder_char_ids: whole_noisy_char_Id,
                              BeamSearch_seq2seq.encoder_char_len: whole_char_len,
                              BeamSearch_seq2seq.decoder_length: target_len,
                              BeamSearch_seq2seq.decoder_inputs: train_shuffle,
                              BeamSearch_seq2seq.decoder_targets: target_shuffle_in}
                        policy_l, entr_l, loss_ppo, ratio, _, policy= sess_beamsearch.run(
                            [BeamSearch_seq2seq.policy_loss, BeamSearch_seq2seq.entropy,
                              BeamSearch_seq2seq.loss, BeamSearch_seq2seq.ratio, BeamSearch_seq2seq.optimize_expr, BeamSearch_seq2seq.predicting_scores], fd)
                    print ('Batch {} PPO_loss: '.format(batch),loss_ppo)
                    # print (" The ratio is :", ratio)

                # for t in range(batch_size):
                #     ref = []
                #     hyp = []
                #     for tar_id in target_shuffle[t]:
                #         if tar_id != EOS and tar_id != PAD:
                #             ref.append(vocab[tar_id])
                #             # print vocab[tar_id]
                #     for pre_id in RL_logits[t]:
                #         if pre_id != EOS and pre_id != PAD:
                #             hyp.append(vocab[pre_id])
                #
                #     hyp_sen = u" ".join(hyp).encode('utf-8')
                #     ref_sen = u" ".join(ref).encode('utf-8')
                #     dic_hyp = {}
                #     dic_hyp[0] = [hyp_sen]
                #     dic_ref = {}
                #     dic_ref[0] = [ref_sen]
                #     sen_bleu, _ = Bleu_obj.compute_score(dic_ref, dic_hyp)
                #     sen_rouge = Rouge_obj.compute_score(dic_ref, dic_hyp)
                #     sen_meteor, _ = Meteor_obj.compute_score(dic_ref, dic_hyp)
                #     sen_cider, _ = cIDER_obj.compute_score(dic_ref, dic_hyp)
                #     Bleu_score1.append(sen_bleu[0])
                #     Bleu_score2.append(sen_bleu[1])
                #     Bleu_score3.append(sen_bleu[2])
                #     Bleu_score4.append(sen_bleu[3])
                #     Rouge_score.append(sen_rouge[0])
                #     Meteor_score.append(sen_meteor)
                #     Cider_score.append(sen_cider)
                #
                # if batch == 0 or batch % 100 == 0:
                #     print(' batch {}'.format(batch))
                #     print('   minibatch reward of training: {}'.format(- loss_ppo))
                #     #print the result
                #     for t in range(5):  ## five sentences
                #         print('Question {}'.format(t))
                #         print("noisy question:")
                #         print(" ".join(map(lambda i: vocab[i], list(source_shuffle[t]))).strip().replace("<EOS>"," "))
                #         print("clean question:")
                #         print(" ".join(map(lambda i: vocab[i], list(target_shuffle[t]))).strip().replace("<EOS>"," "))
                #         print("the generated first question:")
                #         pre_sen = []
                #         for log_id in RL_logits[t, :, 0]:
                #             if log_id != -1 and log_id!=2 and log_id!=0:
                #                 pre_sen.append(log_id)
                #         print(" ".join(map(lambda i: vocab[i], pre_sen)).strip())
                #         print("\n")
                #     print("\n Bleu_score1, Bleu_score2, Bleu_score3, Bleu_score4, Rouge,  Cider\n")
                #     print("   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f}".format(
                #         sum(Bleu_score1) / float(len(Bleu_score1)),
                #         sum(Bleu_score2) / float(len(Bleu_score2)),
                #         sum(Bleu_score3) / float(len(Bleu_score3)),
                #         sum(Bleu_score4) / float(len(Bleu_score4)),
                #         sum(Rouge_score) / float(len(Rouge_score)),
                #         sum(Meteor_score) / float(len(Meteor_score)),
                #         sum(Cider_score) / float(len(Cider_score))
                #     ))
                gc.collect()

                if (batch % 100 == 0 and batch < 100) or (batch % 20 == 0 and batch >100):
                    generated_test_sen = []
                    test_answer_sen = []
                    test_noisy_sen = []
                    test_clean_sen = []
                    # saver.save(sess_beamsearch, BS_ckp_dir)
                    ##testing set result:

                    for batch_test in range(len(test_noisy_Id) / batch_size):
                        idx_t = np.arange(0, len(test_noisy_Id))
                        idx_t = list(np.random.permutation(idx_t))


                        test_source_shuffle, test_source_len, test_source_nemd, test_char_Id, test_char_len_in, test_train_shuffle, test_target_shuffle, test_target_len, test_answer_shuffle, test_ans_len = next_batch(
                            test_noisy_Id, test_noisy_len, test_nemd, test_char_noisy_Id, test_char_noisy_len, test_train_Id, test_ground_truth, test_clean_len, test_answer_Id,
                        test_answer_len, batch_size, batch_test, idx_t)

                        for an in test_answer_shuffle:
                            test_answer_sen.append(vocab[anum] for anum in an)
                        for no in test_source_shuffle:
                            test_noisy_sen.append(vocab[si] for si in no)
                        for cl in test_target_shuffle:
                            test_clean_sen.append(vocab[ci] for ci in cl)

                        test_source_shuffle_in = tf.keras.preprocessing.sequence.pad_sequences(test_source_shuffle, maxlen = max_word,
                                                                                       padding='post', truncating="post",value=EOS)

                        test_source_emd = []
                        for n in test_source_nemd:
                            test_source_emd.append(n[0 : test_source_shuffle_in.shape[1]])
                        test_source_emd = np.asarray(test_source_emd)

                        val_target_shuffle_in=[]
                        for val in test_target_shuffle:
                            val_target_shuffle_in.append([val[0]])

                        val_train_shuffle_in = []
                        for tra in test_train_shuffle:
                            val_train_shuffle_in.append([tra[0]])

                        initial_input_in = [SOS for i in range(batch_size)]
                        # [SOS for i in range(batch_size)]

                        val_target_len_in = np.tile(1, batch_size)

                        test_batch_noisy_char_Id = []
                        for sen_char in test_char_Id:
                            sen_len = len(sen_char)
                            for i in range(len(test_source_shuffle_in[0]) - sen_len):
                                sen_char.append([0] * max_char)  ## fix the char with the length of max_word
                            test_batch_noisy_char_Id.append(sen_char)
                        test_batch_noisy_char_Id = np.asarray(test_batch_noisy_char_Id)

                        test_char_len = []
                        for char_List in test_char_len_in:
                            test_char_len.append(char_List[:len(test_source_shuffle_in[0])])
                        # print "the shape of test_noisy_char_Id ", np.array(test_batch_noisy_char_Id).shape
                        # if np.array(test_batch_noisy_char_Id).shape == (batch_size,):
                        #     continue
                        # print "the shape of test_char_len ", np.array(test_char_len).shape

                        # # print "the shape of source_shuffle ", np.array(test_source_shuffle_in).shape
                        # # print "the shape of source_len ", np.array(test_source_len).shape
                        # # print "the shape of source_emd ", np.array(test_source_emd).shape
                        # # print "the shape of whole_noisy_char_Id ", np.array(test_batch_noisy_char_Id).shape
                        # # print "the shape of whole_char_len ", np.array(test_char_len).shape

                        fd = {BeamSearch_seq2seq.encoder_inputs: test_source_shuffle_in,
                              BeamSearch_seq2seq.encoder_inputs_length: test_source_len,
                              BeamSearch_seq2seq.encoder_emb: test_source_emd,
                              BeamSearch_seq2seq.encoder_char_ids: test_batch_noisy_char_Id,
                              BeamSearch_seq2seq.encoder_char_len: test_char_len}

                        val_id = sess_beamsearch.run([BeamSearch_seq2seq.ids], fd)

                        for t in range(batch_size):
                            ref = []
                            hyp = []
                            for tar_id in test_target_shuffle[t]:
                                if tar_id != EOS and tar_id != PAD and tar_id != SOS:
                                    ref.append(vocab[tar_id])
                                    # print vocab[tar_id]
                            for pre_id in val_id[0][t]:
                                if pre_id != EOS and pre_id != PAD and pre_id != SOS:
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
                            Bleu_score1.append(sen_bleu[0])
                            Bleu_score2.append(sen_bleu[1])
                            Bleu_score3.append(sen_bleu[2])
                            Bleu_score4.append(sen_bleu[3])
                            Rouge_score.append(sen_rouge[0])
                            Meteor_score.append(sen_meteor)
                            Cider_score.append(sen_cider)

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
                    bleu_score = sum(Bleu_score1) / float(len(Bleu_score1))

                    for i in range(len(generated_test_sen[0:2])):
                        print("question"+str(i) + "\n")
                        print("answer: "+" ".join(test_answer_sen[i]) + "\n")
                        print("noisy question: " + " ".join(test_noisy_sen[i]) + "\n")
                        print("clean question: " + " ".join(test_clean_sen[i]) + "\n")
                        print("generated question: " + " ".join(generated_test_sen[i]) + "\n")

                    fname = BS_ckp_dir +str(bleu_score)
                    f=open(fname, "wb")
                    f.write("\n Bleu_score1, Bleu_score2, Bleu_score3, Bleu_score4, Rouge, Meteor  Cider\n")
                    f.write("   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f},   {:.4f} \n".format(
                        sum(Bleu_score1) / float(len(Bleu_score1)),
                        sum(Bleu_score2) / float(len(Bleu_score2)),
                        sum(Bleu_score3) / float(len(Bleu_score3)),
                        sum(Bleu_score4) / float(len(Bleu_score4)),
                        sum(Rouge_score) / float(len(Rouge_score)),
                        sum(Meteor_score) / float(len(Meteor_score)),
                        sum(Cider_score) / float(len(Cider_score))
                    ))
                    for i in range(len(generated_test_sen)):
                        f.write("question"+str(i) + "\n")
                        f.write("answer: "+" ".join(test_answer_sen[i]) + "\n")
                        f.write("noisy question: " + " ".join(test_noisy_sen[i]) + "\n")
                        f.write("clean question: " + " ".join(test_clean_sen[i]) + "\n")
                        f.write("generated question: " + " ".join(generated_test_sen[i]) + "\n")
                    network_vars = tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES, config.PPoscope)
                    saver = tf.train.Saver(sharded=False)
                    saver.save(sess_beamsearch, config.BS_ckp_dir)

# if __name__ == '__main__':
#     RL_tuning_model()


# for epo in range(epoch):
#     for batch in range(max_batches):
#         deltas, returns = get_training_data()
#         deltas =
#         returns =
#         ratio =
#         cur_policy =
#         for prm_epo in range(K):
#             actions_ph: actions,
#             returns_ph: returns,
#             advantages_ph: advantages,
#             old_log_probs_ph: log_probs,
#             fd={deltas, returns, cur_policy}
