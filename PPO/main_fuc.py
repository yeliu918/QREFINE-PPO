import tensorflow as tf
flags = tf.app.flags
from Train import RL_tuning_model
from CharacterBA import run_BA
from QA_similiarity import run_QA
import os
import pprint
pp = pprint.PrettyPrinter()

Dataset_name = "HUAWEI"
Dataset_name = "YAHOO"

Dataset_name1 = "wrongword1"

data_path = "/mnt/WDRed4T/ye/DataR/YAHOO/"
ckp_path = '/mnt/WDRed4T/ye/Qrefine/ckpt/'
'/mnt/WDRed4T/ye/Qrefine/ckpt/QA_sm'

#dataset = "wrongordersmall_"
#dataset = "wrongword_"
# dataset = "wrongorder_"
# dataset = "back_"
# dataset = "threeopt_"
# "wrongorder_", "wrongword_"

## reload the parameter is not correct, because need to use reuse for the weight
for dataset in ["threeopt_"]:
    ##clean all FLAGs
    flags_dict = flags.FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        flags.FLAGS.__delattr__(keys)

    flags.DEFINE_integer("dec_L", 2, "the decade length")
    flags.DEFINE_integer("batch_size", 10, "the batch size")
    flags.DEFINE_integer("epoch", 10, "epoch")
    flags.DEFINE_integer("batches_in_epoch", 100, "the print batch size")
    flags.DEFINE_integer("num_units", 500, "the number of hidden state units")
    flags.DEFINE_integer("char_hidden_units", 100, "the number of hidden state units")
    flags.DEFINE_integer("char_dim", 50, "the number of hidden state units")
    flags.DEFINE_integer("char_num", 44, "the number of chars")
    flags.DEFINE_integer("update_N", 3, "the number of update policy")
    flags.DEFINE_float("learning_rate", 0.1, "the learning rate")

    flags.DEFINE_float("discount_factor", 0.8, "the discount factor of the reward")
    flags.DEFINE_float("lam", 0.9, "the dropout rate")
    flags.DEFINE_float("gamma", 0.9, "the dropout rate")

    flags.DEFINE_float("sen_reward_rate", 0.5, "the rate of sentence reward")
    flags.DEFINE_float("dropout", 0.9, "the dropout rate")
    flags.DEFINE_integer("beam_width", 5, "the size of beam search")
    flags.DEFINE_bool("Bidirection", False, "bidirection")
    flags.DEFINE_bool("Embd_train", False, "Embedding trainable")
    flags.DEFINE_bool("Attention", False, "Attention")
    flags.DEFINE_float("epsilon", 0.2, "The epsilon of clip")
    flags.DEFINE_float("grad_clip", 10.0, "The grad clip")
    flags.DEFINE_float("value_factor", 1, "The value factor")
    flags.DEFINE_float("entropy_factor", 1, "The entropy factor")
    flags.DEFINE_string("PPoscope", "BS_", "The PPO scope")

    flags.DEFINE_string("bert_data_path", "/home/shared/software/uncased_L-12_H-768_A-12/vocab.txt", "The bert_vocab_path")
    flags.DEFINE_string("bert_weight_path", data_path + "bert_w_b", "The bert_weight_bia")
    flags.DEFINE_string("data_dir", data_path + dataset + 'Id', "The processing data file name")
    flags.DEFINE_string("nemd_dir",  data_path + dataset + "bert_noisyemb.npy", "The noisy emd file name")
    flags.DEFINE_string("cemd_dir",  data_path + dataset + "bert_cleanemb.npy", "The clean emd file name")
    flags.DEFINE_string("aemd_dir",  data_path + dataset + "bert_answeremb.npy", "The answer emd file name")

    flags.DEFINE_string("perf_path",  ckp_path + "CharBA/"+ str(dataset) + "/performance_epoch.txt", "The answer emd file name")
    flags.DEFINE_string("main_perf_path",  ckp_path +  "BS/"+ str(dataset) + "/performance_epoch.txt", "The answer emd file name")
    # process_file_name = '/mnt/WDRed4T/ye/DataR/HUAWEI/huawei_Id_830'

    FLAGS = flags.FLAGS
    flags.DEFINE_string("QA_ckp_dir", ckp_path + "QA_sm/" + str(dataset)+ "/qa_similiarity_" + str(FLAGS.num_units), "QA_checkpoint directory")
    flags.DEFINE_string("seq2seq_ckp_dir", ckp_path + "CharBA/"+ str(dataset) + "/CharBA_"+ str(FLAGS.num_units)+"_Bid_"+str(FLAGS.Bidirection) + "_Att_" + str(FLAGS.Attention)+"_EmTrain_"+str(FLAGS.Embd_train)+"_Epoch10", "CharBA_checkpoint directory")
    flags.DEFINE_string("BS_ckp_dir", ckp_path + "BS/"+ str(dataset) + "/BS_"+ str(FLAGS.num_units)+"_Bid_"+str(FLAGS.Bidirection) + "_Att_" + str(FLAGS.Attention)+"_EmTrain_"+str(FLAGS.Embd_train), "CharBA_checkpoint directory")
    flags.DEFINE_string("S2S_ckp_dir", ckp_path + "BS/" + str(dataset) + "/BS_" + str(FLAGS.num_units) + "_Bid_" + str(
        FLAGS.Bidirection) + "_Att_" + str(FLAGS.Attention) + "_EmTrain_" + str(FLAGS.Embd_train),
                        "CharBA_checkpoint directory")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # if 1:
    #     run_QA(data_file= FLAGS.data_dir, emd_file=[FLAGS.nemd_dir, FLAGS.cemd_dir],ckp_dir=FLAGS.QA_ckp_dir)
    if 1:
        model_type = "training"
        run_BA(model_type, config=FLAGS)
    # if 1:
    #     model_type = "training"
    #     run_BA(model_type, config=FLAGS)
    if 1:
        RL_tuning_model(config=FLAGS)

