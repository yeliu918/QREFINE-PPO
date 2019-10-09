import tensorflow as tf
import QR_tuner
import Data_loading

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("epoch", 3, "the epoch number")
tf.app.flags.DEFINE_integer("batch_size", 100, "the batch size")
tf.app.flags.DEFINE_integer("num_units", 200, "the number of hidden state units")
tf.app.flags.DEFINE_float("discount_factor", 0.1, "the discount factor of the reward")
tf.app.flags.DEFINE_float("sen_reward_rate", 2, "the rate of sentence reward")
tf.app.flags.DEFINE_integer("beam_width", 5, "the size of beam search")
tf.app.flags.DEFINE_bool("Bidirection", False, "bidirection")
tf.app.flags.DEFINE_bool("Embd_train", False, "Embedding trainable")
tf.app.flags.DEFINE_bool("Attention", False, "Attention")
tf.app.flags.DEFINE_string("Data_FileName", "../result/Huawei/large_unique0.95_with", "The processing data file name")


train_data_com, val_data_com, test_data_com, max_data = Data_loading.load_data(FLAGS.Data_FileName)


data_comb = [train_data_com, val_data_com, max_data]
for reward_rate in range(1, 5):
    print "the reward rate is:", reward_rate
    FLAGS.sen_reward_rate = reward_rate
    QR_tuner.RL_tuning_model(
        data_comb=data_comb,
        epoch=FLAGS.epoch,
        batch_size=FLAGS.batch_size,
        num_units=FLAGS.num_units,
        beam_width=FLAGS.beam_width,
        discount_factor=FLAGS.discount_factor,
        sen_reward_rate=FLAGS.sen_reward_rate,
        Bidirection=FLAGS.Bidirection,
        Embd_train=FLAGS.Embd_train,
        Attention=FLAGS.Attention)


    # print "starting Seq2Seq training_" + str(Seq2seq_loop)+"_loop"
    # model_cond = "training"
    # Bi_Atten_Seq2Seq.run_graph(
    #              data_comb=train_data_com,
    #              model_cond =model_cond,
    #              batch_size=FLAGS.batch_size,
    #              epoch_seq = Seq2seq_loop,
    #              num_units=FLAGS.num_units,
    #              Bidirection = FLAGS.Bidirection,
    #              Embd_train = FLAGS.Embd_train,
    #              Attention= FLAGS.Attention)
    #
    # print "starting Seq2Seq training_" + str(RL_loop) + "_loop"
