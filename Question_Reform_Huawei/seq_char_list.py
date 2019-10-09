import numpy as np
import tensorflow as tf
import pickle
from tensorflow.contrib import keras
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple
from sklearn.model_selection import train_test_split

# rollout buffer
state_buffer = []
reward_buffer = []
action_buffer = []


class BeamSearch_Seq2seq(object):
    def __init__(self, vocab_size, num_units, beam_width, model_name, embd, Bidirection, Embd_train, char_hidden_units, char_num, char_dim, batch_size,
                 Attention, max_target_length, sen_reward_rate=0.1,
                 learning_rate=0.001):
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.beam_width = beam_width
        self.model_name = model_name
        self.embedding = embd
        self.sen_reward_rate = sen_reward_rate
        self.learning_rate = learning_rate
        self.Bidirection = Bidirection
        self.embd_train = Embd_train
        self.Attention = Attention
        self.max_target_length = max_target_length
        self.batch_size = batch_size
        self.char_num = char_num
        self.char_dim = char_dim
        self.char_hidden_units = char_hidden_units
        self.build_graph()

    def build_graph(self):
        # input placehodlers
        with tf.variable_scope(self.model_name + "model_inputs"):
            self.encoder_inputs = tf.placeholder(shape=(None, None),
                                                 dtype=tf.int32, name='encoder_inputs')
            self.encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32,
                                                        name='encoder_inputs_length')

            self.encoder_char_ids = tf.placeholder(shape=(self.batch_size, None, None), dtype=tf.int32,
                                                   name="encoder_char_ids")
            self.encoder_char_len = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32,
                                                   name="encoder_char_length")  ##batch * word num * char num

            self.decoder_targets = tf.placeholder(shape=(None, None),
                                                  dtype=tf.int32, name='decoder_targets')
            self.decoder_inputs = tf.placeholder(shape=(None, None),
                                                 dtype=tf.int32, name='decoder_targets')
            self.decoder_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_length')
            self.initial_input = tf.placeholder(shape=(None,), dtype=tf.int32, name="infer_initial_input")

        with tf.variable_scope(self.model_name + "embeddings"):
            self.embeddings = tf.get_variable(name="embedding_W", shape=self.embedding.shape,
                                              initializer=tf.constant_initializer(self.embedding),
                                              trainable=self.embd_train)
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)

            _char_embeddings = tf.get_variable(name="_char_embeddings", dtype=tf.float32,
                                               shape=[self.char_num, self.char_dim])  ## char num 44, char dim? 50

            self.char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.encoder_char_ids,
                                                          name="char_embeddings")

            self.decoder_outputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)

        self.encoder()
        self.training()
        self.RL_reward()
        self.Seq_loss()

    def encoder(self):
        ####Encoder
        with tf.variable_scope(self.model_name + "encoder_model"):

            s = tf.shape(self.char_embeddings)  # batch * word * char * embedding
            char_embeddings = tf.reshape(self.char_embeddings,
                                         shape=[s[0] * s[1], s[-2], self.char_dim])
            word_lengths = tf.reshape(self.encoder_char_len, shape=[s[0] * s[1]])

            # bi lstm on chars
            cell_fw = tf.contrib.rnn.LSTMCell(self.char_hidden_units,
                                              state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self.char_hidden_units,
                                              state_is_tuple=True)
            _output = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, char_embeddings,
                sequence_length=word_lengths, dtype=tf.float32)

            # read and concat output
            _, ((_, output_fw), (_, output_bw)) = _output
            output = tf.concat([output_fw, output_bw], axis=-1)

            # shape = (batch size, max sentence length, char hidden size)
            output = tf.reshape(output,
                                shape=[s[0], s[1], 2 * self.char_hidden_units])
            self.word_embeddings_cat = tf.concat([self.encoder_inputs_embedded, output], axis=-1)

            if self.Bidirection == False:
                encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units)
                self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                                                   inputs=self.encoder_inputs_embedded,
                                                                                   sequence_length=self.encoder_inputs_length,
                                                                                   time_major=False,
                                                                                   dtype=tf.float32)
                self.hidden_units = self.num_units

            elif self.Bidirection == True:
                encoder_cell_fw = LSTMCell(self.num_units)
                encoder_cell_bw = LSTMCell(self.num_units)
                ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)) = (
                    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_fw, cell_bw=encoder_cell_bw,
                                                    inputs=self.encoder_inputs_embedded,
                                                    sequence_length=self.encoder_inputs_length,
                                                    dtype=tf.float32, time_major=False))
                # Concatenates tensors along one dimension.
                encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

                encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
                encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

                # TF Tuple used by LSTM Cells for state_size, zero_state, and output state.
                self.encoder_final_state = LSTMStateTuple(c=encoder_final_state_c, h=encoder_final_state_h)
                self.hidden_units = 2 * self.num_units

                # TRAINING DECODER

    def training(self):
        with tf.variable_scope(self.model_name + "decoder_model"):
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_units, reuse=tf.AUTO_REUSE)
            # attention_states: [self.batch_size, max_time, self.num_units]
            # attention_states = tf.transpose(encoder_outputs, [0, 1, 2])
            self.initial_state = self.encoder_final_state

            if self.Attention == True:
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    self.hidden_units, self.encoder_outputs,
                    memory_sequence_length=self.encoder_inputs_length)

                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    decoder_cell, attention_mechanism,
                    attention_layer_size=self.hidden_units)

                zero_state = decoder_cell.zero_state(tf.shape(self.encoder_inputs)[0], tf.float32)

                self.initial_state = zero_state.clone(cell_state=self.encoder_final_state)

            # if self.model_cond == 'training':
            helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_outputs_embedded, self.decoder_length,
                                                       time_major=False)

            output_layer = tf.layers.Dense(self.vocab_size)
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper,
                                                      initial_state=self.initial_state,
                                                      output_layer=output_layer)
            # Dynamic decoding
            # if self.model_cond == "training":
            outputs, s2s_final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False,
                                                                            impute_finished=True,
                                                                            maximum_iterations=tf.shape(self.decoder_targets)[1])

            self.logits = tf.identity(outputs.rnn_output)
            self.ids = tf.identity(outputs.sample_id)
            self.softmax_logits = tf.nn.softmax(self.logits, dim=2)

            Infere_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embeddings,
                                                                     start_tokens=self.initial_input,  ##self.logits.lastid
                                                                     end_token=2)

            initial_state = tf.cond(tf.cast(tf.reduce_sum(self.decoder_length) == 1, tf.bool), lambda: s2s_final_state, lambda: self.initial_state)

            #initial_state2 = s2s_final_state

            decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=Infere_helper,
                                                      initial_state=initial_state,
                                                      output_layer=output_layer)

            outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False,
                                                                        impute_finished=True,
                                                                        maximum_iterations=self.max_target_length - tf.shape(self.decoder_targets)[1])
            self.RL_ids = tf.identity(outputs.sample_id)
            self.policy = tf.identity(outputs.rnn_output)  ##do we need to get max
            self.softmax_policy = tf.nn.softmax(self.policy, dim=2)
            self.max_policy = tf.reduce_max(self.softmax_policy, axis = 2)

    def RL_reward(self):
        # masks = tf.sequence_mask(self.decoder_length, self.max_target_sequence_length, dtype=tf.float32)
        # self.beamsearch_loss = tf.contrib.seq2seq.sequence_loss(logits=self.predicting_logits,
        #                                                         targets=self.decoder_targets, weights=masks)
        # self.train_op = tf.train.AdamOptimizer().minimize(self.beamsearch_loss)
        # word_log_prob = self.predicting_scores  # tf.placeholder(shape=[batch_size, max_target_length, beam_width], dtype=tf.float32, name='log_prob')
        self.word_log_prob = tf.log(self.max_policy)
        self.dis_rewards = tf.placeholder(shape=[None, None], dtype=tf.float32,
                                          name='discounted_rewards')
        # add one for sentence
        # padding = self.sen_reward_rate * tf.ones(
        #     [tf.shape(word_log_prob)[0], 1, tf.shape(word_log_prob)[2]])
        # whole_log_pro = tf.concat([padding, word_log_prob], 1)

        #TODO: ADD the regularization term
        cross_entropy = self.word_log_prob * self.dis_rewards
        entropy = tf.log(self.policy) * self.policy
        self.rl_reward = - tf.reduce_mean(cross_entropy) - tf.reduce_mean(entropy)
        self.RL_train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.rl_reward)

    def Seq_loss(self):
        # padding = -1 * tf.ones(
        #     [tf.shape(self.logits)[0], self.max_target_sequence_length - tf.shape(self.logits)[1],
        #      tf.shape(self.logits)[2]])
        self.loss_seq2seq = tf.contrib.seq2seq.sequence_loss(logits=self.logits,
                                                             targets=self.decoder_targets, weights=tf.cast(
                tf.sequence_mask(tf.fill([tf.shape(self.logits)[0]], tf.shape(self.logits)[1]),
                                 tf.shape(self.logits)[1]), tf.float32))
        self.loss_seq2seq = tf.reduce_mean(self.loss_seq2seq)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss_seq2seq)


        # for i in range(self.beam_width):
        #     padding_decoder = -1 * tf.ones([tf.shape(self.decoder_targets)[0], 2 * tf.reduce_max(self.encoder_inputs_length, axis=0) - tf.shape(self.decoder_targets)[1]])
        #     self.decoder_targets = tf.concat([self.decoder_targets, tf.cast(padding_decoder,tf.int32)],1)
        #     padding = -1 * tf.ones([tf.shape(self.predicting_logits[:,:,i])[0], 2 * tf.reduce_max(self.encoder_inputs_length, axis=0) - tf.shape(self.predicting_logits[:,:,i])[1]],dtype=tf.int32)
        #     self.predicting_logits[:,:,i] = tf.concat([self.predicting_logits[:,:,i], padding],1)
        #
        #     self.beamsearch_loss[i] = tf.nn.softmax_cross_entropy_with_logits(
        #         labels=tf.one_hot(self.predicting_logits[:,:,i], depth=self.vocab_size, dtype=tf.float32),
        #         logits=self.decoder_targets)
        # self.sum_bs_loss = tf.reduce_sum(self.beamsearch_loss)
        # self.Seq_train_op = tf.train.AdamOptimizer().minimize(self.sum_bs_loss)



        # masks = tf.sequence_mask(self.decoder_length, tf.shape(self.predicting_logits[:,:,i])[1], dtype=tf.float32)
        # self.beamsearch_loss[i] = tf.contrib.seq2seq.sequence_loss(logits=self.predicting_logits[:,:,i],
        #                                                            targets=self.target, weights=masks)
