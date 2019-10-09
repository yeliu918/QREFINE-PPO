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
    def __init__(self, vocab_size, num_units, beam_width, model_name, embd, Bidirection, Embd_train,
                 Attention, sen_reward_rate=0.1,
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
        self.build_graph()

    def build_graph(self):
        # input placehodlers
        with tf.variable_scope(self.model_name + "model_inputs"):
            self.encoder_inputs = tf.placeholder(shape=(None, None),
                                                 dtype=tf.int32, name='encoder_inputs')
            self.encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32,
                                                        name='encoder_inputs_length')
            self.decoder_targets = tf.placeholder(shape=(None, None),
                                                  dtype=tf.int32, name='decoder_targets')
            self.decoder_inputs = tf.placeholder(shape=(None, None),
                                                 dtype=tf.int32, name='decoder_targets')
            self.decoder_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_length')

        with tf.variable_scope(self.model_name + "embeddings"):
            self.embeddings = tf.get_variable(name="embedding_W", shape=self.embedding.shape,
                                              initializer=tf.constant_initializer(self.embedding),
                                              trainable=self.embd_train)
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
            self.decoder_outputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)

        self.encoder()
        self.training()
        self.RL_reward()
        self.Seq_loss()


    def encoder(self):
        ####Encoder
        with tf.variable_scope(self.model_name + "encoder_model"):
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
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_units,reuse=tf.AUTO_REUSE)
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
            # elif self.model_cond == 'testing':
            #     helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embeddings,
            #                                                       start_tokens=tf.fill([self.batch_size], 1),
            #                                                       end_token=2)
            output_layer = tf.layers.Dense(self.vocab_size)
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper,
                                                      initial_state=self.initial_state,
                                                      output_layer= output_layer)
            # Dynamic decoding
            # if self.model_cond == "training":
            outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False,
                                                                        impute_finished=True,
                                                                        maximum_iterations=self.decoder_length[0])
            # elif self.model_cond == "testing":
            #     outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False,
            #                                                                 impute_finished=True,
            #                                                                 maximum_iterations=2 * tf.reduce_max(
            #                                                                     self.encoder_inputs_length, axis=0))

            self.logits = tf.identity(outputs.rnn_output)
            self.ids = tf.identity(outputs.sample_id)
            self.softmax_logits = tf.nn.softmax(self.logits, dim=2)

    #     # PREDICTING_DECODER
    #     ## Beam Search Decoder
    # def inference(self):
    #     with tf.variable_scope(self.model_name + "decoder_model"):

            self.initial_state_beam = tf.contrib.seq2seq.tile_batch(self.encoder_final_state, self.beam_width)

            # attention_states: [batch_size, max_time, num_units]
            # attention_states = tf.transpose(encoder_outputs, [0, 1, 2])
            if self.Attention == True:
                enc_rnn_out_beam = tf.contrib.seq2seq.tile_batch(self.encoder_outputs, self.beam_width)
                encoder_length_beam = tf.contrib.seq2seq.tile_batch(self.encoder_inputs_length, self.beam_width)
                self.encoder_final_state_beam = tf.contrib.seq2seq.tile_batch(self.encoder_final_state,
                                                                              multiplier=self.beam_width)

                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    self.hidden_units, memory=enc_rnn_out_beam,
                    memory_sequence_length=encoder_length_beam)

                # attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=2 * self.num_units, memory=enc_rnn_out_beam, normalize=True)

                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    decoder_cell, attention_mechanism,
                    attention_layer_size=self.hidden_units)

                self.initial_state_beam = decoder_cell.zero_state(batch_size=tf.shape(self.encoder_inputs)[0] * self.beam_width,
                                                                  dtype=tf.float32).clone(
                    cell_state=self.encoder_final_state_beam)

            # https://github.com/tensorflow/tensorflow/issues/11598
            predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell, embedding=self.embeddings,
                start_tokens=tf.fill([tf.shape(self.encoder_inputs)[0]], 1), end_token=2,
                #  tf.tile(tf.constant([1], dtype=tf.int32), [self.batch_size])
                initial_state=self.initial_state_beam, beam_width=self.beam_width,
                output_layer= output_layer, length_penalty_weight=0.0)
            # outputs, next_state, next_inputs=predicting_decoder.step(time=,inputs=,state=)
            # tf.contrib.seq2seq.BeamSearchDecoderState

            predicting_decoder_final_output, beam_search_final_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=predicting_decoder,
                impute_finished=False,
                maximum_iterations=2 * tf.reduce_max(self.encoder_inputs_length, axis=0))
            self.predicting_logits = predicting_decoder_final_output.predicted_ids
            self.predicting_scores = predicting_decoder_final_output.beam_search_decoder_output.scores
            self.log_pro = beam_search_final_state.log_probs

    def RL_reward(self):
        # masks = tf.sequence_mask(self.decoder_length, self.max_target_sequence_length, dtype=tf.float32)
        # self.beamsearch_loss = tf.contrib.seq2seq.sequence_loss(logits=self.predicting_logits,
        #                                                         targets=self.decoder_targets, weights=masks)
        # self.train_op = tf.train.AdamOptimizer().minimize(self.beamsearch_loss)
        word_log_prob = self.predicting_scores  # tf.placeholder(shape=[batch_size, max_target_length, beam_width], dtype=tf.float32, name='log_prob')
        self.dis_rewards = tf.placeholder(shape=[None, None, self.beam_width], dtype=tf.float32,
                                          name='discounted_rewards')
        # add one for sentence
        # padding = self.sen_reward_rate * tf.ones(
        #     [tf.shape(word_log_prob)[0], 1, tf.shape(word_log_prob)[2]])
        # whole_log_pro = tf.concat([padding, word_log_prob], 1)

        cross_entropy = word_log_prob * self.dis_rewards
        self.rl_reward = - tf.reduce_sum(cross_entropy)
        self.RL_train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.rl_reward)

    def Seq_loss(self):
        # padding = -1 * tf.ones(
        #     [tf.shape(self.logits)[0], self.max_target_sequence_length - tf.shape(self.logits)[1],
        #      tf.shape(self.logits)[2]])
        self.loss_seq2seq = tf.contrib.seq2seq.sequence_loss(logits=self.logits,
                                                             targets=self.decoder_targets, weights=tf.cast(
                tf.sequence_mask(tf.fill([tf.shape(self.logits)[0]], tf.shape(self.logits)[1]), tf.shape(self.logits)[1]), tf.float32))
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