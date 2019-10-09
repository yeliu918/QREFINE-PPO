import tensorflow as tf


# Train_graph = tf.Graph()
# with Train_graph.as_default():
#     train_reward = Build_model.build_train(config, BeamSearch_seq2seq)
#
#
# with tf.Session(graph=train_reward) as sess_train:
#     print(Train_graph.get_operations())
#     sess_train.run(tf.initialize_all_variables)
#     fd = {Train_graph.returns_ph: returns, Train_graph.advantages_ph: deltas,
#           Train_graph.actions: generated_sen, Train_graph.old_policy: old_policy,
#           BeamSearch_seq2seq.encoder_inputs: source_shuffle,
#           BeamSearch_seq2seq.encoder_inputs_length: source_len,
#           BeamSearch_seq2seq.decoder_length: target_len,
#           BeamSearch_seq2seq.decoder_inputs: train_shuffle,
#           BeamSearch_seq2seq.decoder_targets: target_shuffle_in}
#     loss, _, policy= sess_train.run(
#         [Train_graph.loss, Train_graph.optimize_expr, Train_graph.policy], fd)

class build_train(object):

    # lenth, gamma=1.0, epsilon=0.2, beta=0.01, grad_clip=40.0, value_factor=0.5, entropy_factor=0.01, scope='ppo', reuse=None
    def __init__(self, model, config):
        gamma = config.gamma
        epsilon = config.epsilon
        beta = config.beta
        grad_clip = config.grad_clip
        value_factor = config.value_factor
        entropy_factor = config.entropy_factor
        scope = config.scope
        lenth = config.lenth
        with tf.variable_scope(scope, reuse=None):
            # input placeholders
            returns_ph = tf.placeholder(tf.float32, [None, lenth], name='return')
            advantages_ph = tf.placeholder(tf.float32, [None, lenth], name='advantage')
            actions_ph  = tf.placeholder(tf.float32, [None, lenth], name='advantage')
            old_log_probs_ph = tf. placeholder(tf.float32, [None, lenth], name = "old_policy")

            train_dist, train_value = model.log_pro, model.value

            # network weights
            network_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope)

            # loss
            advantages = tf.reshape(advantages_ph, [-1, 1])
            returns = tf.reshape(returns_ph, [-1, 1])
            with tf.variable_scope('value_loss'):
                value_loss = tf.reduce_mean(tf.square(returns - train_value))
                value_loss *= value_factor
            with tf.variable_scope('entropy'):
                entropy = tf.reduce_mean(train_dist.entropy())
                entropy *= entropy_factor
            with tf.variable_scope('policy_loss'):
                log_prob = train_dist.log_prob(actions_ph)
                ratio = tf.exp(log_prob - old_log_probs_ph)
                ratio = tf.reshape(ratio, [-1, 1])
                surr1 = ratio * advantages
                surr2 = tf.clip_by_value(
                    ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
                surr = tf.minimum(surr1, surr2)
                policy_loss = tf.reduce_mean(surr)
            loss = value_loss - policy_loss - entropy

            # gradients
            gradients = tf.gradients(loss, network_vars)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)
            # update
            grads_and_vars = zip(clipped_gradients, network_vars)
            optimizer = tf.train.AdamOptimizer()
            optimize_expr = optimizer.apply_gradients(grads_and_vars)