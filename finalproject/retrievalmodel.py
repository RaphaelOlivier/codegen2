import theano
import theano.tensor as T
import numpy as np
import logging
import copy

from model import Model
from components import CondAttLSTM
import config
from nn.utils.theano_utils import *


class CondAttLSTMAligner(CondAttLSTM):
    def __init__(self, *args, **kwargs):

        super(CondAttLSTMAligner, self).__init__(*args, **kwargs)

    def _step_align(self,
                    t, xi_t, xf_t, xo_t, xc_t, mask_t, parent_t,
                    h_tm1, c_tm1, hist_h,
                    u_i, u_f, u_o, u_c,
                    c_i, c_f, c_o, c_c,
                    h_i, h_f, h_o, h_c,
                    p_i, p_f, p_o, p_c,
                    att_h_w1, att_w2, att_b2,
                    context, context_mask, context_att_trans,
                    b_u):

        # context: (batch_size, context_size, context_dim)

        # (batch_size, att_layer1_dim)
        h_tm1_att_trans = T.dot(h_tm1, att_h_w1)

        # (batch_size, context_size, att_layer1_dim)
        att_hidden = T.tanh(context_att_trans + h_tm1_att_trans[:, None, :])
        # (batch_size, context_size, 1)
        att_raw = T.dot(att_hidden, att_w2) + att_b2
        att_raw = att_raw.reshape((att_raw.shape[0], att_raw.shape[1]))

        # (batch_size, context_size)
        ctx_att = T.exp(att_raw - T.max(att_raw, axis=-1, keepdims=True))

        if context_mask:
            ctx_att = ctx_att * context_mask

        ctx_att = ctx_att / T.sum(ctx_att, axis=-1, keepdims=True)
        # (batch_size, context_dim)
        scores = ctx_att[:, :, None]
        ctx_vec = T.sum(context * scores, axis=1)
        ##### attention over history #####

        def _attention_over_history():
            hist_h_mask = T.zeros((hist_h.shape[0], hist_h.shape[1]), dtype='int8')
            hist_h_mask = T.set_subtensor(hist_h_mask[:, T.arange(t)], 1)

            hist_h_att_trans = T.dot(hist_h, self.hatt_hist_W1) + self.hatt_b1
            h_tm1_hatt_trans = T.dot(h_tm1, self.hatt_h_W1)

            hatt_hidden = T.tanh(hist_h_att_trans + h_tm1_hatt_trans[:, None, :])
            hatt_raw = T.dot(hatt_hidden, self.hatt_W2) + self.hatt_b2
            hatt_raw = hatt_raw.reshape((hist_h.shape[0], hist_h.shape[1]))
            hatt_exp = T.exp(hatt_raw - T.max(hatt_raw, axis=-1, keepdims=True)) * hist_h_mask
            h_att_weights = hatt_exp / (T.sum(hatt_exp, axis=-1, keepdims=True) + 1e-7)

            # (batch_size, output_dim)
            _h_ctx_vec = T.sum(hist_h * scores, axis=1)

            return _h_ctx_vec

        h_ctx_vec = T.switch(t,
                             _attention_over_history(),
                             T.zeros_like(h_tm1))

        if not config.parent_hidden_state_feed:
            t = 0

        par_h = T.switch(t,
                         hist_h[T.arange(hist_h.shape[0]), parent_t, :],
                         T.zeros_like(h_tm1))

        ##### feed in parent hidden state #####
        if config.tree_attention:
            i_t = self.inner_activation(
                xi_t + T.dot(h_tm1 * b_u[0], u_i) + T.dot(ctx_vec, c_i) + T.dot(par_h, p_i) + T.dot(h_ctx_vec, h_i))
            f_t = self.inner_activation(
                xf_t + T.dot(h_tm1 * b_u[1], u_f) + T.dot(ctx_vec, c_f) + T.dot(par_h, p_f) + T.dot(h_ctx_vec, h_f))
            c_t = f_t * c_tm1 + i_t * self.activation(
                xc_t + T.dot(h_tm1 * b_u[2], u_c) + T.dot(ctx_vec, c_c) + T.dot(par_h, p_c) + T.dot(h_ctx_vec, h_c))
            o_t = self.inner_activation(
                xo_t + T.dot(h_tm1 * b_u[3], u_o) + T.dot(ctx_vec, c_o) + T.dot(par_h, p_o) + T.dot(h_ctx_vec, h_o))
        else:
            i_t = self.inner_activation(
                xi_t + T.dot(h_tm1 * b_u[0], u_i) + T.dot(ctx_vec, c_i) + T.dot(par_h, p_i))  # + T.dot(h_ctx_vec, h_i)
            f_t = self.inner_activation(
                xf_t + T.dot(h_tm1 * b_u[1], u_f) + T.dot(ctx_vec, c_f) + T.dot(par_h, p_f))  # + T.dot(h_ctx_vec, h_f)
            c_t = f_t * c_tm1 + i_t * self.activation(
                xc_t + T.dot(h_tm1 * b_u[2], u_c) + T.dot(ctx_vec, c_c) + T.dot(par_h, p_c))  # + T.dot(h_ctx_vec, h_c)
            o_t = self.inner_activation(
                xo_t + T.dot(h_tm1 * b_u[3], u_o) + T.dot(ctx_vec, c_o) + T.dot(par_h, p_o))  # + T.dot(h_ctx_vec, h_o)
        h_t = o_t * self.activation(c_t)

        h_t = (1 - mask_t) * h_tm1 + mask_t * h_t
        c_t = (1 - mask_t) * c_tm1 + mask_t * c_t

        new_hist_h = T.set_subtensor(hist_h[:, t, :], h_t)

        return h_t, c_t, scores, new_hist_h

    def align(self, X, context, parent_t_seq, init_state=None, init_cell=None, hist_h=None,
              mask=None, context_mask=None, srng=None, time_steps=None):
        assert context_mask.dtype == 'int8', 'context_mask is not int8, got %s' % context_mask.dtype

        # (n_timestep, batch_size)
        mask = self.get_mask(mask, X)
        # (n_timestep, batch_size, input_dim)
        X = X.dimshuffle((1, 0, 2))

        B_w = np.ones((4,), dtype=theano.config.floatX)
        B_u = np.ones((4,), dtype=theano.config.floatX)

        # (n_timestep, batch_size, output_dim)
        xi = T.dot(X * B_w[0], self.W_i) + self.b_i
        xf = T.dot(X * B_w[1], self.W_f) + self.b_f
        xc = T.dot(X * B_w[2], self.W_c) + self.b_c
        xo = T.dot(X * B_w[3], self.W_o) + self.b_o

        # (batch_size, context_size, att_layer1_dim)
        context_att_trans = T.dot(context, self.att_ctx_W1) + self.att_b1

        if init_state:
            # (batch_size, output_dim)
            first_state = T.unbroadcast(init_state, 1)
        else:
            first_state = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)

        if init_cell:
            # (batch_size, output_dim)
            first_cell = T.unbroadcast(init_cell, 1)
        else:
            first_cell = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)

        if not hist_h:
            # (batch_size, n_timestep, output_dim)
            hist_h = alloc_zeros_matrix(X.shape[1], X.shape[0], self.output_dim)

        n_timestep = X.shape[0]
        time_steps = T.arange(n_timestep, dtype='int32')

        # (n_timestep, batch_size)
        parent_t_seq = parent_t_seq.dimshuffle((1, 0))

        [outputs, cells, att_scores, hist_h_outputs], updates = theano.scan(
            self._step_align,
            sequences=[time_steps, xi, xf, xo, xc, mask, parent_t_seq],
            outputs_info=[
                first_state,  # for h
                first_cell,  # for cell
                None,
                hist_h,  # for hist_h
            ],
            non_sequences=[
                self.U_i, self.U_f, self.U_o, self.U_c,
                self.C_i, self.C_f, self.C_o, self.C_c,
                self.H_i, self.H_f, self.H_o, self.H_c,
                self.P_i, self.P_f, self.P_o, self.P_c,
                self.att_h_W1, self.att_W2, self.att_b2,
                context, context_mask, context_att_trans,
                B_u
            ])

        att_scores = att_scores.dimshuffle((1, 0, 2))

        alignments = T.argmax(att_scores, axis=2)

        return alignments


class RetrievalModel(Model):
    def __init__(self, regular_model=None):
        """
        super(RetrievalModel, self).__init__()
        self.decoder_lstm = CondAttLSTMAligner(config.rule_embed_dim + config.node_embed_dim + config.rule_embed_dim,
                                               config.decoder_hidden_dim, config.encoder_hidden_dim, config.attention_hidden_dim,
                                               name='decoder_lstm')
        """
        super(RetrievalModel, self).__init__()

        self.decoder_lstm = CondAttLSTMAligner(config.rule_embed_dim + config.node_embed_dim + config.rule_embed_dim,
                                               config.decoder_hidden_dim, config.encoder_hidden_dim, config.attention_hidden_dim,
                                               name='decoder_lstm')
        # update params for new decoder
        self.params = self.query_embedding.params + self.query_encoder_lstm.params + \
            self.decoder_lstm.params + self.src_ptr_net.params + self.terminal_gen_softmax.params + \
            [self.rule_embedding_W, self.rule_embedding_b, self.node_embedding, self.vocab_embedding_W, self.vocab_embedding_b] + \
            self.decoder_hidden_state_W_rule.params + self.decoder_hidden_state_W_token.params

    def build(self):
        super(RetrievalModel, self).build()
        self.build_aligner()

    def build_aligner(self):
        tgt_action_seq = ndim_itensor(3, 'tgt_action_seq')
        tgt_action_seq_type = ndim_itensor(3, 'tgt_action_seq_type')
        tgt_node_seq = ndim_itensor(2, 'tgt_node_seq')
        tgt_par_rule_seq = ndim_itensor(2, 'tgt_par_rule_seq')
        tgt_par_t_seq = ndim_itensor(2, 'tgt_par_t_seq')

        tgt_node_embed = self.node_embedding[tgt_node_seq]
        query_tokens = ndim_itensor(2, 'query_tokens')
        query_token_embed, query_token_embed_mask = self.query_embedding(
            query_tokens, mask_zero=True)
        batch_size = tgt_action_seq.shape[0]
        max_example_action_num = tgt_action_seq.shape[1]

        tgt_action_seq_embed = T.switch(T.shape_padright(tgt_action_seq[:, :, 0] > 0),
                                        self.rule_embedding_W[tgt_action_seq[:, :, 0]],
                                        self.vocab_embedding_W[tgt_action_seq[:, :, 1]])
        tgt_action_seq_embed_tm1 = tensor_right_shift(tgt_action_seq_embed)
        tgt_par_rule_embed = T.switch(tgt_par_rule_seq[:, :, None] < 0,
                                      T.alloc(0., 1, config.rule_embed_dim),
                                      self.rule_embedding_W[tgt_par_rule_seq])

        if not config.frontier_node_type_feed:
            tgt_node_embed *= 0.
        if not config.parent_action_feed:
            tgt_par_rule_embed *= 0.

        decoder_input = T.concatenate(
            [tgt_action_seq_embed_tm1, tgt_node_embed, tgt_par_rule_embed], axis=-1)
        query_embed = self.query_encoder_lstm(query_token_embed, mask=query_token_embed_mask,
                                              dropout=0, srng=self.srng)

        tgt_action_seq_mask = T.any(tgt_action_seq_type, axis=-1)

        alignments = self.decoder_lstm.align(decoder_input, context=query_embed,
                                             context_mask=query_token_embed_mask,
                                             mask=tgt_action_seq_mask,
                                             parent_t_seq=tgt_par_t_seq,
                                             srng=self.srng)

        alignment_inputs = [query_tokens, tgt_action_seq, tgt_action_seq_type,
                            tgt_node_seq, tgt_par_rule_seq, tgt_par_t_seq]
        self.align = theano.function(alignment_inputs, [alignments])
