import torch
import torch.nn as nn
from torch.nn import Parameter
import math
import numpy as np


class Encoder(nn.Module):
    """Maps a graph represented as an input sequence to a hidden vector
    """

    def __init__(self, input_dim, hidden_dim, use_cuda):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.use_cuda = use_cuda
        self.enc_init_state = self.init_hidden(hidden_dim)

    def forward(self, x, hidden):
        # hidden: (h0, c0)
        output, hidden = self.lstm(x, hidden)
        return output, hidden

    def init_hidden(self, hidden_dim):
        """Trainable initial hidden state"""
        enc_init_hx = Parameter(torch.zeros(hidden_dim), requires_grad=False)
        if self.use_cuda:
            enc_init_hx = enc_init_hx.cuda()

        # enc_init_hx.uniform_(-(1. / math.sqrt(hidden_dim)),
        #        1. / math.sqrt(hidden_dim))

        enc_init_cx = Parameter(torch.zeros(hidden_dim), requires_grad=False)
        if self.use_cuda:
            enc_init_cx = enc_init_cx.cuda()

        # enc_init_cx = nn.Parameter(enc_init_cx)
        # enc_init_cx.uniform_(-(1. / math.sqrt(hidden_dim)),
        #        1. / math.sqrt(hidden_dim))
        return enc_init_hx, enc_init_cx


class Attention(nn.Module):
    """A generic attention module for a decoder in seq2seq"""

    def __init__(self, dim, use_tanh=False, C=10, use_cuda=True):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()

        v = torch.FloatTensor(dim)
        if use_cuda:
            v = v.cuda()
        self.v = nn.Parameter(v, requires_grad=True)
        self.v.data.uniform_(-1. / math.sqrt(dim), 1. / math.sqrt(dim))

    def forward(self, query, ref):
        """
        Args:
            query: is the hidden state of the decoder at the current time step. [batch_size x hidden_dim]
            ref: the set of hidden states from the encoder.
                [sourceL x batch_size x hidden_dim]
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)  # [batch_size x hidden_dim x 1]
        e = self.project_ref(ref)  # [batch_size x hidden_dim x sourceL]
        # expand the query by sourceL
        # [batch x dim x sourceL]
        expanded_q = q.repeat(1, 1, e.size(2))
        # [batch x 1 x hidden_dim]
        v_view = self.v.unsqueeze(0).expand(expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL] = [batch_size x 1 x sourceL]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u
        return e, logits


class Decoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 seq_len,
                 tanh_exploration,
                 use_tanh,
                 decode_type,
                 n_glimpses=1,
                 beam_size=0,
                 use_cuda=True):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.seq_len = seq_len
        self.decode_type = decode_type
        self.beam_size = beam_size
        self.use_cuda = use_cuda

        self.input_weights = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_weights = nn.Linear(hidden_dim, 4 * hidden_dim)

        self.pointer = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration, use_cuda=self.use_cuda)
        self.glimpse = Attention(hidden_dim, use_tanh=False, use_cuda=self.use_cuda)
        self.sm = nn.Softmax(dim=1)

    def apply_mask_to_logits(self, logits, mask, prev_idxs):
        if mask is None:
            mask = torch.zeros(logits.size()).byte()  # dtype=torch.uint8
            if self.use_cuda:
                mask = mask.cuda()

        maskk = mask.clone()

        # to prevent them from being reselected.
        # Or, allow re-selection and penalize in the objective function
        if prev_idxs is not None:
            # set most recently selected idx values to 1
            maskk[list(range(logits.size(0))), prev_idxs] = 1  # awesome!
            logits[maskk] = -np.inf
        return logits, maskk

    def forward(self, decoder_input, embedded_inputs, hidden, context):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size x embedding_dim]. Trainable parameter.
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            hidden: the prev hidden state, size is [batch_size x hidden_dim].
                Initially this is set to (enc_h[-1], enc_c[-1])
            context: encoder outputs, [sourceL x batch_size x hidden_dim]
        """

        def recurrence(x, hidden, logit_mask, prev_idxs):

            hx, cx = hidden  # batch_size x hidden_dim
            # gates: [batch_size x (hidden_dim x 4)]
            gates = self.input_weights(x) + self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)  # batch_size x hidden_dim

            g_l = hy
            for _ in range(self.n_glimpses):
                ref, logits = self.glimpse(g_l, context)
                logits, logit_mask = self.apply_mask_to_logits(logits, logit_mask, prev_idxs)
                # [batch_size x h_dim x sourceL] * [batch_size x sourceL x 1] = [batch_size x h_dim x 1]
                g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
            _, logits = self.pointer(g_l, context)

            logits, logit_mask = self.apply_mask_to_logits(logits, logit_mask, prev_idxs)
            probs = self.sm(logits)
            return hy, cy, probs, logit_mask

        def topk(x, k):
            a = [(idx, e[-1]) for (idx, e) in enumerate(x)]
            for i in range(k):
                for j in range(len(a) - 1 - i):
                    if a[j][-1] > a[j + 1][-1]:
                        a[j], a[j + 1] = a[j + 1], a[j]
            return [x[e[0]] for e in a[-k:]]

        batch_size = context.size(1)
        sourceL = context.size(0)
        outputs = []
        selections = []
        # idxs = None
        idxs = [0] * batch_size  #
        selections.append(torch.LongTensor(idxs).cuda() if self.use_cuda else torch.LongTensor(idxs))   #
        choose_i = torch.LongTensor([0]).cuda() if self.use_cuda else torch.LongTensor([0])
        prob_0 = torch.zeros(batch_size, sourceL).cuda() if self.use_cuda else torch.zeros(batch_size, sourceL)
        prob_0.index_fill_(1, choose_i, 1)
        outputs.append(prob_0)

        mask = None
        if mask is None:
            # dtype=torch.uint8, mask:[batch_size x sourceL]
            mask = torch.zeros(batch_size, self.seq_len).byte()

            choose_i = torch.LongTensor([0])
            mask.index_fill_(1, choose_i, 1)

            if self.use_cuda:
                mask = mask.cuda()

        if self.decode_type == 'stochastic':
            for _ in range(self.seq_len - 1):
                hx, cx, probs, mask = recurrence(decoder_input, hidden, mask, idxs)
                hidden = (hx, cx)
                # select the next inputs for the decoder [batch_size x hidden_dim]
                decoder_input, idxs = self.decode_stochastic(probs, embedded_inputs, selections)
                # use outs to point to next object
                outputs.append(probs)
                selections.append(idxs)

            return (outputs, selections), hidden

        elif self.decode_type == 'beam_search':
            # embedded_inputs: [sourceL x batch_size x embedding_dim]
            # decoder_input: [batch_size x embedding_dim]
            # hidden: [batch_size x hidden_dim]
            # context: [sourceL x batch_size x hidden_dim]
            sel_cands = [[[list(), 0.0]] for _ in range(batch_size)]
            for seq_id in range(self.seq_len - 1):
                # probs: [batch_size x sourceL]
                hx, cx, probs, mask = recurrence(decoder_input, hidden, mask, idxs)
                hidden = (hx, cx)
                # [(beam_size or 1) x batch_size x sourceL]
                probs = probs.view(-1, batch_size, self.seq_len)
                b_or_1 = probs.size(0)
                for b_id in range(batch_size):
                    sequences = sel_cands[b_id]
                    all_candidates = list()
                    for i in range(len(sequences)):
                        seq, score = sequences[i]
                        for k in range(b_or_1):
                            for j in range(len(probs[k][b_id])):
                                candidate = [seq + [j], score + torch.log(probs[k][b_id][j])]
                                all_candidates.append(candidate)
                    # sel_cands[b_id] = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:self.beam_size]
                    sel_cands[b_id] = topk(all_candidates, self.beam_size)
                # candidates_idxs: [beam_size x batch_size]
                sel_cands_idxs = np.array([e[0][-1] for line in sel_cands for e in line]).reshape(-1, self.beam_size).T
                # decoder_input: [(beam_size x batch_size) x embedding_dim]
                idxs = sel_cands_idxs.reshape(-1)
                decoder_input = torch.cat(
                    [embedded_inputs[sel_cands_idxs[i], list(range(batch_size)), :] for i in range(self.beam_size)], 0)

                if seq_id == 0:
                    hidden = (hidden[0].repeat(self.beam_size, 1), hidden[1].repeat(self.beam_size, 1))
                    context = context.repeat(1, self.beam_size, 1)
                    mask = mask.repeat(self.beam_size, 1)

            selections = np.array([sel[0][0] for sel in sel_cands]).reshape(-1, batch_size)

            return (None, selections), None

    def decode_stochastic(self, probs, embedded_inputs, selections):
        """
        Return the next input for the decoder by selecting the
        input corresponding to the max output

        Args:
            probs: [batch_size x sourceL]
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            selections: list of all of the previously selected indices during decoding
       Returns:
            Tensor of size [batch_size x sourceL] containing the embeddings
            from the inputs corresponding to the [batch_size] indices
            selected for this iteration of the decoding, as well as the
            corresponding indicies
        """
        batch_size = probs.size(0)
        # idxs is [batch_size]
        idxs = probs.multinomial(1).squeeze(1)

        # due to race conditions, might need to resample here
        for old_idxs in selections:
            # compare new idxs elementwise with the previous idxs.
            # If any matches, then need to resample
            if old_idxs.eq(idxs).any():
                print('[!] resampling due to race condition')
                idxs = probs.multinomial(1).squeeze(1)
                break

        sels = embedded_inputs[idxs, list(range(batch_size)), :]  # [batch_size x embedding_size]
        return sels, idxs


class PointerNetwork(nn.Module):
    """The pointer network, which is the core seq2seq model
    """

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 seq_len,
                 n_glimpses,
                 tanh_exploration,
                 use_tanh,
                 beam_size,
                 use_cuda):
        super(PointerNetwork, self).__init__()

        self.encoder = Encoder(
            embedding_dim,
            hidden_dim,
            use_cuda)

        self.decoder = Decoder(
            embedding_dim,
            hidden_dim,
            seq_len,
            tanh_exploration=tanh_exploration,
            use_tanh=use_tanh,
            decode_type='stochastic',
            n_glimpses=n_glimpses,
            beam_size=beam_size,
            use_cuda=use_cuda)

        # Trainable initial hidden states
        dec_in_0 = torch.FloatTensor(embedding_dim)
        if use_cuda:
            dec_in_0 = dec_in_0.cuda()

        self.decoder_in_0 = nn.Parameter(dec_in_0)
        self.decoder_in_0.data.uniform_(-1. / math.sqrt(embedding_dim), 1. / math.sqrt(embedding_dim))

    def forward(self, inputs):
        """ Propagate inputs through the network
        Args:
            inputs: [sourceL x batch_size x embedding_dim]
        """

        (encoder_hx, encoder_cx) = self.encoder.enc_init_state
        encoder_hx = encoder_hx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)  # [1 x batch_size x hidden_dim]
        encoder_cx = encoder_cx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)

        # encoder forward pass
        # enc_h: [seq_len x batch_size x hidden_dim], enc_h_t: [1 x batch_size x hidden_dim]
        enc_h, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))

        dec_init_state = (enc_h_t[-1], enc_c_t[-1])

        # repeat decoder_in_0 across batch
        # decoder_input = self.decoder_in_0.unsqueeze(0).repeat(inputs.size(1), 1)  # [batch_size x embedding_dim]
        decoder_input = inputs[0].clone().detach()
        (pointer_probs, input_idxs), dec_hidden_t = self.decoder(decoder_input,
                                                                 inputs,
                                                                 dec_init_state,
                                                                 enc_h)

        return pointer_probs, input_idxs


class CriticNetwork(nn.Module):
    """Useful as a baseline in REINFORCE updates"""

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_process_blocks,
                 tanh_exploration,
                 use_tanh,
                 use_cuda):
        super(CriticNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_process_blocks = n_process_blocks

        self.encoder = Encoder(embedding_dim,
                               hidden_dim,
                               use_cuda)

        self.process_block = Attention(hidden_dim,
                                       use_tanh=use_tanh,
                                       C=tanh_exploration,
                                       use_cuda=use_cuda)
        self.sm = nn.Softmax(dim=1)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # baseline prediction, a single scalar
        )

    def forward(self, inputs):
        """
        Args:
            inputs: [sourceL x batch_size x embedding_dim] of embedded inputs
        """

        (encoder_hx, encoder_cx) = self.encoder.enc_init_state  # [hidden_dim]
        encoder_hx = encoder_hx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)  # [1 x batch_size x hidden_dim]
        encoder_cx = encoder_cx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)

        # encoder forward pass
        enc_outputs, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))

        # grab the hidden state and process it via the process block
        process_block_state = enc_h_t[-1]  # [batch_size x hidden_dim]
        for _ in range(self.n_process_blocks):
            ref, logits = self.process_block(process_block_state, enc_outputs)
            process_block_state = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        # produce the final scalar output
        out = self.decoder(process_block_state)
        return out


class NeuralCombOptRL(nn.Module):
    """
    This module contains the PointerNetwork (actor) and CriticNetwork (critic).
    It requires an application-specific reward function
    """

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 seq_len,
                 n_glimpses,
                 n_process_blocks,
                 tanh_exploration,  # C
                 use_tanh,
                 beam_size,
                 objective_fn,  # reward function
                 is_train,
                 use_cuda):
        super(NeuralCombOptRL, self).__init__()
        self.objective_fn = objective_fn
        self.is_train = is_train
        self.use_cuda = use_cuda

        self.actor_net = PointerNetwork(
            embedding_dim,
            hidden_dim,
            seq_len,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            beam_size,
            use_cuda)

        # utilize critic network
        self.critic_net = CriticNetwork(
            embedding_dim,
            hidden_dim,
            n_process_blocks,
            tanh_exploration,
            False,
            use_cuda)

    # def forward(self, inputs, car, graphs, requests):
    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, sourceL, input_dim]
        """
        batch_size = inputs.size(0)

        # [sourceL x batch_size x embedding_dim]
        embedded_inputs = inputs.permute(1, 0, 2)

        # query the actor net for the input indices
        # making up the output, and the pointer attn
        probs_, action_idxs = self.actor_net(embedded_inputs)
        # probs_: [seq_len x batch_size x seq_len], action_idxs: [seq_len x batch_size]

        # Select the actions (inputs pointed to by the pointer net)
        actions = []

        for action_id in action_idxs:
            actions.append(inputs[list(range(batch_size)), action_id, :])

        if self.is_train:
            # probs_ is a list of len sourceL of [batch_size x sourceL]
            # probs: [sourceL x batch_size]
            probs = []
            for prob, action_id in zip(probs_, action_idxs):
                probs.append(prob[list(range(batch_size)), action_id])
        else:
            # return the list of len sourceL of [batch_size x sourceL]
            probs = probs_

        # get the critic value fn estimates for the baseline
        # [batch_size]
        v = self.critic_net(embedded_inputs)

        action_idxs = torch.cat(action_idxs, 0).view(-1, batch_size).transpose(1, 0).tolist()
        C1 = 0
        C2 = 0
        C4 = 0
        time_penalty = 120  # min

        # [batch_size]
        # batch_node_list = graphs  # 小图的节点列表
        # batch_ser_num_list = []  # mapping table
        #
        # for node_list in batch_node_list:
        #     ser_num_list = []
        #     for node in node_list:
        #         ser_num_list.append(node.serial_number)
        #
        #     batch_ser_num_list.append(ser_num_list)
        #
        # for i, ser_num_list in enumerate(batch_ser_num_list):
        #     action_idxs[i] = [ser_num_list[j] for j in action_idxs[i]]

        # for i, graph in enumerate(graphs):
        #     action_idxs[i] = [graph[j].serial_number for j in action_idxs[i]]
        # R = self.objective_fn(car, action_idxs, graphs, requests, C1, C2, C4, time_penalty)

        # return R, v, probs, actions, action_idxs
        return v, probs, actions, action_idxs
