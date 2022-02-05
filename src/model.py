from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device
import torch


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, padding_idx, n_layers=1, dropout = 0.4, bidirectional = True, discriminator=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=padding_idx)
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=False, bidirectional=self.bidirectional)

    def forward(self, X, attn=None, Rattn=None):
        if X[0].dim()==2:
            embedded = self.embedding(X[0])
        else:
            embedded = X[0].matmul(self.embedding.weight)
        if Rattn is not None:
            len = attn.size(1)
            embedded = embedded[:len, :,:] * Rattn.transpose(1, 0)

        embedded = self.dropout(embedded)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, X[1].cpu(), batch_first=False, enforce_sorted=False)
        output, hidden = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        if self.bidirectional == True:
            output = (output[:, :, :self.hidden_size] +output[:, :, self.hidden_size:])
            hidden = hidden[0]+hidden[1]
        #output = len x batch x dim
        #hidden = batch x dim
        return output, hidden

class MLPAttention(nn.Module):
    def __init__(self, hidden_dim, att_dim, dropout=0.4):
        super(MLPAttention, self).__init__()
        self.W_k = nn.Linear(hidden_dim, att_dim, bias=True)
        self.v = nn.Linear(att_dim, 1, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
#         self.linear = nn.Linear(hidden_dim, 2)

    def forward(self, encoder_outputs, lengths, temp=1):
        V = encoder_outputs
        max_len = encoder_outputs.size(0)
        
        pos_idx = torch.arange(max_len).unsqueeze(0)
        pos_idx = pos_idx.to(lengths.device)
        src_mask = pos_idx[:, :max_len] >= (lengths).unsqueeze(-1)
        encoder_outputs = self.W_k(encoder_outputs)
        encoder_outputs = self.dropout(encoder_outputs)
        encoder_outputs = self.tanh(encoder_outputs)
        scores = self.v(encoder_outputs).squeeze(-1)
        
        ## MASK OUT PAD
        scores = scores.masked_fill(src_mask.T, float("-inf"))
        scores = torch.softmax(scores/temp, dim=0).unsqueeze(0).permute(2, 1, 0)
        
        attn_hidden = torch.bmm(V.permute(1, 2, 0), scores)
        #scores = batch x len x 1
        #attn_hidden = batch x dim x 1
        reverse_scores = 1-scores.squeeze(2)
        reverse_scores = reverse_scores.masked_fill(src_mask, float(0)).view(-1,max_len,1)
        #reverse_scores = batch x len x 1
        
        return scores, attn_hidden.squeeze(2), reverse_scores, src_mask

class SentimentClassifier(nn.Module):
    def __init__(self, hidden_size, num_class=2, dropout=0.3):
        super(SentimentClassifier, self).__init__()
        self.hidden = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.hidden, num_class)
        
    def forward(self, attn_output):
        return self.linear(attn_output).squeeze()



class Generator(nn.Module):
    def __init__(self, enc_cls, mlp, enc):
        super(Generator, self).__init__()
        self.enc_cls = enc_cls
        self.mlp = mlp
        self.enc = enc
        self.mlp.dropout.p=0.2
        self.enc_cls.dropout.p=0.2
        self.z_dim = 500
        self.y_dim = 200

    def forward(self, X):
        output, hidden = self.enc_cls(X)
        scores, attn_hidden, reverse_scores, src_mask = self.mlp(output, X[1], temp=0.4)
        scores = scores.detach()
        reverse_scores = reverse_scores.detach()
        output, hidden = self.enc(X, scores, reverse_scores)
#         content = hidden[:,self.y_dim:]
#         style = hidden[:,:self.y_dim]
        return hidden, output, hidden, src_mask, reverse_scores

# language model for ppl
class GRU_LM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx, dropout = 0.4):
        super(GRU_LM, self).__init__()
        self.hidden_size = hidden_size
        self.emb= nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, text, length):
        x = self.emb(text)
        x = self.dropout(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, length.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.gru(x)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.dropout(output)
        return self.out(output)

#########################################################################

# Decoder is a modification of "https://github.com/IBM/pytorch-seq2seq"

#########################################################################
class BaseRNN(nn.Module):
    SYM_MASK = "MASK"
    SYM_EOS = "EOS"

    def __init__(self, vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout_p = dropout_p

    def forward(self, *args, **kwargs):
        raise NotImplementedError()
class DecoderRNN_LN(BaseRNN):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'
    def __init__(self, vocab_size, max_len, hidden_size,
            sos_id, eos_id, pad_id, style_size, num_class,
            n_layers=1, rnn_cell='gru', bidirectional=False,
            input_dropout_p=0.2, dropout_p=0.2, use_attention=False):
        super(DecoderRNN_LN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p,
                n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.emb_dim = 128
        self.rnn = self.rnn_cell(self.emb_dim, hidden_size, n_layers, batch_first=True, dropout=dropout_p)

        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.sos_id = sos_id

        self.init_input = None
        self.embedding = nn.Embedding(self.output_size, self.emb_dim)
        if use_attention:
            self.attention = Attention(self.hidden_size, self.hidden_size-style_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(0.2)
        self.gamma = nn.Embedding(num_class, style_size)
        self.beta = nn.Embedding(num_class, style_size)
        self.reduce = nn.Linear(hidden_size-style_size, style_size)
    def forward_step(self, input_var, hidden, encoder_outputs, function, mask, gumbel):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)
        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs, mask)
        if gumbel:
            predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)), hard=False, dim=1).view(batch_size, output_size, -1)
        else:
            predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                    function=F.log_softmax, teacher_forcing_ratio=0, mask = None, attn_hidden = None, label=None, gumbel=False, max_length = None):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN_LN.KEY_ATTN_SCORE] = list()

        inputs, batch_size, _ = self._validate_args(inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio)
        attn_hidden = self.reduce(attn_hidden.detach())
        beta = self.beta(label)
        gamma = self.gamma(label)
        mean = torch.mean(attn_hidden, -1, keepdim=True)
        std = torch.std(attn_hidden, -1, unbiased=False, keepdim=True)
        attn_hidden = (attn_hidden-mean)/std
        style_vec = (attn_hidden*gamma)+beta
        style_vec = style_vec.unsqueeze(0)
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_hidden = torch.cat([style_vec, decoder_hidden], dim=-1)
        decoder_hidden = self.dropout(decoder_hidden)
        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN_LN.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)
            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        decoder_input = inputs[:, 0].unsqueeze(1).to(style_vec.device)
        if gumbel:
            function = F.gumbel_softmax
        for di in range(max_length):
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                     function=function, mask = mask, gumbel=gumbel)
            step_output = decoder_output.squeeze(1)
            symbols = decode(di, step_output, step_attn)
            decoder_input = symbols
        ret_dict[DecoderRNN_LN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN_LN.KEY_LENGTH] = lengths.tolist()
        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            max_length = encoder_outputs.size(0)
        else:
            max_length = inputs.size(1) - 1 # minus the start of sequence symbol

        return inputs, batch_size, max_length

class Attention(nn.Module):
    def __init__(self, output_hidden_dim, hidden_dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(hidden_dim*2, output_hidden_dim)
        self.mask = None
        self.linear = nn.Linear(output_hidden_dim, hidden_dim)

    def set_mask(self, mask):
        """
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context, mask):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(0)
        #output = self.linear(output)
        self.set_mask(mask)
        
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        output = self.linear(output)
        attn = torch.bmm(output, context.permute(1, 2, 0))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask.unsqueeze(1), -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context.permute(1, 0, 2))

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = torch.tanh(self.linear_out(combined)).view(batch_size, -1, hidden_size)

        return output, attn