"""
Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""

import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        action_query: Tensor,
        inputs: Tensor,
        inputs_pos_embed,
        action_steps_pos_embed,
    ):
        """
        Inputs for case with image
        """

        memory = self.encoder(inputs=inputs, inputs_pos_embed=inputs_pos_embed)

        hidden_state = self.decoder(
            query=action_query,
            query_pos_embed=action_steps_pos_embed,
            key_value=memory,
            key_value_pos_embed=inputs_pos_embed,
        )

        hidden_state = hidden_state.transpose(0, 1)
        return hidden_state, memory


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, inputs: Tensor, inputs_pos_embed: Optional[Tensor] = None):
        output = inputs

        for layer in self.layers:
            output = layer(inputs=output, inputs_pos_embed=inputs_pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, stop_gradient=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.stop_gradient = stop_gradient

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        query_pos_embed: Optional[Tensor] = None,
        key_value_pos_embed: Optional[Tensor] = None,
    ):

        if self.stop_gradient:
            query = query.detach()
            key_value = key_value.detach()
            key_value_pos_embed = key_value_pos_embed.detach()
            query_pos_embed = query_pos_embed.detach()

        output = query
        for layer in self.layers:
            output = layer(
                query=output,
                key_value=key_value,
                query_pos_embed=query_pos_embed,
                key_value_pos_embed=key_value_pos_embed,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, x: Tensor, pos_embed: Optional[Tensor] = None):
        q = k = self.with_pos_embed(x, pos_embed)
        # No causal mask, full information passing along action + joints + class
        x = x + self.dropout1(self.self_attn(q, k, value=x, need_weights=False)[0])
        x = self.norm1(x)
        x = x + self.dropout2(
            self.linear2(self.dropout(self.activation(self.linear1(x))))
        )
        x = self.norm2(x)
        return x

    def forward_pre(self, x: Tensor, pos_embed: Optional[Tensor] = None):
        x2 = self.norm1(x)
        q = k = self.with_pos_embed(x2, pos_embed)
        x2 = self.self_attn(q, k, value=x2, need_weights=False)[0]
        x = x + self.dropout1(x2)
        x2 = self.norm2(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x = x + self.dropout2(x2)
        return x

    def forward(self, inputs: Tensor, inputs_pos_embed: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(x=inputs, pos_embed=inputs_pos_embed)
        return self.forward_post(x=inputs, pos_embed=inputs_pos_embed)


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        query: Tensor,
        key_value: Tensor,
        key_value_pos_embed: Optional[Tensor] = None,
        query_pos_embed: Optional[Tensor] = None,
    ):
        tgt = query
        memory = key_value
        pos = key_value_pos_embed
        query_pos = query_pos_embed

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, need_weights=False)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        query: Tensor,
        kv: Tensor,
        q_pos_embed: Optional[Tensor] = None,
        kv_pos_embed: Optional[Tensor] = None,
    ):
        q2 = self.norm1(query)
        q = k = self.with_pos_embed(q2, q_pos_embed)
        q2 = self.self_attn(q, k, value=q2, need_weights=False)[0]
        query = query + self.dropout1(q2)
        q2 = self.norm2(query)

        q2 = self.multihead_attn(
            query=self.with_pos_embed(q2, q_pos_embed),
            key=self.with_pos_embed(kv, kv_pos_embed),
            value=kv,
            need_weights=False,
        )[0]

        query = query + self.dropout2(q2)
        q2 = self.norm3(query)
        q2 = self.linear2(self.dropout(self.activation(self.linear1(q2))))
        query = query + self.dropout3(q2)
        return query

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        key_value_pos_embed: Optional[Tensor] = None,
        query_pos_embed: Optional[Tensor] = None,
    ):

        if self.normalize_before:
            return self.forward_pre(
                query=query,
                kv=key_value,
                q_pos_embed=query_pos_embed,
                kv_pos_embed=key_value_pos_embed,
            )

        return self.forward_post(
            query=query,
            key_value=key_value,
            query_pos_embed=query_pos_embed,
            key_value_pos_embed=key_value_pos_embed,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
