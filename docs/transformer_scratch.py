import marimo

__generated_with = "0.13.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.utils.data as data
    import math
    import copy
    return math, mo, nn, torch


@app.cell
def _(mo):
    mo.md(
        r"""
    We need to define the basic building blocks

    - Multihead Attention
    - Position wise FeedForward Networks
    - Positional Encoding
    - ...

    Multi head attention are based on three different set of informations:
    - Values
    - Keys
    - Queries
    All go through a linear layer and then are aggregated through a scaled dot product attention, concatenated and passed once again through a linear layer.
    """
    )
    return


@app.cell
def _(math, nn, torch):
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, num_heads):
            super(MultiHeadAttention, self).__init__()
            # Ensure that the model dimension (d_model is divisible by the number of heads)
            assert d_model % num_heads == 0, "d_model must be divisible by number of heads"

            # Init dimensions
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads # Dimension of each head key, query and value

            # Linear layers
            self.W_q = nn.Linear(d_model, d_model) # query
            self.W_k = nn.Linear(d_model, d_model) # key
            self.W_v = nn.Linear(d_model ,d_model) # value
            self.W_0 = nn.Linear(d_model , d_model) # output

        def scaled_dot_product_attention(self, Q, K, V, mask=None):
            # Calculare attention scores
            attn_scores = torch.matmul(
                Q,
                K.transpose(-2, -1)
            ) / math.sqrt(self.d_k)

            # Apply mask if provided (usefull for preventing attention to certain part
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

            # Sftmax to attain probabilities
            attn_probs = torch.softmax(attn_scores, dim=1)
            # Multiply by values
            output = torch.matmul(attn_probs, V)
            return output

        def split_heads(self, x):
            # Reshape the input to have num_heads for multi-head attention
            batch_size, seq_length, d_model = x.size()
            return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1,2)

        def combine_heads(self, x):
            # Combine the multiple heads ack to original shape
            batch_size, _, seq_length, d_k = x.size()
            return x.transpose(1,2).contiguous().view(
                batch_size, seq_length, self.d_model
            )

        def forward(self, Q,K,V,mask=None):
            # Apply linear transformations and split heads
            Q = self.split_heads(self.W_q(Q))
            K = self.split_heads(self.W_k(K))
            V = self.split_heads(self.W_v(V))

            # Perform scaled dot produc
            attn_output = self.scaled_dot_product_attention(Q,K,V,mask)

            # combine heads and apply output
            output = self.W_o(self.combine_heads(attn_output))

            return output
    return (MultiHeadAttention,)


@app.cell
def _(nn):
    class PositionWiseFeedForward(nn.Module):
        def __init__(self, d_model, d_ff):
            super(PositionWiseFeedForward, self).__init__()
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))
    return


app._unparsable_cell(
    r"""
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_seq_length):
            super(PositionalEncoding, self).__init__()

            pe = torch.zeros(max_seq_length, d_model)
            position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_ model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]
    """,
    name="_"
)


@app.cell
def _():
    from IPython.display import Markdown, display

    display(Markdown(
        "![Transformer Encoder](https://media.datacamp.com/legacy/v1691083306/Figure_2_The_Encoder_part_of_the_transformer_network_Source_image_from_the_original_paper_b0e3ac40fa.png)"
    ))
    return Markdown, display


@app.cell
def _(MultiHeadAttention, PositionaWiseFeedForward, nn):
    class EncoderLayer(nn.Module):
        def __init__(self, d_model, num_heads, d_ff, dropout):
            super(EncoderLayer, self).__init__()
            self.self_attn = MultiHeadAttention(d_model, num_heads)
            self.feed_forward = PositionaWiseFeedForward(d_model, d_ff)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask):
            attn_output = self.self_attn(x,x,x, mask)
            x = self.norm1(x + self.dropout(attn_output))
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))
            return x
    return


@app.cell
def _(Markdown, display):
    display(Markdown(
        "![Transformer Decoder](https://media.datacamp.com/legacy/v1691083444/Figure_3_The_Decoder_part_of_the_Transformer_network_Souce_Image_from_the_original_paper_b90d9e7f66.png)"
    ))

    return


if __name__ == "__main__":
    app.run()
