import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    batch_size = None
    epochs = None
    total_steps = None
    warmup_steps = None
    learning_rate = None
    weight_decay = None
    betas = None
    d_model = None
    n_heads = None
    n_layers = None
    n_pixels = None
    n_clusters = None
    p_drop = None
    make_cluster = None
    device = None

    def __init__(self, config):
        self.batch_size = None
        for k, v in config.items():
            setattr(self, k, v)
        self.device = torch.device("cuda")

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # embedding layers
        # n_clusters: similar to vocab_size of Language transformers
        # d_model: the embedding dimension we want
        # n_pixels: number of pixels
        # pix_embed (Assigning embeddings to clusters)-> (n_cluster, d_model)
        # seq_len = n_pixels*n_pixels
        # pos_embed (Positional embedding) -> (seq_len, d_model)
        self.pix_embed = nn.Embedding(config.n_clusters, config.d_model)
        self.pos_embed = nn.Embedding(config.n_pixels * config.n_pixels, config.d_model)
        self.embed_drop = nn.Dropout(config.p_drop)

    def forward(self, x):   # x -> (batch_size, seq_len)
        seq_length = x.size(1)   # (seq_len)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)  # (seq_len)
        position_ids = position_ids.unsqueeze(0).expand_as(x)  # (batch_size, seq_len) :extrapolating

        pix_embeddings = self.pix_embed(x)  # (batch_size, seq_len, d_model)
        pos_embeddings = self.pos_embed(position_ids)   # (batch_size, seq_len, d_model)
        embeddings = self.embed_drop(pix_embeddings + pos_embeddings)
        return embeddings  # (batch_size, seq_len, d_model)


class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        # d_model: Embedding vector size
        # n_heads: Number of attention heads
        assert config.d_model % config.n_heads == 0, "d_model is not divisible by h"
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads  # Dimension of each head

        # query, key, value
        self.query_w = nn.Linear(config.d_model, config.d_model)
        self.key_w = nn.Linear(config.d_model, config.d_model)
        self.value_w = nn.Linear(config.d_model, config.d_model)
        self.final_embed_w = nn.Linear(config.d_model, config.d_model)

    @staticmethod
    def attention(query, key, value):
        d_head = query.shape[-1]
        seq_len = query.shape[-2]
        attention_mask = torch.full((seq_len, seq_len), -float('inf'), device=query.device, dtype=query.dtype)
        attention_mask = torch.triu(attention_mask, diagonal=1)
        attention_mask = torch.nan_to_num(attention_mask, nan=0.0)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_head) + attention_mask
        attention_scores = F.softmax(attention_scores, dim=-1)     # (batch_size, seq_len, seq_len)
        score = attention_scores @ value     # (batch, h, seq_len, d_k)
        return score, attention_scores

    def forward(self, q, k, v):
        query = self.query_w(q)  # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        key = self.key_w(k)    # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        value = self.value_w(v)  # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, n_head, d_head) --> (batch, n_head, seq_len, d_head)
        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_head).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_head).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_head).transpose(1, 2)

        # Calculate attention
        x, attention_scores = MHA.attention(query, key, value)

        # Combine all the heads together
        # (batch, n_head, seq_len, d_head) --> (batch, seq_len, n_head, d_head) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_head)

        # last linear layer
        x = self.final_embed_w(x)   # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        return x   # (batch, seq_len, d_model)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.p_drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x    # (batch, seq_len, d_model)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MHA(config)
        self.mlp = FeedForward(config)

        # layer normalization
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.ln_2 = nn.LayerNorm(config.d_model)

    def forward(self, x):
        x = self.ln_1(x)
        x = self.attention(x,x,x) + x
        x = self.ln_2(x)
        x = self.mlp(x) + x
        return x    # (batch, seq_len, d_model)



class IGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # start token -> (d_model,)
        self.start_of_image = torch.nn.Parameter(torch.zeros(config.d_model))  # Trainable SOS finds patterns in data
        nn.init.normal_(self.start_of_image)  # Initialise Normally distributed (mu=0, sd=1)

        # embedding layers
        self.embedding = Embeddings(config)

        # Encoder
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])

        # Decoder
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.n_clusters, bias=False)

        self.apply(self._init_weights)
        self.to(config.device)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        # prepend sos token
        # repeat(x.size(0), 1) means
        # x.size(0): first dimension (batch dimension) is expanded to match the batch size
        # 1: second dimension (features) remains the same
        # (d_model,) -> (batch_size, d_model) -> (batch_size,1, d_model)
        start = self.start_of_image.repeat(x.size(0), 1).unsqueeze(1)
        h = self.embedding(x)    # h -> (batch_size, seq_len, d_model)
        # Remove the last token from h and append the SOS at start : seq_len is not changed
        h = torch.cat((start, h[:, :-1, :]), dim=1)    # h -> (batch_size, seq_len, d_model)

        x = self.blocks(h)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits    # (batch, seq_len, n_clusters)



