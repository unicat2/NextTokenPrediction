import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        super(SelfAttentionLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scare = self.head_dim ** -0.5

        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()

        query = self.query_projection(hidden_states) * self.scare
        key = self.key_projection(hidden_states)
        value = self.value_projection(hidden_states)

        def reshape(tensor):
            return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        query = reshape(query).view(batch_size * self.num_heads, seq_len, self.head_dim)
        key = reshape(key).view(batch_size * self.num_heads, seq_len, self.head_dim)
        value = reshape(value).view(batch_size * self.num_heads, seq_len, self.head_dim)

        attention_weights = torch.bmm(query, key.transpose(1, 2))
        MASK_VALUE = torch.finfo(attention_weights.dtype).min
        causal_mask = torch.triu(torch.ones(batch_size * self.num_heads, seq_len, seq_len, device=hidden_states.device),
                                 diagonal=1).bool()
        attention_weights = torch.where(causal_mask, MASK_VALUE, attention_weights)

        attention_probs = F.softmax(attention_weights, dim=-1)
        attention_probs = self.dropout(attention_probs)

        attention_output = torch.bmm(attention_probs, value)
        attention_output = attention_output.view(batch_size, self.num_heads, seq_len, self.head_dim).transpose(1,2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.embedding_dim)
        attention_output = self.output_projection(attention_output)
        attention_output = self.dropout(attention_output)

        attention_output = self.layer_norm(hidden_states + attention_output)

        return attention_output


class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, embedding_dim: int, num_heads: int):
        super(TransformerModel, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(max_len, embedding_dim)
        self.attention_layers = nn.ModuleList([SelfAttentionLayer(embedding_dim, num_heads) for _ in range(2)])
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.embeddings(input_ids) + position_embeddings
        embeddings = self.dropout(embeddings)

        output = embeddings
        for layer in self.attention_layers:
            output = layer(output, attention_mask)
        logits = self.output_layer(output)

        return logits

# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
#
#         with torch.cuda.amp.autocast(enabled=False):
#             attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale
#
#         attn = attn - torch.max(attn, dim=-1, keepdim=True)[0]
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x, attn
#
#
# class Block(nn.Module):
#
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#
#     def forward(self, x, return_attention=False):
#         if return_attention:
#             _, attn = self.attn(self.norm1(x))
#             return attn
#         else:
#             y, _ = self.attn(self.norm1(x))
#             x = x + self.drop_path(y)
#             x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x