import math
import inspect
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F

class LayerNorm(nn.Module):

    def __init__(self, ndim, bias):

        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):

        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):

    """
    1. Split large array into Wk, Wq, Wv matrices
    2. Distribute each of these weight matrices among the heads
    3. Do scaled dot product attention
    4. Join them back
    5. Pass through linear and dropout layers => This layer learns weightage among the heads
    """

    def __init__(self, config):

        super().__init__()
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.dropout = config.dropout

        assert self.n_embed % self.n_head == 0

        # K, Q and V combined output from input embedding vector
        self.attn = nn.Linear(self.n_embed, 3*self.n_embed, bias = config.bias)

        # Projection output from MLP
        self.proj = nn.Linear(self.n_embed, self.n_embed, bias = config.bias)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x):

        batch_size, seq_len, embed_size = x.size()

        # Get Wq, Wk, Wv matrices from input vector
        q, k, v = self.attn(x).split(self.n_embed, dim=2)

        # Splitting into multiple heads
        k = k.view(batch_size, seq_len, self.n_head, embed_size//self.n_head).transpose(1,2)          
        q = q.view(batch_size, seq_len, self.n_head, embed_size//self.n_head).transpose(1,2)    
        v = v.view(batch_size, seq_len, self.n_head, embed_size//self.n_head).transpose(1,2)

        # Scaled dot product attn
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

        # Combining the outputs from the heads into the same shape for input into next block
        y = y.transpose(1,2).contiguous().view(batch_size, seq_len, embed_size)

        # Difference from original trafo architecture
        y = self.residual_dropout(self.proj(y))

        return y

class MLP(nn.Module):
    # Different from original trafo paper's feedforward network
    
    def __init__(self, config):

        super().__init__()
        self.fc = nn.Linear(config.n_embed, 4*config.n_embed, bias=False)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(4*config.n_embed, config.n_embed, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):

        x = self.gelu(self.fc(x))
        x = self.proj(x)
        x = self.dropout(x)
        return x
    

class Block(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.ln1 = LayerNorm(config.n_embed, bias = False)
        self.attention = SelfAttention(config)
        self.ln2 = LayerNorm(config.n_embed, bias = False)
        self.linear = MLP(config)

    def forward(self, x):

        x = x + self.attention(self.ln1(x))
        x = x + self.linear(self.ln2(x))
        return x

@dataclass
class GPTConfig:

    block_size = 32
    vocab_size = 50304 # GPT 2 vocab size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer = 2
    n_head = 4
    n_embed = 64
    dropout = 0.0
    bias = False


class GPT(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                token_embedding = nn.Embedding(config.vocab_size, config.n_embed),
                positional_embedding = nn.Embedding(config.block_size, config.n_embed),
                dropout = nn.Dropout(config.dropout),
                attention_blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                layer_norm_final = LayerNorm(config.n_embed, bias = False)
            )
        )

        # Probability of next output token
        self.language_model_head = nn.Linear(config.n_embed, config.vocab_size, bias = False)

        self.transformer.token_embedding.weight = self.language_model_head.weight # Weight tying, unsure about this line

        # Init all weights
        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*config.n_layer))

        # Report model size
        print("No. of paramaters: ", self.get_num_params()/1e6)

    def forward(self, idx, targets=None):

        device = idx.device
        batch, seq_len = idx.size()

        assert seq_len <= self.config.block_size, f"Cannot forward sequence of length {seq_len}, block size is only {self.config.block_size}"

        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)

        token_embeddings = self.transformer.token_embedding(idx)
        positional_embeddings = self.transformer.positional_embedding(pos)

        # Combine positional and token embeddings
        x = self.transformer.dropout(token_embeddings+positional_embeddings)

        for block in self.transformer.attention_blocks:
            x = block(x)

        x = self.transformer.layer_norm_final(x)

        if targets is not None:
            logits = self.language_model_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.language_model_head(x[:,[-1],:])
            loss = None

        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, betas):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        extra_args = dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.positional_embedding.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)





