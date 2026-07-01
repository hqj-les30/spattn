import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (B, 3, 32, 32) -> (B, 32, 32, 32)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> (B, 64, 32, 32)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),                              # -> (B, 64, 16, 16)

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # -> (B, 128, 16, 16)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),                              # -> (B, 128, 8, 8)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),                                 # -> (B, 128*8*8)
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)                            # -> (B, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class CNN_cifar(nn.Module):
    def __init__(self, n_class=10):
        super(CNN_cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.global_fc1 = nn.Linear(128 * 4 * 4, 128)
        self.global_fc2 = nn.Linear(128, n_class)

        self.reserved = [
            'global_fc1', 'global_fc2'
        ]

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool1(x)

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool2(x)

        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.pool3(x)

        x = x.view(-1, 128 * 4 * 4)

        x = self.global_fc1(x)
        # x = F.relu(self.global_fc1(x))
        x = self.global_fc2(x)

        # return F.softmax(x, dim=1)
        return x

class CNN_fmnist(nn.Module):
    def __init__(self, n_class=10):
        super(CNN_fmnist, self).__init__()
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.convlayer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.global_fc1 = nn.Linear(64*6*6, 128)
        self.global_fc2 = nn.Linear(128, n_class)

        self.reserved = [
            'global_fc1', 'global_fc2', 'convlayer2'
        ]

    def forward(self, x):
        x = self.convlayer1(x)        
        x = self.convlayer2(x)
        x = x.view(-1, 64*6*6)       
        x = self.global_fc1(x)
        x = self.global_fc2(x)
        # x = self.global_fc3(x)
        return x

class UCIHARNet(nn.Module):
    def __init__(self, n_class=6, in_dim=561):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, n_class)
        )

    def forward(self, x):
        return self.net(x)

def set_model_fn(name='cifar10'):
    if name == 'cifar10':
        return CNN_cifar
    elif name == 'fashion':
        return CNN_fmnist
    elif name == 'har':
        return UCIHARNet

# ============================================
# Regular Deep Q-Network
# ============================================

class ScaledLearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        """
        Learnable positional encoding with learnable scale.

        Args:
            d_model (int): Embedding dimension.
            max_len (int): Maximum sequence length.
        """
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with scaled positional encoding added.
        """
        seq_len = x.size(1)
        position_ids = torch.arange(seq_len, device=x.device)
        pe = self.pos_embedding(position_ids).unsqueeze(0)
        return x + self.scale * pe

class MultiStepAgent(nn.Module):
    def __init__(
            self,
            d_raw_feature: int = 128,
            d_embedding: int = 32,
            d_attention: int = 64,
            d_stats: int = 1,
            nhead: int = 2,
            nclasses: int = 10,
            max_user_size: int = 100,
            max_history_size: int = 16,
            use_temporal_attn: bool = True,
            use_spatial_attn: bool = True,
            encoder: str = 'attention',
            enc_input: str = 'abs'
            ):
        super().__init__()
        # component-level ablation switches (C2)
        self.use_temporal_attn = use_temporal_attn
        self.use_spatial_attn = use_spatial_attn
        self.encoder = encoder
        # combo (WT2): encoder input space + /scale divisor conditioning. 'abs'=RP(W_i)
        # (/1024); 'grad'/'scalar' are z-scored to O(1) upstream (/1.0) so the
        # Linear+LayerNorm does not re-crush them into the bias regime.
        self.enc_input = enc_input
        self.d_raw_feature = d_raw_feature
        self.d_embedding = d_embedding
        self.nclasses = nclasses
        # feature extractor of model params
        self.model_param_embedding = nn.Sequential(
            nn.Linear(d_raw_feature, d_raw_feature),
            nn.GELU(),
            nn.LayerNorm(d_raw_feature)
            # nn.BatchNorm1d(d_raw_feature)
        )
        # anticipate output distribution of model param
        self.distribution_anticipation_head = nn.Sequential(
            nn.Linear(d_raw_feature, nclasses),
            nn.Softmax(dim=-1)
        )
        # learn the bias induced by current model id and update amplitude
        self.user_token_embedding = nn.Sequential(
            nn.Linear(d_raw_feature + d_embedding + d_stats, d_attention * 4),
            nn.GELU(),
            nn.Linear(d_attention * 4, d_attention),
            nn.LayerNorm(d_attention)
        )
        # learn the bias induced by current cluster
        self.state_embedding = nn.Sequential(
            nn.Linear(d_raw_feature + d_embedding, d_attention * 4),
            nn.GELU(),
            nn.Linear(d_attention * 4, d_attention),
            nn.LayerNorm(d_attention)
        )
        # id embedding
        self.id_embedding = nn.Sequential(
            nn.Embedding(max_user_size, d_embedding),
            nn.LayerNorm(d_embedding)
        )
        # self attention of user tokens
        self.user_attention = nn.MultiheadAttention(
            embed_dim=d_raw_feature,
            num_heads=nhead,
            dropout=0.0,
            batch_first=True
        )
        # position encoding of temporal tokens
        self.pe = ScaledLearnablePositionalEncoding(d_raw_feature, max_len= max_history_size)
        # self attention of temporal tokens
        self.temproal_attention = nn.MultiheadAttention(
            embed_dim=d_raw_feature,
            num_heads=nhead,
            dropout=0.0,
            batch_first=True
        )
        # cross attention to learn the bias induced by history
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_raw_feature,
            num_heads=nhead,
            dropout=0.0,
            batch_first=True
        )
        # Q_heads
        self.value_head = nn.Sequential(
            nn.Linear(d_raw_feature + d_embedding, (d_raw_feature+d_embedding) * 2),
            nn.GELU(),
            nn.Linear((d_raw_feature + d_embedding) * 2, 1)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear((d_raw_feature+d_embedding) * 2, (d_raw_feature+d_embedding) * 4),
            nn.GELU(),
            nn.Linear((d_raw_feature+d_embedding) * 4, 1)
        )
        # A5c: optional MLP history encoder (replaces the attention stack).
        # Parameter count is matched to the attention encoder it replaces so
        # capacity differences do not confound the comparison.
        if encoder == 'mlp':
            target = sum(p.numel() for m in (self.user_attention, self.temproal_attention,
                                             self.cross_attention, self.pe) for p in m.parameters())
            self.mlp_encoder = self._build_mlp_encoder(
                in_dim=3 * d_raw_feature + d_embedding,
                out_dim=d_raw_feature,
                target_params=target,
            )

    @staticmethod
    def _build_mlp_encoder(in_dim: int, out_dim: int, target_params: int) -> nn.Sequential:
        """2-hidden-layer Linear+GELU MLP whose parameter count approximates target_params."""
        def total(h: int) -> int:
            # 3 linear layers (in->h, h->h, h->out) with biases
            return in_dim * h + h + h * h + h + h * out_dim + out_dim
        best_h, best_diff = 64, None
        for h in range(64, 4097, 32):
            diff = abs(total(h) - target_params)
            if best_diff is None or diff < best_diff:
                best_diff, best_h = diff, h
        h = best_h
        return nn.Sequential(
            nn.Linear(in_dim, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, out_dim),
        )

    def forward(self, x_u: torch.Tensor, x_s: torch.Tensor, indicators: torch.Tensor, distribution: torch.Tensor, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''
        x_u: updated model params of each user
        x_s: global param before update
        indicators: ids of current cluster
        history: global params and action of last round
        '''
        B, N, _ = x_u.shape
        # combo (WT2): grad/scalar inputs are z-scored to O(1) upstream, so divide by
        # 1.0 here; the abs-tuned /1024 would otherwise crush them back into the
        # Linear-bias regime and re-trigger input collapse. abs stays /1024.
        scale = 1024.0 if self.enc_input == 'abs' else 1.0
        history_param = history[:, :, :self.d_raw_feature] / scale
        history_param_emb = self.model_param_embedding(history_param)

        id_emb = self.id_embedding(indicators)

        x_u_grad = x_u - x_s.unsqueeze(1)
        x_u_amp = x_u_grad.norm(dim=-1, keepdim=True)
        x_u_emb = self.model_param_embedding(x_u / scale)

        x_s_emb = self.model_param_embedding(x_s / scale)

        if self.encoder == 'mlp':
            # A5c: MLP history encoder. Per-client concat of
            # [RP(W_i); meanpool{RP(W_j)}; meanpool{RP(W_glob)}; emb(i)].
            cluster_pool = x_u_emb.mean(dim=1, keepdim=True).expand(-1, N, -1)
            temporal_pool = torch.cat([history_param_emb, x_s_emb.unsqueeze(1)], dim=1).mean(dim=1, keepdim=True).expand(-1, N, -1)
            mlp_in = torch.cat([x_u_emb, cluster_pool, temporal_pool, id_emb], dim=-1)
            x_u_token = self.mlp_encoder(mlp_in)
        else:
            temporal_token = torch.cat([history_param_emb, x_s_emb.unsqueeze(1)], dim=1)
            temporal_token = self.pe(temporal_token)

            if self.use_spatial_attn:
                x_u_token = x_u_emb + self.user_attention(
                    query=x_u_emb,
                    key=x_u_emb,
                    value=x_u_emb,
                    key_padding_mask=mask
                )[0]
            else:
                # A3c: w/o spatial attention
                x_u_token = x_u_emb
            if mask is not None:
                x_u_token = x_u_token.masked_fill(mask.unsqueeze(-1), 0.0)
            if self.use_temporal_attn:
                temporal_token = temporal_token + self.temproal_attention(
                    query=temporal_token,
                    key=temporal_token,
                    value=temporal_token,
                    attn_mask=nn.Transformer.generate_square_subsequent_mask(temporal_token.size(1)).to(x_s.device)
                )[0]
            # cross attention is retained (isolates the self-attention blocks only)
            x_u_token = x_u_token + self.cross_attention(
                query=x_u_token,
                key=temporal_token,
                value=temporal_token
            )[0]

        # common tail: concat id_emb -> pool -> dueling heads
        if mask is not None:
            x_u_token = x_u_token.masked_fill(mask.unsqueeze(-1), 0.0)
            id_emb = id_emb.masked_fill(mask.unsqueeze(-1), 0.0)
        x_u_token = torch.cat([x_u_token, id_emb], dim=-1)
        if mask is not None:
            x_u_token_pool = x_u_token.sum(dim=1, keepdim=True) / (~mask).sum(dim=1, keepdim=True).float().unsqueeze(-1)
        else:
            x_u_token_pool = x_u_token.mean(dim=1, keepdim=True)
        x_u_condition = torch.cat([x_u_token, x_u_token_pool.expand(-1, N, -1)], dim=-1) # (B, N, D_attention*2)
        adv = self.advantage_head(x_u_condition)
        adv = adv - adv.mean(dim=1, keepdim=True)
        val = self.value_head(x_u_token.mean(dim=1))
        q = val.unsqueeze(1) + adv

        output_distribution = self.distribution_anticipation_head(x_s_emb)
        return q, output_distribution

class MultiStepAgentNoemb(MultiStepAgent):
    def __init__(
            self,
            **kwargs
            ):
        super().__init__(**kwargs)
        d_raw_feature = self.d_raw_feature
        self.value_head = nn.Sequential(
            nn.Linear(d_raw_feature, (d_raw_feature) * 2),
            nn.GELU(),
            nn.Linear((d_raw_feature) * 2, 1)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear((d_raw_feature) * 2, (d_raw_feature) * 4),
            nn.GELU(),
            nn.Linear((d_raw_feature) * 4, 1)
        )
        self.id_embedding = None
    
    def forward(self, x_u: torch.Tensor, x_s: torch.Tensor, indicators: torch.Tensor, distribution: torch.Tensor, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''
        x_u: updated model params of each user
        x_s: global param before update
        indicators: ids of current cluster
        history: global params and action of last round
        '''
        # Placeholder for forward method
        B, N, _ = x_u.shape
        # combo-attention (WT2): mirror the /scale divisor conditioning of the base
        # forward so -q noemb also honors --enc-input (default 'abs' = /1024 = master).
        scale = 1024.0 if self.enc_input == 'abs' else 1.0
        # history_action = history[:, :, self.d_raw_feature:].to(torch.long)
        # history_action_emb = self.id_embedding(history_action).mean(dim=2)
        history_param = history[:, :, :self.d_raw_feature] / scale
        history_param_emb = self.model_param_embedding(history_param)
        # history_token = torch.cat(
        #     [history_param_emb, history_action_emb], dim=-1
        # )

        # id_emb = self.id_embedding(indicators)

        x_u_grad = x_u - x_s.unsqueeze(1)
        x_u_amp = x_u_grad.norm(dim=-1, keepdim=True)
        x_u_emb = self.model_param_embedding(x_u / scale)
        # x_u_token = torch.cat([x_u_emb, id_emb, torch.log(x_u_amp + 1e-8)], dim=-1) # (B, N, D_raw + D_emb + D_stats)
        # x_u_token = self.user_token_embedding(x_u_token)

        x_s_emb = self.model_param_embedding(x_s / scale)
        # x_s_token = torch.cat([x_s_emb, id_emb.mean(dim=1)], dim=-1) # (B, D_raw + D_emb)
        # temporal_token = torch.cat([history_token, x_s_token.unsqueeze(1)], dim=1) # (B, 2, D_raw + D_emb)
        temporal_token = torch.cat([history_param_emb, x_s_emb.unsqueeze(1)], dim=1)
        # temporal_token = self.state_embedding(temporal_token)
        temporal_token = self.pe(temporal_token)

        x_u_token = x_u_emb + self.user_attention(
            query=x_u_emb,
            key=x_u_emb,
            value=x_u_emb,
            key_padding_mask=mask
        )[0]
        if mask is not None:
            x_u_token = x_u_token.masked_fill(mask.unsqueeze(-1), 0.0)
        temporal_token = temporal_token + self.temproal_attention(
            query=temporal_token,
            key=temporal_token,
            value=temporal_token,
            attn_mask=nn.Transformer.generate_square_subsequent_mask(temporal_token.size(1)).to(x_s.device)
        )[0]

        x_u_token = x_u_token + self.cross_attention(
            query=x_u_token,
            key=temporal_token,
            value=temporal_token
        )[0]
        if mask is not None:
            x_u_token = x_u_token.masked_fill(mask.unsqueeze(-1), 0.0)
            # id_emb = id_emb.masked_fill(mask.unsqueeze(-1), 0.0)
        # x_u_token = torch.cat(
        #     [x_u_token, id_emb], dim=-1
        # )
        if mask is not None:
            x_u_token_pool = x_u_token.sum(dim=1, keepdim=True) / (~mask).sum(dim=1, keepdim=True).float().unsqueeze(-1)
        else:
            x_u_token_pool = x_u_token.mean(dim=1, keepdim=True)
        x_u_condition = torch.cat([x_u_token, x_u_token_pool.expand(-1, N, -1)], dim=-1) # (B, N, D_attention*2)
        adv = self.advantage_head(x_u_condition)
        adv = adv - adv.mean(dim=1, keepdim=True)
        val = self.value_head(x_u_token.mean(dim=1))
        q = val.unsqueeze(1) + adv

        output_distribution = self.distribution_anticipation_head(x_s_emb)
        return q, output_distribution


class MultiStepAgentVec(MultiStepAgent):
    """WT5 Path B: per-class (vectorized) dueling Q-net.

    Rebuilds value/advantage heads to output ``nclasses`` dims so the inherited
    forward yields ``q`` of shape (B, N, C) instead of (B, N, 1). The attention
    encoder (spatial/temporal/cross) is inherited unchanged, so the only difference
    vs MultiStepAgent is the per-class output structure.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        d = self.d_raw_feature + self.d_embedding
        C = self.nclasses
        self.value_head = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, C)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(d * 2, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, C)
        )


def set_qnet_fn(name='mlp'):
    if name == 'noemb':
        # return DuelingQNetwork
        return MultiStepAgentNoemb
    elif name == 'singleT':
        return FLA_SingleTransformer_Agent
    elif name.lower() == 'selfcross':
        return SelfCrossAttentionAgent
    elif name.lower() == 'matrix':
        return MatrixQAgent
    elif name.lower() == 'multistep':
        return MultiStepAgent
    elif name.lower() == 'multistep_vec':
        # WT5 Path B: per-class dueling Q-net -> q of shape (B, N, C)
        return MultiStepAgentVec
