import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram
from torchaudio import functional as aF

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = (self.dilation[0] * (self.kernel_size[0] - 1) + (1 - self.stride[0]), 0) # https://github.com/lucidrains/audiolm-pytorch/issues/8#issuecomment-1293727896

    def forward(self, x):
        return self._conv_forward(F.pad(x, self.causal_padding), self.weight, self.bias)

class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upsample_factor = self.stride[0]

    def forward(self, x):
        n = x.shape[-1]
        out = super().forward(x)
        return out[..., :(n * self.upsample_factor)]


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, dilation=dilation),
            nn.ELU(),
            CausalConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        )

    
    def forward(self, x):
        return F.elu(x + self.layers(x))

    
class EncoderBlock(nn.Module):
    def __init__(self, N, S):
        super().__init__()
        
        self.layers = nn.Sequential(
            ResidualUnit(in_channels=N//2, out_channels=N//2, dilation=1),
            ResidualUnit(in_channels=N//2, out_channels=N//2, dilation=3),
            ResidualUnit(in_channels=N//2, out_channels=N//2, dilation=9),
            CausalConv1d(kernel_size=2*S, in_channels=N//2, out_channels=N, stride=S)
        )

    def forward(self, x):
        return self.layers(x)
    
class Encoder(nn.Module):
    def __init__(self, C, embedding_dim, strides=(2, 4, 5, 8)):
        super().__init__()

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=1, out_channels=C, kernel_size=7),
            EncoderBlock(N=2*C, S=strides[0]),
            EncoderBlock(N=4*C, S=strides[1]),
            EncoderBlock(N=8*C, S=strides[2]),
            EncoderBlock(N=16*C, S=strides[3]),
            CausalConv1d(kernel_size=3, in_channels=16*C, out_channels=embedding_dim)
        )

        self.film = None # TODO

    def forward(self, x):
        x = self.layers(x)
        x = x.permute(0, 2, 1)
        return x
    

class DecoderBlock(nn.Module):
    def __init__(self, out_channels, S):
        super().__init__()
        self.layers = nn.Sequential(
            CausalConvTranspose1d(kernel_size=2*S, in_channels=2*out_channels,out_channels=out_channels, stride=S),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=1),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=3),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=9)
        )

    def forward(self, x):
        return self.layers(x)
    
class Decoder(nn.Module):
    def __init__(self, C, embedding_dim, strides=(8, 5, 4, 2)):
        super().__init__()
        self.film = None # TODO

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=embedding_dim, out_channels=16*C, kernel_size=7),
            DecoderBlock(out_channels=8*C, S=strides[0]),
            DecoderBlock(out_channels=4*C, S=strides[1]),
            DecoderBlock(out_channels=2*C, S=strides[2]),
            DecoderBlock(out_channels=  C, S=strides[3]),
            CausalConv1d(in_channels=C, out_channels=1, kernel_size=7)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.layers(x)
        return x
    
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_dim, codebook_size, num_groups=1, gamma=0.99, dead_codebook_ema_threshold=2):

        if codebook_dim % num_groups != 0:
            raise ValueError("codebook_dim must be divisible by num_groups")

        super().__init__()
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.num_groups = num_groups
        self.register_buffer('codebook', torch.empty(num_groups, codebook_size, codebook_dim // num_groups))
        self.register_buffer('N', torch.ones(num_groups, codebook_size))
        self.register_buffer('m', self.codebook.detach().clone())
        self.saved_input   = [[] for _ in range(self.num_groups)]
        self.saved_indices = [[] for _ in range(self.num_groups)]

        self.gamma = gamma
        self.dead_codebook_ema_threshold = dead_codebook_ema_threshold

        for group in self.codebook:
            torch.nn.init.kaiming_uniform_(group)

    def forward(self, x):

        B, S, D = x.shape
        G = self.num_groups
        x_grouped = x.view(B, S, G, D // G)
        quantized = torch.empty_like(x_grouped)

        for g, group in enumerate(self.codebook):
            x_g = x_grouped[:, :, g, :]    # (B, S, D // G)
            self.saved_input[g].append(x_g)  

            x_norm = (x_g ** 2).sum(dim=-1, keepdim=True)                  # (B,S,1)
            codebook_norm = (group ** 2).sum(dim=-1)             # (N,)
            dist = x_norm + codebook_norm.unsqueeze(0).unsqueeze(0) - 2 * x_g @ group.t()   # (B,S,N)

            indices = torch.argmin(dist, dim=-1)
            self.saved_indices[g].append(indices)                     # (B,S)
            quantized[:, :, g, :] = group[indices]                       # (B, S, D//G)

        quantized = quantized.view(B, S, D)   # (B, S, G, D // G) - > (B, S, D)

        # B, S, D = x.shape
        # G = self.num_groups
        
        # x_grouped = x.view(B, S, G, D // G)
        
        # # Compute norms
        # x_norm = (x_grouped ** 2).sum(dim=-1, keepdim=True)  # (B, S, G, 1)
        # codebook_norm = (self.codebook ** 2).sum(dim=-1)  # (G, N)
        
        # # Vectorized dot product using einsum
        # dot_products = torch.einsum('bsgd,gnd->bsgn', x_grouped, self.codebook)
        
        # # Compute distances
        # dist = x_norm + codebook_norm.unsqueeze(0).unsqueeze(0) - 2 * dot_products
        
        # # Find indices and quantize
        # indices = torch.argmin(dist, dim=-1)  # (B, S, G)
        
        # # Advanced indexing for quantization
        # quantized = self.codebook[torch.arange(G).view(1, 1, G), indices]
        
        # # Save for backward pass
        # for g in range(G):
        #     self.saved_input[g].append(x_grouped[:, :, g, :])
        #     self.saved_indices[g].append(indices[:, :, g].clone())
        
        # quantized = quantized.view(B, S, D)
         
        return x + (quantized - x).detach() # straight through estimator
    
    def update_codebook(self):

        for g in range(self.num_groups):

            saved_input_detached = torch.cat(self.saved_input[g], dim=0).detach().flatten(0,1) # (B, S, D//G) -> (N*B, S, D//G) -> (B*S*N, D//G) 
            saved_indices = torch.cat(self.saved_indices[g], dim=0).flatten() # (B, S) -> (N*B, S) -> (N*B*S,)

            mapped_idxs, inverse_indices = torch.unique(saved_indices, return_inverse=True, sorted=True)  # (num_unique_idxs,) , (B*S,)
            group_sizes = torch.bincount(inverse_indices)
            group_sums = torch.zeros(len(mapped_idxs), self.codebook_dim // self.num_groups, 
                            device=saved_input_detached.device, dtype=saved_input_detached.dtype)  # (num_unique_idxs, D//G)
            group_sums.scatter_add_(dim=0, index=inverse_indices.unsqueeze(1).expand(-1, self.codebook_dim // self.num_groups), 
                                    src=saved_input_detached)

            all_indices = torch.arange(self.codebook_size).to(mapped_idxs.device)
            mask = ~torch.isin(all_indices, mapped_idxs) 
            not_mapped_idxs = all_indices[mask]

            self.N[g, mapped_idxs] = self.gamma * self.N[g, mapped_idxs] + (1-self.gamma) * group_sizes
            self.m[g, mapped_idxs] = self.gamma * self.m[g, mapped_idxs] + (1-self.gamma) * group_sums
            self.codebook[g].copy_(self.m[g] / self.N[g, :, None])

            self.N[g, not_mapped_idxs] *= self.gamma
            self.m[g, not_mapped_idxs] *= self.gamma

        self._prune_unused_codes()

        self.saved_input = [[] for _ in range(self.num_groups)]
        self.saved_indices = [[] for _ in range(self.num_groups)]

    def _prune_unused_codes(self):
        to_be_pruned = self.N < self.dead_codebook_ema_threshold
        if not torch.any(to_be_pruned):
            return

        for g in range(self.num_groups):
            # Flatten input to choose random replacements
            replacement_candidates = torch.cat(self.saved_input[g], dim=0).detach().flatten(0,1)
            num_replacement_candidates = len(replacement_candidates)
            num_to_replace = to_be_pruned[g].sum()

            excess = num_to_replace - num_replacement_candidates
            if excess > 0 :
                true_indices = torch.nonzero(to_be_pruned[g], as_tuple=False).squeeze(1)
                drop_indices = true_indices[torch.randperm(true_indices.size(0))[:excess]]
                to_be_pruned[g, drop_indices] = False
                num_to_replace = num_replacement_candidates

            replacement_idxs = torch.randperm(num_replacement_candidates)[:num_to_replace]
            replacements = replacement_candidates[replacement_idxs]


            # Update codebook, m, and N in-place using masks
            self.codebook[g, to_be_pruned[g]] = replacements
            self.m[g, to_be_pruned[g]] = replacements
            self.N[g, to_be_pruned[g]] = self.dead_codebook_ema_threshold

    def get_commmit_loss(self):
        loss = 0
        for g in range(self.num_groups):
            x = self.saved_input[g][-1]
            quantized = self.codebook[g, self.saved_indices[g][-1]]
            loss = loss + F.mse_loss(quantized.detach(), x)

        return loss

        
class ResidualVectorQuantizer(nn.Module):
    def __init__(self, n_quantizers, codebook_dim, codebook_size, num_groups=1, gamma=0.99, use_quantizer_dropout=True):
        super().__init__()
        self.nquantizers = n_quantizers
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.gamma = gamma
        self.use_quantizer_dropout = use_quantizer_dropout
        self.vqs = nn.ModuleList([VectorQuantizer(codebook_dim=codebook_dim, codebook_size=codebook_size, num_groups=num_groups, gamma=gamma) for _ in range(self.nquantizers)])
        self.nq = self.nquantizers

    def forward(self, x):
        self.nq = torch.randint(1, self.nquantizers + 1, (1,)).item() if (self.use_quantizer_dropout and self.training) else self.nquantizers
        yhat = torch.zeros_like(x)
        residual = x
        for vq in self.vqs[:self.nq]:
            quantized = vq(residual)        # includes straight-through estimator
            yhat += quantized               # quantized = residual + (e - residual).detach()
            residual = residual - quantized # keep residual differentiable
        return yhat
    
    def update_codebook(self):
        for vq in self.vqs[:self.nq]:
            vq.update_codebook()


class ResidualUnit2D(nn.Module):
    def __init__(self, in_channels, C, m, downsample_factors):
        super().__init__()
        self.st, self.sf = downsample_factors

        self.layers = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3),
                      in_channels=in_channels, 
                      out_channels=C, 
                      padding=1),
            nn.ELU(),
            nn.Conv2d(kernel_size=(self.sf+2, self.st+2),
                      in_channels=C,
                      out_channels=m * C,
                      stride=(self.sf, self.st))
        )
        self.skip_connection = nn.Conv2d(in_channels=in_channels,
                                         out_channels=m*C,
                                         kernel_size=(1, 1),
                                         stride=(self.sf, self.st))

    def forward(self, x):
        return F.elu(self.skip_connection(x) + self.layers(F.pad(x, (self.st+1, 0, self.sf+1, 0))))

class STFTDiscriminator(nn.Module):
    def __init__(self, C, n_fft=1024, H=256, W=1024):
        super().__init__()
        self.H = H
        self.W = W
        self.n_fft = n_fft

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(7 , 7)),
                nn.ELU()
            ),
            ResidualUnit2D(in_channels=32 , C=  C, m=2, downsample_factors=(1, 2)),
            ResidualUnit2D(in_channels=2*C, C=2*C, m=2, downsample_factors=(2, 2)),
            ResidualUnit2D(in_channels=4*C, C=4*C, m=1, downsample_factors=(1, 2)),
            ResidualUnit2D(in_channels=4*C, C=4*C, m=2, downsample_factors=(2, 2)),
            ResidualUnit2D(in_channels=8*C, C=8*C, m=1, downsample_factors=(1, 2)),
            ResidualUnit2D(in_channels=8*C, C=8*C, m=2, downsample_factors=(2, 2)),
            nn.Conv2d(in_channels=16*C, out_channels=1, kernel_size=((W//2)//(2**6), 1))
        ])


    def forward(self, x):
        x = x.squeeze() # (B, 1, T) -> (B, T)
        x = torch.view_as_real(torch.stft(x,
                                          n_fft=self.n_fft,
                                          hop_length=self.H,
                                          win_length=self.W,
                                          window=torch.hann_window(self.W).to(x.device),
                                          return_complex=True))

        x = x.permute(0, 3, 1, 2) # (batch, N, time, channels) -> (batch, channels, N, time)
        feature_maps = []
        for layer in self.layers:
            x = layer(x)
            feature_maps.append(x)
        return feature_maps


class WaveDiscriminatorBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(kernel_size=15, stride=1, in_channels=1, out_channels=16),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.Conv1d(kernel_size=41, stride=4, groups=4, in_channels=16, out_channels=64),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.Conv1d(kernel_size=41, stride=4, groups=16, in_channels=64, out_channels=256),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.Conv1d(kernel_size=41, stride=4, groups=64, in_channels=256, out_channels=1024),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.Conv1d(kernel_size=41, stride=4, groups=256, in_channels=1024, out_channels=1024),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.Conv1d(kernel_size=5, stride=1, in_channels=1024, out_channels=1024),
                nn.LeakyReLU(),
            ),
            nn.Sequential(
                nn.Conv1d(kernel_size=3, stride=1, in_channels=1024, out_channels=1)
            )
        ])

    def forward(self, x):
        feature_maps = []
        for layer in self.layers:
            x = layer(x)
            feature_maps.append(x)
        return feature_maps
    
class WaveDiscriminator(nn.Module):
    def __init__(self, downsample_factors = [2, 4]):
        super().__init__()
    
        K = len(downsample_factors) + 1
        self.Ds = nn.ModuleList([WaveDiscriminatorBlock() for _ in range(K)])
        self.poolers = nn.ModuleList([nn.Identity()] + [nn.AvgPool1d(kernel_size=4, stride=s) for s in downsample_factors])

    def forward(self, x):
        outputs = []
        for pooler, discriminator in zip(self.poolers, self.Ds):
            y = discriminator(pooler(x))
            outputs.append(y)
        return outputs
    
def discriminator_loss(Dx, DGx):
    """
    Compute the discriminator loss

    Args:
        Dx (List[torch.tensor]): A length K list of torch tensors, representing the outputs of each of the 
                                 K discriminators when given original audio sample x. D(x). Each element of the list 
                                 will have shape (B, T_k) where T_k is dependent on position in the list
        DGx(List[torch.tensor]): A length K list of torch tensor, representing the outputs of each of the 
                                 K discriminators when given generated audio sample, xhat.  D(G(x)) D(x). Each element of the list 
                                 will have shape (B, T_k) where T_k is dependent on position in the list
    """
    K  = len(Dx)
    B  = Dx[0].shape[0]               
    losses = torch.empty(K, 2, B)

    for k, (real, fake) in enumerate(zip(Dx, DGx)):
        losses[k, 0] = torch.mean(F.relu(1-real), dim=-1) # (B, T_k) -> (B,)
        losses[k, 1] = torch.mean(F.relu(1+fake), dim=-1) # (B, T_k) -> (B,)

    loss = losses.mean(dim=(0, -1)).sum() # (K, 2, B) Mean -> (2,) Sum -> (1,)

    return loss

def generator_adversarial_loss(DGx):
    """
    Compute the adversarial generator loss

    Args:
        DGx(List[torch.tensor]): A length K list of torch tensor, representing the outputs of each of the 
                                 K discriminators when given generated audio sample, xhat.  D(G(x)). Each element of the list 
                                 will have shape (B, T_k) where T_k is dependent on position in the list
    """  
    K  = len(DGx)
    B  = DGx[0].shape[0]               
    losses = torch.empty(K, B)

    for k, fake in enumerate(DGx):
        losses[k] = torch.mean(F.relu(1-fake), dim=-1) # (B, T_k) -> (B,)

    loss = losses.mean() # (K, B) -> (1,)

    return loss

def generator_feature_loss(Dx_intermediates, DGx_intermediates):
    """
    Compute the generator feature loss

    Args:
        Dx_intermediates (List[ List[torch.tensor] ]) : List of length K. k-th item is a list of  feature maps 
                                          from the forward pass of the k-th discriminator on real sample
        DGx_intermediates (List[ List[torch.tensor] ]) : List of length K. k-th item is a list of feature maps 
                                          from the forward pass of the k-th discriminator on generated sample
    TODO:  might be better to just do l1 loss over these maps
    """
    K = len(Dx_intermediates)
    B = Dx_intermediates[0][0].shape[0]
    losses = torch.empty(K,B)
    for k, (Dx_featmaps, DGx_featmaps) in enumerate(zip(Dx_intermediates, DGx_intermediates)):

        L = len(Dx_featmaps) # Number of layers (i.e. number of feature maps)
        layer_losses = torch.empty(L, B)

        for l, (real, fake) in enumerate(zip(Dx_featmaps, DGx_featmaps)):
            # Reduce all dimensions except batch (0) and last (e.g., time) dimension
            dims_to_reduce = torch.arange(real.ndim)[1:-1].tolist() # need to reduce all dims except batch dim(first) and time dim (last)
            diff = torch.abs(real-fake)
            reduced = torch.sum(diff, dim=dims_to_reduce)        # Sum over extraneous axes (B, ..., T) -> (B, T)
            layer_losses[l] = torch.mean(reduced, dim=-1)    # Mean over time (B, T) -> (B,)
        
        losses[k] = layer_losses.mean(dim=0) # Mean over number of layers (L, B) -> (B,)

    loss = losses.mean() # Mean over batch (B) and discriminators (K)   (K, B) -> (1,)

    return loss

def generator_reconstruction_loss(x, Gx):
    return F.mse_loss(x, Gx)

def generator_multispectral_reconstruction_loss(x, Gx, windows=None, eps=1e-20):
    """
    Compute the reconstruction loss

    Args:
        x (torch.tensor): batched raw waveforms of shape (B, T)
        Gx (torch.tensor): batched fake waveforms of shape (B, T)
    
    Returns:
        loss (float): mel-spectrogram reconstruction loss
    """
    if not windows:
        windows = torch.tensor([2**i for i in range(6, 12)], dtype=torch.long)
    
    alphas = (windows // 2) ** 0.5
    loss = 0
    for alpha, s in zip(alphas, windows):
        real_spec = _compute_mel_spectrogram(x, sample_rate=24000, s=s, n_mels=64)
        fake_spec = _compute_mel_spectrogram(Gx, sample_rate=24000, s=s, n_mels=64)
        l1_loss = torch.linalg.vector_norm(real_spec - fake_spec, ord=1, dim=-2).mean()
        l2_loss = torch.linalg.vector_norm(torch.log(real_spec.clamp(min=eps)) - torch.log(fake_spec.clamp(min=eps)), ord=2, dim=-2).mean()
        loss = loss + l1_loss + alpha * l2_loss

    return loss 

def generator_loss(G, x, Gx, Dx=None, DGx=None, weights=None):
    """
    Compute the total generator loss which is a weighted combination of the adversarial, feature, and reconstruction losses

    Args:
        G (nn.Module) : the soundstream model
        x (torch.tensor): shape (B, T) batched raw audio waveforms
        Gx (torch.tensor): shape (B, T) batched fake audio waveforms
        Dx (List[List[torch.tensor]]): Feature maps from layers of each discriminator. k-th item is list
                                       with the feature maps from k-th discriminator
        DGx (List[List[torch.tensor]]): same as above, but for a waveform from the generator

    
    Returns:
        loss, (adv_loss, feat_loss, multi_spec_loss, rec_loss, com_loss)
    """
    if weights is None:
        weights = (1, 100, 1e-2, 1, 1)

    
    if (Dx is not None) and  (DGx is not None):
        DGx_logits        = [feat_maps[-1].squeeze() for feat_maps in DGx]
        Dx_intermediates  = [feat_maps[:-1] for feat_maps in Dx]
        DGx_intermediates = [feat_maps[:-1] for feat_maps in DGx]

        adv_loss        = generator_adversarial_loss(DGx_logits)
        feat_loss       = generator_feature_loss(Dx_intermediates, DGx_intermediates)

    else:
        adv_loss = torch.tensor([0.]).to(x.device)
        feat_loss = torch.tensor([0.]).to(x.device)


    multi_spec_loss = generator_multispectral_reconstruction_loss(x, Gx)
    rec_loss        = generator_reconstruction_loss(x, Gx)
    com_loss        = commit_loss(G)

    loss = weights[0]*adv_loss + weights[1]*feat_loss + weights[2]*multi_spec_loss + weights[3]*rec_loss + weights[4]*com_loss

    return loss, (adv_loss.detach(), feat_loss.detach(), multi_spec_loss.detach(), rec_loss.detach(), com_loss.detach())

def commit_loss(soundstream_model):
    """
    Compute the commitment loss of each quantizer and return the sum over all quantizers
    """
    rvq = soundstream_model.rvq

    loss = 0
    for vq in rvq.vqs[:rvq.nq]:
        loss = loss + vq.get_commmit_loss()

    return loss

def _compute_mel_spectrogram(x, sample_rate, s, n_mels):
    specgram = aF.spectrogram(waveform=x,
                              pad=0,
                              window=torch.hann_window(s, device=x.device),
                              n_fft=max(s, 512),
                              hop_length=s//4,
                              win_length=s,
                              power=2.0,
                              normalized=False,
                              center=True,
                              pad_mode="reflect",
                              onesided=True)
    fb = aF.melscale_fbanks(n_freqs=max(s, 512)//2 + 1,
                            f_min=0.0,
                            f_max=float(sample_rate//2),
                            n_mels=n_mels,
                            sample_rate=sample_rate).to(x.device)
    mel_specgram = torch.matmul(specgram.transpose(-1, -2), fb).transpose(-1, -2)

    return mel_specgram

class SoundStream(nn.Module):
    """Instance of SoundStream Model

    Args:
        C (int): channels
        embedding_dim (int): embedding dimension (i.e. dimension of latent space)
        n_quantizers (int): number of vector quantizers in the residual vector quantizer
        codebook_size (int): number of codebook vectors for each vector quantizer
        
        gamma (float): exponential moving average parameter for dictionary updates of codebook vectors. Default 0.99
    """
    def __init__(self,
                 C=32,
                 strides = (3, 4, 5, 8),
                 embedding_dim = 512,
                 n_quantizers = 8,
                 codebook_size = 1024,
                 num_groups = 1,
                 rq_ema_gamma=0.99,
                 use_quantizer_dropout=True):
        super().__init__()
        self.embedding_ratio = strides[0] * strides[1] * strides[2] * strides[3]
        self.enc = Encoder(C=C, embedding_dim=embedding_dim, strides=strides)
        self.rvq = ResidualVectorQuantizer(n_quantizers=n_quantizers,
                                           codebook_dim=embedding_dim,
                                           codebook_size=codebook_size,
                                           num_groups=num_groups,
                                           gamma=rq_ema_gamma,
                                           use_quantizer_dropout=use_quantizer_dropout)
        self.dec = Decoder(C=C, embedding_dim=embedding_dim, strides=strides[::-1])

    def forward(self, x):
        x = self.enc(x)
        self.saved_encoding = x.detach().clone()  # for debugging, can be removed
        x = self.rvq(x)
        x = self.dec(x)

        return x



if __name__ == "__main__":

    # from torch.profiler import profile, record_function, ProfilerActivity
    # import time

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = SoundStream(num_groups=2).to(device)
    # model.eval()

    # x = torch.randn(128,1,24000).to(device)

    # # with profile(activities=[
    # #         ProfilerActivity.CPU,
    # #         ProfilerActivity.CUDA],  # Only include CUDA if using GPU
    # #         record_shapes=True) as prof:
        
    # #     with record_function("model_inference"):
    # with torch.no_grad():
    #     model(x)
    #     start = time.perf_counter()
    #     model.rvq.update_codebook()
    #     end = time.perf_counter()

    #     print(end-start)

    # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

    import time

    B, S, D = (16, 50, 512)

    vq = VectorQuantizer(codebook_dim=512, codebook_size=1024, num_groups=2)

    x = torch.randn(B, S, D)

    
    y = vq(x)
    
    start = time.perf_counter()
    vq.update_codebook()
    end = time.perf_counter()  
     

    print(end-start)
