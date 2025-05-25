import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram
from torchaudio import functional as aF

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = (self.dilation[0] * (self.kernel_size[0] - 1) + (1 - self.stride[0]),
                               0)

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
        self.C = C

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
    def __init__(self, codebook_dim, codebook_size, gamma=0.99, dead_codebook_ema_threshold=2):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.register_buffer('codebook', torch.empty(codebook_size, codebook_dim))
        self.register_buffer('N', torch.ones(codebook_size))
        self.register_buffer('m', self.codebook.detach().clone())
        self.saved_input = None
        self.saved_indices = None

        self.gamma = gamma
        self.dead_codebook_ema_threshold = dead_codebook_ema_threshold

        torch.nn.init.kaiming_uniform_(self.codebook)

    def forward(self, x):
        self.saved_input = x
        x_norm = (x ** 2).sum(dim=-1, keepdim=True)                  # (B,S,1)
        codebook_norm = (self.codebook ** 2).sum(dim=-1)             # (N,)
        dist = x_norm + codebook_norm.unsqueeze(0).unsqueeze(0) - 2 * x @ self.codebook.t()   # (B,S,N)

        self.saved_indices = torch.argmin(dist, dim=-1)                     # (B,S)
        quantized = self.codebook[self.saved_indices]                       # (B, S, D)

        return x + (quantized - x).detach()
    
    def update_codebook(self):
        mapped_idxs = self.saved_indices.unique(sorted=True)
        all_indices = torch.arange(self.codebook_size).to(mapped_idxs.device)
        mask = ~torch.isin(all_indices, mapped_idxs) 
        not_mapped_idxs = all_indices[mask]

        group_sizes = torch.empty(len(mapped_idxs)).to(mapped_idxs.device)
        group_sums = torch.empty(len(mapped_idxs), self.codebook_dim).to(mapped_idxs.device)

        saved_input_detached = self.saved_input.detach()

        for i, idx in enumerate(mapped_idxs):
            group = saved_input_detached[self.saved_indices==idx]
            group_sizes[i] = group.shape[0]
            group_sums[i] = group.sum(dim=0)

        self.N[mapped_idxs] = self.gamma * self.N[mapped_idxs] + (1-self.gamma) * group_sizes
        self.m[mapped_idxs] = self.gamma * self.m[mapped_idxs] + (1-self.gamma) * group_sums
        self.codebook.copy_(self.m / self.N[:, None])

        self.N[not_mapped_idxs] *= self.gamma
        self.m[not_mapped_idxs] *= self.gamma

        self.prune_unused_codes()

    def prune_unused_codes(self):
        to_be_pruned = self.N < self.dead_codebook_ema_threshold
        if not torch.any(to_be_pruned):
            return

        # Flatten input to choose random replacements
        replacement_candidates = self.saved_input.detach().flatten(0, 1)
        replacement_idxs = torch.randperm(len(replacement_candidates))[:to_be_pruned.sum()]

        replacements = replacement_candidates[replacement_idxs]

        # Update codebook, m, and N in-place using masks
        self.codebook[to_be_pruned] = replacements
        self.m[to_be_pruned] = replacements
        self.N[to_be_pruned] = self.dead_codebook_ema_threshold
        


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, n_quantizers, codebook_dim, codebook_size, gamma=0.99, use_quantizer_dropout=True):
        super().__init__()
        self.nquantizers = n_quantizers
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.gamma = gamma
        self.use_quantizer_dropout = use_quantizer_dropout
        self.vqs = nn.ModuleList([VectorQuantizer(codebook_dim=codebook_dim, codebook_size=codebook_size, gamma=gamma) for _ in range(self.nquantizers)])
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
        self.C = C
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
        x = torch.view_as_real(torch.stft(x,
                                          n_fft=self.n_fft,
                                          hop_length=self.H,
                                          win_length=self.W,
                                          window = torch.hann_window(self.W).to(x.device),
                                          return_complex=True)
                                          )
        x = x.permute(0, 3, 1, 2)
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
        x = x.unsqueeze(1)
        feature_maps = []
        for layer in self.layers:
            x = layer(x)
            feature_maps.append(x)
        return feature_maps
    
class WaveDiscriminator(nn.Module):
    def __init__(self, downsample_factors = [2, 4]):
        super().__init__()
    
        self.K = len(downsample_factors) + 1
        self.Ds = nn.ModuleList([WaveDiscriminatorBlock() for _ in range(self.K)])
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
    losses = torch.empty(K, B)

    for i, output in enumerate(Dx):
        losses[i] = torch.mean((1-output).clamp(min=0), dim=-1) # (B,)
    loss1 = losses.mean()


    for i, output in enumerate(DGx):
        losses[i] = torch.mean((1+output).clamp(min=0), dim=-1) # (B,)
    loss2 = losses.mean()

    return loss1 + loss2

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

    for i, output in enumerate(DGx):
        losses[i] = torch.mean((1-output).clamp(min=0), dim=-1) # (B,)
    loss = losses.mean()

    return loss

def generator_feature_loss(Dx, DGx):
    """
    Compute the generator feature loss

    Args:
        Dx (List[ List[torch.tensor] ]) : List of length K. k-th item is a list of torch tensors representing feature maps 
                                          from the forward pass of the k-th discriminator
    """
    K = len(Dx)
    B = Dx[0][0].shape[0]
    losses = torch.empty(K,B)
    for j, (out1, out2) in enumerate(zip(Dx, DGx)):
        L = len(out1)
        mylist = torch.empty(L, B)
        for k, (real, fake) in enumerate(zip(out1, out2)):
            dims_to_reduce = tuple(i for i in range(real.ndim) if i != 0 and i != real.ndim - 1)
            real_minus_fake = (real-fake).sum(dims_to_reduce)
            mylist[k] = torch.mean(torch.abs(real_minus_fake), dim=-1)
        
        losses[j] = mylist.mean(dim=0) # shape (B,)
    loss = losses.mean()
    print(loss)
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
        loss += l1_loss + alpha * l2_loss

    return loss 

def generator_loss(x, Gx, Dx, DGx, weights=None):
    """
    Compute the total generator loss which is a weighted combination of the adversarial, feature, and reconstruction losses

    Args:
        x (torch.tensor): shape (B, T) batched raw audio waveforms
        Gx (torch.tensor): shape (B, T) batched fake audio waveforms
        Dx (List[List[torch.tensor]]): Feature maps from internal layers of each discriminator. k-th item is list
                                       with the feature maps from k-th discriminator
        DGx (List[List[torch.tensor]]): same as above, but for a waveform from the generator
    """
    if not weights:
        weights = [1, 100, 0.02]

    disc_outputs = [disc[-1].squeeze() for disc in DGx]

    adv_loss = generator_adversarial_loss(disc_outputs)
    feat_loss = generator_feature_loss(Dx, DGx)
    rec_loss = generator_reconstruction_loss(x, Gx)

    loss = weights[0]*adv_loss + weights[1]*feat_loss + weights[2]*rec_loss

    return loss

def commit_loss(soundstream_model):
    """
    Compute the commitment loss of each quantizer and return the sum over all quantizers
    """
    rvq = soundstream_model.rvq

    loss = 0
    for vq in rvq.vqs[:rvq.nq]:
        x = vq.saved_input
        quantized = vq.codebook[vq.saved_indices]
        loss = loss + F.mse_loss(quantized.detach(), x)

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
                 rq_ema_gamma=0.99,
                 use_quantizer_dropout=True):
        super().__init__()
        self.embedding_ratio = strides[0] * strides[1] * strides[2] * strides[3]
        self.enc = Encoder(C=C, embedding_dim=embedding_dim, strides=strides)
        self.rvq = ResidualVectorQuantizer(n_quantizers=n_quantizers,
                                           codebook_dim=embedding_dim,
                                           codebook_size=codebook_size,
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
    model = SoundStream()
    model.eval()

    x = torch.randn(2, 24319)

    y = model(x)

    print(y.shape)