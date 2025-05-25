import torch
import torchaudio
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR

from tqdm import tqdm

from model import SoundStream, STFTDiscriminator, WaveDiscriminator, generator_loss, discriminator_loss
from model import generator_reconstruction_loss, generator_multispectral_reconstruction_loss, commit_loss

import torch
from torch.utils.tensorboard import SummaryWriter

num_epochs = 1
batch_size = 128
max_grad_norm = 0.5
warmup_steps = 1000
lr=2e-4
betas = (0.9, 0.99)
max_iter = 50000
rq_ema_gamma = 0.95

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()


G = SoundStream(C=32,
                rq_ema_gamma=rq_ema_gamma).to(device)
# stft_D = STFTDiscriminator(C=16).to(device)
# wave_D = WaveDiscriminator().to(device)


G_optimizer = torch.optim.Adam(G.parameters()     , lr=lr, betas=betas)
G_warmup_scheduler = LinearLR(G_optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
G_constant_scheduler = ConstantLR(G_optimizer, factor=1.0, total_iters=1) 
G_scheduler = SequentialLR(G_optimizer, schedulers=[G_warmup_scheduler, G_constant_scheduler], milestones=[warmup_steps])

# D_optimizer = torch.optim.Adam(list(stft_D.parameters()) + list(wave_D.parameters()), lr=1e-4, betas=(0.5, 0.9))

def collate_fn(batch, target_length=24000):
    processed_batch = []
    for item, _, _, _, _, _, _ in batch:
        # item: shape (1, 1, L)
        current_length = item.shape[-1]
        if current_length < target_length:
            # Pad at the end along the last dimension
            pad_amount = target_length - current_length
            padded = F.pad(item, (0, pad_amount))  # Pad only last dimension
        else:
            # Truncate along the last dimension
            padded = item[..., :target_length]
        processed_batch.append(padded)
    
    return torch.stack(processed_batch)

librtts_data = torchaudio.datasets.LIBRITTS('.', download=False)
data_loader = torch.utils.data.DataLoader(
    librtts_data,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=7)

n_iter = 0
done = False
while not done:
    for x in tqdm(data_loader):

        x = x.to(device)
        # # Train Discriminators
        # with torch.no_grad():
        #     Gx = G(x)
        
        # Dx =  [item[-1] for item in [stft_D(x) ] + wave_D(x) ] # only take the output feature map
        # DGx = [item[-1] for item in [stft_D(Gx)] + wave_D(Gx)] # only take the output feature map

        # D_loss = discriminator_loss(Dx, DGx)

        # D_optimizer.zero_grad()
        # D_loss.backward()
        # D_optimizer.step()

        # Train generator
        Gx = G(x)

        multi_spec_loss = generator_multispectral_reconstruction_loss(x, Gx)
        recon_loss = generator_reconstruction_loss(x, Gx)
        com_loss = commit_loss(G)
        G_loss = multi_spec_loss + recon_loss + com_loss

        current_lr = G_scheduler.get_last_lr()[0]  # returns a list (one per param group)
        mean_enc_norm = torch.linalg.vector_norm(G.saved_encoding, dim=-1, ord=2).mean().item()
        writer.add_scalar("Encoder/mean_l2norm_embedding", mean_enc_norm, n_iter)

        writer.add_scalar("Loss/multi_spec_loss", multi_spec_loss, n_iter)
        writer.add_scalar("Loss/recon_loss", recon_loss, n_iter)
        writer.add_scalar("Loss/commit_loss", com_loss, n_iter)
        writer.add_scalar("Scheduler/lr", current_lr, n_iter)

        for i, vq in enumerate(G.rvq.vqs):
            mean_norm = torch.linalg.vector_norm(vq.codebook, dim=-1, ord=2).mean().item()  # norm per vector
            writer.add_scalar(f"Quantizers/l2norm_codebook_{i}", mean_norm, n_iter)
           


        # with torch.no_grad():
        #     Dx  = [stft_D(x) ] + wave_D(x )
        #     DGx = [stft_D(Gx)] + wave_D(Gx)

        # G_loss = generator_loss(x, Gx, Dx, DGx)
        G_optimizer.zero_grad()
        G_loss.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), max_grad_norm)
        G_optimizer.step()
        G_scheduler.step()
        G.rvq.update_codebook()

        n_iter += 1
        if n_iter >= max_iter:
            done = True
            break

torch.save(G.state_dict(), 'soundstream_weights.pth')
writer.flush()
writer.close()