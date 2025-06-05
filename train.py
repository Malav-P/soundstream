import torch
import torchaudio
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR

from tqdm import tqdm
import warnings
import math

from model import SoundStream, STFTDiscriminator, WaveDiscriminator, generator_loss, discriminator_loss
from model import generator_reconstruction_loss, generator_multispectral_reconstruction_loss, commit_loss

from torch.utils.tensorboard import SummaryWriter

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

def generator_warmup(batch_size=128,
                     max_grad_norm=0.5,
                     warmup_steps=1000,
                     G_lr=2e-4,
                     G_betas=(0.9, 0.99),
                     n_iter=0,
                     max_iter=50000,
                     rq_ema_gamma=0.95,
                     use_quantizer_dropout=True,
                     C=32,
                     save_every=5000,
                     weights=(0., 0., 1.0, 1.0, 1.0),
                     log_dir=None,
                     save_dir=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir)

    G = SoundStream(C=C,
                    rq_ema_gamma=rq_ema_gamma,
                    use_quantizer_dropout=use_quantizer_dropout
                    ).to(device)

    save_dir = '' if save_dir is None else save_dir

    try:
        checkpoint_G = torch.load(save_dir + f'soundstream_{n_iter//1000}k.pth', map_location=device, weights_only=True)
        G.load_state_dict(checkpoint_G)

    except FileNotFoundError as e:
        warning_msg = f"\033[93m[WARNING] {e}\033[0m"
        warnings.warn(warning_msg)

    G_optimizer = torch.optim.Adam(G.parameters(), lr=G_lr, betas=G_betas)
    G_warmup_scheduler = LinearLR(G_optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
    G_constant_scheduler = ConstantLR(G_optimizer, factor=1.0, total_iters=1) 
    G_scheduler = SequentialLR(G_optimizer, schedulers=[G_warmup_scheduler, G_constant_scheduler], milestones=[warmup_steps])

    librtts_data = torchaudio.datasets.LIBRITTS('.', download=False)
    data_loader = torch.utils.data.DataLoader(
        librtts_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=7)

    done = False
    while not done:
        for x in tqdm(data_loader):

            x = x.to(device)
            Gx = G(x)


            ######################### GENERATOR TRAINING ##########################
            G_loss, (adv_loss, feat_loss, multi_spec_loss, recon_loss, com_loss) = generator_loss(G, x, Gx, weights=weights)

            G_optimizer.zero_grad()
            G_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_grad_norm)
            G_optimizer.step()
            G_scheduler.step()

            G.rvq.update_codebook()

            ##### LOGGING ######################

            current_lr    = G_scheduler.get_last_lr()[0] 
            writer.add_scalar("Scheduler/lr", current_lr, n_iter)

            mean_enc_norm = torch.linalg.vector_norm(G.saved_encoding, dim=-1, ord=2).mean().item()
            writer.add_scalar("Encoder/mean_l2norm_embedding", mean_enc_norm, n_iter)

            writer.add_scalar("Loss/adv_loss", adv_loss, n_iter)
            writer.add_scalar("Loss/feat_loss", feat_loss, n_iter)
            writer.add_scalar("Loss/multi_spec_loss", multi_spec_loss, n_iter)
            writer.add_scalar("Loss/recon_loss", recon_loss, n_iter)
            writer.add_scalar("Loss/commit_loss", com_loss, n_iter)
            writer.add_scalar("Loss/gen_loss", G_loss, n_iter)


            for i, vq in enumerate(G.rvq.vqs):
                mean_norm = torch.linalg.vector_norm(vq.codebook, dim=-1, ord=2).mean().item()  # norm per vector
                writer.add_scalar(f"Quantizers/l2norm_codebook_{i}", mean_norm, n_iter)

                probs  = vq.N / torch.sum(vq.N)
                codebook_entropy = - (probs * torch.log2(probs)).sum().item() / math.log2(vq.codebook_size)   # 1 means uniform distribution (GOOD), 0 means dirac delta distribution (BAD)
                writer.add_scalar(f"Quantizers/entropy_codebook_{i}", codebook_entropy, n_iter)


            
            n_iter += 1

            if n_iter % save_every == 0:
                torch.save(G.state_dict(), save_dir + f'soundstream_{n_iter//1000}k.pth')

            if n_iter >= max_iter:
                done = True
                break

    
    writer.flush()
    writer.close()

def adversarial_training(batch_size=128,
                         max_grad_norm=0.5,
                         G_lr=2e-4,
                         D_lr=1e-4,
                         G_betas=(0.9, 0.99),
                         D_betas=(0.5, 0.9),
                         n_iter=50000,
                         max_iter=95000,
                         rq_ema_gamma=0.95,
                         use_quantizer_dropout=True,
                         C=32,
                         save_every=5000,
                         weights=(1.0, 0.01, 0.01, 1.0, 1.0),
                         log_dir=None,
                         save_dir=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir)

    save_dir = '' if save_dir is None else save_dir

    G = SoundStream(C=C,
                    rq_ema_gamma=rq_ema_gamma,
                    use_quantizer_dropout=use_quantizer_dropout).to(device)
    stft_D = STFTDiscriminator(C=C).to(device)
    wave_D = WaveDiscriminator().to(device)

    try:
        checkpoint_G = torch.load(save_dir + f'soundstream_{n_iter//1000}k.pth', map_location=device, weights_only=True)
        G.load_state_dict(checkpoint_G)

    except FileNotFoundError as e:
        warning_msg = f"\033[93m[WARNING] {e}\033[0m"
        warnings.warn(warning_msg)

    try:
        checkpoint_stft_D = torch.load(save_dir + f'stft_{n_iter//1000}k.pth'  , map_location=device, weights_only=True)
        stft_D.load_state_dict(checkpoint_stft_D)

    except FileNotFoundError as e:
        warning_msg = f"\033[93m[WARNING] {e}\033[0m"
        warnings.warn(warning_msg)

    try:
        checkpoint_wave_D = torch.load(save_dir + f'wave_{n_iter//1000}k.pth'  , map_location=device, weights_only=True)
        wave_D.load_state_dict(checkpoint_wave_D)

    except FileNotFoundError as e:
        warning_msg = f"\033[93m[WARNING] {e}\033[0m"
        warnings.warn(warning_msg)

    G_optimizer = torch.optim.Adam(G.parameters(), lr=G_lr, betas=G_betas)
    D_optimizer = torch.optim.Adam(list(stft_D.parameters()) + list(wave_D.parameters()), lr=D_lr, betas=D_betas)

    librtts_data = torchaudio.datasets.LIBRITTS('.', download=False)
    data_loader = torch.utils.data.DataLoader(
        librtts_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=7)


    done = False
    while not done:
        for x in tqdm(data_loader):

            x = x.to(device)

            Gx = G(x)
            Dx =  [stft_D(x) ] + wave_D(x) 

            ######################### DISCRIMINATOR TRAINING #########################

            DGx = [stft_D(Gx.detach())] + wave_D(Gx.detach())

            Dx_outputs_only  = [item[-1].squeeze() for item in Dx ]
            DGx_outputs_only = [item[-1].squeeze() for item in DGx]

            D_loss = discriminator_loss(Dx_outputs_only, DGx_outputs_only)

            D_optimizer.zero_grad()
            D_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(stft_D.parameters()) + list(wave_D.parameters()), max_grad_norm)
            D_optimizer.step()

            ######################### GENERATOR TRAINING ##########################

            DGx = [stft_D(Gx)] + wave_D(Gx)
            Dx = [[d.detach() for d in featmaps] for featmaps in Dx]

            G_loss, (adv_loss, feat_loss, multi_spec_loss, recon_loss, com_loss) = generator_loss(G, x, Gx, Dx, DGx, weights=weights)

            G_optimizer.zero_grad()
            G_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_grad_norm)
            G_optimizer.step()


            G.rvq.update_codebook()

            ##### LOGGING ######################

            writer.add_scalar("Loss/disc_loss", D_loss, n_iter)

            mean_enc_norm = torch.linalg.vector_norm(G.saved_encoding, dim=-1, ord=2).mean().item()
            writer.add_scalar("Encoder/mean_l2norm_embedding", mean_enc_norm, n_iter)

            writer.add_scalar("Loss/adv_loss", adv_loss, n_iter)
            writer.add_scalar("Loss/feat_loss", feat_loss, n_iter)
            writer.add_scalar("Loss/multi_spec_loss", multi_spec_loss, n_iter)
            writer.add_scalar("Loss/recon_loss", recon_loss, n_iter)
            writer.add_scalar("Loss/commit_loss", com_loss, n_iter)
            writer.add_scalar("Loss/gen_loss", G_loss, n_iter)


            for i, vq in enumerate(G.rvq.vqs):
                mean_norm = torch.linalg.vector_norm(vq.codebook, dim=-1, ord=2).mean().item()  # norm per vector
                writer.add_scalar(f"Quantizers/l2norm_codebook_{i}", mean_norm, n_iter)

                probs  = vq.N / torch.sum(vq.N)
                codebook_entropy = - (probs * torch.log2(probs)).sum().item() / math.log2(vq.codebook_size)   # 1 means uniform distribution (GOOD), 0 means dirac delta distribution (BAD)
                writer.add_scalar(f"Quantizers/entropy_codebook_{i}", codebook_entropy, n_iter)

            n_iter += 1

            if n_iter % save_every == 0:
                torch.save(G.state_dict(), save_dir + f'soundstream_{n_iter//1000}k.pth')
                torch.save(stft_D.state_dict(), save_dir + f'stft_{n_iter//1000}k.pth')
                torch.save(wave_D.state_dict(), save_dir + f'wave_{n_iter//1000}k.pth')

            if n_iter >= max_iter:
                done = True
                break



    writer.flush()
    writer.close()


if __name__ == "__main__":

    generator_warmup(log_dir="runs/generator_warmup/",
                     save_dir="checkpoints/generator_warmup/")
