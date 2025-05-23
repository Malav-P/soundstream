import torch
import tqdm
import torchaudio
from model import SoundStream, STFTDiscriminator, WaveDiscriminator, generator_loss, discriminator_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = SoundStream(C=16).to(device)
stft_D = STFTDiscriminator(C=16).to(device)
wave_D = WaveDiscriminator().to(device)

G_optimizer    = torch.optim.Adam(G.parameters()     , lr=0.001)
stft_optimizer = torch.optim.Adam(stft_D.parameters(), lr=0.001)
wave_optimizer = torch.optim.Adam(wave_D.parameters(), lr=0.001)

num_epochs = 1

librtts_data = torchaudio.datasets.LIBRITTS('.', download=False)
data_loader = torch.utils.data.DataLoader(
    librtts_data,
    batch_size=1,
    shuffle=True)

for _ in range(num_epochs):
    for x, sample_rate, _, _, _, _, _ in tqdm.tqdm(data_loader):
        
        x = x.squeeze(0)
        new_len = (x.shape[-1] // G.embedding_ratio) * G.embedding_ratio
        x = x[..., :new_len].to(device)

        # Train Discriminators
        with torch.no_grad():
            Gx = G(x)
        
        Dx =  [item[-1] for item in [stft_D(x) ] + wave_D(x) ] # only take the output feature map
        DGx = [item[-1] for item in [stft_D(Gx)] + wave_D(Gx)] # only take the output feature map

        D_loss = discriminator_loss(Dx, DGx)

        stft_optimizer.zero_grad()
        wave_optimizer.zero_grad()
        D_loss.backward()
        stft_optimizer.step()
        wave_optimizer.step()

        # Train generator
        Gx = G(x)
        with torch.no_grad():
            Dx  = [stft_D(x) ] + wave_D(x )
            DGx = [stft_D(Gx)] + wave_D(Gx)

        G_loss = generator_loss(x, Gx, Dx, DGx)

        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()
        G.rvq.update_codebook()