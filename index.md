## Dataset
We use LibriTTS dataset, specifically with `train-clean-100` subset. It can be downloaded and used in pytorch with the following line:

```python
libritts_dataset = torchaudio.datasets.LIBRITTS('.', download=True)
```
## Training

Traning is done a single A100-80GB GPU with 8 CPU threads. 7 threads are used for dataloading.

We first did generator warmup ( by calling `generator_warmup()` in `train.py`) for 50k iterations. Parameters for training were

```python
batch_size=128,
max_grad_norm=0.5,      # gradients are clipped to avoid excessively large updates
warmup_steps=1000,      # number of warmup steps for linear warmup schedule
G_lr=2e-4,              # learning rate
G_betas=(0.9, 0.99),    # adam optimizer betas
rq_ema_gamma=0.95,      # vector quantizer codebook exponential moving average update gamma
use_quantizer_dropout=True, # wheter to use quantizer dropout
C=32,                       # soundstream channel parameter
weights=(0., 0., 1.0, 1.0, 1.0), # (adversarial loss, feature loss, multi_spectral loss, reconstruction loss, commitment loss)
```

We then followed with adversarial training (`adversarial_training()` in `train.py`) for another 50k iterations.

```python
batch_size=128,
max_grad_norm=0.5,
G_lr=2e-4,
D_lr=1e-4,
G_betas=(0.9, 0.99),
D_betas=(0.5, 0.9),
rq_ema_gamma=0.95,
use_quantizer_dropout=True,
C=32,
weights=(1.0, 0.1, 0.01, 1.0, 1.0), # (adversarial loss, feature loss, multi_spectral loss, reconstruction loss, commitment loss)
```

### 🎧 Real vs. Fake Audio Samples

We report a ViSQOL score of 2.40 which is well below the numbers reported in the original soundstream paper. Not sure why. Suggestions welcome.

| Timestep | Real Audio | Fake (Generated) Audio |
|----------|------------|------------------------|
| 50k      | [🔊 Real 50k](media/real_50k.wav) | [🤖 Fake 50k](media/fake_50k.wav) |
| 80k      | [🔊 Real 80k](media/real_80k.wav) | [🤖 Fake 80k](media/fake_80k.wav) |
| 100k     | [🔊 Real 100k](media/real_100k.wav) | [🤖 Fake 100k](media/fake_100k.wav) |

### Training Curves

For codebook diversity metric, score of 1 implies codebooks are used uniformly (GOOD). Score of 0 implies only one codebook vector is being used (BAD).

| Metric   |                    | 
|----------|------------------------------|
| Loss  | ![Loss 1](media/loss1.png) |
|       | ![Loss 2](media/loss2.png) |
| Encoder Embeddings L2 Norm| ![Enc 50k](media/enc_norm.png) |
| Codebook Diversity | ![Entropy 1](media/entropy1.png)  |
|                    | ![Entropy 2](media/entropy2.png)  |
|                    | ![Entropy 3](media/entropy3.png)  |




