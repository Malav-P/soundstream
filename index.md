## Overview
This repository implements the Soundstream neural audio codec from scratch in pytorch, staying as true as possible to the [paper](https://arxiv.org/abs/2107.03312). We provide an implemenration of the model in `model.py` and training code in `train.py`. 

![](./media/soundstream.png)


## Dataset
We use the LibriTTS dataset, specifically the `train-clean-100` subset. It can be downloaded and used in pytorch with the following line:

```python
libritts_dataset = torchaudio.datasets.LIBRITTS('.', download=True)
```
## Training

Traning is done a single Nvidia L40S GPU with 8 CPU threads. 7 threads are used for dataloading.

We first did generator warmup ( by calling `generator_warmup()` in `train.py`) for 50k iterations (~ 4-5 hours). Parameters for training were

```python
batch_size=128,
max_grad_norm=0.5,      # gradients are clipped to avoid excessively large updates
warmup_steps=1000,      # number of warmup steps for linear warmup schedule
G_lr=2e-4,              # learning rate
G_betas=(0.9, 0.99),    # adam optimizer betas
rq_ema_gamma=0.95,      # vector quantizer codebook exponential moving average update gamma
use_quantizer_dropout=True, # whether to use quantizer dropout
C=32,                       # soundstream channel parameter
weights=(0., 0., 1.0, 1.0, 1.0), # (adversarial loss, feature loss, multi_spectral loss, reconstruction loss, commitment loss)
```

We then followed with adversarial training (`adversarial_training()` in `train.py`) for another 50k iterations (~ 2-3 hours).

```python
batch_size=16,
max_grad_norm=0.5,
G_lr=1e-4,
D_lr=1e-4,
G_betas=(0.9, 0.99),
D_betas=(0.5, 0.9),
update_D_every=1,                   # update the discriminator every ___ iterations
update_codebook_every=8,            # update the RVQ codebooks every ___ iterations
rq_ema_gamma=0.95,
use_quantizer_dropout=True,
C=32,
weights=(1.0, 0.1, 0.01, 1.0, 1.0), # (adversarial loss, feature loss, multi_spectral loss, reconstruction loss, commitment loss)
```

### 🎧 Real vs. Fake Audio Samples

We take 100 test samples from the `dev-clean` subset of LibriTTS for testing. We report a ViSQOL score which is calculated as the mean of MOS-LQO scores of the 100 samples. Not sure why the ViSQOL score is well below the reported number by the authors. Insights into why are welcome.

| Real Audio                        | Iteration No.|  Fake (Generated) Audio             | VISQOL|
|-----------------------------------|--------------|-------------------------------------|-------|
|[🔊 Real 50k](media/5259.wav) | 50k               |  [🤖 Fake 50k](media/5259_50k.wav)   |2.3049 |
|                              | 80k               |  [🤖 Fake 80k](media/5259_80k.wav)   |1.9633 |
|                              | 100k              |  [🤖 Fake 100k](media/5259_100k.wav) |2.2364 |

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
| Codebook Mean L2 Norm |![Norm 1](media/codebooknorm1.png) |
|                       |![Norm 2](media/codebooknorm2.png) |



### Training Tips

- ensure that the product of `batch_size` and `update_codebook_every` is equal to 128. Anything less than this and we experience the encoder embeddings explode as training progresses.

