## Training Progress

### 🎧 Real vs. Fake Audio Samples

| Timestep | Real Audio | Fake (Generated) Audio |
|----------|------------|------------------------|
| 50k      | [🔊 Real 50k](media/real_50k.wav) | [🤖 Fake 50k](media/fake_50k.wav) |
| 80k      | [🔊 Real 80k](media/real_80k.wav) | [🤖 Fake 80k](media/fake_80k.wav) |
| 100k     | [🔊 Real 100k](media/real_100k.wav) | [🤖 Fake 100k](media/fake_100k.wav) |

### Training Curves

For codebook diversity metric, score of 1 implies codebooks are used unformly (GOOD). Score of 0 implies only one codebook vector is being used (BAD).

| Metric   |                    | 
|----------|------------------------------|
| Loss  | ![Loss 1](media/loss1.png) |
|       | ![Loss 2](media/loss2.png) |
| Encoder Embeddings L2 Norm| ![Enc 50k](media/enc_norm.png) |
| Codebook Diversity | ![Entropy 1](media/entropy1.png)  |
|                    | ![Entropy 2](media/entropy2.png)  |
|                    | ![Entropy 3](media/entropy3.png)  |




