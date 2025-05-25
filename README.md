## Tensorboard Setup

Run this locally to set up port forwarding:
```bash
ssh -N -f -L localhost:16006:localhost:6006 mpatel636@login-ice.pace.gatech.edu 
```

Run this on the remote server:
```bash
sbatch train.sbatch # begins training
sbatch tensorboard.sbatch # begins tensorboard logging
```

Then go into your browser and type in `http://localhost:16006`