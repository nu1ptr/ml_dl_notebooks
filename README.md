**Warning!: Currently cleaning up these notebooks (somewhat in a "hacky" state), so view and use at your own peril.**

**Note: This is actively being worked on, some parts might not work and there are additional notebooks to be added.**

**Hardware specs used to train models: 64GB RAM, 48GB GPU, 20 threads (With smaller datasets can just toss it all into the RAM or GPU)**

# General Todo:
1. Create docker container.
2. Upload all trained pytorch models (Requires a significant amount of cloud storage)
3. Add Latent Exploration Visualization Outputs to all implemented models.
4. Cleanup RealNVP (Almost Done)
5. Clenup Glow
6. Clean DRAW
7. Add WaveNet


[All Trained Pytorch Models (Google Drive)](https://drive.google.com/drive/folders/1Uh4wO9dwKD3WlqFVZXUaTnG4v2MfeYpW?usp=sharing)

# Attention Models
1. [DRAW: A Recurrent Neural Network For Image Generation](https://arxiv.org/pdf/1502.04623.pdf)

# Flow Models
1. [NICE: Non-Linear Independent Components Estimation](https://arxiv.org/pdf/1410.8516.pdf)
2. [Density Estimation Using Real NVP](https://arxiv.org/pdf/1605.08803.pdf)
3. [Glow: Generative Flows with Invertible 1x1 Convolutions](https://arxiv.org/pdf/1807.03039.pdf)

## Tips / Tricks & Resources
1. [Understanding why pinned-memory is faster for DataLoader](https://leimao.github.io/blog/Page-Locked-Host-Memory-Data-Transfer/#:~:text=With%20paged%20memory%2C%20the%20specific,not%20communicate%20with%20hard%20drive.)
2. [Why Gradient Clipping Accelerates Training](https://openreview.net/pdf?id=BJgnXpVYwS)
3. [Pytorch Performance Tuning Guide](https://www.youtube.com/watch?v=9mS1fIYj1So)
4. cuDNN AutoTuner for CNNs
`torch.backends.cudnn.benchmark = True`
5. Increase batch size to max out GPU Memory but also remember to tune learning rates or switch to optimizer designed for large-batch training due to convergence issues.
6. Disable bias for convolutions directly followed by a batch norm since batch norm mean calculations cancel out the bias of convolution.
7. Use parameter.grad = None instead of model.zero_grad() since it does note execute memset for every parameter and memory is zeroed-out by allocator which is more efficient.
`for param in model.parameters():
    param.grad = None`
8. Use Fuse pointwise operations via @torch.jit.script. Pointwise operations are memory bound and require multiple kernel calls. jit will fuse these kernels reducing overhead.
