# Humanoid Walking RL

Training a 3D humanoid to walk using reinforcement learning with GPU-accelerated physics simulation.

## Results

**Trained humanoid walking 45 meters with reward 8925:**

https://github.com/user-attachments/assets/humanoid_walking_final.mp4

The video in `output/humanoid_walking_final.mp4` shows the trained policy with camera tracking.

### Training Progression (Brax + PPO on A100)

| Step | Reward | Status |
|------|--------|--------|
| 0 | 90 | Falls immediately |
| 3.6M | 409 | Learning to balance |
| 10.8M | 700 | Walking attempts |
| 18M | 1,049 | Walking |
| 25.2M | 2,535 | Running |
| 32.4M | 2,767 | Stable running |

**Final evaluation: 1061 steps, 8925 reward, 45.3m walked**

## Architecture

- **Physics Engine**: [Brax](https://github.com/google/brax) (Google DeepMind) - JAX-based GPU-accelerated physics
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Environment**: Humanoid (17 joints, 376-dim observation, 17-dim action)
- **Parallel Environments**: 4,096 running simultaneously on GPU
- **Training Time**: ~8 minutes on NVIDIA A100 80GB (including JIT compilation)

## Files

| File | Description |
|------|-------------|
| `train_brax.py` | Main training script using Brax GPU physics + PPO |
| `render_video.py` | Renders trained policy with MuJoCo camera tracking |
| `train_mujoco.py` | Alternative: MuJoCo CPU-based training with Stable-Baselines3 |
| `requirements.txt` | Python dependencies |
| `output/` | Trained model and rendered videos |

## Quick Start

### Prerequisites

- NVIDIA GPU (A100 recommended, RTX 3090+ works)
- Python 3.10+
- CUDA 12+

### Install

```bash
pip install brax "jax[cuda12]" imageio imageio-ffmpeg Pillow
```

### Train

```bash
python train_brax.py
```

This runs 30M timesteps across 4096 parallel environments on GPU. Training takes ~8 minutes on A100.

### Render Video

After training, render the walking video with camera tracking:

```bash
MUJOCO_GL=osmesa python render_video.py
```

Requires `libosmesa6-dev` for headless rendering:
```bash
apt install libosmesa6-dev
```

## Alternative: MuJoCo CPU Training

For CPU-based training using MuJoCo + Stable-Baselines3:

```bash
pip install "gymnasium[mujoco]" stable-baselines3 imageio imageio-ffmpeg Pillow rich
MUJOCO_GL=osmesa python train_mujoco.py --timesteps 5000000 --n-envs 32
```

Note: CPU-based training is significantly slower and requires more timesteps.

## How It Works

1. **Brax** simulates 4,096 copies of the humanoid simultaneously on the GPU
2. **PPO** collects experience from all environments in parallel
3. The policy network (MLP) learns to map observations to joint torques
4. **Reward** = forward velocity + survival bonus - control cost
5. Over 30M steps, the humanoid progresses from falling to walking to running
