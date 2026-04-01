"""
3D Humanoid Walking via GPU-Accelerated Reinforcement Learning
==============================================================
Uses Brax (Google DeepMind) with JAX for GPU-accelerated physics.
4096 parallel environments running entirely on GPU.
Single training run with recorded final policy.
"""

import os
import time
import functools
import numpy as np
import jax
from jax import random
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.io import model as brax_model
import imageio
from PIL import Image, ImageDraw, ImageFont

# Must be set before MuJoCo import
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

from brax.io import image as brax_image  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_TIMESTEPS = 30_000_000
NUM_EVALS = 10
NUM_ENVS = 4096
EPISODE_LENGTH = 1000
RENDER_STEPS = 500
RENDER_HEIGHT = 480
RENDER_WIDTH = 640
FPS = 50
OUTPUT_DIR = "/workspace/rl_walking/output"


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------
def _get_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _overlay(frame, text):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = _get_font(18)
    bbox = draw.textbbox((10, 10), text, font=font)
    draw.rectangle([bbox[0]-4, bbox[1]-4, bbox[2]+4, bbox[3]+4], fill=(0, 0, 0))
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def _title_card(h, w, text, n_frames=30):
    card = np.full((h, w, 3), 30, dtype=np.uint8)
    img = Image.fromarray(card)
    draw = ImageDraw.Draw(img)
    font = _get_font(28)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, (h - th) // 2), text, fill=(255, 255, 255), font=font)
    return [np.array(img)] * n_frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  3D Humanoid Walking - GPU-Accelerated RL")
    print("=" * 65)
    print(f"  Engine         : Brax (JAX GPU physics)")
    print(f"  Algorithm      : PPO")
    print(f"  Parallel envs  : {NUM_ENVS:,}")
    print(f"  Total steps    : {NUM_TIMESTEPS:,}")
    print(f"  GPU            : {jax.devices()[0]}")
    print("=" * 65)

    env = envs.get_environment(env_name="humanoid", backend="generalized")

    # Track progress
    progress_log = []

    def progress_fn(num_steps, metrics):
        reward = metrics.get("eval/episode_reward", 0)
        progress_log.append({"step": num_steps, "reward": float(reward)})
        elapsed = time.time() - t0
        print(f"  step={num_steps:>12,}  reward={reward:>8.1f}  time={elapsed:.0f}s")

    # Configure PPO
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=NUM_TIMESTEPS,
        num_evals=NUM_EVALS,
        reward_scaling=0.1,
        episode_length=EPISODE_LENGTH,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=5,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=NUM_ENVS,
        batch_size=2048,
        seed=0,
    )

    # --- Train ---
    print("\nTraining (JIT compilation on first call, please wait)...\n")
    t0 = time.time()

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=progress_fn,
    )

    train_time = time.time() - t0
    print(f"\nTraining completed in {train_time:.1f}s")

    # Save model
    model_path = os.path.join(OUTPUT_DIR, "humanoid_params")
    brax_model.save_params(model_path, params)
    print(f"Model saved: {model_path}")

    # --- Record final trained policy ---
    print("\nRecording final trained policy...")
    inference_fn = make_inference_fn(params)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_infer = jax.jit(inference_fn)

    key = random.PRNGKey(42)
    state = jit_reset(rng=key)
    trajectory = [state.pipeline_state]
    total_reward = 0.0

    for _ in range(RENDER_STEPS):
        key, subkey = random.split(key)
        act = jit_infer(state.obs, subkey)
        action = act[0] if isinstance(act, tuple) else act
        state = jit_step(state, action)
        trajectory.append(state.pipeline_state)
        total_reward += float(state.reward)
        if float(state.done):
            break

    print(f"  Episode: {len(trajectory)} steps, reward={total_reward:.1f}")

    # --- Render ---
    print(f"  Rendering {len(trajectory)} frames...")
    frames = brax_image.render_array(
        sys=env.sys,
        trajectory=trajectory,
        height=RENDER_HEIGHT,
        width=RENDER_WIDTH,
        camera=-1,
    )
    print(f"  Got {len(frames)} frames")

    # --- Compile video ---
    final_video = os.path.join(OUTPUT_DIR, "humanoid_learning_to_walk.mp4")
    print(f"\nCompiling video...")
    writer = imageio.get_writer(final_video, fps=FPS, codec="libx264", quality=8)

    # Intro
    for f in _title_card(RENDER_HEIGHT, RENDER_WIDTH,
                         "Humanoid Learning to Walk (Brax GPU)", 75):
        writer.append_data(f)

    # Training progress card
    if progress_log:
        lines = [f"Training Progress ({train_time:.0f}s on {jax.devices()[0]}):"]
        for p in progress_log:
            lines.append(f"  Step {p['step']:>10,}: Reward {p['reward']:>7.1f}")
        progress_text = "\n".join(lines)
        # Show each line as a frame sequence
        for f in _title_card(RENDER_HEIGHT, RENDER_WIDTH,
                             f"Final Reward: {progress_log[-1]['reward']:.0f}", 60):
            writer.append_data(f)

    # Main walking episode
    label = f"Trained Policy | Reward: {total_reward:.0f} | {len(trajectory)} steps"
    for frame in frames:
        writer.append_data(_overlay(frame, label))

    # Outro
    for f in _title_card(RENDER_HEIGHT, RENDER_WIDTH, "Training Complete!", 60):
        writer.append_data(f)

    writer.close()
    size_mb = os.path.getsize(final_video) / (1024 * 1024)
    print(f"Video saved: {final_video} ({size_mb:.1f} MB)")

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  Training Summary")
    print("=" * 65)
    for p in progress_log:
        print(f"  Step {p['step']:>12,} | Reward: {p['reward']:>8.1f}")
    print(f"\n  Total time : {train_time:.1f}s")
    print(f"  Video      : {final_video}")
    print(f"  Model      : {model_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
