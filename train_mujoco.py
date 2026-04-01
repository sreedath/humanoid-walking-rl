"""
3D Humanoid Walking via Reinforcement Learning
===============================================
Uses MuJoCo Humanoid-v4 with PPO (Proximal Policy Optimization).
Records evaluation episodes throughout training to show learning progression,
then compiles them into a single video.
"""

import os
import sys
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import imageio
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_TIMESTEPS = 1_500_000
RECORD_INTERVAL = 150_000
MAX_EPISODE_STEPS = 1000
N_ENVS = 32
OUTPUT_DIR = "output"
RECORDINGS_DIR = os.path.join(OUTPUT_DIR, "episodes")
FINAL_VIDEO = os.path.join(OUTPUT_DIR, "humanoid_learning_to_walk.mp4")
FPS = 30


def parse_args():
    parser = argparse.ArgumentParser(description="Train a humanoid to walk with PPO")
    parser.add_argument(
        "--timesteps", type=int, default=DEFAULT_TIMESTEPS,
        help=f"Total training timesteps (default: {DEFAULT_TIMESTEPS:,})",
    )
    parser.add_argument(
        "--record-interval", type=int, default=RECORD_INTERVAL,
        help=f"Record an episode every N steps (default: {RECORD_INTERVAL:,})",
    )
    parser.add_argument(
        "--n-envs", type=int, default=N_ENVS,
        help=f"Number of parallel environments (default: {N_ENVS})",
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a saved model to resume training from",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def make_env(rank, seed=42):
    def _init():
        env = gym.make("Humanoid-v4")
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


# ---------------------------------------------------------------------------
# Recording callback
# ---------------------------------------------------------------------------
class RecordProgressCallback(BaseCallback):
    """Records a deterministic evaluation episode at fixed training intervals."""

    def __init__(self, record_freq, save_dir, verbose=1):
        super().__init__(verbose)
        self.record_freq = record_freq
        self.save_dir = save_dir
        self.recordings = []
        os.makedirs(save_dir, exist_ok=True)

    def _on_training_start(self):
        self._record_episode()

    def _on_step(self):
        if self.num_timesteps % self.record_freq < self.training_env.num_envs:
            self._record_episode()
        return True

    def _record_episode(self):
        env = gym.make("Humanoid-v4", render_mode="rgb_array")
        obs, _ = env.reset(seed=0)
        frames = []
        total_reward = 0.0

        for _ in range(MAX_EPISODE_STEPS):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            frames.append(env.render())
            if terminated or truncated:
                break

        env.close()

        idx = len(self.recordings)
        frames_path = os.path.join(self.save_dir, f"episode_{idx:04d}.npy")
        np.save(frames_path, np.array(frames, dtype=np.uint8))

        self.recordings.append({
            "timestep": self.num_timesteps,
            "reward": total_reward,
            "n_frames": len(frames),
            "frames_path": frames_path,
        })

        if self.verbose:
            print(
                f"\n>>> [Recording {idx}] "
                f"Step: {self.num_timesteps:>10,} | "
                f"Reward: {total_reward:>8.1f} | "
                f"Frames: {len(frames)}"
            )


# ---------------------------------------------------------------------------
# Video compilation
# ---------------------------------------------------------------------------
def _get_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _overlay_text(frame, text):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = _get_font(20)
    bbox = draw.textbbox((10, 10), text, font=font)
    draw.rectangle(
        [bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4],
        fill=(0, 0, 0),
    )
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def _make_title_card(shape, text, duration_frames=20):
    """Creates solid-color title card frames."""
    frames = []
    card = np.zeros(shape, dtype=np.uint8)
    card[:, :] = (30, 30, 30)
    img = Image.fromarray(card)
    draw = ImageDraw.Draw(img)
    font = _get_font(32)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (shape[1] - tw) // 2
    y = (shape[0] - th) // 2
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    card_frame = np.array(img)
    for _ in range(duration_frames):
        frames.append(card_frame)
    return frames


def compile_video(recordings, output_path, fps=FPS):
    """Stitches recorded episodes into a single progression video."""
    print("\nCompiling progression video...")

    if not recordings:
        print("No recordings to compile.")
        return

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    n = len(recordings)

    # Intro card
    sample = np.load(recordings[0]["frames_path"])
    h, w, c = sample[0].shape
    for f in _make_title_card((h, w, c), "Humanoid Learning to Walk (PPO)", 60):
        writer.append_data(f)

    for i, rec in enumerate(recordings):
        frames = np.load(rec["frames_path"])
        step = rec["timestep"]
        reward = rec["reward"]
        label = f"Step: {step:,}  |  Reward: {reward:.0f}  |  Episode {i + 1}/{n}"

        # Title card between episodes
        for f in _make_title_card(
            (h, w, c),
            f"Episode {i + 1} / {n}   -   Step {step:,}",
            duration_frames=25,
        ):
            writer.append_data(f)

        # Subsample middle episodes to keep video length reasonable
        if 0 < i < n - 1 and len(frames) > 200:
            step_size = max(1, len(frames) // 200)
            frames = frames[::step_size]

        for frame in frames:
            writer.append_data(_overlay_text(frame, label))

    # Outro
    for f in _make_title_card((h, w, c), "Training Complete!", 60):
        writer.append_data(f)

    writer.close()
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Video saved: {output_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    output_dir = args.output
    recordings_dir = os.path.join(output_dir, "episodes")
    final_video = os.path.join(output_dir, "humanoid_learning_to_walk.mp4")
    os.makedirs(recordings_dir, exist_ok=True)

    print("=" * 60)
    print("  3D Humanoid Walking - Reinforcement Learning")
    print("=" * 60)
    print(f"  Algorithm      : PPO")
    print(f"  Environment    : Humanoid-v4 (MuJoCo)")
    print(f"  Total steps    : {args.timesteps:,}")
    print(f"  Record every   : {args.record_interval:,} steps")
    print(f"  Parallel envs  : {args.n_envs}")
    print(f"  Output dir     : {output_dir}")
    if args.resume:
        print(f"  Resuming from  : {args.resume}")
    print("=" * 60)

    # Create vectorized training environment
    env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])

    # PPO with hyperparameters tuned for Humanoid locomotion
    if args.resume:
        print(f"\nLoading model from {args.resume} ...")
        model = PPO.load(args.resume, env=env, device="auto")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=2048,
            n_epochs=20,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": {"pi": [512, 512, 512], "vf": [512, 512, 512]}},
            verbose=1,
            device="auto",
        )

    callback = RecordProgressCallback(
        record_freq=args.record_interval,
        save_dir=recordings_dir,
    )

    # --- Train ---
    print("\nStarting training...\n")
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)

    # Save final model
    model_path = os.path.join(output_dir, "humanoid_ppo_final")
    model.save(model_path)
    print(f"\nModel saved: {model_path}")

    # Record one last episode
    callback._record_episode()

    # Compile progression video
    compile_video(callback.recordings, final_video)

    env.close()

    # Summary
    print("\n" + "=" * 60)
    print("  Training Summary")
    print("=" * 60)
    for i, rec in enumerate(callback.recordings):
        print(
            f"  Episode {i + 1:>3}: "
            f"Step {rec['timestep']:>10,} | "
            f"Reward: {rec['reward']:>8.1f}"
        )
    print("=" * 60)
    print(f"\n  Final video : {final_video}")
    print(f"  Final model : {model_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
