"""
YouTube Shorts: Humanoid Learning to Walk
==========================================
Single script that trains a humanoid with Brax PPO, captures intermediate
policy checkpoints via policy_params_fn, records episodes at each stage,
renders 9:16 vertical video with MuJoCo camera tracking, and compiles
a ~48-second YouTube Shorts video showing the learning progression.
"""

import os
import time
import functools
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.io import model as brax_model
import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # Video format (YouTube Shorts 9:16)
    width: int = 1080
    height: int = 1920
    fps: int = 60

    # Training
    num_timesteps: int = 30_000_000
    num_evals: int = 20
    num_envs: int = 4096
    episode_length: int = 1000

    # Camera (portrait framing — closer to humanoid)
    cam_distance: float = 3.5
    cam_elevation: float = -15.0
    cam_azimuth: float = 90.0
    cam_lookat_z: float = 1.0

    # YouTube Shorts safe zones (pixels from edge)
    safe_top: int = 300
    safe_bottom: int = 1248
    safe_left: int = 100
    safe_right: int = 980

    # Paths
    output_dir: str = "/workspace/rl_walking/output"


# ---------------------------------------------------------------------------
# Checkpoint storage
# ---------------------------------------------------------------------------
@dataclass
class Checkpoint:
    step: int
    reward: float
    params: Any  # JAX pytree copied to host


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------
def _get_font(size: int) -> ImageFont.FreeTypeFont:
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Phase 1: Train with checkpoint capture
# ---------------------------------------------------------------------------
def train_with_checkpoints(config: Config, env):
    """Train PPO and capture intermediate params via policy_params_fn."""

    checkpoints: list[Checkpoint] = []
    progress_metrics: dict[int, float] = {}
    t0 = time.time()

    def progress_fn(num_steps: int, metrics: dict) -> None:
        reward = float(metrics.get("eval/episode_reward", 0))
        progress_metrics[num_steps] = reward
        elapsed = time.time() - t0
        print(f"  step={num_steps:>12,}  reward={reward:>8.1f}  time={elapsed:.0f}s")

    def policy_params_fn(current_step: int, make_policy, params) -> None:
        host_params = jax.device_get(params)
        checkpoints.append(Checkpoint(
            step=current_step,
            reward=0.0,
            params=host_params,
        ))
        print(f"  [checkpoint {len(checkpoints)}] step={current_step:,}")

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=config.num_timesteps,
        num_evals=config.num_evals,
        reward_scaling=0.1,
        episode_length=config.episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=5,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=config.num_envs,
        batch_size=2048,
        seed=0,
    )

    print("\nTraining (JIT compile on first call, please wait)...\n")

    make_inference_fn, final_params, _ = train_fn(
        environment=env,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
    )

    train_time = time.time() - t0
    print(f"\nTraining done in {train_time:.0f}s  ({len(checkpoints)} checkpoints)")

    # Backfill rewards from progress_metrics
    for ckpt in checkpoints:
        closest_step = min(progress_metrics.keys(),
                          key=lambda s: abs(s - ckpt.step),
                          default=0)
        if closest_step:
            ckpt.reward = progress_metrics[closest_step]

    return checkpoints, make_inference_fn, final_params, train_time


# ---------------------------------------------------------------------------
# Phase 2: Select narrative checkpoints
# ---------------------------------------------------------------------------
def select_narrative(checkpoints: list[Checkpoint]):
    """Pick checkpoints that tell the failure→walking story.
    Returns list of (Checkpoint, label, max_rollout_steps)."""

    n = len(checkpoints)
    if n < 6:
        return [(c, f"Step {c.step:,}", 300) for c in checkpoints]

    return [
        (checkpoints[0],          "First Attempts",     180),
        (checkpoints[n // 10],    "Early Training",     180),
        (checkpoints[n // 4],     "Finding Balance",    240),
        (checkpoints[n // 2],     "Learning to Walk",   240),
        (checkpoints[3 * n // 4], "Getting Better",     300),
        (checkpoints[-1],         "Fully Trained",      900),
    ]


# ---------------------------------------------------------------------------
# Phase 3: Record episode
# ---------------------------------------------------------------------------
def record_episode(env, make_inference_fn, params, num_steps: int, seed: int = 42):
    """Run one episode, return (pipeline_states, positions, total_reward)."""

    inference_fn = make_inference_fn(params)
    jit_infer = jax.jit(inference_fn)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    key = random.PRNGKey(seed)
    state = jit_reset(rng=key)

    pipeline_states = [state.pipeline_state]
    positions = []
    total_reward = 0.0

    pos = state.pipeline_state.x.pos[0]
    positions.append((float(pos[0]), float(pos[1]), float(pos[2])))

    for _ in range(num_steps):
        key, subkey = random.split(key)
        act = jit_infer(state.obs, subkey)
        action = act[0] if isinstance(act, tuple) else act
        state = jit_step(state, action)
        pipeline_states.append(state.pipeline_state)
        total_reward += float(state.reward)

        pos = state.pipeline_state.x.pos[0]
        positions.append((float(pos[0]), float(pos[1]), float(pos[2])))

        if float(state.done):
            break

    return pipeline_states, positions, total_reward


# ---------------------------------------------------------------------------
# Phase 4: MuJoCo rendering setup
# ---------------------------------------------------------------------------
def setup_renderer(env, config: Config):
    """Create a MuJoCo renderer with clean grey environment."""

    mj_model = env.sys.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Framebuffer must be >= render size
    mj_model.vis.global_.offwidth = max(config.width, mj_model.vis.global_.offwidth)
    mj_model.vis.global_.offheight = max(config.height, mj_model.vis.global_.offheight)

    # Clean grey floor — remove textures
    for i in range(mj_model.nmat):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_MATERIAL, i)
        if name and "floor" in name.lower():
            mj_model.mat_texid[i] = -1
            mj_model.mat_rgba[i] = [0.55, 0.55, 0.55, 1.0]

    for i in range(mj_model.ngeom):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and "floor" in name.lower():
            mj_model.geom_rgba[i] = [0.55, 0.55, 0.55, 1.0]
            mj_model.geom_matid[i] = -1

    renderer = mujoco.Renderer(mj_model, height=config.height, width=config.width)
    return mj_model, mj_data, renderer


def render_frame(mj_model, mj_data, renderer, pstate, torso_pos, config: Config):
    """Render a single frame with camera tracking."""

    q = np.array(pstate.q)
    qd = np.array(pstate.qd)
    if len(q.shape) > 1:
        q = q[0]
        qd = qd[0]
    mj_data.qpos[:len(q)] = q
    mj_data.qvel[:len(qd)] = qd
    mujoco.mj_forward(mj_model, mj_data)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[0] = torso_pos[0]
    cam.lookat[1] = torso_pos[1]
    cam.lookat[2] = config.cam_lookat_z
    cam.distance = config.cam_distance
    cam.elevation = config.cam_elevation
    cam.azimuth = config.cam_azimuth

    renderer.update_scene(mj_data, camera=cam)
    return renderer.render().copy()


# ---------------------------------------------------------------------------
# Phase 5: Text overlays (safe-zone aware)
# ---------------------------------------------------------------------------
def _centered_x(draw, text, font, config: Config) -> int:
    """Get x position to center text in safe zone."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    return max(config.safe_left, (config.safe_left + config.safe_right - tw) // 2)


def _text_height(draw, text, font) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[3] - bbox[1]


def _clamp_text(text: str, max_chars: int = 30) -> str:
    """Ensure text doesn't overflow by truncating if needed."""
    return text[:max_chars] if len(text) > max_chars else text


def compose_frame(
    frame: np.ndarray,
    label: str,
    step_text: str,
    config: Config,
    progress: float = 0.0,
) -> np.ndarray:
    """Add text overlays within YouTube safe zones."""

    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_label = _get_font(36)
    font_info = _get_font(28)

    label = _clamp_text(label, 28)
    step_text = _clamp_text(step_text, 35)

    # Label — top of safe zone
    x = _centered_x(draw, label, font_label, config)
    y = config.safe_top + 16
    th = _text_height(draw, label, font_label)
    bbox = draw.textbbox((0, 0), label, font=font_label)
    tw = bbox[2] - bbox[0]
    draw.rounded_rectangle(
        [x - 16, y - 8, x + tw + 16, y + th + 8],
        radius=12, fill=(0, 0, 0, 170),
    )
    draw.text((x, y), label, fill=(255, 255, 255, 255), font=font_label)

    # Step info below label
    x2 = _centered_x(draw, step_text, font_info, config)
    y2 = y + th + 24
    th2 = _text_height(draw, step_text, font_info)
    bbox2 = draw.textbbox((0, 0), step_text, font=font_info)
    tw2 = bbox2[2] - bbox2[0]
    draw.rounded_rectangle(
        [x2 - 12, y2 - 6, x2 + tw2 + 12, y2 + th2 + 6],
        radius=10, fill=(0, 0, 0, 140),
    )
    draw.text((x2, y2), step_text, fill=(210, 210, 210, 255), font=font_info)

    # Thin progress bar
    if progress > 0:
        bar_y = config.safe_top + 4
        bar_w = int((config.safe_right - config.safe_left) * min(progress, 1.0))
        draw.rectangle(
            [config.safe_left, bar_y, config.safe_left + bar_w, bar_y + 3],
            fill=(80, 220, 120, 200),
        )

    result = Image.alpha_composite(img, overlay).convert("RGB")
    return np.array(result)


def make_title_card(
    lines: list[str],
    config: Config,
    duration_sec: float = 2.5,
    font_sizes: list[int] | None = None,
    bg_color: tuple = (20, 20, 25),
) -> list[np.ndarray]:
    """Generate centered title card frames."""

    n_frames = int(duration_sec * config.fps)
    card = np.full((config.height, config.width, 3), bg_color, dtype=np.uint8)
    img = Image.fromarray(card)
    draw = ImageDraw.Draw(img)

    if font_sizes is None:
        font_sizes = [48] + [34] * (len(lines) - 1)

    # Measure total height
    spacing = 28
    line_data = []
    total_h = 0
    for text, fsize in zip(lines, font_sizes):
        font = _get_font(fsize)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        line_data.append((text, font, tw, th))
        total_h += th
    total_h += spacing * (len(lines) - 1)

    # Center in safe zone
    start_y = (config.safe_top + config.safe_bottom - total_h) // 2

    for text, font, tw, th in line_data:
        x = (config.width - tw) // 2
        # Clamp to safe zone
        x = max(config.safe_left, min(x, config.safe_right - tw))
        draw.text((x, start_y), text, fill=(255, 255, 255), font=font)
        start_y += th + spacing

    frame = np.array(img)
    return [frame] * n_frames


def make_transition(duration_sec: float, config: Config) -> list[np.ndarray]:
    """Brief dark fade between segments."""
    n = int(duration_sec * config.fps)
    frame = np.full((config.height, config.width, 3), 15, dtype=np.uint8)
    return [frame] * n


# ---------------------------------------------------------------------------
# Phase 6: Upsample 50Hz physics to 60fps
# ---------------------------------------------------------------------------
def upsample_50_to_60(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Duplicate every 5th frame to go from 50 to 60 fps."""
    result = []
    for i, f in enumerate(frames):
        result.append(f)
        if (i + 1) % 5 == 0:
            result.append(f)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 60)
    print("  YouTube Shorts: Humanoid Learning to Walk")
    print("=" * 60)
    print(f"  GPU          : {jax.devices()[0]}")
    print(f"  Resolution   : {config.width}x{config.height} (9:16)")
    print(f"  FPS          : {config.fps}")
    print(f"  Training     : {config.num_timesteps:,} steps, {config.num_envs} envs")
    print(f"  Checkpoints  : {config.num_evals}")
    print("=" * 60)

    # --- Environment ---
    env = envs.get_environment(env_name="humanoid", backend="generalized")

    # --- Phase 1: Train ---
    checkpoints, make_inference_fn, final_params, train_time = \
        train_with_checkpoints(config, env)

    # Save final model
    model_path = os.path.join(config.output_dir, "humanoid_params")
    brax_model.save_params(model_path, final_params)
    print(f"Model saved: {model_path}")

    # --- Phase 2: Select narrative ---
    narrative = select_narrative(checkpoints)
    print(f"\nSelected {len(narrative)} episodes for video:")
    for ckpt, label, steps in narrative:
        print(f"  {label:20s}  step={ckpt.step:>10,}  reward={ckpt.reward:>8.1f}")

    # --- Phase 3-6: Record, render, compose, write ---
    output_path = os.path.join(config.output_dir, "humanoid_shorts.mp4")
    writer = imageio.get_writer(
        output_path,
        fps=config.fps,
        codec="libx264",
        quality=9,
        pixelformat="yuv420p",
        macro_block_size=8,
        output_params=["-preset", "slow", "-crf", "18"],
    )

    # Setup MuJoCo renderer (reuse across all episodes)
    mj_model, mj_data, renderer = setup_renderer(env, config)

    # --- Intro card ---
    print("\nWriting intro card...")
    for f in make_title_card(
        ["Teaching an AI", "to Walk", "", "Brax PPO  |  4096 Envs"],
        config,
        duration_sec=2.5,
        font_sizes=[56, 56, 20, 30],
    ):
        writer.append_data(f)

    total_frames_written = int(2.5 * config.fps)

    # --- Episode segments ---
    for seg_idx, (ckpt, label, max_steps) in enumerate(narrative):
        print(f"\n--- Segment {seg_idx+1}/{len(narrative)}: {label} ---")

        # Transition
        for f in make_transition(0.3, config):
            writer.append_data(f)
        total_frames_written += int(0.3 * config.fps)

        # Record episode
        print(f"  Recording (max {max_steps} steps)...")
        states, positions, reward = record_episode(
            env, make_inference_fn, ckpt.params, max_steps,
        )
        actual_steps = len(states)
        print(f"  Got {actual_steps} steps, reward={reward:.1f}")

        # Pad very short episodes (< 60 frames) by holding last frame
        min_frames = 60
        while len(states) < min_frames:
            states.append(states[-1])
            positions.append(positions[-1])

        # Render and write frame-by-frame (memory efficient)
        print(f"  Rendering & composing {len(states)} frames...")
        progress = ckpt.step / config.num_timesteps
        step_text = f"Step {ckpt.step:,}  |  Reward {ckpt.reward:.0f}"
        frame_count = 0
        upsample_buf = []

        for i in range(len(states)):
            raw = render_frame(mj_model, mj_data, renderer, states[i], positions[i], config)
            upsample_buf.append(raw)

            # Upsample: every 5 physics frames, emit 6 video frames
            if len(upsample_buf) == 5:
                for uf in upsample_buf:
                    composed = compose_frame(uf, label, step_text, config, progress)
                    writer.append_data(composed)
                    frame_count += 1
                # Duplicate last frame for 50→60 upsample
                composed = compose_frame(upsample_buf[-1], label, step_text, config, progress)
                writer.append_data(composed)
                frame_count += 1
                upsample_buf = []

        # Flush remaining frames
        for uf in upsample_buf:
            composed = compose_frame(uf, label, step_text, config, progress)
            writer.append_data(composed)
            frame_count += 1

        total_frames_written += frame_count
        print(f"  Wrote {frame_count} video frames ({frame_count/config.fps:.1f}s)")

        # Free memory
        del states, positions

    renderer.close()

    # --- Stats card ---
    print("\nWriting stats card...")
    for f in make_transition(0.3, config):
        writer.append_data(f)
    final_ckpt = checkpoints[-1]
    for f in make_title_card(
        [f"Trained in {train_time:.0f} seconds",
         f"Final Reward: {final_ckpt.reward:.0f}",
         "",
         f"30M steps  |  4,096 environments",
         f"NVIDIA A100 80GB"],
        config,
        duration_sec=2.5,
        font_sizes=[40, 44, 20, 30, 28],
    ):
        writer.append_data(f)
    total_frames_written += int(2.8 * config.fps)

    # --- Outro ---
    for f in make_title_card(
        ["GPU-Accelerated RL",
         "",
         "google/brax  |  deepmind"],
        config,
        duration_sec=2.0,
        font_sizes=[42, 20, 28],
    ):
        writer.append_data(f)
    total_frames_written += int(2.0 * config.fps)

    writer.close()

    duration = total_frames_written / config.fps
    size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print("\n" + "=" * 60)
    print("  YouTube Shorts Video Complete")
    print("=" * 60)
    print(f"  File     : {output_path}")
    print(f"  Duration : {duration:.1f}s")
    print(f"  Size     : {size_mb:.1f} MB")
    print(f"  Format   : {config.width}x{config.height} @ {config.fps}fps")
    print(f"  Training : {train_time:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
