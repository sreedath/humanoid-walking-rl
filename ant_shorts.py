"""
YouTube Shorts: Ant Learning to Walk
=====================================
A 4-legged ant creature learning to move — fun, spider-like locomotion.
Faster to train than humanoid, visually interesting gait patterns.
"""

import os
import time
import functools
from dataclasses import dataclass
from typing import Any

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


@dataclass
class Config:
    width: int = 1080
    height: int = 1920
    fps: int = 60
    num_timesteps: int = 30_000_000
    num_evals: int = 20
    num_envs: int = 4096
    episode_length: int = 1000
    # Camera: closer for ant (smaller creature)
    cam_distance: float = 3.0
    cam_elevation: float = -25.0
    cam_azimuth: float = 60.0  # Slight angle for 3D depth
    cam_lookat_z: float = 0.3
    safe_top: int = 300
    safe_bottom: int = 1248
    safe_left: int = 100
    safe_right: int = 980
    output_dir: str = "/workspace/rl_walking/output"


@dataclass
class Checkpoint:
    step: int
    reward: float
    params: Any


def _get_font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def compose_frame(frame, label, step_text, config):
    img = Image.fromarray(frame).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(ov)
    fl = _get_font(36); fs = _get_font(28)
    bb = draw.textbbox((0,0), label, font=fl)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    x = max(config.safe_left, (config.safe_left+config.safe_right-tw)//2)
    y = config.safe_top + 16
    draw.rounded_rectangle([x-16,y-8,x+tw+16,y+th+8], radius=12, fill=(0,0,0,170))
    draw.text((x,y), label[:28], fill=(255,255,255,255), font=fl)
    bb2 = draw.textbbox((0,0), step_text, font=fs)
    tw2, th2 = bb2[2]-bb2[0], bb2[3]-bb2[1]
    x2 = max(config.safe_left, (config.safe_left+config.safe_right-tw2)//2)
    y2 = y + th + 24
    draw.rounded_rectangle([x2-12,y2-6,x2+tw2+12,y2+th2+6], radius=10, fill=(0,0,0,140))
    draw.text((x2,y2), step_text[:35], fill=(210,210,210,255), font=fs)
    return np.array(Image.alpha_composite(img, ov).convert("RGB"))


def make_title_card(lines, config, dur=2.5, font_sizes=None, bg=(20,20,25)):
    n = int(dur * config.fps)
    card = np.full((config.height, config.width, 3), bg, dtype=np.uint8)
    img = Image.fromarray(card)
    draw = ImageDraw.Draw(img)
    if font_sizes is None:
        font_sizes = [48] + [34]*(len(lines)-1)
    sp = 28; ld = []; th = 0
    for t, fs in zip(lines, font_sizes):
        f = _get_font(fs); bb = draw.textbbox((0,0),t,font=f)
        tw, h = bb[2]-bb[0], bb[3]-bb[1]; ld.append((t,f,tw,h)); th += h
    th += sp*(len(lines)-1)
    sy = (config.safe_top + config.safe_bottom - th)//2
    for t, f, tw, h in ld:
        x = max(config.safe_left, (config.width-tw)//2)
        draw.text((x, sy), t, fill=(255,255,255), font=f)
        sy += h + sp
    return [np.array(img)] * n


def make_transition(dur, config):
    n = int(dur * config.fps)
    return [np.full((config.height, config.width, 3), 15, dtype=np.uint8)] * n


def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 60)
    print("  Ant Learning to Walk - GPU-Accelerated RL")
    print("=" * 60)
    print(f"  GPU: {jax.devices()[0]}")

    env = envs.get_environment(env_name="ant", backend="generalized")

    checkpoints = []
    progress_metrics = {}
    t0 = time.time()

    def progress_fn(num_steps, metrics):
        reward = float(metrics.get("eval/episode_reward", 0))
        progress_metrics[num_steps] = reward
        print(f"  step={num_steps:>12,}  reward={reward:>8.1f}  time={time.time()-t0:.0f}s")

    def policy_params_fn(current_step, make_policy, params):
        host_params = jax.device_get(params)
        checkpoints.append(Checkpoint(step=current_step, reward=0.0, params=host_params))

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=config.num_timesteps,
        num_evals=config.num_evals,
        reward_scaling=10,
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
        seed=2,
    )

    print("\nTraining...\n")
    make_inference_fn, final_params, _ = train_fn(
        environment=env,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
    )
    train_time = time.time() - t0
    print(f"\nDone in {train_time:.0f}s ({len(checkpoints)} checkpoints)")

    for ckpt in checkpoints:
        closest = min(progress_metrics.keys(), key=lambda s: abs(s - ckpt.step), default=0)
        if closest: ckpt.reward = progress_metrics[closest]

    brax_model.save_params(os.path.join(config.output_dir, "ant_params"), final_params)

    n = len(checkpoints)
    narrative = [
        (checkpoints[0],          "First Wiggles",     180),
        (checkpoints[max(1,n//8)], "Baby Steps",        180),
        (checkpoints[n//4],       "Finding Legs",      240),
        (checkpoints[n//2],       "Scurrying Along",   300),
        (checkpoints[3*n//4],     "Getting Fast",      300),
        (checkpoints[-1],         "Speed Demon",       900),
    ] if n >= 6 else [(c, f"Step {c.step:,}", 300) for c in checkpoints]

    # Renderer
    mj_model = env.sys.mj_model
    mj_data = mujoco.MjData(mj_model)
    mj_model.vis.global_.offwidth = max(config.width, mj_model.vis.global_.offwidth)
    mj_model.vis.global_.offheight = max(config.height, mj_model.vis.global_.offheight)
    for i in range(mj_model.nmat):
        nm = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_MATERIAL, i)
        if nm and "floor" in nm.lower():
            mj_model.mat_texid[i] = -1; mj_model.mat_rgba[i] = [0.55,0.55,0.55,1.0]
    for i in range(mj_model.ngeom):
        nm = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if nm and "floor" in nm.lower():
            mj_model.geom_rgba[i] = [0.55,0.55,0.55,1.0]; mj_model.geom_matid[i] = -1
    renderer = mujoco.Renderer(mj_model, height=config.height, width=config.width)

    output_path = os.path.join(config.output_dir, "ant_shorts.mp4")
    writer = imageio.get_writer(
        output_path, fps=config.fps, codec="libx264",
        quality=9, pixelformat="yuv420p", macro_block_size=8,
        output_params=["-preset", "slow", "-crf", "18"],
    )
    total_frames = 0

    for f in make_title_card(
        ["Bug Learning", "to WALK", "", "4 Legs | Brax PPO"],
        config, dur=3.0, font_sizes=[52, 56, 20, 28],
    ):
        writer.append_data(f); total_frames += 1

    for seg_idx, (ckpt, label, max_steps) in enumerate(narrative):
        print(f"\n--- {label} ---")
        for f in make_transition(0.3, config):
            writer.append_data(f); total_frames += 1

        inference_fn = make_inference_fn(ckpt.params)
        jit_infer = jax.jit(inference_fn)
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)

        key = random.PRNGKey(seg_idx * 7 + 42)
        state = jit_reset(rng=key)
        pipeline_states = [state.pipeline_state]
        positions = [(float(state.pipeline_state.x.pos[0][0]),
                      float(state.pipeline_state.x.pos[0][1]),
                      float(state.pipeline_state.x.pos[0][2]))]
        ep_reward = 0.0

        for _ in range(max_steps):
            key, subkey = random.split(key)
            act = jit_infer(state.obs, subkey)
            action = act[0] if isinstance(act, tuple) else act
            state = jit_step(state, action)
            pipeline_states.append(state.pipeline_state)
            ep_reward += float(state.reward)
            p = state.pipeline_state.x.pos[0]
            positions.append((float(p[0]), float(p[1]), float(p[2])))
            if float(state.done): break

        while len(pipeline_states) < 60:
            pipeline_states.append(pipeline_states[-1])
            positions.append(positions[-1])

        print(f"  {len(pipeline_states)} steps, reward={ep_reward:.1f}")

        step_text = f"Step {ckpt.step:,} | Reward {ckpt.reward:.0f}"
        fc = 0
        for i, pstate in enumerate(pipeline_states):
            q = np.array(pstate.q); qd = np.array(pstate.qd)
            if len(q.shape) > 1: q = q[0]; qd = qd[0]
            mj_data.qpos[:len(q)] = q; mj_data.qvel[:len(qd)] = qd
            mujoco.mj_forward(mj_model, mj_data)

            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[0] = positions[i][0]
            cam.lookat[1] = positions[i][1]
            cam.lookat[2] = config.cam_lookat_z
            cam.distance = config.cam_distance
            cam.elevation = config.cam_elevation
            cam.azimuth = config.cam_azimuth

            renderer.update_scene(mj_data, camera=cam)
            raw = renderer.render().copy()
            comp = compose_frame(raw, label, step_text, config)
            writer.append_data(comp); fc += 1
            if (i+1) % 5 == 0:
                writer.append_data(comp); fc += 1

        total_frames += fc
        print(f"  Wrote {fc} frames ({fc/config.fps:.1f}s)")
        del pipeline_states, positions

    renderer.close()

    for f in make_transition(0.3, config):
        writer.append_data(f); total_frames += 1
    for f in make_title_card(
        [f"Trained in {train_time:.0f}s", f"Reward: {checkpoints[-1].reward:.0f}",
         "", "30M steps | A100 GPU"],
        config, dur=2.5, font_sizes=[40, 44, 20, 28],
    ):
        writer.append_data(f); total_frames += 1
    for f in make_title_card(
        ["GPU-Accelerated RL", "", "google/brax"],
        config, dur=2.0, font_sizes=[42, 20, 28],
    ):
        writer.append_data(f); total_frames += 1

    writer.close()
    duration = total_frames / config.fps
    size_mb = os.path.getsize(output_path) / (1024**2)
    print(f"\n{'='*60}")
    print(f"  Ant Shorts Complete")
    print(f"  File: {output_path}")
    print(f"  Duration: {duration:.1f}s | Size: {size_mb:.1f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
