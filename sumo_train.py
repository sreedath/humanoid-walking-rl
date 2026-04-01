"""
AI Sumo Wrestling: Two Humanoids on a Ring
===========================================
Custom Brax environment with two humanoids on an elevated circular disc.
Self-play PPO training: one policy controls both agents symmetrically.
When pushed off the ring edge, the humanoid falls due to gravity.
Renders a YouTube Shorts video (1080x1920) showing the learning progression.
"""

import os
import time
import functools
import textwrap
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import brax
from brax import envs
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as brax_mjcf
from brax.io import model as brax_model
from brax.training.agents.ppo import train as ppo
import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # Video (YouTube Shorts 9:16)
    width: int = 1080
    height: int = 1920
    fps: int = 60
    # Training
    num_timesteps: int = 15_000_000
    num_evals: int = 15
    num_envs: int = 512
    episode_length: int = 500
    # Sumo ring
    ring_radius: float = 3.0
    ring_height: float = 2.0
    # Camera
    cam_distance: float = 7.0
    cam_elevation: float = -30.0
    cam_azimuth: float = 90.0
    cam_lookat_z: float = 2.5
    # Safe zones
    safe_top: int = 300
    safe_bottom: int = 1248
    safe_left: int = 100
    safe_right: int = 980
    # Paths
    output_dir: str = "/workspace/rl_walking/output"


# ---------------------------------------------------------------------------
# MJCF XML: Two humanoids on a circular elevated ring
# ---------------------------------------------------------------------------
def _humanoid_body(prefix: str, pos_x: float, pos_y: float, rgba: str) -> str:
    """Generate MJCF XML for one humanoid body with prefixed names."""
    z = 3.5  # ring_height + humanoid spawn height
    return f"""
        <body name="{prefix}torso" pos="{pos_x} {pos_y} {z}">
            <joint armature="0" damping="0" limited="false" name="{prefix}root" pos="0 0 0" stiffness="0" type="free"/>
            <geom fromto="0 -.07 0 0 .07 0" name="{prefix}torso1" size="0.07" type="capsule" rgba="{rgba}"/>
            <geom name="{prefix}head" pos="0 0 .19" size=".09" type="sphere" user="258" rgba="{rgba}"/>
            <geom fromto="-.01 -.06 -.12 -.01 .06 -.12" name="{prefix}uwaist" size="0.06" type="capsule" rgba="{rgba}"/>
            <body name="{prefix}lwaist" pos="-.01 0 -0.260" quat="1.000 0 -0.002 0">
                <geom fromto="0 -.06 0 0 .06 0" name="{prefix}lwaist_g" size="0.06" type="capsule" rgba="{rgba}"/>
                <joint armature="0.02" axis="0 0 1" damping="5" name="{prefix}abdomen_z" pos="0 0 0.065" range="-45 45" stiffness="20" type="hinge"/>
                <joint armature="0.02" axis="0 1 0" damping="5" name="{prefix}abdomen_y" pos="0 0 0.065" range="-75 30" stiffness="10" type="hinge"/>
                <body name="{prefix}pelvis" pos="0 0 -0.165" quat="1.000 0 -0.002 0">
                    <joint armature="0.02" axis="1 0 0" damping="5" name="{prefix}abdomen_x" pos="0 0 0.1" range="-35 35" stiffness="10" type="hinge"/>
                    <geom fromto="-.02 -.07 0 -.02 .07 0" name="{prefix}butt" size="0.09" type="capsule" rgba="{rgba}"/>
                    <body name="{prefix}right_thigh" pos="0 -0.1 -0.04">
                        <joint armature="0.01" axis="1 0 0" damping="5" name="{prefix}right_hip_x" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
                        <joint armature="0.01" axis="0 0 1" damping="5" name="{prefix}right_hip_z" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
                        <joint armature="0.008" axis="0 1 0" damping="5" name="{prefix}right_hip_y" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
                        <geom fromto="0 0 0 0 0.01 -.34" name="{prefix}right_thigh1" size="0.06" type="capsule" rgba="{rgba}"/>
                        <body name="{prefix}right_shin" pos="0 0.01 -0.403">
                            <joint armature="0.006" axis="0 -1 0" name="{prefix}right_knee" pos="0 0 .02" range="-160 -2" type="hinge"/>
                            <geom fromto="0 0 0 0 0 -.3" name="{prefix}right_shin1" size="0.049" type="capsule" rgba="{rgba}"/>
                            <body name="{prefix}right_foot" pos="0 0 -0.45">
                                <geom contype="1" conaffinity="1" name="{prefix}right_foot" pos="0 0 0.1" size="0.075" type="sphere" rgba="{rgba}"/>
                            </body>
                        </body>
                    </body>
                    <body name="{prefix}left_thigh" pos="0 0.1 -0.04">
                        <joint armature="0.01" axis="-1 0 0" damping="5" name="{prefix}left_hip_x" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
                        <joint armature="0.01" axis="0 0 -1" damping="5" name="{prefix}left_hip_z" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
                        <joint armature="0.01" axis="0 1 0" damping="5" name="{prefix}left_hip_y" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
                        <geom fromto="0 0 0 0 -0.01 -.34" name="{prefix}left_thigh1" size="0.06" type="capsule" rgba="{rgba}"/>
                        <body name="{prefix}left_shin" pos="0 -0.01 -0.403">
                            <joint armature="0.006" axis="0 -1 0" name="{prefix}left_knee" pos="0 0 .02" range="-160 -2" stiffness="1" type="hinge"/>
                            <geom fromto="0 0 0 0 0 -.3" name="{prefix}left_shin1" size="0.049" type="capsule" rgba="{rgba}"/>
                            <body name="{prefix}left_foot" pos="0 0 -0.45">
                                <geom contype="1" conaffinity="1" name="{prefix}left_foot" type="sphere" size="0.075" pos="0 0 0.1" rgba="{rgba}"/>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
            <body name="{prefix}right_upper_arm" pos="0 -0.17 0.06">
                <joint armature="0.0068" axis="2 1 1" name="{prefix}right_shoulder1" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
                <joint armature="0.0051" axis="0 -1 1" name="{prefix}right_shoulder2" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
                <geom fromto="0 0 0 .16 -.16 -.16" name="{prefix}right_uarm1" size="0.04 0.16" type="capsule" rgba="{rgba}"/>
                <body name="{prefix}right_lower_arm" pos=".18 -.18 -.18">
                    <joint armature="0.0028" axis="0 -1 1" name="{prefix}right_elbow" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
                    <geom fromto="0.01 0.01 0.01 .17 .17 .17" name="{prefix}right_larm" size="0.031" type="capsule" rgba="{rgba}"/>
                    <geom name="{prefix}right_hand" pos=".18 .18 .18" size="0.04" type="sphere" rgba="{rgba}"/>
                </body>
            </body>
            <body name="{prefix}left_upper_arm" pos="0 0.17 0.06">
                <joint armature="0.0068" axis="2 -1 1" name="{prefix}left_shoulder1" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
                <joint armature="0.0051" axis="0 1 1" name="{prefix}left_shoulder2" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
                <geom fromto="0 0 0 .16 .16 -.16" name="{prefix}left_uarm1" size="0.04 0.16" type="capsule" rgba="{rgba}"/>
                <body name="{prefix}left_lower_arm" pos=".18 .18 -.18">
                    <joint armature="0.0028" axis="0 -1 -1" name="{prefix}left_elbow" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
                    <geom fromto="0.01 -0.01 0.01 .17 -.17 .17" name="{prefix}left_larm" size="0.031" type="capsule" rgba="{rgba}"/>
                    <geom name="{prefix}left_hand" pos=".18 -.18 .18" size="0.04" type="sphere" rgba="{rgba}"/>
                </body>
            </body>
        </body>"""


def _humanoid_actuators(prefix: str) -> str:
    joints = [
        ("abdomen_y", 100), ("abdomen_z", 100), ("abdomen_x", 100),
        ("right_hip_x", 100), ("right_hip_z", 100), ("right_hip_y", 300),
        ("right_knee", 200),
        ("left_hip_x", 100), ("left_hip_z", 100), ("left_hip_y", 300),
        ("left_knee", 200),
        ("right_shoulder1", 25), ("right_shoulder2", 25), ("right_elbow", 25),
        ("left_shoulder1", 25), ("left_shoulder2", 25), ("left_elbow", 25),
    ]
    lines = []
    for jname, gear in joints:
        lines.append(
            f'        <motor gear="{gear}" joint="{prefix}{jname}" name="{prefix}{jname}"/>'
        )
    return "\n".join(lines)


def build_sumo_xml(config: Config) -> str:
    r = config.ring_radius
    h = config.ring_height
    agent0 = _humanoid_body("a0_", -1.5, 0.0, "0.2 0.4 0.8 1.0")   # blue
    agent1 = _humanoid_body("a1_",  1.5, 0.0, "0.8 0.2 0.2 1.0")   # red
    act0 = _humanoid_actuators("a0_")
    act1 = _humanoid_actuators("a1_")

    return f"""<mujoco model="sumo">
    <compiler angle="degree" inertiafromgeom="true"/>
    <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom condim="3" friction="1.5 .1 .1" rgba="0.5 0.5 0.5 1.0"/>
        <motor ctrllimited="true" ctrlrange="-.4 .4"/>
    </default>
    <option iterations="8" timestep="0.003" gravity="0 0 -9.81"/>
    <custom>
        <numeric data="2500" name="constraint_limit_stiffness"/>
        <numeric data="27000" name="constraint_stiffness"/>
        <numeric data="30" name="constraint_ang_damping"/>
        <numeric data="80" name="constraint_vel_damping"/>
        <numeric data="-0.05" name="ang_damping"/>
        <numeric data="0.5" name="joint_scale_pos"/>
        <numeric data="0.1" name="joint_scale_ang"/>
        <numeric data="0" name="spring_mass_scale"/>
        <numeric data="1" name="spring_inertia_scale"/>
        <numeric data="20" name="matrix_inv_iterations"/>
        <numeric data="15" name="solver_maxls"/>
    </custom>
    <size nkey="5" nuser_geom="1"/>
    <visual>
        <map fogend="10" fogstart="6"/>
        <global offwidth="1080" offheight="1920"/>
    </visual>
    <asset>
        <texture builtin="gradient" height="100" rgb1="0.85 0.88 0.95" rgb2="0.6 0.65 0.75" type="skybox" width="100"/>
        <material name="ring_mat" rgba="0.65 0.6 0.55 1.0" specular="0.3" shininess="0.5"/>
        <material name="ground_mat" rgba="0.35 0.35 0.35 1.0"/>
    </asset>
    <worldbody>
        <light cutoff="100" diffuse="1 1 1" dir="0 0 -1.3" directional="true" exponent="1" pos="0 0 5" specular=".2 .2 .2"/>
        <light diffuse="0.6 0.6 0.6" dir="0 -1 -0.5" directional="true" pos="0 3 5"/>

        <!-- Ground below the ring -->
        <geom name="ground" type="plane" size="20 20 0.1" pos="0 0 0" rgba="0.35 0.35 0.35 1.0"
              conaffinity="1" contype="1" condim="3" friction="1 .1 .1"/>

        <!-- Sumo ring: elevated square platform (box used for Brax compatibility) -->
        <body name="ring" pos="0 0 {h}">
            <geom name="ring_floor" type="box" size="{r} {r} 0.05"
                  rgba="0.65 0.6 0.55 1.0" conaffinity="1" contype="1"
                  condim="3" friction="1.5 .1 .1" mass="10000"/>
        </body>

        <!-- Agent 0 (blue) -->
{agent0}

        <!-- Agent 1 (red) -->
{agent1}
    </worldbody>

    <actuator>
{act0}
{act1}
    </actuator>
</mujoco>"""


# ---------------------------------------------------------------------------
# Custom Sumo Environment
# ---------------------------------------------------------------------------
class SumoEnv(PipelineEnv):
    """Two humanoids sumo wrestling on a square ring.
    Self-play: single policy controls both agents.
    """

    def __init__(self, config: Config = None, backend: str = "generalized", **kwargs):
        if config is None:
            config = Config()
        self._config = config
        self._ring_radius = config.ring_radius
        self._ring_height = config.ring_height

        xml = build_sumo_xml(config)
        sys = brax_mjcf.loads(xml)

        # n_frames=5 means 5 physics substeps per env step
        super().__init__(sys=sys, backend=backend, n_frames=5)
        self._n_actuators_per_agent = 17

    def reset(self, rng: jax.Array) -> State:
        qpos = self.sys.init_q
        qvel = jnp.zeros(self.sys.qd_size())

        rng, rng_noise = random.split(rng)
        noise = random.uniform(rng_noise, shape=(self.sys.q_size(),),
                               minval=-0.005, maxval=0.005)
        qpos = qpos + noise

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state)

        reward = jnp.float32(0)
        done = jnp.float32(0)
        # Empty info — let PipelineEnv manage its own keys
        return State(pipeline_state, obs, reward, done, {})

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        # Torso positions (body indices depend on XML order)
        a0_pos = pipeline_state.x.pos[1]   # a0_torso
        a1_pos = pipeline_state.x.pos[15]  # a1_torso

        # Distance from center (box ring: max of |x|, |y|)
        a0_dist = jnp.maximum(jnp.abs(a0_pos[0]), jnp.abs(a0_pos[1]))
        a1_dist = jnp.maximum(jnp.abs(a1_pos[0]), jnp.abs(a1_pos[1]))

        a0_height = a0_pos[2]
        a1_height = a1_pos[2]
        ring_r = self._ring_radius
        ring_h = self._ring_height

        # Off-ring detection
        a0_off = jnp.logical_or(a0_dist > ring_r - 0.1, a0_height < ring_h - 0.5)
        a1_off = jnp.logical_or(a1_dist > ring_r - 0.1, a1_height < ring_h - 0.5)

        # Reward (agent 0 perspective)
        survive = jnp.float32(0.5)
        push = (a1_dist - a0_dist) * 0.3
        win = jnp.where(a1_off, jnp.float32(10.0), jnp.float32(0.0))
        lose = jnp.where(a0_off, jnp.float32(-10.0), jnp.float32(0.0))
        dist_between = jnp.sqrt(jnp.sum((a0_pos[:2] - a1_pos[:2])**2))
        approach = -dist_between * 0.1
        ctrl = -0.01 * jnp.sum(action**2)
        reward = survive + push + win + lose + approach + ctrl

        # Done if off ring or collapsed
        done = jnp.logical_or(a0_off, a1_off).astype(jnp.float32)
        done = jnp.maximum(done,
            jnp.logical_or(a0_height < ring_h + 0.3,
                           a1_height < ring_h + 0.3).astype(jnp.float32))

        obs = self._get_obs(pipeline_state)

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
        )

    def _get_obs(self, pipeline_state) -> jax.Array:
        return jnp.concatenate([pipeline_state.q, pipeline_state.qd])

    @property
    def observation_size(self) -> int:
        return self.sys.q_size() + self.sys.qd_size()

    @property
    def action_size(self) -> int:
        return self._n_actuators_per_agent * 2


# ---------------------------------------------------------------------------
# Checkpoint storage
# ---------------------------------------------------------------------------
@dataclass
class Checkpoint:
    step: int
    reward: float
    params: Any


# ---------------------------------------------------------------------------
# Font / video helpers (same as render_shorts.py)
# ---------------------------------------------------------------------------
def _get_font(size: int):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def compose_frame(frame, label, step_text, config):
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_l = _get_font(36)
    font_s = _get_font(28)

    # Label
    bbox = draw.textbbox((0, 0), label, font=font_l)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = max(config.safe_left, (config.safe_left + config.safe_right - tw) // 2)
    y = config.safe_top + 16
    draw.rounded_rectangle([x-16, y-8, x+tw+16, y+th+8], radius=12, fill=(0,0,0,170))
    draw.text((x, y), label[:28], fill=(255,255,255,255), font=font_l)

    # Step text
    bbox2 = draw.textbbox((0, 0), step_text, font=font_s)
    tw2 = bbox2[2] - bbox2[0]
    th2 = bbox2[3] - bbox2[1]
    x2 = max(config.safe_left, (config.safe_left + config.safe_right - tw2) // 2)
    y2 = y + th + 24
    draw.rounded_rectangle([x2-12, y2-6, x2+tw2+12, y2+th2+6], radius=10, fill=(0,0,0,140))
    draw.text((x2, y2), step_text[:35], fill=(210,210,210,255), font=font_s)

    return np.array(Image.alpha_composite(img, overlay).convert("RGB"))


def make_title_card(lines, config, duration_sec=2.5, font_sizes=None, bg=(20,20,25)):
    n_frames = int(duration_sec * config.fps)
    card = np.full((config.height, config.width, 3), bg, dtype=np.uint8)
    img = Image.fromarray(card)
    draw = ImageDraw.Draw(img)
    if font_sizes is None:
        font_sizes = [48] + [34] * (len(lines) - 1)
    spacing = 28
    line_data = []
    total_h = 0
    for text, fsize in zip(lines, font_sizes):
        font = _get_font(fsize)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        line_data.append((text, font, tw, th))
        total_h += th
    total_h += spacing * (len(lines) - 1)
    start_y = (config.safe_top + config.safe_bottom - total_h) // 2
    for text, font, tw, th in line_data:
        x = max(config.safe_left, (config.width - tw) // 2)
        draw.text((x, start_y), text, fill=(255,255,255), font=font)
        start_y += th + spacing
    return [np.array(img)] * n_frames


def make_transition(duration_sec, config):
    n = int(duration_sec * config.fps)
    return [np.full((config.height, config.width, 3), 15, dtype=np.uint8)] * n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 60)
    print("  AI Sumo Wrestling - GPU-Accelerated RL")
    print("=" * 60)
    print(f"  GPU          : {jax.devices()[0]}")
    print(f"  Ring         : radius={config.ring_radius}m, height={config.ring_height}m")
    print(f"  Training     : {config.num_timesteps:,} steps, {config.num_envs} envs")
    print("=" * 60)

    # --- Create environment ---
    print("\nCreating sumo environment...")
    env = SumoEnv(config, backend="generalized")
    print(f"  obs_size={env.observation_size}, act_size={env.action_size}")

    # Register for Brax training
    envs.register_environment("sumo", lambda **kwargs: SumoEnv(config, **kwargs))

    # --- Train with checkpoints ---
    checkpoints: list[Checkpoint] = []
    progress_metrics: dict[int, float] = {}
    t0 = time.time()

    def progress_fn(num_steps, metrics):
        reward = float(metrics.get("eval/episode_reward", 0))
        progress_metrics[num_steps] = reward
        elapsed = time.time() - t0
        print(f"  step={num_steps:>12,}  reward={reward:>8.1f}  time={elapsed:.0f}s")

    def policy_params_fn(current_step, make_policy, params):
        host_params = jax.device_get(params)
        checkpoints.append(Checkpoint(step=current_step, reward=0.0, params=host_params))
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
        batch_size=512,
        seed=0,
    )

    print("\nTraining (JIT compile first, please wait)...\n")
    make_inference_fn, final_params, _ = train_fn(
        environment=env,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
    )
    train_time = time.time() - t0
    print(f"\nTraining done in {train_time:.0f}s ({len(checkpoints)} checkpoints)")

    # Backfill rewards
    for ckpt in checkpoints:
        closest = min(progress_metrics.keys(), key=lambda s: abs(s - ckpt.step), default=0)
        if closest:
            ckpt.reward = progress_metrics[closest]

    # Save model
    model_path = os.path.join(config.output_dir, "sumo_params")
    brax_model.save_params(model_path, final_params)

    # --- Select narrative checkpoints ---
    n = len(checkpoints)
    if n >= 6:
        narrative = [
            (checkpoints[0],          "First Attempts",   200),
            (checkpoints[n//8],       "Early Training",   200),
            (checkpoints[n//4],       "Learning to Push", 300),
            (checkpoints[n//2],       "Mid Training",     300),
            (checkpoints[3*n//4],     "Getting Competitive", 400),
            (checkpoints[-1],         "Trained Fighters", 500),
        ]
    else:
        narrative = [(c, f"Step {c.step:,}", 300) for c in checkpoints]

    print(f"\nSelected {len(narrative)} episodes for video")

    # --- Record, render, compile video ---
    output_path = os.path.join(config.output_dir, "sumo_shorts.mp4")
    writer = imageio.get_writer(
        output_path, fps=config.fps, codec="libx264",
        quality=9, pixelformat="yuv420p", macro_block_size=8,
        output_params=["-preset", "slow", "-crf", "18"],
    )

    # Setup MuJoCo renderer
    xml_str = build_sumo_xml(config)
    mj_model = mujoco.MjModel.from_xml_string(xml_str)
    mj_data = mujoco.MjData(mj_model)
    mj_model.vis.global_.offwidth = max(config.width, mj_model.vis.global_.offwidth)
    mj_model.vis.global_.offheight = max(config.height, mj_model.vis.global_.offheight)
    renderer = mujoco.Renderer(mj_model, height=config.height, width=config.width)

    total_video_frames = 0

    # Intro
    print("\nWriting intro...")
    for f in make_title_card(
        ["AI Sumo Wrestling", "", "Two Humanoids  |  One Ring", "", "Brax PPO  |  Self-Play"],
        config, duration_sec=3.0, font_sizes=[52, 20, 36, 20, 28],
    ):
        writer.append_data(f)
        total_video_frames += 1

    # Episode segments
    for seg_idx, (ckpt, label, max_steps) in enumerate(narrative):
        print(f"\n--- Segment {seg_idx+1}/{len(narrative)}: {label} ---")

        # Transition
        for f in make_transition(0.3, config):
            writer.append_data(f)
            total_video_frames += 1

        # Record episode
        print(f"  Recording (max {max_steps} steps)...")
        inference_fn = make_inference_fn(ckpt.params)
        jit_infer = jax.jit(inference_fn)
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)

        key = random.PRNGKey(seg_idx * 7 + 42)
        state = jit_reset(rng=key)
        pipeline_states = [state.pipeline_state]
        ep_reward = 0.0

        for step_i in range(max_steps):
            key, subkey = random.split(key)
            act = jit_infer(state.obs, subkey)
            action = act[0] if isinstance(act, tuple) else act
            state = jit_step(state, action)
            pipeline_states.append(state.pipeline_state)
            ep_reward += float(state.reward)

            # Check if someone fell off — record a few more frames then stop
            if float(state.done):
                # Record 30 more frames of the fall
                for _ in range(30):
                    key, subkey = random.split(key)
                    try:
                        act = jit_infer(state.obs, subkey)
                        action = act[0] if isinstance(act, tuple) else act
                        state = jit_step(state, action)
                        pipeline_states.append(state.pipeline_state)
                    except Exception:
                        break
                break

        # Pad short episodes
        while len(pipeline_states) < 60:
            pipeline_states.append(pipeline_states[-1])

        actual_steps = len(pipeline_states)
        print(f"  Got {actual_steps} steps, reward={ep_reward:.1f}")

        # Render and write
        print(f"  Rendering {actual_steps} frames...")
        step_text = f"Step {ckpt.step:,}  |  Reward {ckpt.reward:.0f}"
        frame_count = 0

        for i, pstate in enumerate(pipeline_states):
            q = np.array(pstate.q)
            qd = np.array(pstate.qd)
            if len(q.shape) > 1:
                q = q[0]
                qd = qd[0]
            mj_data.qpos[:len(q)] = q
            mj_data.qvel[:len(qd)] = qd
            mujoco.mj_forward(mj_model, mj_data)

            # Camera: track midpoint between both agents
            a0_pos = mj_data.body("a0_torso").xpos
            a1_pos = mj_data.body("a1_torso").xpos
            mid = (a0_pos + a1_pos) / 2

            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[0] = float(mid[0])
            cam.lookat[1] = float(mid[1])
            cam.lookat[2] = config.cam_lookat_z
            cam.distance = config.cam_distance
            cam.elevation = config.cam_elevation
            cam.azimuth = config.cam_azimuth

            renderer.update_scene(mj_data, camera=cam)
            raw_frame = renderer.render().copy()

            composed = compose_frame(raw_frame, label, step_text, config)
            writer.append_data(composed)
            frame_count += 1

            # Upsample 50->60: duplicate every 5th frame
            if (i + 1) % 5 == 0:
                writer.append_data(composed)
                frame_count += 1

        total_video_frames += frame_count
        print(f"  Wrote {frame_count} frames ({frame_count/config.fps:.1f}s)")
        del pipeline_states

    renderer.close()

    # Stats card
    for f in make_transition(0.3, config):
        writer.append_data(f)
        total_video_frames += 1
    for f in make_title_card(
        [f"Trained in {train_time:.0f}s",
         f"30M steps  |  Self-Play PPO",
         "", "NVIDIA A100 80GB"],
        config, duration_sec=2.5, font_sizes=[40, 32, 20, 28],
    ):
        writer.append_data(f)
        total_video_frames += 1

    # Outro
    for f in make_title_card(
        ["GPU-Accelerated RL", "", "google/brax"],
        config, duration_sec=2.0, font_sizes=[42, 20, 28],
    ):
        writer.append_data(f)
        total_video_frames += 1

    writer.close()
    duration = total_video_frames / config.fps
    size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print("\n" + "=" * 60)
    print("  Sumo Wrestling Video Complete")
    print("=" * 60)
    print(f"  File     : {output_path}")
    print(f"  Duration : {duration:.1f}s")
    print(f"  Size     : {size_mb:.1f} MB")
    print(f"  Format   : {config.width}x{config.height} @ {config.fps}fps")
    print(f"  Training : {train_time:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
