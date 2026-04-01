"""
Render a trained Brax humanoid policy with camera tracking.
Loads saved params, runs a long episode, renders with MuJoCo
camera following the humanoid.
"""

import os
import numpy as np
import jax
from jax import random
from brax import envs
from brax.io import model as brax_model
import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EPISODE_STEPS = 2000
HEIGHT = 480
WIDTH = 640
FPS = 60
CAM_DISTANCE = 6.0
CAM_ELEVATION = -20
CAM_AZIMUTH = 90       # Side view
OUTPUT_DIR = "/workspace/rl_walking/output"
MODEL_PATH = "/workspace/rl_walking/output/humanoid_params"
VIDEO_PATH = os.path.join(OUTPUT_DIR, "humanoid_walking_final.mp4")


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------
def _get_font(size):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def overlay(frame, text):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = _get_font(22)
    bbox = draw.textbbox((10, 10), text, font=font)
    draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], fill=(0, 0, 0))
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def title_card(h, w, text, n=45):
    card = np.full((h, w, 3), 30, dtype=np.uint8)
    img = Image.fromarray(card)
    draw = ImageDraw.Draw(img)
    font = _get_font(36)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, (h - th) // 2), text, fill=(255, 255, 255), font=font)
    return [np.array(img)] * n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Rendering trained humanoid with camera tracking")
    print("=" * 60)

    # 1. Load environment and trained params
    print("\nLoading environment and model...")
    env = envs.get_environment(env_name="humanoid", backend="generalized")
    params = brax_model.load_params(MODEL_PATH)
    print(f"  Model loaded from {MODEL_PATH}")

    # 2. Build inference function via a minimal training call
    import functools
    from brax.training.agents.ppo import train as ppo_train

    print("  Building inference function (quick JIT)...")
    mini_train = functools.partial(
        ppo_train.train,
        num_timesteps=1,
        num_evals=1,
        episode_length=10,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=5,
        num_minibatches=1,
        num_updates_per_batch=1,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        reward_scaling=0.1,
        num_envs=4,
        batch_size=4,
        seed=0,
    )
    make_inference_fn, _, _ = mini_train(environment=env)

    inference_fn = make_inference_fn(params)
    jit_infer = jax.jit(inference_fn)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # 3. Run episode
    print(f"\nRunning episode ({EPISODE_STEPS} steps)...")
    key = random.PRNGKey(42)
    state = jit_reset(rng=key)

    # Collect trajectory as pipeline states AND extract positions
    trajectory = [state.pipeline_state]
    positions = []  # Track humanoid position for camera
    total_reward = 0.0

    # Get initial position
    x_pos = float(state.pipeline_state.x.pos[0][0])  # torso x
    y_pos = float(state.pipeline_state.x.pos[0][1])  # torso y
    z_pos = float(state.pipeline_state.x.pos[0][2])  # torso z
    positions.append((x_pos, y_pos, z_pos))

    for step in range(EPISODE_STEPS):
        key, subkey = random.split(key)
        act = jit_infer(state.obs, subkey)
        action = act[0] if isinstance(act, tuple) else act
        state = jit_step(state, action)
        trajectory.append(state.pipeline_state)
        total_reward += float(state.reward)

        x_pos = float(state.pipeline_state.x.pos[0][0])
        y_pos = float(state.pipeline_state.x.pos[0][1])
        z_pos = float(state.pipeline_state.x.pos[0][2])
        positions.append((x_pos, y_pos, z_pos))

        if float(state.done):
            print(f"  Episode ended at step {step+1}")
            break

    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Steps survived: {len(trajectory)}")
    start_x = positions[0][0]
    end_x = positions[-1][0]
    print(f"  Distance walked: {abs(end_x - start_x):.1f}m")

    # 4. Render with camera tracking using MuJoCo directly
    print(f"\nRendering {len(trajectory)} frames at {WIDTH}x{HEIGHT}...")

    # Get MuJoCo model from Brax system
    mj_model = env.sys.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Set offscreen framebuffer size to match render size
    mj_model.vis.global_.offwidth = max(WIDTH, mj_model.vis.global_.offwidth)
    mj_model.vis.global_.offheight = max(HEIGHT, mj_model.vis.global_.offheight)

    # Create renderer
    renderer = mujoco.Renderer(mj_model, height=HEIGHT, width=WIDTH)

    # Make floor grey (no texture)
    for i in range(mj_model.nmat):
        mat_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_MATERIAL, i)
        if mat_name and "floor" in mat_name.lower():
            mj_model.mat_texid[i] = -1  # Remove texture
            mj_model.mat_rgba[i] = [0.6, 0.6, 0.6, 1.0]  # Grey

    # Also try to set the ground plane color directly
    for i in range(mj_model.ngeom):
        geom_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and "floor" in geom_name.lower():
            mj_model.geom_rgba[i] = [0.6, 0.6, 0.6, 1.0]
            mj_model.geom_matid[i] = -1  # Remove material/texture

    frames = []
    for i, pstate in enumerate(trajectory):
        # Set MuJoCo state from Brax pipeline state
        q = np.array(pstate.q)
        qd = np.array(pstate.qd)
        if len(q.shape) > 1:
            q = q[0]
            qd = qd[0]
        mj_data.qpos[:len(q)] = q
        mj_data.qvel[:len(qd)] = qd
        mujoco.mj_forward(mj_model, mj_data)

        # Camera tracking: follow the humanoid via update_scene camera param
        torso_pos = positions[i]

        # Use MuJoCo's built-in camera with lookat tracking
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[0] = torso_pos[0]
        cam.lookat[1] = torso_pos[1]
        cam.lookat[2] = 1.0
        cam.distance = CAM_DISTANCE
        cam.elevation = CAM_ELEVATION
        cam.azimuth = CAM_AZIMUTH

        renderer.update_scene(mj_data, camera=cam)
        frame = renderer.render()
        frames.append(frame.copy())

        if (i + 1) % 200 == 0:
            print(f"  Rendered {i+1}/{len(trajectory)} frames")

    renderer.close()
    print(f"  Rendered {len(frames)} frames total")

    # 5. Compile video
    print(f"\nCompiling video to {VIDEO_PATH}...")
    writer = imageio.get_writer(VIDEO_PATH, fps=FPS, codec="libx264",
                                quality=9, pixelformat="yuv420p")

    # Intro
    for f in title_card(HEIGHT, WIDTH, "Humanoid Walking - Brax RL", 90):
        writer.append_data(f)

    # Episode info card
    info = (f"Reward: {total_reward:.0f}  |  "
            f"Steps: {len(trajectory)}  |  "
            f"Distance: {abs(end_x - start_x):.1f}m")
    for f in title_card(HEIGHT, WIDTH, info, 60):
        writer.append_data(f)

    # Main walking video with overlay
    for i, frame in enumerate(frames):
        dist = abs(positions[min(i, len(positions)-1)][0] - positions[0][0])
        label = (f"Step: {i}/{len(frames)}  |  "
                 f"Reward: {total_reward:.0f}  |  "
                 f"Distance: {dist:.1f}m")
        writer.append_data(overlay(frame, label))

    # Outro
    for f in title_card(HEIGHT, WIDTH, "Training Complete!", 90):
        writer.append_data(f)

    writer.close()
    size_mb = os.path.getsize(VIDEO_PATH) / (1024 * 1024)
    print(f"\nVideo saved: {VIDEO_PATH} ({size_mb:.1f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
