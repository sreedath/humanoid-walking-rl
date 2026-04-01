"""
YouTube Shorts: Humanoid Learning to STAND UP
===============================================
Humanoid starts on the ground and learns to stand upright.
Dramatic progression — from lying flat to standing tall.
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
    num_timesteps: int = 50_000_000
    num_evals: int = 20
    num_envs: int = 4096
    episode_length: int = 1000
    cam_distance: float = 4.0
    cam_elevation: float = -20.0
    cam_azimuth: float = 70.0  # Slight 3/4 angle for dramatic standup
    cam_lookat_z: float = 0.8  # Lower — starts on ground
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
        if os.path.exists(p): return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def compose_frame(frame, label, step_text, config):
    img = Image.fromarray(frame).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(ov)
    fl = _get_font(36); fs = _get_font(28)
    bb = draw.textbbox((0,0), label, font=fl); tw, th = bb[2]-bb[0], bb[3]-bb[1]
    x = max(config.safe_left, (config.safe_left+config.safe_right-tw)//2); y = config.safe_top+16
    draw.rounded_rectangle([x-16,y-8,x+tw+16,y+th+8], radius=12, fill=(0,0,0,170))
    draw.text((x,y), label[:28], fill=(255,255,255,255), font=fl)
    bb2 = draw.textbbox((0,0), step_text, font=fs); tw2, th2 = bb2[2]-bb2[0], bb2[3]-bb2[1]
    x2 = max(config.safe_left, (config.safe_left+config.safe_right-tw2)//2); y2 = y+th+24
    draw.rounded_rectangle([x2-12,y2-6,x2+tw2+12,y2+th2+6], radius=10, fill=(0,0,0,140))
    draw.text((x2,y2), step_text[:35], fill=(210,210,210,255), font=fs)
    return np.array(Image.alpha_composite(img, ov).convert("RGB"))


def make_title_card(lines, config, dur=2.5, font_sizes=None, bg=(20,20,25)):
    n = int(dur*config.fps); card = np.full((config.height,config.width,3), bg, dtype=np.uint8)
    img = Image.fromarray(card); draw = ImageDraw.Draw(img)
    if font_sizes is None: font_sizes = [48]+[34]*(len(lines)-1)
    sp = 28; ld = []; th = 0
    for t, fs in zip(lines, font_sizes):
        f = _get_font(fs); bb = draw.textbbox((0,0),t,font=f)
        tw, h = bb[2]-bb[0], bb[3]-bb[1]; ld.append((t,f,tw,h)); th += h
    th += sp*(len(lines)-1); sy = (config.safe_top+config.safe_bottom-th)//2
    for t, f, tw, h in ld:
        x = max(config.safe_left, (config.width-tw)//2)
        draw.text((x,sy), t, fill=(255,255,255), font=f); sy += h+sp
    return [np.array(img)]*n


def make_transition(dur, config):
    return [np.full((config.height,config.width,3),15,dtype=np.uint8)]*int(dur*config.fps)


def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    print("="*60+"\n  Humanoid Standup - GPU RL\n"+"="*60)
    print(f"  GPU: {jax.devices()[0]}")

    env = envs.get_environment(env_name="humanoidstandup", backend="generalized")
    checkpoints = []; pm = {}; t0 = time.time()

    def pfn(ns, m):
        r = float(m.get("eval/episode_reward",0)); pm[ns] = r
        print(f"  step={ns:>12,}  reward={r:>8.1f}  time={time.time()-t0:.0f}s")
    def ppfn(cs, mp, p):
        checkpoints.append(Checkpoint(step=cs, reward=0.0, params=jax.device_get(p)))

    tf = functools.partial(ppo.train, num_timesteps=config.num_timesteps, num_evals=config.num_evals,
        reward_scaling=0.01, episode_length=config.episode_length, normalize_observations=True,
        action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4,
        discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, num_envs=config.num_envs,
        batch_size=2048, seed=5)

    print("\nTraining...\n")
    mif, fp, _ = tf(environment=env, progress_fn=pfn, policy_params_fn=ppfn)
    tt = time.time()-t0; print(f"\nDone in {tt:.0f}s ({len(checkpoints)} ckpts)")

    for c in checkpoints:
        cl = min(pm.keys(), key=lambda s: abs(s-c.step), default=0)
        if cl: c.reward = pm[cl]

    brax_model.save_params(os.path.join(config.output_dir, "standup_params"), fp)

    n = len(checkpoints)
    nar = [
        (checkpoints[0],          "Lying on Ground",   200),
        (checkpoints[max(1,n//8)], "First Wiggles",     200),
        (checkpoints[n//4],       "Trying to Rise",    300),
        (checkpoints[n//2],       "Almost There",      300),
        (checkpoints[3*n//4],     "Getting Up!",       400),
        (checkpoints[-1],         "Standing Tall",     600),
    ] if n >= 6 else [(c, f"Step {c.step:,}", 300) for c in checkpoints]

    mm = env.sys.mj_model; md = mujoco.MjData(mm)
    mm.vis.global_.offwidth = max(config.width, mm.vis.global_.offwidth)
    mm.vis.global_.offheight = max(config.height, mm.vis.global_.offheight)
    for i in range(mm.nmat):
        nm = mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_MATERIAL, i)
        if nm and "floor" in nm.lower(): mm.mat_texid[i]=-1; mm.mat_rgba[i]=[0.55,0.55,0.55,1.0]
    for i in range(mm.ngeom):
        nm = mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_GEOM, i)
        if nm and "floor" in nm.lower(): mm.geom_rgba[i]=[0.55,0.55,0.55,1.0]; mm.geom_matid[i]=-1
    rr = mujoco.Renderer(mm, height=config.height, width=config.width)

    op = os.path.join(config.output_dir, "standup_shorts.mp4")
    w = imageio.get_writer(op, fps=config.fps, codec="libx264", quality=9,
        pixelformat="yuv420p", macro_block_size=8, output_params=["-preset","slow","-crf","18"])
    tf2 = 0

    for f in make_title_card(["Humanoid Learning","to STAND UP","","From Ground to Feet","","Brax PPO | GPU"],
        config, dur=3.0, font_sizes=[48,52,20,36,20,28]):
        w.append_data(f); tf2 += 1

    for si, (ckpt, lab, ms) in enumerate(nar):
        print(f"\n--- {lab} ---")
        for f in make_transition(0.3, config): w.append_data(f); tf2 += 1
        inf = mif(ckpt.params); ji = jax.jit(inf); jr = jax.jit(env.reset); js = jax.jit(env.step)
        key = random.PRNGKey(si*7+42); st = jr(rng=key)
        ps = [st.pipeline_state]; pos = [(float(st.pipeline_state.x.pos[0][0]),
            float(st.pipeline_state.x.pos[0][1]), float(st.pipeline_state.x.pos[0][2]))]
        er = 0.0
        for _ in range(ms):
            key, sk = random.split(key); a = ji(st.obs, sk)
            act = a[0] if isinstance(a, tuple) else a; st = js(st, act)
            ps.append(st.pipeline_state); er += float(st.reward)
            p = st.pipeline_state.x.pos[0]; pos.append((float(p[0]),float(p[1]),float(p[2])))
            if float(st.done): break
        while len(ps)<60: ps.append(ps[-1]); pos.append(pos[-1])
        print(f"  {len(ps)} steps, reward={er:.1f}")
        st2 = f"Step {ckpt.step:,} | Reward {ckpt.reward:.0f}"; fc = 0
        for i, pst in enumerate(ps):
            q = np.array(pst.q); qd = np.array(pst.qd)
            if len(q.shape)>1: q=q[0]; qd=qd[0]
            md.qpos[:len(q)]=q; md.qvel[:len(qd)]=qd; mujoco.mj_forward(mm,md)
            cm = mujoco.MjvCamera(); cm.type=mujoco.mjtCamera.mjCAMERA_FREE
            cm.lookat[0]=pos[i][0]; cm.lookat[1]=pos[i][1]; cm.lookat[2]=config.cam_lookat_z
            cm.distance=config.cam_distance; cm.elevation=config.cam_elevation; cm.azimuth=config.cam_azimuth
            rr.update_scene(md, camera=cm); raw=rr.render().copy()
            comp=compose_frame(raw, lab, st2, config); w.append_data(comp); fc+=1
            if (i+1)%5==0: w.append_data(comp); fc+=1
        tf2+=fc; print(f"  Wrote {fc} frames ({fc/config.fps:.1f}s)")
        del ps, pos

    rr.close()
    for f in make_transition(0.3,config): w.append_data(f); tf2+=1
    for f in make_title_card([f"Trained in {tt:.0f}s",f"Reward: {checkpoints[-1].reward:.0f}","","50M | A100"],
        config, dur=2.5, font_sizes=[40,44,20,28]):
        w.append_data(f); tf2+=1
    for f in make_title_card(["GPU-Accelerated RL","","google/brax"],config,dur=2.0,font_sizes=[42,20,28]):
        w.append_data(f); tf2+=1
    w.close()
    dur=tf2/config.fps; sz=os.path.getsize(op)/(1024**2)
    print(f"\n{'='*60}\n  Standup Complete\n  {op}\n  {dur:.1f}s | {sz:.1f} MB\n{'='*60}")

if __name__ == "__main__":
    main()
