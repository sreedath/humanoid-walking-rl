"""
YouTube Shorts: Half-Cheetah Learning to RUN
==============================================
A fast 2D running creature — develops smooth, cat-like gait.
One of the most satisfying RL locomotion tasks visually.
"""

import os, time, functools
from dataclasses import dataclass
from typing import Any
import numpy as np, jax, jax.numpy as jnp
from jax import random
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.io import model as brax_model
import mujoco, imageio
from PIL import Image, ImageDraw, ImageFont

@dataclass
class Config:
    width: int = 1080; height: int = 1920; fps: int = 60
    num_timesteps: int = 30_000_000; num_evals: int = 20; num_envs: int = 4096
    episode_length: int = 1000
    cam_distance: float = 3.5; cam_elevation: float = -10.0
    cam_azimuth: float = 90.0; cam_lookat_z: float = 0.3
    safe_top: int = 300; safe_bottom: int = 1248
    safe_left: int = 100; safe_right: int = 980
    output_dir: str = "/workspace/rl_walking/output"

@dataclass
class Checkpoint:
    step: int; reward: float; params: Any

def _gf(s):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if os.path.exists(p): return ImageFont.truetype(p,s)
    return ImageFont.load_default()

def cf(fr,lb,st,c):
    img=Image.fromarray(fr).convert("RGBA"); ov=Image.new("RGBA",img.size,(0,0,0,0))
    d=ImageDraw.Draw(ov); fl=_gf(36); fs=_gf(28)
    bb=d.textbbox((0,0),lb,font=fl); tw,th=bb[2]-bb[0],bb[3]-bb[1]
    x=max(c.safe_left,(c.safe_left+c.safe_right-tw)//2); y=c.safe_top+16
    d.rounded_rectangle([x-16,y-8,x+tw+16,y+th+8],radius=12,fill=(0,0,0,170))
    d.text((x,y),lb[:28],fill=(255,255,255,255),font=fl)
    bb2=d.textbbox((0,0),st,font=fs); tw2,th2=bb2[2]-bb2[0],bb2[3]-bb2[1]
    x2=max(c.safe_left,(c.safe_left+c.safe_right-tw2)//2); y2=y+th+24
    d.rounded_rectangle([x2-12,y2-6,x2+tw2+12,y2+th2+6],radius=10,fill=(0,0,0,140))
    d.text((x2,y2),st[:35],fill=(210,210,210,255),font=fs)
    return np.array(Image.alpha_composite(img,ov).convert("RGB"))

def mtc(ls,c,dur=2.5,fss=None,bg=(20,20,25)):
    n=int(dur*c.fps); cd=np.full((c.height,c.width,3),bg,dtype=np.uint8)
    img=Image.fromarray(cd); d=ImageDraw.Draw(img)
    if fss is None: fss=[48]+[34]*(len(ls)-1)
    sp=28; ld=[]; th=0
    for t,fs in zip(ls,fss):
        f=_gf(fs); bb=d.textbbox((0,0),t,font=f)
        tw,h=bb[2]-bb[0],bb[3]-bb[1]; ld.append((t,f,tw,h)); th+=h
    th+=sp*(len(ls)-1); sy=(c.safe_top+c.safe_bottom-th)//2
    for t,f,tw,h in ld:
        x=max(c.safe_left,(c.width-tw)//2)
        d.text((x,sy),t,fill=(255,255,255),font=f); sy+=h+sp
    return [np.array(img)]*n

def mtr(dur,c): return [np.full((c.height,c.width,3),15,dtype=np.uint8)]*int(dur*c.fps)

def main():
    c=Config(); os.makedirs(c.output_dir,exist_ok=True)
    print("="*60+"\n  Half-Cheetah - Fast Runner\n"+"="*60)
    print(f"  GPU: {jax.devices()[0]}")
    env=envs.get_environment(env_name="halfcheetah",backend="generalized")
    ckpts=[]; pm={}; t0=time.time()
    def pfn(ns,m):
        r=float(m.get("eval/episode_reward",0)); pm[ns]=r
        print(f"  step={ns:>12,}  reward={r:>8.1f}  time={time.time()-t0:.0f}s")
    def ppfn(cs,mp,p): ckpts.append(Checkpoint(step=cs,reward=0.0,params=jax.device_get(p)))
    tf=functools.partial(ppo.train,num_timesteps=c.num_timesteps,num_evals=c.num_evals,
        reward_scaling=5,episode_length=c.episode_length,normalize_observations=True,
        action_repeat=1,unroll_length=5,num_minibatches=32,num_updates_per_batch=4,
        discounting=0.97,learning_rate=3e-4,entropy_cost=1e-2,num_envs=c.num_envs,
        batch_size=2048,seed=6)
    print("\nTraining...\n")
    mif,fp,_=tf(environment=env,progress_fn=pfn,policy_params_fn=ppfn)
    tt=time.time()-t0; print(f"\nDone in {tt:.0f}s ({len(ckpts)} ckpts)")
    for ck in ckpts:
        cl=min(pm.keys(),key=lambda s:abs(s-ck.step),default=0)
        if cl: ck.reward=pm[cl]
    brax_model.save_params(os.path.join(c.output_dir,"cheetah_params"),fp)
    n=len(ckpts)
    nar=[(ckpts[0],"First Stride",180),(ckpts[max(1,n//8)],"Warming Up",180),
         (ckpts[n//4],"Finding Rhythm",240),(ckpts[n//2],"Galloping",300),
         (ckpts[3*n//4],"Near Top Speed",300),(ckpts[-1],"Full Speed",900)
    ] if n>=6 else [(ck,f"Step {ck.step:,}",300) for ck in ckpts]
    mm=env.sys.mj_model; md=mujoco.MjData(mm)
    mm.vis.global_.offwidth=max(c.width,mm.vis.global_.offwidth)
    mm.vis.global_.offheight=max(c.height,mm.vis.global_.offheight)
    for i in range(mm.nmat):
        nm=mujoco.mj_id2name(mm,mujoco.mjtObj.mjOBJ_MATERIAL,i)
        if nm and "floor" in nm.lower(): mm.mat_texid[i]=-1; mm.mat_rgba[i]=[0.55,0.55,0.55,1.0]
    for i in range(mm.ngeom):
        nm=mujoco.mj_id2name(mm,mujoco.mjtObj.mjOBJ_GEOM,i)
        if nm and "floor" in nm.lower(): mm.geom_rgba[i]=[0.55,0.55,0.55,1.0]; mm.geom_matid[i]=-1
    rr=mujoco.Renderer(mm,height=c.height,width=c.width)
    op=os.path.join(c.output_dir,"cheetah_shorts.mp4")
    w=imageio.get_writer(op,fps=c.fps,codec="libx264",quality=9,pixelformat="yuv420p",
        macro_block_size=8,output_params=["-preset","slow","-crf","18"])
    tf2=0
    for f in mtc(["Half-Cheetah","Learning to RUN","","Cat-Like Gait | Brax PPO"],c,dur=3.0,
        fss=[48,52,20,28]): w.append_data(f); tf2+=1
    for si,(ck,lb,ms) in enumerate(nar):
        print(f"\n--- {lb} ---")
        for f in mtr(0.3,c): w.append_data(f); tf2+=1
        inf=mif(ck.params); ji=jax.jit(inf); jr=jax.jit(env.reset); js=jax.jit(env.step)
        key=random.PRNGKey(si*7+42); st=jr(rng=key)
        ps=[st.pipeline_state]; pos=[(float(st.pipeline_state.x.pos[0][0]),
            float(st.pipeline_state.x.pos[0][1]),float(st.pipeline_state.x.pos[0][2]))]
        er=0.0
        for _ in range(ms):
            key,sk=random.split(key); a=ji(st.obs,sk)
            act=a[0] if isinstance(a,tuple) else a; st=js(st,act)
            ps.append(st.pipeline_state); er+=float(st.reward)
            p=st.pipeline_state.x.pos[0]; pos.append((float(p[0]),float(p[1]),float(p[2])))
            if float(st.done): break
        while len(ps)<60: ps.append(ps[-1]); pos.append(pos[-1])
        print(f"  {len(ps)} steps, reward={er:.1f}")
        st2=f"Step {ck.step:,} | Reward {ck.reward:.0f}"; fc=0
        for i,pst in enumerate(ps):
            q=np.array(pst.q); qd=np.array(pst.qd)
            if len(q.shape)>1: q=q[0]; qd=qd[0]
            md.qpos[:len(q)]=q; md.qvel[:len(qd)]=qd; mujoco.mj_forward(mm,md)
            cm=mujoco.MjvCamera(); cm.type=mujoco.mjtCamera.mjCAMERA_FREE
            cm.lookat[0]=pos[i][0]; cm.lookat[1]=pos[i][1]; cm.lookat[2]=c.cam_lookat_z
            cm.distance=c.cam_distance; cm.elevation=c.cam_elevation; cm.azimuth=c.cam_azimuth
            rr.update_scene(md,camera=cm); raw=rr.render().copy()
            comp=cf(raw,lb,st2,c); w.append_data(comp); fc+=1
            if (i+1)%5==0: w.append_data(comp); fc+=1
        tf2+=fc; print(f"  Wrote {fc} frames ({fc/c.fps:.1f}s)")
        del ps,pos
    rr.close()
    for f in mtr(0.3,c): w.append_data(f); tf2+=1
    for f in mtc([f"Trained in {tt:.0f}s",f"Reward: {ckpts[-1].reward:.0f}","","30M | A100"],
        c,dur=2.5,fss=[40,44,20,28]): w.append_data(f); tf2+=1
    for f in mtc(["GPU-Accelerated RL","","google/brax"],c,dur=2.0,fss=[42,20,28]):
        w.append_data(f); tf2+=1
    w.close()
    dur=tf2/c.fps; sz=os.path.getsize(op)/(1024**2)
    print(f"\n{'='*60}\n  Cheetah Complete\n  {op}\n  {dur:.1f}s | {sz:.1f} MB\n{'='*60}")

if __name__=="__main__": main()
