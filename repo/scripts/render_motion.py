"""Render existing Kimodo joints on CPU; never generate or modify motion."""
import json, math, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
W,H=1280,720
BG=(13,20,34); WHITE=(232,238,248); MUTED=(145,161,183)
LEFT=(60,207,225); RIGHT=(255,160,87); CENTER=(183,164,255)
TARGET=(255,204,51); GENERATED=(56,213,237)
def font(size):
    try: return ImageFont.load_default(size=size)
    except TypeError: return ImageFont.load_default()
F={s:font(s) for s in (16,18,22,30)}
def text(d,xy,value,size=18,color=WHITE):
    d.text(xy,value,font=F[size],fill=color)
import argparse
parser=argparse.ArgumentParser(description="Render an existing Kimodo smoke-test NPZ into MP4/GIF; CPU only.")
parser.add_argument("--input",required=True,help="Existing .npz with matching .json report")
args=parser.parse_args()
input_path=Path(args.input)
assert input_path.suffix==".npz" and input_path.is_file()
assert input_path.resolve().is_relative_to((ROOT/"outputs").resolve()), "Input must be in project outputs."
for stem in (input_path.with_suffix(""),):
    kind=stem.parent.name
    dest=stem.with_name(stem.name+"_skeleton")
    targets=[dest.with_suffix(ext) for ext in (".mp4",".gif",".png")]
    assert not any(p.exists() for p in targets), "Existing preview protected"
    meta=json.loads(stem.with_suffix(".json").read_text())
    with np.load(stem.with_suffix(".npz"),allow_pickle=False) as data:
        points=data["posed_joints"].copy()
    T,J,_=points.shape
    fps=float(meta["fps"])
    assert T>=2 and fps>0
    expected_duration=T/fps
    frame_hash=__import__("hashlib").sha256(stem.with_suffix(".npz").read_bytes()).hexdigest()
    parents=meta["joint_parents"]; names=meta["joint_names"]
    assert len(parents)==J and np.isfinite(points).all()
    colors=[LEFT if "left" in n.lower() else RIGHT if "right" in n.lower() else CENTER for n in names]
    keyframes=np.asarray(meta.get("keyframe_indices",[]),dtype=int)
    target_points=np.asarray(meta.get("keyframe_targets",[]),dtype=float)
    if len(keyframes):
        assert target_points.shape==(len(keyframes),J,3) and np.isfinite(target_points).all()
        assert np.all((keyframes>=0)&(keyframes<T)) and len(set(keyframes.tolist()))==len(keyframes)
        colors=[GENERATED]*J
        bounds_points=np.concatenate([points,target_points],axis=0)
    else:
        assert target_points.size==0, "Targets need matching keyframe indices"
        bounds_points=points
    # Kimodo native coordinate system: Y up, Z forward. Cameras remain fixed.
    right=np.array([.8,0,-.6]); up=np.array([-.22,.93,-.30]); up/=np.linalg.norm(up)
    cams=[np.stack([right,up],axis=1),np.array([[1,0],[0,1],[0,0]],float)]
    panels=[(30,130,615,620),(665,130,1250,620)]
    projections=[]
    for cam,box in zip(cams,panels):
        p=bounds_points@cam
        lo=p.reshape(-1,2).min(0); hi=p.reshape(-1,2).max(0)
        mid=(lo+hi)/2
        scale=min((box[2]-box[0]-100)/max(hi[0]-lo[0],.5),(box[3]-box[1]-80)/max(hi[1]-lo[1],.5))
        def project(x,cam=cam,mid=mid,scale=scale,box=box):
            v=np.asarray(x)@cam
            v=(v-mid)*scale
            return np.stack([(box[0]+box[2])/2+v[...,0],(box[1]+box[3])/2-v[...,1]],axis=-1)
        projections.append(project)
    cmd=["ffmpeg","-hide_banner","-loglevel","error","-n","-f","rawvideo","-pix_fmt","rgb24",
         "-s",f"{W}x{H}","-r",str(meta["fps"]),"-i","pipe:0","-an","-c:v","libx264",
         "-preset","fast","-crf","19","-threads","2","-pix_fmt","yuv420p","-movflags","+faststart",str(targets[0])]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    gifframes=[]; checks=[]
    gridmin=np.floor(bounds_points[:,:,[0,2]].reshape(-1,2).min(0))-.5
    gridmax=np.ceil(bounds_points[:,:,[0,2]].reshape(-1,2).max(0))+.5
    for frame in range(T):
        im=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(im)
        text(d,(32,22),kind.upper()+"  |  Kimodo motion preview",30)
        label = meta.get("prompt") or "No text conditioning"
        text(d,(32,66),f"{label}  /  {J} joints  /  {expected_duration:.1f} s  /  {fps:g} FPS",18,MUTED)
        text(d,(32,104),"3D VIEW",18); text(d,(670,104),"FRONT VIEW",18)
        for view,(box,project) in enumerate(zip(panels,projections)):
            d.rounded_rectangle(box,radius=14,fill=(18,29,46),outline=(37,53,74),width=1)
            # Clip grid and skeleton to this viewport.
            layer=Image.new("RGB",(W,H),(18,29,46)); ld=ImageDraw.Draw(layer)
            if view==0:
                for x in np.arange(gridmin[0],gridmax[0]+.01,.5):
                    a,b=project([[x,0,gridmin[1]],[x,0,gridmax[1]]])
                    ld.line([tuple(a),tuple(b)],fill=(39,55,73),width=1)
                for z in np.arange(gridmin[1],gridmax[1]+.01,.5):
                    a,b=project([[gridmin[0],0,z],[gridmax[0],0,z]])
                    ld.line([tuple(a),tuple(b)],fill=(39,55,73),width=1)
            else:
                y=project([[0,0,0]])[0,1]
                ld.line([(box[0],y),(box[2],y)],fill=(50,69,90),width=2)
            # Root trajectory from the beginning; absolute translation is retained.
            trail=project(points[:frame+1,0,:])
            if len(trail)>1: ld.line([tuple(v) for v in trail],fill=(102,121,145),width=2)
            xy=project(points[frame])
            for j,parent in enumerate(parents):
                if parent<0: continue
                a,b=xy[parent],xy[j]
                ld.line([tuple(a),tuple(b)],fill=colors[j],width=4)
            for j,(x,y) in enumerate(xy):
                r=3 if J==77 and ("Hand" in names[j] or "Eye" in names[j]) else 4
                ld.ellipse((x-r,y-r,x+r,y+r),fill=colors[j])
            # Keep constraints fixed in world space; never align them to the moving root.
            # Draw yellow targets last so coincident constrained poses remain visible.
            for keyframe,target in zip(keyframes,target_points):
                target_xy=project(target)
                for j,parent in enumerate(parents):
                    if parent>=0:
                        ld.line([tuple(target_xy[parent]),tuple(target_xy[j])],fill=TARGET,width=3)
                for x,y in target_xy:
                    ld.ellipse((x-3,y-3,x+3,y+3),fill=TARGET)
                anchor=target_xy[np.argmin(target_xy[:,1])]
                text(ld,(anchor[0]+8,anchor[1]-22),f"TARGET {keyframe/fps:.2f}s",16,TARGET)
            im.paste(layer.crop((box[0]+2,box[1]+2,box[2]-2,box[3]-2)),(box[0]+2,box[1]+2))
        if len(keyframes):
            text(d,(34,643),"GENERATED",16,GENERATED)
            text(d,(175,643),"KEYFRAME TARGET (fixed position)",16,TARGET)
        else:
            text(d,(34,643),"LEFT",16,LEFT); text(d,(112,643),"RIGHT",16,RIGHT)
            text(d,(205,643),"CENTER",16,CENTER)
        text(d,(780,643),f"Frame {frame+1:02d}/{T}   |   t = {frame/fps:.2f} s",18)
        d.rounded_rectangle((32,683,1248,690),radius=3,fill=(43,57,77))
        d.rounded_rectangle((32,683,32+int(1216*(frame+1)/T),690),radius=3,fill=LEFT)
        for keyframe in keyframes:
            x=32+int(1216*keyframe/max(T-1,1))
            d.line((x,678,x,695),fill=TARGET,width=3)
        proc.stdin.write(im.tobytes())
        if frame%2==0:
            gifframes.append(im.resize((768,432),Image.Resampling.LANCZOS))
        if frame in (0,(T-1)//2,T-1): checks.append(im.resize((640,360),Image.Resampling.LANCZOS))
    proc.stdin.close()
    assert proc.wait(timeout=30)==0
    # GIF shows every second source frame; timing preserves the source duration.
    ends=[round(min((i+1)*2,T)/fps*100)*10 for i in range(len(gifframes))]
    durations=[ends[0]]+[ends[i]-ends[i-1] for i in range(1,len(ends))]
    assert all(d>0 for d in durations)
    gifframes[0].save(targets[1],save_all=True,append_images=gifframes[1:],duration=durations,loop=0,optimize=False)
    sheet=Image.new("RGB",(1920,360),BG)
    for i,im in enumerate(checks): sheet.paste(im,(i*640,0))
    sheet.save(targets[2])
    probe=subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
          "-show_entries","stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration",
          "-of","json",str(targets[0])],text=True)
    info=json.loads(probe)["streams"][0]
    assert int(info["nb_read_frames"])==T and abs(float(info["duration"])-expected_duration)<1/fps
    subprocess.run(["ffmpeg","-v","error","-threads","2","-i",str(targets[0]),"-f","null","-"],check=True)
    with Image.open(targets[1]) as gif:
        assert gif.n_frames==len(gifframes)
        duration=0
        for i in range(gif.n_frames): gif.seek(i); duration+=gif.info["duration"]
        assert abs(duration-expected_duration*1000)<=10
    assert __import__("hashlib").sha256(stem.with_suffix(".npz").read_bytes()).hexdigest()==frame_hash
    print(kind.upper(),json.dumps(info),[(str(p),p.stat().st_size) for p in targets],flush=True)
