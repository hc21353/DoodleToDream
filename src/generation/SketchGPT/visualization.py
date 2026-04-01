


def toks_to_strokes(toks):
    step_size=0.03; polylines=[]; current_pts=[]; x,y=0.5,0.5
    for t in toks:
        if t in (TOKEN_BOS,TOKEN_PAD): continue
        if t==TOKEN_EOS:
            if len(current_pts)>=2: polylines.append(current_pts)
            break
        if t==TOKEN_SEP:
            if len(current_pts)>=2: polylines.append(current_pts)
            current_pts=[(x,y)]; continue
        pid=t-SPECIAL_TOKENS
        if 0<=pid<N_PRIMITIVES:
            d=PRIMITIVES[pid]
            if not current_pts: current_pts.append((x,y))
            x=float(np.clip(x+d[0]*step_size,0.0,1.0))
            y=float(np.clip(y+d[1]*step_size,0.0,1.0))
            current_pts.append((x,y))
    if len(current_pts)>=2: polylines.append(current_pts)
    return polylines


def draw(polylines,ax,title="",color="black",smooth=True):
    if not polylines:
        ax.axis('off'); ax.set_title(title,fontsize=8); return
    for pts in polylines:
        if len(pts)<2: continue
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
        if smooth and len(pts)>=4:
            try:
                from scipy.interpolate import splprep,splev
                tck,u=splprep([xs,ys],s=0,k=min(3,len(pts)-1))
                u_new=np.linspace(0,1,len(pts)*5)
                xs_s,ys_s=splev(u_new,tck)
                ax.plot(xs_s,ys_s,color=color,lw=1.5,
                        solid_capstyle='round',solid_joinstyle='round')
            except Exception:
                ax.plot(xs,ys,color=color,lw=1.5,
                        solid_capstyle='round',solid_joinstyle='round')
        else:
            ax.plot(xs,ys,color=color,lw=1.5,
                    solid_capstyle='round',solid_joinstyle='round')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.invert_yaxis(); ax.set_aspect('equal')
    ax.axis('off'); ax.set_title(title,fontsize=8)


def draw_original_quickdraw(drawing,ax,title="",color="black"):
    all_x,all_y=[],[]
    for s in drawing: all_x.extend(s[0]); all_y.extend(s[1])
    if not all_x: return
    xr=max(all_x)-min(all_x) or 1; yr=max(all_y)-min(all_y) or 1; PAD=0.05
    for s in drawing:
        xs=[(x-min(all_x))/xr*(1-2*PAD)+PAD for x in s[0]]
        ys=[(y-min(all_y))/yr*(1-2*PAD)+PAD for y in s[1]]
        ax.plot(xs,ys,color=color,lw=1.5,
                solid_capstyle='round',solid_joinstyle='round')
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.invert_yaxis()
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title,fontsize=8)


def _savefig(fname):
    path=os.path.join(OUTPUT_DIR,fname)
    plt.savefig(path,dpi=150,bbox_inches='tight')
    print(f"  [saved] {path}"); plt.close()


def _plot_loss(tr,va,title,fname):
    fig,ax=plt.subplots(figsize=(7,3)); ep=range(1,len(tr)+1)
    ax.plot(ep,tr,'b-o',ms=3,label='Train')
    ax.plot(ep,va,'r-o',ms=3,label='Val')
    ax.set(xlabel='Epoch',ylabel='NLL Loss',title=title)
    ax.legend(); ax.grid(alpha=.3); plt.tight_layout(); _savefig(fname)


def _polylines_to_pil(polylines, img_size=256, color="black"):
    dpi=100; sz=img_size/dpi
    fig,ax=plt.subplots(figsize=(sz,sz),dpi=dpi)
    fig.patch.set_facecolor('white')
    draw(polylines,ax,title="",color=color)
    fig.tight_layout(pad=0)
    buf=io.BytesIO()
    fig.savefig(buf,format='png',dpi=dpi,
                bbox_inches='tight',pad_inches=0.02,facecolor='white')
    plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGB").resize(
        (img_size,img_size),Image.LANCZOS)


def _load_font(size=12):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]:
        try: return ImageFont.truetype(path, size)
        except Exception: pass
    return ImageFont.load_default()



def show_raw_samples(n=4):
    cmap=matplotlib.cm.get_cmap('tab10',len(CLASSES))
    COLS=[matplotlib.colors.to_hex(cmap(i)) for i in range(len(CLASSES))]
    fig,axes=plt.subplots(len(CLASSES),n,
                          figsize=(n*2.8,len(CLASSES)*2.8))
    for ci,cname in enumerate(CLASSES):
        drawings=download_ndjson(cname,n*5)
        samples=random.sample(drawings,min(n,len(drawings)))
        for j,d in enumerate(samples):
            draw_original_quickdraw(d,axes[ci][j],
                                    title=f"{cname}\n(original)",
                                    color=COLS[ci])
    plt.suptitle("QuickDraw Original Data",y=1.01)
    plt.tight_layout(); _savefig("raw_samples.png")


def show_generated(cls_models: dict, cls_datasets: dict, device, n=4):
    cmap=matplotlib.cm.get_cmap('tab10',len(CLASSES))
    COLS=[matplotlib.colors.to_hex(cmap(i)) for i in range(len(CLASSES))]

    fig,axes=plt.subplots(len(CLASSES)*2, n,
                          figsize=(n*2.8, len(CLASSES)*5.6))

    for ci,cname in enumerate(CLASSES):
        model  = cls_models[cname]
        ds     = cls_datasets[cname]
        idxs   = random.sample(range(len(ds)), min(n, len(ds)))

        for j,idx in enumerate(idxs):
            toks_tensor,_ = ds.items[idx]
            full_toks = toks_tensor.tolist()
            real_len  = next((i for i,t in enumerate(full_toks)
                              if t==TOKEN_EOS), len(full_toks))


            draw(toks_to_strokes(full_toks), axes[ci*2][j],
                 f"{cname}\n(original)", COLS[ci])


            prompt_len = max(5, int(real_len*PROMPT_RATIO))
            prompt     = [t for t in full_toks[:prompt_len]
                          if t not in (TOKEN_EOS, TOKEN_PAD)]
            gen_polylines = toks_to_strokes(
                generate(model, device, prompt=prompt))
            draw(gen_polylines, axes[ci*2+1][j],
                 f"{cname}\n(generated #{j+1})", COLS[ci])

    plt.suptitle(
        f"Original vs Generated  [per-class model, prompt={int(PROMPT_RATIO*100)}%]",
        y=1.01)
    plt.tight_layout()
    _savefig("generated_sketches.png")


def save_sequential_strokes(cls_models: dict, cls_datasets: dict, device,
                             n_per_class=10, img_size=256):
    print(f"\n[Sequential] {n_per_class}/class × {len(CLASSES)} = "
          f"{n_per_class*len(CLASSES)} total  →  {SEQ_DIR}")

    all_finals = {c:[] for c in CLASSES}
    total = 0

    for ci,cname in enumerate(CLASSES):
        model  = cls_models[cname]
        ds     = cls_datasets[cname]
        idxs   = random.sample(range(len(ds)), min(n_per_class, len(ds)))
        cls_dir= os.path.join(SEQ_DIR, cname.replace(" ","_"))
        os.makedirs(cls_dir, exist_ok=True)

        for si,idx in enumerate(idxs):
            toks_tensor,_ = ds.items[idx]
            full_toks  = toks_tensor.tolist()
            real_len   = next((i for i,t in enumerate(full_toks)
                               if t==TOKEN_EOS), len(full_toks))
            prompt_len = max(5, int(real_len*PROMPT_RATIO))
            prompt     = [t for t in full_toks[:prompt_len]
                          if t not in (TOKEN_EOS, TOKEN_PAD)]
            gen_toks   = generate(model, device, prompt=prompt)
            polylines  = toks_to_strokes(gen_toks)
            if not polylines: continue

            sample_dir = os.path.join(cls_dir, f"sample_{si:02d}")
            os.makedirs(sample_dir, exist_ok=True)


            for k in range(1, len(polylines)+1):
                _polylines_to_pil(polylines[:k], img_size).save(
                    os.path.join(sample_dir, f"stroke_{k:03d}.png"))


            final_pil = _polylines_to_pil(polylines, img_size)
            final_pil.save(os.path.join(sample_dir, "stroke_all.png"))
            all_finals[cname].append(final_pil)


            stroke_data=[{"stroke_id":i,
                           "points":[[float(x),float(y)] for x,y in pl]}
                          for i,pl in enumerate(polylines)]
            with open(os.path.join(sample_dir,"strokes.json"),"w") as f:
                json.dump({"class":cname,"sample_id":si,
                           "n_strokes":len(polylines),
                           "img_size":img_size,
                           "prompt_ratio":PROMPT_RATIO,
                           "strokes":stroke_data}, f, indent=2)


            records=[(sid,pid,x,y)
                     for sid,pl in enumerate(polylines)
                     for pid,(x,y) in enumerate(pl)]
            np.save(os.path.join(sample_dir,"strokes.npy"),
                    np.array(records,
                             dtype=[('stroke',np.int32),('point',np.int32),
                                    ('x',np.float32),('y',np.float32)]))
            total+=1
            print(f"  [{total:3d}/{n_per_class*len(CLASSES)}] "
                  f"{cname} sample_{si:02d}: {len(polylines)} strokes")

        if all_finals[cname]:
            _save_class_preview(all_finals[cname], cname, cls_dir, img_size)

    _save_overview(all_finals, CLASSES, SEQ_DIR, img_size, n_per_class)
    print(f"\n[Sequential] Done: {total} samples")
    print(f"  overview  : {SEQ_DIR}/overview.png")
    print(f"  per-class : {SEQ_DIR}/<class>/preview.png")


def _save_class_preview(pil_list, cname, cls_dir, img_size):
    n=len(pil_list); pad=4
    W=n*(img_size+pad)+pad; H=img_size+2*pad+22
    canvas=Image.new("RGB",(W,H),(245,245,245))
    for i,img in enumerate(pil_list):
        canvas.paste(img,(pad+i*(img_size+pad),pad))
    d=PILDraw.Draw(canvas)
    d.text((pad,H-18),cname,fill=(40,40,40),font=_load_font(13))
    path=os.path.join(cls_dir,"preview.png")
    canvas.save(path); print(f"  [preview] {path}")


def _save_overview(all_finals, classes, seq_dir, img_size, n_per_class):
    pad=4; label_w=110; title_h=28
    W=label_w+n_per_class*(img_size+pad)+pad
    H=title_h+len(classes)*(img_size+pad)+pad
    canvas=Image.new("RGB",(W,H),(255,255,255))
    d=PILDraw.Draw(canvas)
    font_ti=_load_font(14); font_sm=_load_font(11)
    d.text((pad,6),
           f"Generated Sketches Overview  "
           f"[per-class model, prompt={int(PROMPT_RATIO*100)}%  "
           f"{n_per_class} samples/class]",
           fill=(20,20,20), font=font_ti)
    for ri,cname in enumerate(classes):
        y0=title_h+ri*(img_size+pad)
        d.text((pad,y0+img_size//2-7), cname, fill=(50,50,50), font=font_sm)
        for ci,img in enumerate(all_finals.get(cname,[])[:n_per_class]):
            canvas.paste(img,(label_w+ci*(img_size+pad),y0))
    path=os.path.join(seq_dir,"overview.png")
    canvas.save(path); print(f"  [overview] {path}")
