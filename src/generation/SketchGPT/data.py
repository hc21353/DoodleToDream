



def make_loader(dataset, batch_size, shuffle=False, sampler=None, drop_last=False):
    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=(shuffle and sampler is None), sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=drop_last,
        persistent_workers=True,
    )


def download_ndjson(class_name: str, max_n: int,
                    recognized_only: bool = True) -> list:
    url_name  = class_name.replace(" ", "%20")
    url       = (f"https://storage.googleapis.com/quickdraw_dataset"
                 f"/full/simplified/{url_name}.ndjson")
    save_path = os.path.join(DATA_DIR, f"{class_name.replace(' ','_')}.ndjson")

    if not os.path.exists(save_path):
        print(f"  Downloading: {class_name} ...")
        r = requests.get(url, stream=True); r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(8192): f.write(chunk)
    else:
        print(f"  Cache: {class_name}")

    sketches = []; total_seen = 0
    with open(save_path) as f:
        for line in f:
            total_seen += 1
            data = json.loads(line)
            if recognized_only and not data.get("recognized", True):
                continue
            sketches.append(data["drawing"])
            if len(sketches) >= max_n: break
    print(f"  Loaded: {len(sketches)} (checked {total_seen})")
    return sketches


def drawing_to_stroke3(drawing: list) -> np.ndarray:
    pts, prev_x, prev_y = [], 0, 0
    for stroke in drawing:
        xs, ys = stroke[0], stroke[1]
        for i in range(len(xs)):
            pts.append([xs[i]-prev_x, ys[i]-prev_y,
                        1 if i==len(xs)-1 else 0])
            prev_x, prev_y = xs[i], ys[i]
    return np.array(pts, dtype=np.float32)


def normalize_stroke3(s3: np.ndarray) -> np.ndarray:
    abs_xy = np.cumsum(s3[:, :2], axis=0)
    mn, mx = abs_xy.min(0), abs_xy.max(0)
    denom  = np.where(mx-mn < 1e-8, 1.0, mx-mn)
    norm   = (abs_xy - mn) / denom
    delta  = np.diff(norm, axis=0, prepend=norm[:1])
    out    = s3.copy(); out[:, :2] = delta
    return out


def build_primitives(n: int) -> np.ndarray:
    angles = np.linspace(0, 2*math.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)

PRIMITIVES = build_primitives(N_PRIMITIVES)


def prim_id(dx: float, dy: float) -> int:
    vec = np.array([dx, dy]); n = np.linalg.norm(vec)
    if n < 1e-8: return 0
    return int(np.argmax(PRIMITIVES @ (vec / n)))


def scale_factor(dx: float, dy: float) -> int:
    L = math.sqrt(dx**2 + dy**2)
    if L < 1e-8: return 1
    return max(1, min(8, math.ceil(L / PRIM_LENGTH)))


def tokenize(s3: np.ndarray) -> list:
    tokens = [TOKEN_BOS]
    for i, (dx, dy, lift) in enumerate(s3):
        tok = SPECIAL_TOKENS + prim_id(dx, dy)
        tokens.extend([tok] * scale_factor(dx, dy))
        if lift == 1 and i < len(s3)-1:
            tokens.append(TOKEN_SEP)
    tokens.append(TOKEN_EOS)
    return tokens


def run_eda(classes, n_sample=500):
    global PRIMITIVES
    all_angles, all_lengths, all_stroke_lens = [], [], []
    for cls_name in classes:
        drawings = download_ndjson(cls_name, n_sample*3)
        count = 0
        for d in drawings:
            s3 = drawing_to_stroke3(d)
            if len(s3) < 5: continue
            s3 = normalize_stroke3(s3)
            for dx, dy, _ in s3:
                L = math.sqrt(dx**2+dy**2)
                if L > 1e-8:
                    all_angles.append(math.atan2(dy, dx))
                    all_stroke_lens.append(L)
            all_lengths.append(len(tokenize(s3)))
            count += 1
            if count >= n_sample: break

    all_angles      = np.array(all_angles)
    all_lengths     = np.array(all_lengths)
    all_stroke_lens = np.array(all_stroke_lens)

    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    axes[0].hist(np.degrees(all_angles), bins=36, range=(-180,180),
                 color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axhline(len(all_angles)/36, color='red', ls='--',
                    label='Uniform baseline')
    axes[0].set(xlabel='Direction (deg)', ylabel='Count',
                title='Stroke Direction Distribution')
    axes[0].legend(); axes[0].grid(alpha=.3)

    p50_l = np.percentile(all_stroke_lens, 50)
    axes[1].hist(all_stroke_lens, bins=60,
                 range=(0, np.percentile(all_stroke_lens,99)),
                 color='coral', edgecolor='white', alpha=0.8)
    axes[1].axvline(p50_l, color='red', ls='--', lw=2,
                    label=f'median={p50_l:.4f}')
    axes[1].set(xlabel='Stroke length (normalized)',
                title='Stroke Length Distribution')
    axes[1].legend(); axes[1].grid(alpha=.3)

    p95 = np.percentile(all_lengths, 95)
    axes[2].hist(all_lengths, bins=50, color='mediumpurple',
                 edgecolor='white', alpha=0.8)
    axes[2].axvline(p95, color='red', ls='--', lw=2,
                    label=f'95th={p95:.0f}')
    axes[2].set(xlabel='Token length', title='Token Length Distribution')
    axes[2].legend(); axes[2].grid(alpha=.3)

    plt.suptitle(f"EDA: {', '.join(classes)}")
    plt.tight_layout()
    _savefig("eda_analysis.png")



    rec_n_prim   = N_PRIMITIVES
    rec_prim_len = float(p50_l)
    raw_max      = int(p95)
    rec_max_seq  = min(2**math.ceil(math.log2(max(raw_max,64))),
                       MAX_SEQ_HARD_LIMIT)
    print(f"  N_PRIMITIVES={rec_n_prim}  PRIM_LENGTH={rec_prim_len:.5f}  "
          f"MAX_SEQ={rec_max_seq}")
    return rec_n_prim, rec_max_seq, rec_prim_len


class SketchDataset(Dataset):
    def __init__(self, tokens_list, labels, max_seq):
        self.items = []
        for toks, lbl in zip(tokens_list, labels):
            toks = list(toks[:max_seq-1])
            if TOKEN_EOS not in toks: toks.append(TOKEN_EOS)
            toks += [TOKEN_PAD]*(max_seq-len(toks))
            self.items.append((torch.tensor(toks, dtype=torch.long), int(lbl)))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


class PartialSketchDataset(Dataset):
    def __init__(self, tokens_list, labels, max_seq,
                 min_ratio=0.1, max_ratio=0.9):
        self.max_seq = max_seq
        self.min_ratio = min_ratio; self.max_ratio = max_ratio
        self.raw = []
        for toks, lbl in zip(tokens_list, labels):
            toks = list(toks[:max_seq-1])
            if TOKEN_EOS not in toks: toks.append(TOKEN_EOS)
            eos_idx = toks.index(TOKEN_EOS)
            self.raw.append((toks[:eos_idx+1], int(lbl)))
    def __len__(self): return len(self.raw)
    def __getitem__(self, i):
        toks, lbl = self.raw[i]; n = len(toks)
        if n < 4:
            cut = n
        else:
            min_cut = max(2, int(n*self.min_ratio))
            max_cut = max(min_cut+1, int(n*self.max_ratio))
            cut = random.randint(min_cut, max_cut)
        partial = toks[:cut]
        if TOKEN_EOS not in partial: partial.append(TOKEN_EOS)
        partial += [TOKEN_PAD]*(self.max_seq-len(partial))
        return torch.tensor(partial[:self.max_seq], dtype=torch.long), lbl


def build_datasets(classes, n_train, n_val, n_test, n_pretrain, max_seq):
    per_cls = []
    for cls_name in classes:
        needed   = n_train+n_val+n_test
        drawings = download_ndjson(cls_name, needed*3)
        cls_toks = []
        for d in drawings:
            s3 = drawing_to_stroke3(d)
            if len(s3) < 5: continue
            s3 = normalize_stroke3(s3); toks = tokenize(s3)
            if len(toks) < 5: continue
            cls_toks.append(toks)
            if len(cls_toks) >= needed: break
        if len(cls_toks) < needed:
            print(f"    {cls_name}: {len(cls_toks)} (target {needed})")
        per_cls.append(cls_toks)

    tr_toks, tr_labs, va_toks, va_labs, te_toks, te_labs = [],[],[],[],[],[]
    cls_counts = []
    for ci, cls_toks in enumerate(per_cls):
        random.shuffle(cls_toks); n = len(cls_toks)
        t1 = min(n_train, int(n*0.5)); t2 = min(n_val, int(n*0.25))
        tr_toks.extend(cls_toks[:t1]);             tr_labs.extend([ci]*t1)
        va_toks.extend(cls_toks[t1:t1+t2]);       va_labs.extend([ci]*t2)
        te_toks.extend(cls_toks[t1+t2:t1+t2*2]); te_labs.extend([ci]*t2)
        cls_counts.append(t1)
        print(f"  {classes[ci]}: train={t1}, val={t2}, test={t2}")

    cls_w = [1.0/c for c in cls_counts]
    def shuf(a,b):
        c=list(zip(a,b)); random.shuffle(c)
        return zip(*c) if c else ([],[])

    tr_t,tr_l = shuf(tr_toks,tr_labs)
    va_t,va_l = shuf(va_toks,va_labs)
    te_t,te_l = shuf(te_toks,te_labs)
    tr_t2,tr_l2 = list(tr_t),list(tr_l)
    sw = [cls_w[l] for l in tr_l2]

    train_ds   = SketchDataset(tr_t2, tr_l2, max_seq)
    val_ds     = SketchDataset(list(va_t), list(va_l), max_seq)
    test_ds    = SketchDataset(list(te_t), list(te_l), max_seq)
    print(f"\n  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    return train_ds, val_ds, test_ds, sw


def build_class_dataset(cls_name: str, n_train: int, n_val: int,
                        max_seq: int):
    needed   = n_train + n_val
    drawings = download_ndjson(cls_name, needed*3)
    cls_toks = []
    for d in drawings:
        s3 = drawing_to_stroke3(d)
        if len(s3) < 5: continue
        s3 = normalize_stroke3(s3); toks = tokenize(s3)
        if len(toks) < 5: continue
        cls_toks.append(toks)
        if len(cls_toks) >= needed: break

    if len(cls_toks) < needed:
        print(f"    {cls_name}: {len(cls_toks)} (target {needed})")

    random.shuffle(cls_toks)
    n  = len(cls_toks)
    t1 = min(n_train, int(n*0.67))
    t2 = n - t1


    tr_toks = cls_toks[:t1];  tr_labs = [0]*t1
    va_toks = cls_toks[t1:];  va_labs = [0]*t2

    partial_tr = PartialSketchDataset(tr_toks, tr_labs, max_seq)
    partial_va = PartialSketchDataset(va_toks, va_labs, max_seq)

    full_tr    = SketchDataset(tr_toks, tr_labs, max_seq)

    print(f"  {cls_name}: ft_train={t1}, ft_val={t2}")
    return partial_tr, partial_va, full_tr
