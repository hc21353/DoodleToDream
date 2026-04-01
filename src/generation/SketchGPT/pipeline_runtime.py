

def main(skip_eda=False, skip_pretrain=False, skip_finetune=False):
    global N_PRIMITIVES, VOCAB_SIZE, MAX_SEQ, PRIM_LENGTH, PRIMITIVES

    print("="*60)
    if torch.cuda.is_available():
        print(f"  SketchGPT — GPU {GPU_ID}: {torch.cuda.get_device_name(GPU_ID)}")
    else:
        print("  SketchGPT — CPU mode")
    print(f"  Classes : {len(CLASSES)}   Model: L={N_LAYERS}/A={N_HEADS}/H={D_MODEL}")
    print(f"  Output  : {OUTPUT_DIR}")
    print(f"  Paper 3.4: per-class fine-tune → {len(CLASSES)} models")
    print("="*60)


    if skip_eda and os.path.exists(EDA_PATH):
        print("\n[Step 0] Load EDA")
        with open(EDA_PATH) as f: eda=json.load(f)
        if eda["n_primitives"]!=N_PRIMITIVES:
            raise ValueError("N_PRIMITIVES mismatch — delete checkpoints/ and rerun")
        N_PRIMITIVES=eda["n_primitives"]; MAX_SEQ=eda["max_seq"]
        PRIM_LENGTH=eda["prim_length"]
        VOCAB_SIZE=SPECIAL_TOKENS+N_PRIMITIVES
        PRIMITIVES=build_primitives(N_PRIMITIVES)
        print(f"  N_PRIMITIVES={N_PRIMITIVES}  MAX_SEQ={MAX_SEQ}  "
              f"PRIM_LENGTH={PRIM_LENGTH:.5f}")
    else:
        print("\n[Step 0] Run EDA")
        rec_n_prim,rec_max_seq,rec_prim_len=run_eda(CLASSES,n_sample=500)
        N_PRIMITIVES=rec_n_prim; VOCAB_SIZE=SPECIAL_TOKENS+N_PRIMITIVES
        MAX_SEQ=min(rec_max_seq,MAX_SEQ_HARD_LIMIT); PRIM_LENGTH=rec_prim_len
        PRIMITIVES=build_primitives(N_PRIMITIVES)
        with open(EDA_PATH,"w") as f:
            json.dump({"n_primitives":N_PRIMITIVES,"max_seq":MAX_SEQ,
                       "prim_length":PRIM_LENGTH,"classes":CLASSES},f,indent=2)


    print("\n[Step 1] Build datasets (all classes, for pre-training)")
    train_ds,val_ds,test_ds,sw = build_datasets(
        CLASSES, N_TRAIN_PER_CLASS, N_VAL_PER_CLASS, N_TEST_PER_CLASS,
        N_PRETRAIN_PER_CLASS, MAX_SEQ)


    print("\n[Step 2] Init model")
    model = make_model()


    if skip_pretrain and os.path.exists(PRETRAIN_PATH):
        print("\n[Step 3] Load pre-train checkpoint")
        model.load_state_dict(torch.load(PRETRAIN_PATH, map_location=DEVICE,
                                         weights_only=True))
        print(f"  loaded: {PRETRAIN_PATH}")
    else:
        print("\n[Step 3] Pre-training (all classes mixed)")
        model = pretrain(model, train_ds, val_ds, DEVICE, sample_weights=sw)


    print(f"\n[Step 4] Per-class fine-tuning  ({len(CLASSES)} classes)")
    cls_models   = {}
    cls_datasets = {}

    for ci, cname in enumerate(CLASSES):
        ft_path = finetune_path(cname)
        print(f"\n  [{ci+1}/{len(CLASSES)}] {cname}")

        if skip_finetune and os.path.exists(ft_path):

            print(f"  Skip — loading {ft_path}")
            m = make_model()
            m.load_state_dict(torch.load(ft_path, map_location=DEVICE,
                                          weights_only=True))

            _, _, full_tr = build_class_dataset(
                cname, N_TRAIN_PER_CLASS, N_VAL_PER_CLASS, MAX_SEQ)
        else:

            m = finetune_class(cname, PRETRAIN_PATH, DEVICE)
            _, _, full_tr = build_class_dataset(
                cname, N_TRAIN_PER_CLASS, N_VAL_PER_CLASS, MAX_SEQ)

        cls_models[cname]   = m
        cls_datasets[cname] = full_tr


    print("\n[Step 5-A] Raw QuickDraw samples")
    show_raw_samples(n=4)


    print("\n[Step 5-B] Generated sketches (per-class model)")
    show_generated(cls_models, cls_datasets, DEVICE, n=4)


    print("\n[Step 5-C] Sequential stroke images (per-class model)")
    save_sequential_strokes(cls_models, cls_datasets, DEVICE,
                            n_per_class=10, img_size=256)

    print(f"\n Done!  All outputs → {OUTPUT_DIR}")
    print(f"  generated_sketches.png         : original vs generated (per-class)")
    print(f"  sequential/overview.png        : 100개 전체 한눈에 보기")
    print(f"  sequential/<class>/preview.png : 클래스별 10개")
    print(f"  sequential/<class>/sample_XX/  : stroke 누적 이미지 + JSON/NPY")
    print(f"\n  checkpoints/")
    print(f"    pt_best.pt                   : 공통 pre-train")
    for cname in CLASSES:
        print(f"    gen_{cname.replace(' ','_')}.pt")


if __name__ == "__main__":


    main()








