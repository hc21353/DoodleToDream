


def pretrain(model, train_ds, val_ds, device, sample_weights=None,
             epochs=PRETRAIN_EPOCHS, lr=PRETRAIN_LR,
             batch=PRETRAIN_BATCH, save_path=PRETRAIN_PATH):
    print(f"\n{'='*55}")
    print(f"  PHASE 1: Pre-training  (ep={epochs} lr={lr})")
    print(f"  All {len(CLASSES)} classes mixed, no class discrimination")
    print(f"{'='*55}")
    sampler=None
    if sample_weights is not None:
        sampler=WeightedRandomSampler(
            torch.tensor(sample_weights,dtype=torch.float),
            len(train_ds),replacement=True)
    tr=make_loader(train_ds,batch,sampler=sampler,
                   shuffle=(sampler is None),drop_last=True)
    va=make_loader(val_ds,batch)
    opt=AdamW(model.parameters(),lr=lr,weight_decay=0.01,betas=(0.9,0.95))
    sch=CosineAnnealingLR(opt,T_max=epochs,eta_min=lr*0.1)
    tr_ls,va_ls,best,no_imp=[],[],float('inf'),0
    for ep in range(1,epochs+1):
        model.train(); ep_loss=0.0
        bar=tqdm(tr,desc=f"PT {ep}/{epochs}",ncols=90)
        for toks,_ in bar:
            toks=toks.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                loss=lm_loss(model(toks),toks)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            scaler.step(opt); scaler.update()
            ep_loss+=loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}",
                            lr=f"{sch.get_last_lr()[0]:.5f}")
        sch.step()
        va_loss=eval_lm(model,va,device); tr_loss=ep_loss/len(tr)
        tr_ls.append(tr_loss); va_ls.append(va_loss)
        print(f"  Ep{ep:3d} | train={tr_loss:.4f} val={va_loss:.4f} "
              f"ppl={math.exp(min(va_loss,20)):.1f}")
        if va_loss<best:
            best=va_loss; no_imp=0; torch.save(model.state_dict(),save_path)
        else:
            no_imp+=1
            if no_imp>=EARLY_STOP_PATIENCE:
                print(f"  ⏹ Early stopping (ep={ep})"); break
    _plot_loss(tr_ls,va_ls,"Pre-training Loss","pretrain_loss.png")
    model.load_state_dict(torch.load(save_path,map_location=device,
                                     weights_only=True))
    print(f" Pre-training done  best_val={best:.4f}  saved: {save_path}")
    return model


def finetune_class(cls_name: str, pretrain_path: str, device,
                   epochs=FINETUNE_GEN_EPOCHS, lr=FINETUNE_GEN_LR,
                   batch=PRETRAIN_BATCH) -> nn.Module:
    save_path = finetune_path(cls_name)

    print(f"\n{'='*55}")
    print(f"  PHASE 2: Fine-tuning — {cls_name}")
    print(f"  (ep={epochs} lr={lr}  saved→ {save_path})")
    print(f"{'='*55}")


    partial_tr, partial_va, _ = build_class_dataset(
        cls_name, N_TRAIN_PER_CLASS, N_VAL_PER_CLASS, MAX_SEQ)


    model = make_model()
    model.load_state_dict(torch.load(pretrain_path, map_location=device,
                                     weights_only=True))

    tr  = make_loader(partial_tr, batch, shuffle=True, drop_last=True)
    va  = make_loader(partial_va, batch)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.1)

    tr_ls,va_ls,best,no_imp=[],[],float('inf'),0
    for ep in range(1,epochs+1):
        model.train(); ep_loss=0.0
        bar=tqdm(tr,desc=f"FT[{cls_name}] {ep}/{epochs}",ncols=90)
        for toks,_ in bar:
            toks=toks.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                loss=lm_loss(model(toks),toks)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            scaler.step(opt); scaler.update()
            ep_loss+=loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")
        sch.step()
        va_loss=eval_lm(model,va,device); tr_loss=ep_loss/len(tr)
        tr_ls.append(tr_loss); va_ls.append(va_loss)
        print(f"  Ep{ep:3d} | train={tr_loss:.4f} val={va_loss:.4f}")
        if va_loss<best:
            best=va_loss; no_imp=0; torch.save(model.state_dict(),save_path)
        else:
            no_imp+=1
            if no_imp>=EARLY_STOP_PATIENCE:
                print(f"  ⏹ Early stopping (ep={ep})"); break

    _plot_loss(tr_ls, va_ls,
               f"Fine-tuning Loss [{cls_name}]",
               f"ft_loss_{cls_name.replace(' ','_')}.png")
    model.load_state_dict(torch.load(save_path,map_location=device,
                                     weights_only=True))
    print(f" Fine-tuning done [{cls_name}]  best_val={best:.4f}")
    return model



@torch.no_grad()
def generate(model, device, prompt=None, max_new=None,
             temperature=TEMPERATURE, top_k=TOP_K,
             min_new_tokens=MIN_NEW_TOKENS):
    if max_new is None: max_new=MAX_SEQ
    model.eval()
    toks=list(prompt) if prompt else [TOKEN_BOS]
    generated=0
    for _ in range(max_new):
        ctx=torch.tensor([toks[-MAX_SEQ:]],dtype=torch.long,device=device)
        logit=model(ctx)[0,-1]
        logit[TOKEN_PAD]=float('-inf')
        if generated<min_new_tokens: logit[TOKEN_EOS]=float('-inf')
        logit/=temperature
        if top_k>0:
            v,_=logit.topk(min(top_k,logit.size(-1)))
            logit[logit<v[-1]]=float('-inf')
        tok=torch.multinomial(F.softmax(logit,-1),1).item()
        toks.append(tok); generated+=1
        if tok==TOKEN_EOS: break
    return toks
