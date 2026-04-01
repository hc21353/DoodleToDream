


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq, dropout):
        super().__init__()
        self.n_heads=n_heads; self.d_head=d_model//n_heads
        self.qkv=nn.Linear(d_model,3*d_model,bias=False)
        self.proj=nn.Linear(d_model,d_model,bias=False)
        self.attn_drop=nn.Dropout(dropout)
        self.resid_drop=nn.Dropout(dropout)
        mask=torch.tril(torch.ones(max_seq,max_seq))
        self.register_buffer("causal_mask",mask.view(1,1,max_seq,max_seq))
    def forward(self,x):
        B,L,D=x.shape
        q,k,v=self.qkv(x).split(D,dim=2)
        def h(t): return t.view(B,L,self.n_heads,self.d_head).transpose(1,2)
        q,k,v=h(q),h(k),h(v)
        s=(q@k.transpose(-2,-1))/math.sqrt(self.d_head)
        s=s.masked_fill(self.causal_mask[:,:,:L,:L]==0,float('-inf'))
        w=self.attn_drop(torch.softmax(s,dim=-1))
        return self.resid_drop(self.proj(
            (w@v).transpose(1,2).contiguous().view(B,L,D)))


class TransformerBlock(nn.Module):
    def __init__(self,d_model,n_heads,d_ff,max_seq,dropout):
        super().__init__()
        self.ln1=nn.LayerNorm(d_model)
        self.attn=CausalSelfAttention(d_model,n_heads,max_seq,dropout)
        self.ln2=nn.LayerNorm(d_model)
        self.mlp=nn.Sequential(nn.Linear(d_model,d_ff),nn.GELU(),
                               nn.Linear(d_ff,d_model),nn.Dropout(dropout))
    def forward(self,x):
        x=x+self.attn(self.ln1(x)); x=x+self.mlp(self.ln2(x)); return x


class SketchGPT(nn.Module):
    def __init__(self,vocab_size,d_model,n_heads,n_layers,d_ff,max_seq,dropout):
        super().__init__()
        self.tok_emb=nn.Embedding(vocab_size,d_model)
        self.pos_emb=nn.Embedding(max_seq,d_model)
        self.drop=nn.Dropout(dropout)
        self.blocks=nn.ModuleList([
            TransformerBlock(d_model,n_heads,d_ff,max_seq,dropout)
            for _ in range(n_layers)])
        self.ln_f=nn.LayerNorm(d_model)
        self.lm_head=nn.Linear(d_model,vocab_size,bias=False)
        self.apply(self._init_weights)
        n_p=sum(p.numel() for p in self.parameters())
        print(f"  [SketchGPT] L={n_layers}/A={n_heads}/H={d_model} "
              f"params={n_p:,} ({n_p/1e6:.1f}M)")
    def _init_weights(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,0.0,0.02)
        if isinstance(m,nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
    def forward(self,tokens):
        B,L=tokens.shape
        pos=torch.arange(L,device=tokens.device).unsqueeze(0)
        x=self.drop(self.tok_emb(tokens)+self.pos_emb(pos))
        for blk in self.blocks: x=blk(x)
        return self.lm_head(self.ln_f(x))


def make_model():
    return SketchGPT(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, max_seq=MAX_SEQ, dropout=DROPOUT
    ).to(DEVICE)


def lm_loss(logits,tokens):
    return F.cross_entropy(
        logits[:,:-1].contiguous().view(-1,logits.size(-1)),
        tokens[:,1:].contiguous().view(-1), ignore_index=TOKEN_PAD)

@torch.no_grad()
def eval_lm(model,loader,device):
    model.eval(); total,n=0.0,0
    for toks,_ in loader:
        toks=toks.to(device)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            total+=lm_loss(model(toks),toks).item()
        n+=1
    return total/max(n,1)
