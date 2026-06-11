import numpy as np, pandas as pd, torch, torch.nn as nn, warnings
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from itertools import product
warnings.filterwarnings('ignore')

META_CATEGORIES = {
    'floral':['floral','rose','jasmin','lily','muguet','violet','hyacinth','geranium','lavender','orangeflower','chamomile','hawthorn'],
    'fruity':['fruity','apple','apricot','banana','berry','cherry','grape','grapefruit','lemon','melon','orange','peach','pear','pineapple','plum','raspberry','strawberry','tropical','black currant','fruit skin'],
    'sweet':['sweet','vanilla','caramellic','honey','chocolate','cocoa','coconut','creamy','buttery','milky','dairy'],
    'woody':['woody','cedar','sandalwood','pine','vetiver','terpenic','balsamic','cortex'],
    'green':['green','grassy','herbal','leafy','hay','tea','fresh','cucumber','vegetable','weedy'],
    'spicy':['spicy','cinnamon','clove','warm','pungent','sharp','cooling','mint','camphoreous'],
    'animal_musk':['animal','musk','leathery','fishy','sweaty','meaty','beefy','musty'],
    'earthy':['earthy','mushroom','nutty','hazelnut','roasted','coffee','tobacco','smoky','popcorn'],
    'citrus':['citrus','bergamot','ozone','clean','soapy'],
    'chemical':['solvent','ethereal','metallic','medicinal','phenolic','sulfurous','gassy','burnt','oily'],
    'gourmand':['almond','malty','rummy','brandy','cognac','winey','cooked','potato','savory','celery','tomato','radish','onion','garlic','cabbage','cheesy'],
    'powdery_amber':['amber','powdery','anisic','coumarinic','orris','waxy','aldehydic','ketonic','lactonic'],
}

def load_and_split(csv='hmcn_dataset.csv', seed=42):
    df = pd.read_csv(csv)
    fine_cols = [c for c in df.columns if c.startswith('fine_')]
    meta_cols = [c for c in df.columns if c.startswith('meta_')]
    feat_cols = [c for c in df.columns if c not in fine_cols+meta_cols+['SMILES']]
    stds = df[feat_cols].std()
    feat_cols = stds[stds>0].index.tolist()
    X  = df[feat_cols].values.astype(np.float32)
    Y1 = df[fine_cols].values.astype(np.float32)
    Y2 = df[meta_cols].values.astype(np.float32)
    fine_names = [c.replace('fine_','') for c in fine_cols]
    meta_names = [c.replace('meta_','') for c in meta_cols]

    msss = MultilabelStratifiedShuffleSplit(1, test_size=0.2, random_state=seed)
    tv,te = next(msss.split(X,Y2))
    msss2 = MultilabelStratifiedShuffleSplit(1, test_size=0.1/0.8, random_state=seed)
    tr,va = next(msss2.split(X[tv],Y2[tv]))

    sc = StandardScaler()
    Xtr = sc.fit_transform(X[tv][tr]); Xva = sc.transform(X[tv][va]); Xte = sc.transform(X[te])
    return (Xtr,Y1[tv][tr],Y2[tv][tr], Xva,Y1[tv][va],Y2[tv][va],
            Xte,Y1[te],Y2[te], fine_names, meta_names)

def build_pairs(fn,mn):
    fi={n:i for i,n in enumerate(fn)}; mi={n:i for i,n in enumerate(mn)}
    return [(fi[m],mi[meta]) for meta,mems in META_CATEGORIES.items()
            for m in mems if m in fi and meta in mi]

class Block(nn.Module):
    def __init__(self,idim,gdim,ldim,nl,dr):
        super().__init__()
        self.gfc=nn.Sequential(nn.Linear(gdim+idim,gdim),nn.BatchNorm1d(gdim),nn.ReLU(),nn.Dropout(dr))
        self.tr =nn.Sequential(nn.Linear(gdim,ldim),nn.ReLU(),nn.Dropout(dr))
        self.out=nn.Linear(ldim,nl)
    def forward(self,x,A):
        A=self.gfc(torch.cat([A,x],1)); return A, torch.sigmoid(self.out(self.tr(A)))

class HMCNF(nn.Module):
    def __init__(self,idim,nf,nm,gd,ld,dr,beta):
        super().__init__()
        self.beta=beta
        self.proj=nn.Sequential(nn.Linear(idim,gd),nn.BatchNorm1d(gd),nn.ReLU(),nn.Dropout(dr))
        self.l1=Block(idim,gd,ld,nf,dr); self.l2=Block(idim,gd,ld,nm,dr)
        self.go=nn.Linear(gd,nf+nm)
    def forward(self,x):
        A=self.proj(x); A,P1=self.l1(x,A); A,P2=self.l2(x,A)
        PG=torch.sigmoid(self.go(A))
        PF=self.beta*torch.cat([P1,P2],1)+(1-self.beta)*PG
        return PF,P1,P2,PG

def bce(P,Y,e=1e-7): P=torch.clamp(P,e,1-e); return -torch.mean(Y*torch.log(P)+(1-Y)*torch.log(1-P))
def vloss(P1,P2,pairs):
    L=torch.tensor(0.,device=P1.device)
    for fi,mi in pairs: L=L+torch.mean(torch.clamp(P1[:,fi]-P2[:,mi],min=0)**2)
    return L/max(len(pairs),1)
def loss_fn(PF,P1,P2,PG,Y1,Y2,pairs,lv):
    return bce(P1,Y1)+bce(P2,Y2)+bce(PG,torch.cat([Y1,Y2],1))+lv*vloss(P1,P2,pairs)

def find_thr(preds,targets):
    cands=np.linspace(0.05,0.95,19); thr=np.full(targets.shape[1],0.5)
    for i in range(targets.shape[1]):
        if targets[:,i].sum()==0: continue
        best=0.
        for tau in cands:
            f=f1_score(targets[:,i],(preds[:,i]>=tau).astype(int),zero_division=0)
            if f>best: best=f; thr[i]=tau
    return thr

@torch.no_grad()
def collect(model,loader,device):
    model.eval(); fp,ft,mp,mt=[],[],[],[]
    for Xb,Y1b,Y2b in loader:
        _,P1,P2,_=model(Xb.to(device))
        fp.append(P1.cpu().numpy()); ft.append(Y1b.numpy())
        mp.append(P2.cpu().numpy()); mt.append(Y2b.numpy())
    return np.vstack(fp),np.vstack(ft),np.vstack(mp),np.vstack(mt)

def mauc(p,t): return np.mean([roc_auc_score(t[:,i],p[:,i]) for i in range(t.shape[1]) if t[:,i].sum()>0])
def mf1(p,t,thr): return f1_score(t,(p>=thr[np.newaxis,:]).astype(int),average='macro',zero_division=0)

def trial(cfg,data,pairs,device,epochs=120,patience=20):
    Xtr,Y1tr,Y2tr,Xva,Y1va,Y2va,Xte,Y1te,Y2te=data[:9]
    def T(*a): return [torch.tensor(x,dtype=torch.float32) for x in a]
    trl=DataLoader(TensorDataset(*T(Xtr,Y1tr,Y2tr)),batch_size=32,shuffle=True)
    val=DataLoader(TensorDataset(*T(Xva,Y1va,Y2va)),batch_size=128)
    tel=DataLoader(TensorDataset(*T(Xte,Y1te,Y2te)),batch_size=128)

    m=HMCNF(Xtr.shape[1],Y1tr.shape[1],Y2tr.shape[1],
             cfg['gd'],cfg['ld'],cfg['dr'],0.5).to(device)
    opt=torch.optim.Adam(m.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=8,factor=0.5)

    best_auc,best_st,cnt=0.,None,0
    for ep in range(1,epochs+1):
        m.train()
        for Xb,Y1b,Y2b in trl:
            Xb,Y1b,Y2b=Xb.to(device),Y1b.to(device),Y2b.to(device)
            opt.zero_grad()
            PF,P1,P2,PG=m(Xb)
            loss_fn(PF,P1,P2,PG,Y1b,Y2b,pairs,cfg['lv']).backward()
            nn.utils.clip_grad_norm_(m.parameters(),1.)
            opt.step()
        fp_v,ft_v,mp_v,mt_v=collect(m,val,device)
        auc=mauc(mp_v,mt_v); sched.step(-auc)
        if auc>best_auc: best_auc=auc; best_st={k:v.cpu().clone() for k,v in m.state_dict().items()}; cnt=0
        else:
            cnt+=1
            if cnt>=patience: break

    m.load_state_dict(best_st); m.to(device)
    fp_v,ft_v,mp_v,mt_v=collect(m,val,device)
    fthr=find_thr(fp_v,ft_v); mthr=find_thr(mp_v,mt_v)
    fp_t,ft_t,mp_t,mt_t=collect(m,tel,device)
    return {'val_auc':best_auc,'meta_auc':mauc(mp_t,mt_t),
            'meta_f1':mf1(mp_t,mt_t,mthr),'fine_auc':mauc(fp_t,ft_t),
            'fine_f1':mf1(fp_t,ft_t,fthr),'ep':ep}

# Focused grid — 24 configs, ~10 min on CPU
GRID = {
    'gd': [64, 128],
    'ld': [32, 64],
    'dr': [0.5, 0.6, 0.7],
    'lr': [1e-3],
    'wd': [1e-4, 1e-3],
    'lv': [0.1],
}

data = load_and_split(csv='hmcn_dataset.csv')
pairs = build_pairs(data[9], data[10])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

keys=list(GRID.keys()); vals=list(GRID.values())
configs=[dict(zip(keys,c)) for c in product(*vals)]
print(f"Configs: {len(configs)}\n")
print(f"{'#':>3} {'gd':>4} {'ld':>4} {'dr':>4} {'wd':>6} | {'valAUC':>7} {'tstAUC':>7} {'metaF1':>7} {'fineAUC':>8}")
print('-'*60)

results=[]
best_auc=0.; best_cfg=None; best_m=None
for i,cfg in enumerate(configs,1):
    r=trial(cfg,data,pairs,device)
    results.append({**cfg,**r})
    mark=' ◄' if r['val_auc']>best_auc else ''
    if r['val_auc']>best_auc: best_auc=r['val_auc']; best_cfg=cfg; best_m=r
    print(f"{i:3d} {cfg['gd']:4d} {cfg['ld']:4d} {cfg['dr']:4.1f} {cfg['wd']:6.0e} | "
          f"{r['val_auc']:7.4f} {r['meta_auc']:7.4f} {r['meta_f1']:7.4f} {r['fine_auc']:8.4f}{mark}")

pd.DataFrame(results).sort_values('val_auc',ascending=False).to_csv('hmcn_tuning_results.csv',index=False)
print(f"\nBEST: gd={best_cfg['gd']} ld={best_cfg['ld']} dr={best_cfg['dr']} wd={best_cfg['wd']}")
print(f"  val AUC={best_auc:.4f} | test meta AUC={best_m['meta_auc']:.4f} F1={best_m['meta_f1']:.4f} | fine AUC={best_m['fine_auc']:.4f} F1={best_m['fine_f1']:.4f}")
print("Saved → hmcn_tuning_results.csv")
