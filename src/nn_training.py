"""Training loop shared by the FT-Transformer and the LSTM."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LabelSmoothingBCE(nn.Module):
    def __init__(self, s=0.1):
        super().__init__()
        self.s = s

    def forward(self, logits, targets):
        return nn.functional.binary_cross_entropy_with_logits(logits, targets * (1 - self.s) + 0.5 * self.s)


class StockDataset(Dataset):
    def __init__(self, xr, xf, y=None):
        self.xr = torch.tensor(xr, dtype=torch.float32)
        self.xf = torch.tensor(xf, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.xr)

    def __getitem__(self, i):
        return (self.xr[i], self.xf[i], self.y[i]) if self.y is not None else (self.xr[i], self.xf[i])


def scale_fold(rtr, rval, ftr, fval, rte, fte):
    """Standardise returns and engineered inputs with statistics of the training fold only."""
    sc_r, sc_f = StandardScaler().fit(rtr), StandardScaler().fit(ftr)
    return (sc_r.transform(rtr), sc_r.transform(rval), sc_f.transform(ftr),
            sc_f.transform(fval), sc_r.transform(rte), sc_f.transform(fte))


def train_epoch(model, loader, opt, crit):
    model.train()
    total = 0.0
    for xr, xf, yb in loader:
        xr, xf, yb = xr.to(DEVICE), xf.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(xr, xf), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        preds.append(torch.sigmoid(model(batch[0].to(DEVICE), batch[1].to(DEVICE))).cpu().numpy())
        if len(batch) == 3:
            labels.append(batch[2].numpy())
    preds = np.concatenate(preds)
    acc = float(((preds > 0.5) == (np.concatenate(labels) > 0.5)).mean()) if labels else None
    return preds, acc


def train_fold(make_model, rtr, ftr, ytr, rval, fval, yval, rte, fte,
               lr, wd, bs, epochs, patience, smooth, workers=2):
    """Train with early stopping on the validation fold; return its accuracy and the predictions."""
    tl = DataLoader(StockDataset(rtr, ftr, ytr), batch_size=bs, shuffle=True, num_workers=workers,
                    pin_memory=DEVICE.type == "cuda")
    vl = DataLoader(StockDataset(rval, fval, yval), batch_size=bs * 2, shuffle=False, num_workers=workers)
    tel = DataLoader(StockDataset(rte, fte), batch_size=bs * 2, shuffle=False, num_workers=workers)

    model = make_model().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = LabelSmoothingBCE(smooth)

    best_acc, best_state, no_improve = 0.0, None, 0
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, tl, opt, crit)
        _, acc = predict(model, vl)
        sched.step()
        if acc > best_acc:
            best_acc, no_improve = acc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        print(f"    epoch {epoch:>2} | loss {loss:.4f} | val acc {acc:.4f}{' *' if no_improve == 0 else ''}", flush=True)
        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    oof_p, _ = predict(model, vl)
    test_p, _ = predict(model, tel)
    return best_acc, oof_p, test_p
