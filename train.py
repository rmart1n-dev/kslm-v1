import math
import time
import json
import signal
import sys
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from transformers import AutoTokenizer
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.set_float32_matmul_precision("high")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.bfloat16

@dataclass
class ModelConfig:
    vocab_size:  int   = 50_257
    n_positions: int   = 1024
    n_embd:      int   = 768
    n_layer:     int   = 12
    n_head:      int   = 12
    mlp_ratio:   float = 4.0
    dropout:     float = 0.0
    use_bias:    bool  = True

@dataclass
class TrainConfig:
    data_file:      str  = "data/anonymized_train.txt"
    tokenizer_path: str  = "tokenizer"
    output_dir:     str  = "kllm_final"

    seq_len:    int = 1024
    batch_size: int = 32
    grad_accum: int = 2

    num_epochs:    int   = 3
    learning_rate: float = 3e-4
    warmup_steps:  int   = 500
    weight_decay:  float = 0.01
    grad_clip:     float = 1.0

    num_workers: int  = 4
    pin_memory:  bool = True

    log_every:  int = 50
    save_every: int = 2_000
    seed:       int = 42

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head   = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn   = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.use_bias)
        self.c_proj   = nn.Linear(cfg.n_embd, cfg.n_embd,     bias=cfg.use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=-1)
        def reshape(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)

class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden      = int(cfg.n_embd * cfg.mlp_ratio)
        self.c_fc   = nn.Linear(cfg.n_embd, hidden,     bias=cfg.use_bias)
        self.c_proj = nn.Linear(hidden,     cfg.n_embd, bias=cfg.use_bias)
        self.act    = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp  = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class KSLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg     = cfg
        self.wte     = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe     = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.blocks  = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f    = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight   

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x    = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

class TokenDataset(Dataset):
    def __init__(self, data_file: str, tokenizer, seq_len: int, seed: int):
        print("Tokenising corpus (streaming)...")
        all_ids: list = []
        chunk_size = 50_000

        def flush(buf):
            enc = tokenizer(buf, add_special_tokens=True)["input_ids"]
            for ids in enc:
                clean = [int(i) for i in ids if i is not None]
                if clean:
                    all_ids.extend(clean)
                    if tokenizer.eos_token_id is not None:
                        all_ids.append(int(tokenizer.eos_token_id))

        with open(data_file, "r", encoding="utf-8", errors="replace") as f:
            buf = []
            for line in f:
                stripped = line.rstrip("\n")
                if stripped.strip():
                    buf.append(stripped)
                if len(buf) >= chunk_size:
                    flush(buf); buf = []
            if buf:
                flush(buf)

        tokens = np.array(all_ids, dtype=np.int32)
        print(f"Total tokens: {len(tokens):,}")

        stride = seq_len
        n      = (len(tokens) - 1) // stride
        self.chunks = [
            tokens[i * stride : i * stride + seq_len + 1]
            for i in range(n)
            if i * stride + seq_len + 1 <= len(tokens)
        ]
        np.random.default_rng(seed).shuffle(self.chunks)
        print(f"Packed into {len(self.chunks):,} chunks of {seq_len + 1} tokens.")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return torch.from_numpy(self.chunks[idx].astype(np.int64))

def lr_schedule(step: int, cfg: TrainConfig, total_steps: int) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(total_steps - cfg.warmup_steps, 1)
    return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

METRICS_FILE = "metrics.jsonl"

class MetricsLogger:
    def __init__(self, output_dir: str):
        self.path    = Path(output_dir) / METRICS_FILE
        self.records: list = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        self.records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass   

            if self.records:
                print(f"  Loaded {len(self.records)} prior metric records from {self.path}")

    def log(self, record: dict):
        self.records.append(record)
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def all(self) -> list:
        return self.records

RESUME_FILE = "training_state.json"

def _verify_checkpoint(ckpt_dir: Path) -> bool:
    for fname in ("weights.pt", "optimizer.pt", "model_config.json"):
        p = ckpt_dir / fname
        if not p.exists():
            print(f"  Checkpoint verify failed: {p} missing")
            return False
        if p.stat().st_size == 0:
            print(f"  Checkpoint verify failed: {p} is empty")
            return False
    return True

def save_checkpoint(
    model:        KSLM,
    optimizer:    torch.optim.Optimizer,
    scaler:       GradScaler,
    step:         int,
    epoch:        int,
    batch_offset: int,
    total_steps:  int,
    train_cfg:    TrainConfig,
    model_cfg:    ModelConfig,
    prev_tag:     Optional[str] = None,
    is_pause:     bool = False,
) -> str:
    tag     = f"step_{step:07d}"
    out_dir = Path(train_cfg.output_dir)
    final   = out_dir / tag
    tmp     = out_dir / f"{tag}.tmp"

    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    try:
        raw = model
        torch.save(raw.state_dict(),       tmp / "weights.pt")
        torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
        torch.save(cast(GradScaler, scaler).state_dict(), tmp / "scaler.pt")
        (tmp / "model_config.json").write_text(json.dumps(asdict(model_cfg)))
    except Exception as e:
        print(f"  Checkpoint write error: {e} — keeping previous checkpoint.")
        shutil.rmtree(tmp, ignore_errors=True)
        return prev_tag or ""

    if not _verify_checkpoint(tmp):
        print("  Checkpoint failed verification — keeping previous checkpoint.")
        shutil.rmtree(tmp, ignore_errors=True)
        return prev_tag or ""

    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)

    state = {
        "checkpoint_tag":          tag,
        "previous_checkpoint_tag": prev_tag,
        "global_step":             step,
        "epoch":                   epoch,
        "batch_offset":            batch_offset,
        "total_steps":             total_steps,
        "is_pause":                is_pause,
        "saved_at":                time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    state_path = out_dir / RESUME_FILE
    tmp_json   = state_path.with_suffix(".tmp")
    tmp_json.write_text(json.dumps(state, indent=2))
    tmp_json.replace(state_path)

    if prev_tag and prev_tag != tag:
        old = out_dir / prev_tag
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)

    label = "Pause checkpoint" if is_pause else "Checkpoint"
    print(f"  {label} saved -> {final}/  (epoch {epoch}, step {step})")
    return tag

def load_checkpoint(
    model:     KSLM,
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
    train_cfg: TrainConfig,
):
    state_path = Path(train_cfg.output_dir) / RESUME_FILE
    if not state_path.exists():
        return 0, 1, 0, None, None

    try:
        state = json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"  Could not read {state_path}: {e}  - starting fresh.")
        return 0, 1, 0, None, None

    tag  = state["checkpoint_tag"]
    ckpt = Path(train_cfg.output_dir) / tag

    if not ckpt.exists():
        print(f"  Checkpoint dir {ckpt} missing - starting fresh.")
        return 0, 1, 0, None, None

    if not _verify_checkpoint(ckpt):
        print(f"  Checkpoint {tag} failed verification - starting fresh.")
        return 0, 1, 0, None, None

    raw = model
    raw.load_state_dict(
        torch.load(str(ckpt / "weights.pt"),   map_location=DEVICE, weights_only=True)
    )
    optimizer.load_state_dict(
        torch.load(str(ckpt / "optimizer.pt"), map_location="cpu",  weights_only=True)
    )
    scaler_path = ckpt / "scaler.pt"
    if scaler_path.exists():
        cast(GradScaler, scaler).load_state_dict(
            torch.load(str(scaler_path), weights_only=True)
        )

    print(f"  Resumed from {ckpt}/  "
          f"(epoch {state['epoch']}, step {state['global_step']})")
    return (
        state["global_step"],
        state["epoch"],
        state["batch_offset"],
        state.get("total_steps"),
        tag,
    )

def save_final(model: KSLM, tokenizer, model_cfg: ModelConfig, train_cfg: TrainConfig):
    out = Path(train_cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = model
    torch.save(raw.state_dict(), str(out / "weights.pt"))
    (out / "model_config.json").write_text(json.dumps(asdict(model_cfg)))
    tokenizer.save_pretrained(str(out))
    print(f"\nFinal model saved to {out}/")

def _ema(values: list, weight: float = 0.92) -> list:
    out, last = [], None
    for v in values:
        last = v if last is None else last * weight + v * (1.0 - weight)
        out.append(last)
    return out

def plot_training_report(metrics: list, output_dir: str, total_steps: int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("matplotlib not installed - skipping plots.")
        return

    if not metrics:
        print("No metric records - skipping plots.")
        return

    steps      = [r["step"]          for r in metrics]
    losses     = [r["loss"]          for r in metrics]
    ppls       = [r["ppl"]           for r in metrics]
    lrs        = [r["lr"]            for r in metrics]
    gnorms     = [r["gnorm"]         for r in metrics]
    tok_secs   = [r.get("tok_sec", 0) for r in metrics]
    epoch_seq  = [r["epoch"]         for r in metrics]

    smooth_loss  = _ema(losses)
    smooth_ppl   = _ema(ppls)
    smooth_gnorm = _ema(gnorms)

    epoch_data: dict = {}
    for r in metrics:
        epoch_data.setdefault(r["epoch"], []).append(r["loss"])
    epoch_ids   = sorted(epoch_data)
    epoch_means = [float(np.mean(epoch_data[e])) for e in epoch_ids]

    BG     = "#0d1117"
    PANEL  = "#161b22"
    GRID   = "#21262d"
    TEXT   = "#c9d1d9"
    C1     = "#7c3aed"   

    C2     = "#0ea5e9"   

    C3     = "#ec4899"   

    C4     = "#10b981"   

    C5     = "#f59e0b"   

    BARS   = [C1, C2, C3, C4, C5, "#6366f1"]

    plt.rcParams.update({
        "figure.facecolor": BG,   "axes.facecolor":  PANEL,
        "axes.edgecolor":   GRID, "axes.labelcolor": TEXT,
        "axes.titlecolor":  TEXT, "xtick.color":     TEXT,
        "ytick.color":      TEXT, "text.color":      TEXT,
        "grid.color":       GRID, "grid.linewidth":  0.5,
        "font.family":      "monospace",
        "font.size":        8.5,  "axes.titlesize":  10,
        "axes.titleweight": "bold","axes.labelsize":  8,
        "lines.linewidth":  1.8,
    })

    fig = plt.figure(figsize=(20, 11), dpi=150)
    fig.patch.set_facecolor(BG)
    fig.suptitle("KSLM — Training Report", fontsize=16, fontweight="bold",
                 color=TEXT, y=0.99)

    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32,
                  left=0.06, right=0.97, top=0.93, bottom=0.09)

    def make_ax(row, col, title, xlabel="Step", ylabel=""):
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(title, pad=8)
        ax.set_xlabel(xlabel, labelpad=4)
        if ylabel:
            ax.set_ylabel(ylabel, labelpad=4)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        return ax

    ax1 = make_ax(0, 0, "Training Loss", ylabel="Cross-Entropy")
    ax1.plot(steps, losses,      color=C1, alpha=0.2, lw=0.9, label="raw")
    ax1.plot(steps, smooth_loss, color=C1, lw=2.2,            label="EMA")

    prev_ep = None
    for s, ep in zip(steps, epoch_seq):
        if ep != prev_ep and prev_ep is not None:
            ax1.axvline(s, color=C3, lw=0.8, ls=":", alpha=0.8)
        prev_ep = ep
    ax1.legend(framealpha=0, fontsize=8, loc="upper right")
    if losses:
        ax1.set_ylim(bottom=max(0.0, min(losses) * 0.85))

    ax2 = make_ax(0, 1, "Perplexity", ylabel="PPL")
    ax2.plot(steps, ppls,       color=C2, alpha=0.2, lw=0.9, label="raw")
    ax2.plot(steps, smooth_ppl, color=C2, lw=2.2,            label="EMA")
    ax2.legend(framealpha=0, fontsize=8, loc="upper right")
    if ppls and max(ppls) / max(min(ppls), 1e-9) > 15:
        ax2.set_yscale("log")
        ax2.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax2.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax3 = make_ax(0, 2, "Learning-Rate Schedule", ylabel="LR")
    ax3.plot(steps, lrs, color=C3, lw=2.0)
    ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2e"))
    if lrs and steps:
        peak = max(lrs)
        warmup_end_s = next(
            (s for s, lr in zip(steps, lrs) if lr >= peak * 0.999), None
        )
        if warmup_end_s and warmup_end_s > steps[0]:
            ax3.axvspan(steps[0], warmup_end_s, alpha=0.13, color=C3, label="warmup")
            ax3.legend(framealpha=0, fontsize=8)

    ax4 = make_ax(1, 0, "Gradient Norm", ylabel="||g||")
    ax4.plot(steps, gnorms,        color=C4, alpha=0.2, lw=0.9, label="raw")
    ax4.plot(steps, smooth_gnorm,  color=C4, lw=2.2,            label="EMA")
    ax4.legend(framealpha=0, fontsize=8, loc="upper right")

    ax5 = make_ax(1, 1, "Throughput", ylabel="Tokens / sec")
    valid = [(s, t) for s, t in zip(steps, tok_secs) if t > 0]
    if valid:
        vs, vt = zip(*valid)
        ax5.plot(vs, vt,               color=C5, alpha=0.2, lw=0.9, label="raw")
        ax5.plot(vs, _ema(list(vt), 0.85), color=C5, lw=2.2,       label="EMA")
        ax5.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda v, _: f"{v/1e3:.0f}k" if v >= 1000 else f"{int(v)}"
            )
        )
        ax5.legend(framealpha=0, fontsize=8)
    else:
        ax5.text(0.5, 0.5, "No throughput data recorded",
                 transform=ax5.transAxes, ha="center", va="center",
                 color=TEXT, alpha=0.4, fontsize=9)

    ax6 = make_ax(1, 2, "Per-Epoch Mean Loss", xlabel="Epoch", ylabel="Mean Loss")
    x_labels = [str(e) for e in epoch_ids]
    bar_cols  = [BARS[i % len(BARS)] for i in range(len(epoch_ids))]
    bars = ax6.bar(x_labels, epoch_means, color=bar_cols,
                   width=0.52, edgecolor=BG, linewidth=1.0)
    y_pad = max(epoch_means) * 0.015 if epoch_means else 0.01
    for bar, val in zip(bars, epoch_means):
        ax6.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_pad,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=8, color=TEXT,
        )
    ax6.set_ylim(0, max(epoch_means) * 1.18 if epoch_means else 1)

    if metrics:
        parts = [
            f"Total steps: {steps[-1]:,}",
            f"Final loss (EMA): {smooth_loss[-1]:.4f}",
            f"Best loss: {min(losses):.4f}",
            f"Final PPL (EMA): {smooth_ppl[-1]:.2f}",
        ]
        if valid:
            avg_thr = float(np.mean([t for _, t in valid[-20:]]))
            parts.append(f"Avg throughput: {avg_thr/1e3:.1f}k tok/s")
        fig.text(
            0.5, 0.002, "   |   ".join(parts),
            ha="center", va="bottom", fontsize=7.5,
            color=TEXT, alpha=0.65, fontfamily="monospace",
        )

    out_path = Path(output_dir) / "training_report.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Training report saved -> {out_path}")

_PAUSE_REQUESTED = False

def _handle_sigint(sig, frame):
    global _PAUSE_REQUESTED
    if not _PAUSE_REQUESTED:
        print("\n\nPause requested - finishing current step then saving...")
        _PAUSE_REQUESTED = True
    else:
        print("\nForce-quit (no checkpoint written).")
        sys.exit(1)

signal.signal(signal.SIGINT, _handle_sigint)

def train():
    global _PAUSE_REQUESTED

    tcfg = TrainConfig()
    mcfg = ModelConfig()

    torch.manual_seed(tcfg.seed)
    Path(tcfg.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tcfg.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    mcfg.vocab_size = len(tokenizer)

    dataset = TokenDataset(tcfg.data_file, tokenizer, tcfg.seq_len, tcfg.seed)
    loader  = DataLoader(
        dataset,
        batch_size         = tcfg.batch_size,
        shuffle            = True,
        num_workers        = tcfg.num_workers,
        pin_memory         = tcfg.pin_memory,
        drop_last          = True,
        persistent_workers = (tcfg.num_workers > 0),
    )

    raw_model: KSLM = KSLM(mcfg).to(DEVICE).to(DTYPE)
    print(f"Model parameters : {raw_model.num_params() / 1e6:.1f} M  |  "
          f"device : {DEVICE}  |  dtype : {DTYPE}")

    print("Compiling model with torch.compile...")
    model = torch.compile(raw_model, mode="reduce-overhead")

    steps_per_epoch = len(loader)
    total_steps     = steps_per_epoch * tcfg.num_epochs

    decay_params    = [p for n, p in raw_model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in raw_model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": tcfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=tcfg.learning_rate, betas=(0.9, 0.95), fused=True,
    )

    scaler = GradScaler(enabled=(DTYPE == torch.float16))

    metrics_logger = MetricsLogger(tcfg.output_dir)

    global_step, start_epoch, start_batch_offset, saved_total, current_ckpt_tag = \
        load_checkpoint(raw_model, optimizer, scaler, tcfg)

    if saved_total is not None:
        total_steps = saved_total

    print(f"\nTraining for {tcfg.num_epochs} epochs "
          f"({total_steps:,} total optimiser steps).")
    if global_step > 0:
        print(f"Resuming from step {global_step} / epoch {start_epoch} "
              f"(batch offset {start_batch_offset}).\n")

    tokens_per_step = tcfg.batch_size * tcfg.seq_len * tcfg.grad_accum

    for epoch in range(start_epoch, tcfg.num_epochs + 1):
        model.train()
        pbar = tqdm(loader, total=steps_per_epoch,
                    desc=f"Epoch {epoch}/{tcfg.num_epochs}")

        accum_loss   = 0.0
        micro_step   = 0
        batch_cursor = 0
        skip               = start_batch_offset if epoch == start_epoch else 0
        start_batch_offset = 0

        step_t0 = time.perf_counter()

        for batch in pbar:
            if skip > 0:
                skip -= 1
                batch_cursor += 1
                continue

            batch   = batch.to(DEVICE, non_blocking=True)
            inputs  = batch[:, :-1]
            targets = batch[:, 1:]

            with autocast(device_type="cuda", dtype=DTYPE):
                logits = model(inputs)
                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits.view(B * T, V),
                    targets.reshape(B * T),
                ) / tcfg.grad_accum

            scaler.scale(loss).backward()

            accum_loss   += loss.item()
            micro_step   += 1
            batch_cursor += 1

            if micro_step % tcfg.grad_accum == 0:
                scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), tcfg.grad_clip
                ).item()

                new_lr = lr_schedule(global_step, tcfg, total_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = new_lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                avg_loss   = accum_loss
                accum_loss = 0.0
                global_step += 1

                step_dt = time.perf_counter() - step_t0
                tok_sec = tokens_per_step / max(step_dt, 1e-9)
                step_t0 = time.perf_counter()

                if global_step % tcfg.log_every == 0:
                    ppl = math.exp(min(avg_loss, 20))
                    pbar.set_postfix(
                        loss  = f"{avg_loss:.4f}",
                        ppl   = f"{ppl:.1f}",
                        lr    = f"{new_lr:.2e}",
                        gnorm = f"{gnorm:.2f}",
                        tok_s = f"{tok_sec/1e3:.1f}k",
                        step  = global_step,
                    )
                    metrics_logger.log({
                        "step":    global_step,
                        "epoch":   epoch,
                        "loss":    avg_loss,
                        "ppl":     ppl,
                        "lr":      new_lr,
                        "gnorm":   gnorm,
                        "tok_sec": tok_sec,
                        "ts":      time.strftime("%Y-%m-%d %H:%M:%S"),
                    })

                if global_step % tcfg.save_every == 0:
                    current_ckpt_tag = save_checkpoint(
                        raw_model, optimizer, scaler,
                        step=global_step, epoch=epoch,
                        batch_offset=batch_cursor,
                        total_steps=total_steps,
                        train_cfg=tcfg, model_cfg=mcfg,
                        prev_tag=current_ckpt_tag,
                    )

                if _PAUSE_REQUESTED:
                    current_ckpt_tag = save_checkpoint(
                        raw_model, optimizer, scaler,
                        step=global_step, epoch=epoch,
                        batch_offset=batch_cursor,
                        total_steps=total_steps,
                        train_cfg=tcfg, model_cfg=mcfg,
                        prev_tag=current_ckpt_tag,
                        is_pause=True,
                    )
                    print("\nTraining paused. Re-run this script to resume.")
                    sys.exit(0)

    save_final(raw_model, tokenizer, mcfg, tcfg)
    print(f"\nTraining complete - {global_step:,} optimiser steps total.")

    plot_training_report(
        metrics_logger.all(),
        output_dir  = tcfg.output_dir,
        total_steps = global_step,
    )

if __name__ == "__main__":
    train()
