"""L_inf PGD targeted attack against OpenAI CLIP on a single CIFAR-10 image.

Perturbation is constrained in the [0, 1] pixel space (NOT the CLIP-normalized
space). CLIP's normalization is wrapped into the forward pass so the epsilon
budget has the usual semantics.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets
from CLIP import clip

# ---------------- config ----------------
DATASET = "CIFAR10"
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# PGD hyper-parameters (L_inf, pixel-space [0, 1])
EPS = 8 / 255         # perturbation budget
ALPHA = 2 / 255       # step size
STEPS = 30            # number of PGD iterations
RANDOM_START = True   # uniform init inside the L_inf ball
TARGETED = True
TARGET_LABEL = 1      # automobile
SAMPLE_INDEX = 1      # which test-set image to attack

SNAPSHOT_EVERY = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("result", exist_ok=True)

# ---------------- model ----------------
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

text_tokens = clip.tokenize(CLASS_NAMES).to(device)

# CLIP normalization constants — used to map between [0,1] and model input.
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                         device=device).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                        device=device).view(1, 3, 1, 1)


def normalize(x):
    return (x - CLIP_MEAN) / CLIP_STD


def denormalize(x):
    return x * CLIP_STD + CLIP_MEAN


# ---------------- data ----------------
if DATASET == "CIFAR10":
    test_data = datasets.CIFAR10(root="./data", train=False,
                                 transform=preprocess, download=True)
else:
    raise ValueError(f"Unsupported dataset: {DATASET}")

image_normed, true_label = test_data[SAMPLE_INDEX]
image_normed = image_normed.unsqueeze(0).to(device)
# Recover the image in [0, 1] pixel space.
x_clean = denormalize(image_normed).clamp(0.0, 1.0)

print(f"original label : {CLASS_NAMES[true_label]}")
print(f"target label   : {CLASS_NAMES[TARGET_LABEL]}  (targeted={TARGETED})")


# ---------------- PGD ----------------
def pgd_linf(x_clean, true_label, target_label,
             eps=EPS, alpha=ALPHA, steps=STEPS,
             random_start=RANDOM_START, targeted=TARGETED,
             snapshot_every=SNAPSHOT_EVERY):
    """L_inf PGD in pixel space. Returns (x_adv, snapshots)."""
    label_for_loss = torch.tensor(
        [target_label if targeted else true_label],
        device=device, dtype=torch.long,
    )

    if random_start:
        delta = torch.empty_like(x_clean).uniform_(-eps, eps)
        delta = ((x_clean + delta).clamp(0.0, 1.0) - x_clean).detach()
    else:
        delta = torch.zeros_like(x_clean)

    snapshots = []  # (step, x_adv_np, loss, p_true, p_target)

    for step in range(steps):
        delta.requires_grad_(True)
        x_adv = x_clean + delta
        logits_per_image, _ = model(normalize(x_adv), text_tokens)
        loss = F.cross_entropy(logits_per_image.float(), label_for_loss)
        grad = torch.autograd.grad(loss, delta)[0]

        with torch.no_grad():
            # targeted  : minimize CE w.r.t. target  -> step along -sign(grad)
            # untargeted: maximize CE w.r.t. true    -> step along +sign(grad)
            step_dir = -grad.sign() if targeted else grad.sign()
            delta = delta + alpha * step_dir
            delta = delta.clamp(-eps, eps)
            delta = ((x_clean + delta).clamp(0.0, 1.0) - x_clean).detach()

        if step % snapshot_every == 0 or step == steps - 1:
            with torch.no_grad():
                probs = logits_per_image.softmax(dim=-1)[0]
                p_true = probs[true_label].item()
                p_tgt = probs[target_label].item()
            x_adv_np = (x_clean + delta).detach().cpu().numpy()
            snapshots.append((step, x_adv_np, loss.item(), p_true, p_tgt))
            print(f"step={step:3d}  loss={loss.item():.4f}  "
                  f"p[true={CLASS_NAMES[true_label]}]={p_true:.4f}  "
                  f"p[target={CLASS_NAMES[target_label]}]={p_tgt:.4f}")

    x_adv = (x_clean + delta).detach()
    return x_adv, snapshots


# ---------------- plotting helpers ----------------
def make_plot_from_preds(ax, orig_preds, mod_preds, class_labels,
                         colors=("#1f4e79", "#c0392b")):
    width = 0.42
    for i in range(orig_preds.shape[0]):
        v_orig, v_mod = orig_preds[i], mod_preds[i]
        label1 = label2 = ""
        alpha_orig = alpha_mod = 0.55
        if np.argmax(mod_preds) == i:
            alpha_mod = 1.0
            label2 = "adversarial"
        if np.argmax(orig_preds) == i:
            alpha_orig = 1.0
            label1 = "original"
        ax.fill_between([0, v_mod], [i, i], [i + width, i + width],
                        color=colors[1], label=label2, alpha=alpha_mod)
        ax.fill_between([0, v_orig], [i - width, i - width], [i, i],
                        color=colors[0], label=label1, alpha=alpha_orig)
    ax.set_yticks(range(len(class_labels)))
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("probability")
    ax.set_xlim([0, 1.02])
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)


def predict_probs(x):
    with torch.no_grad():
        logits, _ = model(normalize(x), text_tokens)
    return logits.softmax(dim=-1).cpu().numpy()[0]


# ---------------- run attack ----------------
x_adv, snapshots = pgd_linf(x_clean, true_label, TARGET_LABEL)
x_clean_np = x_clean.detach().cpu().numpy()
x_adv_np = x_adv.detach().cpu().numpy()

orig_preds = predict_probs(x_clean)
adv_preds = predict_probs(x_adv)

# ---------------- per-step visualization (2x2) ----------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

for step, snap_np, loss_val, _, _ in snapshots:
    snap_tensor = torch.from_numpy(snap_np).to(device).to(x_clean.dtype)
    step_preds = predict_probs(snap_tensor)

    delta_np = snap_np[0] - x_clean_np[0]
    # δ ∈ [-ε, ε]  →  [0, 1] per channel, preserving sign (0.5 == no change)
    delta_rgb = np.clip(delta_np / (2 * EPS) + 0.5, 0.0, 1.0).transpose(1, 2, 0)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.5),
                             gridspec_kw={"hspace": 0.28, "wspace": 0.18})
    fig.suptitle(f"PGD step {step}     loss = {loss_val:.4f}     "
                 f"ε = {EPS:.4f}  α = {ALPHA:.4f}",
                 fontsize=13, y=0.995)

    axes[0, 0].imshow(x_clean_np[0].clip(0, 1).transpose(1, 2, 0))
    axes[0, 0].set_title(f"original  —  pred: {CLASS_NAMES[int(np.argmax(orig_preds))]}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(snap_np[0].clip(0, 1).transpose(1, 2, 0))
    axes[0, 1].set_title(f"adversarial  —  pred: {CLASS_NAMES[int(np.argmax(step_preds))]}")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(delta_rgb)
    axes[1, 0].set_title(f"adversarial − original  (rescaled, [−ε, ε] → [0, 1])")
    axes[1, 0].axis("off")

    make_plot_from_preds(axes[1, 1], orig_preds, step_preds, CLASS_NAMES)
    axes[1, 1].set_title("class probability")

    fig.savefig(f"result/adv_{step}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------- final summary plot ----------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                         gridspec_kw={"width_ratios": [1, 1, 1.4]})
axes[0].imshow(x_clean_np[0].clip(0, 1).transpose(1, 2, 0))
axes[0].set_title("original\n" + CLASS_NAMES[int(np.argmax(orig_preds))], fontsize=14)
axes[0].axis("off")

axes[1].imshow(x_adv_np[0].clip(0, 1).transpose(1, 2, 0))
axes[1].set_title("adversarial\n" + CLASS_NAMES[int(np.argmax(adv_preds))], fontsize=14)
axes[1].axis("off")

make_plot_from_preds(axes[2], orig_preds, adv_preds, CLASS_NAMES)
axes[2].set_title("class probability")

fig.tight_layout()
fig.savefig("result/result.png", dpi=130, bbox_inches="tight")
plt.close(fig)

linf = float(np.abs(x_adv_np - x_clean_np).max())
print()
print(f"L_inf perturbation : {linf:.6f}  (budget {EPS:.6f})")
print(f"clean prediction   : {CLASS_NAMES[int(np.argmax(orig_preds))]}")
print(f"adv   prediction   : {CLASS_NAMES[int(np.argmax(adv_preds))]}")
