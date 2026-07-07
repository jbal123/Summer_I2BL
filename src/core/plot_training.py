
# --- path bootstrap (auto-added during repo reorg): make sibling modules importable ---
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "features", "day1_da_only", "single_analyte", "multi_analyte", "models"):
    _p = _os.path.join(_SRC_DIR, _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path bootstrap ---

import numpy as np
import matplotlib.pyplot as plt


def plot_training(history, save_path='training_curves.png', title=None, show=True):
    """
    Plot train (and optionally val) loss curves from a history dict.

    history keys:
        'epochs'     — list of epoch numbers at which loss was recorded
        'train_loss' — list of training MSE values
        'val_loss'   — list of validation MSE values (may be empty)

    Uses log scale on y-axis when loss spans more than one order of magnitude.
    """
    epochs     = history['epochs']
    train_loss = history['train_loss']
    val_loss   = history.get('val_loss', [])

    if not epochs:
        print("plot_training: no history to plot.")
        return

    has_val = len(val_loss) == len(epochs)

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(epochs, train_loss, color='steelblue', lw=1.5, label='Train')
    if has_val:
        ax.plot(epochs, val_loss, color='darkorange', lw=1.5, label='Val')

        # Mark minimum val loss
        best_idx  = int(np.argmin(val_loss))
        best_ep   = epochs[best_idx]
        best_loss = val_loss[best_idx]
        ax.axvline(best_ep, color='darkorange', lw=0.8, ls='--', alpha=0.6)
        ax.scatter([best_ep], [best_loss], color='darkorange', s=60, zorder=5,
                   label=f'Best val: {best_loss:.4f} @ epoch {best_ep}')

    # Log scale when range spans >1 order of magnitude
    all_losses = train_loss + (val_loss if has_val else [])
    lo, hi     = min(all_losses), max(all_losses)
    if hi > 0 and lo > 0 and hi / lo > 10:
        ax.set_yscale('log')
        ax.set_ylabel('MSE Loss (scaled, log)', fontsize=11)
    else:
        ax.set_ylabel('MSE Loss (scaled)', fontsize=11)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_title(title or 'Training Convergence', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate final losses in the top-right corner
    final_train = train_loss[-1]
    note = f'Final train: {final_train:.4f}'
    if has_val:
        note += f'\nFinal val:   {val_loss[-1]:.4f}'
    ax.text(0.98, 0.97, note, transform=ax.transAxes,
            ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curve saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)
