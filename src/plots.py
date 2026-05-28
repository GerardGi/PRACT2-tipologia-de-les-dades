"""
Helpers per a figures: estil consistent + desar sempre PNG + PDF a DPI=200
amb títol, llegenda i unitats.

Tots els notebooks han de fer servir `save_fig` per garantir la convenció.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


FIGURES_DIR = Path("outputs/figures")
DPI = 200


def save_fig(fig: plt.Figure, name: str, figures_dir: Path = FIGURES_DIR,
             dpi: int = DPI) -> tuple[Path, Path]:
    """Desa la figura a `figures_dir/{name}.png` i `figures_dir/{name}.pdf`.

    Retorna els dos paths. Crea la carpeta si no existeix. `name` sense extensió.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / f"{name}.png"
    pdf = figures_dir / f"{name}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, dpi=dpi, bbox_inches="tight")
    return png, pdf
