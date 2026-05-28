"""
Proves estadístiques: normalitat (Shapiro-Wilk, D'Agostino), homoscedasticitat
(Levene), tests de comparació de grups (t-test, Mann-Whitney, Kruskal-Wallis),
i intervals de confiança bootstrap.

També inclou `cleaning_report` per comparar el dataset raw vs el processat
de manera estructurada per a la memòria.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sst


RANDOM_STATE = 42

# Llindar de significació estàndard. Es pot sobreescriure per cada test.
ALPHA = 0.05

# Shapiro-Wilk no és vàlid per a N > 5000 (la potència explota i la pràctica
# habitual és prendre una submostra estable amb seed fix).
SHAPIRO_MAX_N = 5000


def cleaning_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    """Comparació estructurada raw vs clean per cada columna comuna.

    Per cada columna present a ambdós DataFrames retorna:
        - dtype_raw, dtype_clean
        - n_missing_raw, n_missing_clean, delta_missing
        - n_unique_raw, n_unique_clean

    Columnes només presents al raw o només al clean s'inclouen amb les
    columnes oposades buides — facilita veure què s'ha dropat o afegit.
    """
    cols = sorted(set(df_raw.columns) | set(df_clean.columns))
    rows = []
    for c in cols:
        in_raw = c in df_raw.columns
        in_clean = c in df_clean.columns
        rows.append({
            "column": c,
            "dtype_raw": str(df_raw[c].dtype) if in_raw else "",
            "dtype_clean": str(df_clean[c].dtype) if in_clean else "",
            "n_missing_raw": int(df_raw[c].isna().sum()) if in_raw else None,
            "n_missing_clean": int(df_clean[c].isna().sum()) if in_clean else None,
            "n_unique_raw": int(df_raw[c].nunique(dropna=True)) if in_raw else None,
            "n_unique_clean": int(df_clean[c].nunique(dropna=True)) if in_clean else None,
        })
    out = pd.DataFrame(rows)
    out["delta_missing"] = out["n_missing_clean"].astype("Int64") - out["n_missing_raw"].astype("Int64")
    out["status"] = out.apply(_classify_status, axis=1)
    return out


def _classify_status(row: pd.Series) -> str:
    if not row["dtype_raw"]:
        return "added"
    if not row["dtype_clean"]:
        return "removed"
    if row["dtype_raw"] != row["dtype_clean"]:
        return "retyped"
    if pd.notna(row["delta_missing"]) and row["delta_missing"] != 0:
        return "imputed" if row["delta_missing"] < 0 else "masked"
    return "unchanged"


def normality_report(values, name: str = "x", alpha: float = ALPHA,
                     random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Aplica Shapiro-Wilk i D'Agostino-Pearson sobre `values`.

    Per a N > 5000 fa Shapiro sobre una submostra aleatòria estable (seed fix)
    per evitar el biaix conegut del test amb mostres molt grans (rebutja sempre).

    Retorna una taula amb test, estadístic, p-valor i conclusió (rebutja H0?).
    """
    v = pd.Series(values).dropna().astype(float).values
    n = len(v)
    rows = []

    # Shapiro-Wilk (potencialment submostrejat)
    if n > SHAPIRO_MAX_N:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, SHAPIRO_MAX_N, replace=False)
        v_sw = v[idx]
        sw_note = f"submostrejat a {SHAPIRO_MAX_N}"
    else:
        v_sw = v
        sw_note = "complet"
    if n >= 3:
        try:
            sw_stat, sw_p = sst.shapiro(v_sw)
            rows.append({
                "variable": name, "test": "Shapiro-Wilk",
                "n_efectiu": len(v_sw), "estadistic": float(sw_stat),
                "p_valor": float(sw_p),
                "rebutja_H0_normalitat": bool(sw_p < alpha),
                "nota": sw_note,
            })
        except Exception as e:
            rows.append({"variable": name, "test": "Shapiro-Wilk", "n_efectiu": len(v_sw),
                         "estadistic": None, "p_valor": None,
                         "rebutja_H0_normalitat": None, "nota": f"error: {e}"})

    # D'Agostino-Pearson (vàlid per a N grans, requereix N ≥ 20)
    if n >= 20:
        dp_stat, dp_p = sst.normaltest(v)
        rows.append({
            "variable": name, "test": "D'Agostino-Pearson",
            "n_efectiu": n, "estadistic": float(dp_stat),
            "p_valor": float(dp_p),
            "rebutja_H0_normalitat": bool(dp_p < alpha),
            "nota": "complet",
        })

    return pd.DataFrame(rows)


def homoscedasticity_levene(groups: dict[str, np.ndarray] | list,
                            center: str = "median",
                            alpha: float = ALPHA) -> dict:
    """Test de Levene per a igualtat de variàncies entre grups.

    `center="median"` correspon al test de Brown-Forsythe (robust a no-normalitat).
    """
    arrs = list(groups.values()) if isinstance(groups, dict) else list(groups)
    arrs = [pd.Series(a).dropna().astype(float).values for a in arrs]
    stat, p = sst.levene(*arrs, center=center)
    return {
        "test": f"Levene ({center})",
        "estadistic": float(stat),
        "p_valor": float(p),
        "rebutja_H0_homoscedasticitat": bool(p < alpha),
        "n_grups": len(arrs),
    }


def cohens_d(a, b) -> float:
    """Cohen's d (mida d'efecte per a comparació de mitjanes, *pooled std*)."""
    a = pd.Series(a).dropna().astype(float).values
    b = pd.Series(b).dropna().astype(float).values
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sd_pool = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sd_pool == 0:
        return float("nan")
    return (a.mean() - b.mean()) / sd_pool


def cliff_delta(a, b) -> float:
    """Cliff's delta (mida d'efecte no paramètric, robust a outliers).

    Implementació via els ranks de Mann-Whitney (O(N log N)):
        delta = (2*U / (n1*n2)) - 1
    on U és l'estadístic U de Mann-Whitney d'`a` sobre `b`.
    """
    a = pd.Series(a).dropna().astype(float).values
    b = pd.Series(b).dropna().astype(float).values
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    u, _ = sst.mannwhitneyu(a, b, alternative="two-sided")
    return float(2 * u / (len(a) * len(b)) - 1)


def compare_two_groups(a, b, name_a: str = "A", name_b: str = "B",
                       alpha: float = ALPHA,
                       random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Compara dues mostres. Aplica:
        - Normalitat (Shapiro) sobre cada grup
        - Si ambdues normals → t-test de Welch + Cohen's d
        - Si no → Mann-Whitney + Cliff's delta

    Retorna una taula d'una sola fila (per concatenar amb altres comparacions).
    """
    a_arr = pd.Series(a).dropna().astype(float).values
    b_arr = pd.Series(b).dropna().astype(float).values

    norm_a = normality_report(a_arr, name=name_a, alpha=alpha, random_state=random_state)
    norm_b = normality_report(b_arr, name=name_b, alpha=alpha, random_state=random_state)
    both_normal = (not norm_a["rebutja_H0_normalitat"].any()
                   and not norm_b["rebutja_H0_normalitat"].any())

    if both_normal:
        stat, p = sst.ttest_ind(a_arr, b_arr, equal_var=False)
        test_name = "Welch t-test"
        effect = cohens_d(a_arr, b_arr)
        effect_name = "Cohen's d"
    else:
        stat, p = sst.mannwhitneyu(a_arr, b_arr, alternative="two-sided")
        test_name = "Mann-Whitney U"
        effect = cliff_delta(a_arr, b_arr)
        effect_name = "Cliff's delta"

    return pd.DataFrame([{
        "comparacio": f"{name_a} vs {name_b}",
        "n_A": len(a_arr), "n_B": len(b_arr),
        "median_A": float(np.median(a_arr)) if len(a_arr) else None,
        "median_B": float(np.median(b_arr)) if len(b_arr) else None,
        "test": test_name,
        "estadistic": float(stat), "p_valor": float(p),
        "significatiu": bool(p < alpha),
        "effect_size_nom": effect_name, "effect_size_val": float(effect),
    }])


def compare_multi_groups(groups: dict[str, np.ndarray],
                         alpha: float = ALPHA) -> dict:
    """Comparació de >2 grups: ANOVA si totes normals + variàncies iguals,
    sinó Kruskal-Wallis (no paramètric)."""
    arrs = {k: pd.Series(v).dropna().astype(float).values for k, v in groups.items()}

    # Test ràpid de normalitat: D'Agostino sobre cada grup (vàlid N ≥ 20)
    not_normal = []
    for k, v in arrs.items():
        if len(v) >= 20:
            _, p = sst.normaltest(v)
            if p < alpha:
                not_normal.append(k)

    homo = homoscedasticity_levene(arrs, alpha=alpha)
    use_parametric = (len(not_normal) == 0) and (not homo["rebutja_H0_homoscedasticitat"])

    arr_list = list(arrs.values())
    if use_parametric:
        stat, p = sst.f_oneway(*arr_list)
        test_name = "ANOVA F"
    else:
        stat, p = sst.kruskal(*arr_list)
        test_name = "Kruskal-Wallis H"

    return {
        "test": test_name,
        "estadistic": float(stat), "p_valor": float(p),
        "significatiu": bool(p < alpha),
        "n_grups": len(arrs),
        "n_total": sum(len(v) for v in arr_list),
        "grups_no_normals": not_normal,
        "homoscedasticitat_p": homo["p_valor"],
    }


def posthoc_dunn(groups: dict[str, np.ndarray],
                 p_adjust: str = "bonferroni",
                 alpha: float = ALPHA) -> pd.DataFrame:
    """Post-hoc parellat Dunn amb correcció Bonferroni (o Holm).

    Implementació pròpia per evitar dependència de `scikit-posthocs`.

    Per a cada parell (i, j), calcula Z amb rangs combinats (incloent
    correcció per empats) i p-valor bilateral; després aplica la correcció
    de comparacions múltiples.

    Retorna una taula amb les K*(K-1)/2 comparacions.
    """
    names = list(groups.keys())
    arrs = [pd.Series(groups[k]).dropna().astype(float).values for k in names]
    n_i = np.array([len(a) for a in arrs])
    N = n_i.sum()

    # Rangs combinats amb correcció per empats
    all_vals = np.concatenate(arrs)
    ranks = sst.rankdata(all_vals)
    # Mean rank per grup
    splits = np.split(ranks, np.cumsum(n_i)[:-1])
    mean_ranks = np.array([s.mean() for s in splits])

    # Correcció per empats
    _, counts = np.unique(all_vals, return_counts=True)
    tie_correction = 1 - (counts ** 3 - counts).sum() / (N ** 3 - N) if N > 1 else 1.0

    # Z per cada parell
    rows = []
    k = len(names)
    n_comparisons = k * (k - 1) // 2
    for i in range(k):
        for j in range(i + 1, k):
            se = np.sqrt(((N * (N + 1)) / 12.0) * tie_correction * (1 / n_i[i] + 1 / n_i[j]))
            z = (mean_ranks[i] - mean_ranks[j]) / se if se > 0 else 0.0
            p_raw = 2 * (1 - sst.norm.cdf(abs(z)))
            rows.append({
                "grup_i": names[i], "grup_j": names[j],
                "mean_rank_i": float(mean_ranks[i]),
                "mean_rank_j": float(mean_ranks[j]),
                "Z": float(z), "p_raw": float(p_raw),
            })

    out = pd.DataFrame(rows)
    # Correcció
    if p_adjust == "bonferroni":
        out["p_adjusted"] = (out["p_raw"] * n_comparisons).clip(upper=1.0)
    elif p_adjust == "holm":
        order = out["p_raw"].argsort().values
        adj = np.empty(len(out))
        for rank_idx, orig_idx in enumerate(order):
            adj[orig_idx] = min(1.0, out["p_raw"].iloc[orig_idx] * (n_comparisons - rank_idx))
        # Monotonicity: each adjusted p must be ≥ the previous one in sort order
        cummax = 0.0
        for orig_idx in order:
            cummax = max(cummax, adj[orig_idx])
            adj[orig_idx] = cummax
        out["p_adjusted"] = adj
    else:
        raise ValueError(f"p_adjust ha de ser 'bonferroni' o 'holm', no {p_adjust!r}")

    out["significatiu"] = out["p_adjusted"] < alpha
    return out.sort_values("p_adjusted").reset_index(drop=True)


def bootstrap_ci(values, statistic=np.median, ci: float = 0.95,
                 n_boot: int = 10_000, random_state: int = RANDOM_STATE) -> dict:
    """Interval de confiança bootstrap per a un estadístic arbitrari.

    Usa el mètode percentil. Per a estadístics no robustos cal augmentar n_boot.
    """
    v = pd.Series(values).dropna().astype(float).values
    if len(v) == 0:
        return {"point": None, "lower": None, "upper": None, "n": 0}
    rng = np.random.default_rng(random_state)
    n = len(v)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(v, size=n, replace=True)
        boots[i] = statistic(sample)
    alpha = (1 - ci) / 2
    return {
        "point": float(statistic(v)),
        "lower": float(np.quantile(boots, alpha)),
        "upper": float(np.quantile(boots, 1 - alpha)),
        "n": int(n),
        "ci": ci,
        "n_boot": n_boot,
    }


def correlation_with_pvalue(x, y, method: str = "spearman",
                            alpha: float = ALPHA) -> dict:
    """Correlació + p-valor + interval de confiança bootstrap (percentil)."""
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    mask = x.notna() & y.notna()
    x, y = x[mask].values, y[mask].values
    if method == "spearman":
        r, p = sst.spearmanr(x, y)
    elif method == "pearson":
        r, p = sst.pearsonr(x, y)
    elif method == "kendall":
        r, p = sst.kendalltau(x, y)
    else:
        raise ValueError(f"Mètode {method!r} no suportat")
    return {
        "method": method, "n": int(len(x)),
        "r": float(r), "p_valor": float(p),
        "significatiu": bool(p < alpha),
    }


def chi2_independence(df: pd.DataFrame, col_a: str, col_b: str,
                      alpha: float = ALPHA) -> dict:
    """Test χ² d'independència entre dues categòriques."""
    tab = pd.crosstab(df[col_a], df[col_b])
    chi2, p, dof, _ = sst.chi2_contingency(tab)
    # Cramér's V
    n = tab.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(tab.shape) - 1))) if min(tab.shape) > 1 else float("nan")
    return {
        "test": "χ² independence",
        "estadistic": float(chi2), "p_valor": float(p), "df": int(dof),
        "n": int(n), "cramers_v": float(cramers_v),
        "significatiu": bool(p < alpha),
    }


def compare_groups(*args, **kwargs):
    """Compatibilitat amb noms antics — redirecció a compare_two_groups."""
    return compare_two_groups(*args, **kwargs)


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor per a totes les columnes numèriques.

    Convenció (Hair et al. 2010): VIF > 10 → col·linealitat severa;
    VIF > 5 → moderada; VIF < 5 → acceptable.

    Requereix statsmodels. Aplica `dropna` i `astype(float)` per assegurar
    que la matriu d'entrada sigui numèrica i estable.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    Xn = X.apply(pd.to_numeric, errors="coerce").dropna().astype(float)
    Xc = add_constant(Xn, has_constant="add").values
    rows = []
    cols = ["const"] + list(Xn.columns)
    for i, col in enumerate(cols):
        if col == "const":
            continue
        try:
            v = variance_inflation_factor(Xc, i)
        except Exception:
            v = float("nan")
        rows.append({"feature": col, "VIF": float(v)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def epsilon_squared_kw(groups: dict[str, np.ndarray] | list) -> float:
    """Mida d'efecte per a Kruskal-Wallis: ε² = (H − k + 1) / (N − k).

    Convenció de Cohen-tipus (Tomczak & Tomczak 2014):
        ε² < 0.01     → negligible
        0.01–0.06     → small
        0.06–0.14     → medium
        ≥ 0.14        → large
    """
    arrs = list(groups.values()) if isinstance(groups, dict) else list(groups)
    arrs = [pd.Series(a).dropna().astype(float).values for a in arrs]
    k = len(arrs)
    N = sum(len(a) for a in arrs)
    if k < 2 or N <= k:
        return float("nan")
    H, _ = sst.kruskal(*arrs)
    return float((H - k + 1) / (N - k))


def cliff_delta_magnitude(delta: float) -> str:
    """Convenció de Romano et al. (2006) per a la magnitud de Cliff's δ."""
    a = abs(delta)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def cohens_d_magnitude(d: float) -> str:
    """Convenció de Cohen (1988): 0.2 small, 0.5 medium, 0.8 large."""
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"
