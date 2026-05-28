# Pràctica 2 — Tipologia i cicle de vida de les dades

**Autors:** Gerard Ginés Pérez · Guillem Maluenda
**Assignatura:** Tipologia i cicle de vida de les dades — UOC, MSc Data Science
**Curs:** 2025-2026

## Descripció

Aquest repositori conté el cicle complet de tractament del dataset
generat a la [Pràctica 1](https://github.com/GerardGi/PRACT1-tipologia-de-les-dades)
(scraping de pisos.com a Catalunya, 10.864 anuncis × 34 columnes):

1. **EDA** — distribucions, missings, anomalies, correlacions.
2. **Neteja i feature engineering** — recuperació de coords, imputació,
   província per *spatial join*, log-transforms, `dist_bcn_km`, etc.
3. **Estadística inferencial** — normalitat, Levene, Kruskal-Wallis + Dunn,
   bootstrap, Mann-Whitney, χ², correlació Spearman, VIF.
4. **Modelat supervisat** — regressió sobre `log(price_per_m2)` (OLS,
   Ridge/Lasso, XGBoost amb Optuna i validació espacial *leave-one-province-out*).
5. **Modelat no supervisat** — K-means amb selecció de *k* per colze + silueta.

**Variable objectiu del modelat:** `log_price_per_m2`.

## Estructura del repositori

```
pract2-tipologia/
├── data/
│   ├── raw/
│   │   ├── pisos_catalunya_20260405.csv     # CSV original P1 (immutable)
│   │   └── geo/
│   │       └── gadm41_ESP_2.json.zip        # GADM ESP nivell 2 (províncies)
│   └── processed/
│       ├── pisos_clean.parquet              # Sortida FASE 2 (43 cols)
│       └── pisos_clean.csv                  # Equivalent en CSV
├── notebooks/
│   ├── 01_eda.ipynb                         # FASE 1 — EDA
│   ├── 02_neteja.ipynb                      # FASE 2 — Neteja + features
│   ├── 03_estadistica_inferencial.ipynb     # FASE 3 — Inferència
│   ├── 04_modelat_regressio.ipynb           # FASE 4 — Regressió
│   └── 05_clustering_kmeans.ipynb           # FASE 5 — Clustering
├── src/                                     # Mòduls reutilitzables
│   ├── cleaning.py                          # Anomalies, recuperació coords, masking
│   ├── features.py                          # Província, log-transforms, dist_bcn, etc.
│   ├── models.py                            # Pipelines de regressió i clustering
│   ├── stats.py                             # Shapiro/Levene/Kruskal/Dunn/bootstrap
│   └── plots.py                             # Helpers de figures (PNG + PDF, DPI=200)
├── outputs/                                 # Generat (gitignored)
│   ├── figures/                             # 29 figures PNG + PDF
│   └── tables/                              # CSVs amb resultats de tests i mètriques
├── memoria/                                 # Memòria LaTeX (vegeu Overleaf)
├── pyproject.toml                           # Dependències (Poetry)
├── poetry.lock                              # Versions exactes
└── README.md
```

## Requisits

- **Python 3.11** (testejat amb 3.11.x)
- **[Poetry](https://python-poetry.org/) 2.x**
- Sistema operatiu: Windows / macOS / Linux. A Windows les rodes
  pre-compilades de `geopandas`/`shapely`/`pyogrio` ja inclouen GDAL/PROJ;
  **no cal cap instal·lació nativa**.
- Espai en disc: ≈ 1 GB (entorn virtual + dades + figures generades).

## Instal·lació

```bash
# 1. Clonar
git clone https://github.com/GerardGi/PRACT2-tipologia-de-les-dades.git
cd PRACT2-tipologia-de-les-dades

# 2. Crear entorn i instal·lar dependències exactes (poetry.lock)
poetry install

# 3. Registrar l'entorn com a kernel de Jupyter
#    (els notebooks tenen kernelspec.name="python3"; cal que aquest
#    "python3" apunti al Python 3.11 de Poetry, no a un altre intèrpret
#    del sistema. La forma més robusta és registrar un kernel explícit.)
poetry run python -m ipykernel install --user \
    --name pract2-tipologia \
    --display-name "Python 3.11 (pract2-tipologia)"

# 4. Verificar que els fitxers de dades hi són
#    (haurien d'aparèixer al clone — venen tracked al repo)
#    data/raw/pisos_catalunya_20260405.csv    (~4 MB)
#    data/raw/geo/gadm41_ESP_2.json.zip       (~250 KB)
#    data/processed/pisos_clean.parquet       (~4 MB, sortida FASE 2)
```

A Jupyter Lab, abans d'executar els notebooks, **selecciona el kernel
"Python 3.11 (pract2-tipologia)"** al desplegable de dalt a la dreta.

## Reproducció completa (de raw a model)

Executa els notebooks **en ordre**. Cadascun consumeix la sortida de l'anterior.
Pots fer-ho un per un o tots de cop des de CLI.

### Opció A — Jupyter Lab (interactiu)

```bash
poetry run jupyter lab notebooks/
```

i obre els notebooks de l'1 al 5 i prem **Run All** a cadascun.

### Opció B — CLI (headless, reproductible)

Cal forçar `--ExecutePreprocessor.kernel_name=pract2-tipologia` perquè
`nbconvert` no agafi un `python3` qualsevol del sistema:

```bash
poetry run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=pract2-tipologia \
  --ExecutePreprocessor.timeout=1800 \
  notebooks/01_eda.ipynb \
  notebooks/02_neteja.ipynb \
  notebooks/03_estadistica_inferencial.ipynb \
  notebooks/04_modelat_regressio.ipynb \
  notebooks/05_clustering_kmeans.ipynb
```

A Windows PowerShell (mateix ordre amb backtick com a continuador):

```powershell
poetry run jupyter nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.kernel_name=pract2-tipologia `
  --ExecutePreprocessor.timeout=1800 `
  notebooks/01_eda.ipynb `
  notebooks/02_neteja.ipynb `
  notebooks/03_estadistica_inferencial.ipynb `
  notebooks/04_modelat_regressio.ipynb `
  notebooks/05_clustering_kmeans.ipynb
```

> El notebook 04 amb Optuna pot trigar uns minuts. Si vols un *smoke test*
> ràpid, executa només els notebooks 01 i 02 — ja regenera `pisos_clean.parquet`
> i totes les figures de FASE 1-2.

### Flux de dades

| Pas | Notebook | Input | Output principal |
|-----|----------|-------|------------------|
| 1   | `01_eda.ipynb` | `data/raw/pisos_catalunya_20260405.csv` | `outputs/figures/01_…10_…` · `outputs/tables/anomalies.csv` |
| 2   | `02_neteja.ipynb` | raw + `data/raw/geo/gadm41_ESP_2.json.zip` | **`data/processed/pisos_clean.{parquet,csv}`** · `outputs/tables/cleaning_report.csv` · figures `11–14` |
| 3   | `03_estadistica_inferencial.ipynb` | `pisos_clean.parquet` | `outputs/tables/h1_…h8_…csv` · `vif_numeric_features.csv` · figures `15–19` |
| 4   | `04_modelat_regressio.ipynb` | `pisos_clean.parquet` | `models_comparison_test.csv` · `xgb_optuna_result.csv` · `spatial_cv_xgb_tuned.csv` · figures `20–24` |
| 5   | `05_clustering_kmeans.ipynb` | `pisos_clean.parquet` | `kmeans_cluster_profile.csv` · `kmeans_scan.csv` · figures `25–29` |

> **Nota**: `outputs/` està a `.gitignore`. Es regenera completament executant
> els 5 notebooks. `data/processed/pisos_clean.parquet` **sí** està committed
> per poder saltar la FASE 2 si només vols reproduir els notebooks 3-5.

## Reproductibilitat — què queda fixat

- `random_state=42` a tot arreu (mostratges, *splits*, K-means, XGBoost, Optuna).
- `poetry.lock` fixa **totes** les versions transitives.
- Pipeline de neteja **bit-perfect determinista**: executar el notebook 02
  des de zero produeix un `pisos_clean.parquet` **byte-idèntic** al committed
  (verificat el 2026-05-27 amb `git diff --stat` → buit).
- Figures: PNG + PDF a DPI=200 amb `src.plots.save_fig`.
- Taules: CSV amb `encoding="utf-8-sig"` (compatible Excel).

## Reproduir només la neteja (sense Jupyter)

Si només vols regenerar `data/processed/pisos_clean.parquet` des de zero:

```bash
poetry run python -c "
import pandas as pd; from pathlib import Path
import sys; sys.path.insert(0, 'src')
import cleaning, features

df = pd.read_csv('data/raw/pisos_catalunya_20260405.csv',
                 delimiter=';', encoding='utf-8-sig')
df = cleaning.recover_coordinates(df)
df = cleaning.flag_anomalies(df)
df = cleaning.mask_unscraped_details(df)
df = cleaning.drop_constant_cols(df)
df = features.province_by_spatial_join(df,
        gadm_path='data/raw/geo/gadm41_ESP_2.json.zip')
df['antiguitat_anys'] = df['construction_year'].map(features.parse_construction_year)
df['floor_num']       = df['floor'].map(features.parse_floor)
df['energy_cert_ord'] = df['energy_cert'].map(features.energy_cert_to_ord)
df = features.add_log_features(df)
df = features.add_distance_to_bcn(df)
df = features.add_market_density(df)

Path('data/processed').mkdir(parents=True, exist_ok=True)
df.to_parquet('data/processed/pisos_clean.parquet', index=False)
df.to_csv('data/processed/pisos_clean.csv', index=False, encoding='utf-8-sig')
print('OK:', df.shape)
"
```

Sortida esperada: `OK: (10864, 43)`.

## Resolució de problemes

- **`ValueError: numpy.dtype size changed, may indicate binary incompatibility`**
  → el notebook s'està executant amb un Python d'un altre entorn (típic:
  Python 3.9 del sistema). Registra el kernel del `.venv` (pas 3 de la
  instal·lació) i selecciona'l a Jupyter Lab, o usa
  `--ExecutePreprocessor.kernel_name=pract2-tipologia` amb `nbconvert`.
- **`ModuleNotFoundError: src`** dins d'un notebook → reinicia el kernel
  després de `poetry install`. Els notebooks injecten `PROJECT_ROOT` a
  `sys.path` automàticament a la primera cel·la.
- **`ParserError: Expected 6 fields…`** llegint el CSV cru → recorda
  `delimiter=";"` i `encoding="utf-8-sig"`.
- **`geopandas`/`pyogrio` falla al carregar el ZIP** → assegura't que la
  ruta és la del ZIP, no la del JSON extret: `zip://data/raw/geo/gadm41_ESP_2.json.zip`
  (el codi ja ho fa). A Linux pot caldre `pip install pyogrio>=0.8`.
- **XGBoost amb Optuna triga molt** → la cerca per defecte usa 30 *trials*.
  Es pot reduir editant `N_TRIALS` al notebook 04.

## Dataset original (Pràctica 1)

- Repo: <https://github.com/GerardGi/PRACT1-tipologia-de-les-dades>
- DOI: [10.5281/zenodo.19429081](https://doi.org/10.5281/zenodo.19429081)
- Llicència del dataset: CC BY-NC-SA 4.0
- Llicència del codi d'aquest repo: MIT

## Repo de la Pràctica 2

URL: <https://github.com/GerardGi/PRACT2-tipologia-de-les-dades>
