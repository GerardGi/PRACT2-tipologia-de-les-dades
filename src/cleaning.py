"""
Funcions de detecció d'anomalies i neteja del dataset de pisos.com.

Aquest mòdul centralitza les regles de validació que s'apliquen al CSV cru
generat a la P1, per tal que el notebook EDA, el notebook de neteja i els
notebooks de modelat comparteixin la mateixa lògica.

Convencions:
- Cap funció modifica el DataFrame d'entrada in-place; sempre retornen còpies
  o màscares booleanes.
- Les regles són explícites (límits hard-coded) perquè s'han de poder
  justificar a la memòria.
- Els NaN no es consideren anomalies en aquest mòdul — es tracten a banda
  amb el mapa de missings.
"""

from __future__ import annotations

import pandas as pd


# Catalunya s'estén aproximadament entre 40.5°N–42.9°N i 0.15°E–3.33°E.
# Marges una mica laxos per absorbir municipis frontera.
LAT_MIN, LAT_MAX = 40.0, 43.0
LON_MIN, LON_MAX = 0.0, 4.0

# Per sobre, probablement finca rústica/parcel·la i no habitatge en venda.
AREA_MAX_M2 = 5000

# Per sota, gairebé sempre error de publicació o cessió de drets,
# no transacció de mercat (habitatges < 20 k€ pràcticament no existeixen a Catalunya).
PRICE_MIN_EUR = 20_000


def anomaly_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Retorna un dict de màscares booleanes (una per cada tipus d'anomalia).

    Les màscares són `True` exactament on l'anomalia es compleix; els valors
    NaN es propaguen com a `False` (no són anomalia, són missing — es tracten
    a banda amb missingno).

    Tipus retornats:
        - lat_out_of_range: `latitude` fora de [40, 43] (probable error d'escala)
        - lon_out_of_range: `longitude` fora de [0, 4] (probable error d'escala)
        - area_huge: `area_m2 > 5000` (probables finques rústiques)
        - price_low: `0 < price_eur < 20000` (probable error de publicació)
        - price_zero: `price_eur == 0` ("consultar preu")
    """
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    area = pd.to_numeric(df["area_m2"], errors="coerce")
    price = pd.to_numeric(df["price_eur"], errors="coerce")

    masks = {
        "lat_out_of_range": ((lat < LAT_MIN) | (lat > LAT_MAX)).fillna(False),
        "lon_out_of_range": ((lon < LON_MIN) | (lon > LON_MAX)).fillna(False),
        "area_huge": (area > AREA_MAX_M2).fillna(False),
        "price_low": ((price > 0) & (price < PRICE_MIN_EUR)).fillna(False),
        "price_zero": (price == 0).fillna(False),
    }
    return masks


_DESCRIPCIONS = {
    "lat_out_of_range": f"latitude fora de [{LAT_MIN}, {LAT_MAX}] — error d'escala",
    "lon_out_of_range": f"longitude fora de [{LON_MIN}, {LON_MAX}] — error d'escala",
    "area_huge": f"area_m2 > {AREA_MAX_M2} — finca rústica probable",
    "price_low": f"0 < price_eur < {PRICE_MIN_EUR} — error de publicació probable",
    "price_zero": "price_eur == 0 — 'consultar preu'",
}


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna una taula resum: tipus, descripció, conteig, % sobre N."""
    n = len(df)
    masks = anomaly_masks(df)
    rows = [
        {
            "tipus": k,
            "descripcio": _DESCRIPCIONS[k],
            "conteig": int(m.sum()),
            "pct": round(100 * m.sum() / n, 3),
        }
        for k, m in masks.items()
    ]
    return pd.DataFrame(rows).sort_values("conteig", ascending=False).reset_index(drop=True)


# Columnes constants identificades a l'EDA — no aporten cap senyal
CONSTANT_COLS = ["operation", "is_new_development"]

# Camps que només s'omplen en mode --with-details del scraper de la P1.
# Quan l'anunci es va processar en mode ràpid, energy_cert és NaN i aquests
# booleans surten com a False (per defecte de la P1) — no com a NaN.
# Això contamina el modelat: cal restaurar-los com a NaN.
DETAIL_BOOLEAN_COLS = [
    "has_terrace", "has_balcony", "has_parking",
    "has_air_conditioning", "has_swimming_pool", "has_garden",
]

# Camps booleans realment del LLISTAT (sempre presents, no afectats pel bug)
LISTING_BOOLEAN_COLS = ["has_elevator"]


def recover_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Recupera lat/lon fora rang dividint per 1000.

    Inspecció EDA: els 142 anuncis fora rang tenen valors com 40812 / 3183
    (en lloc de 40.812 / 3.183) — patró consistent d'un factor 1000.
    """
    out = df.copy()
    lat = pd.to_numeric(out["latitude"], errors="coerce")
    lon = pd.to_numeric(out["longitude"], errors="coerce")

    bad_lat = ((lat < LAT_MIN) | (lat > LAT_MAX)) & lat.notna()
    bad_lon = ((lon < LON_MIN) | (lon > LON_MAX)) & lon.notna()

    lat = lat.mask(bad_lat, lat / 1000)
    lon = lon.mask(bad_lon, lon / 1000)

    out["latitude"] = lat
    out["longitude"] = lon
    return out


def drop_constant_cols(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Elimina les columnes constants identificades a l'EDA."""
    cols = cols or CONSTANT_COLS
    present = [c for c in cols if c in df.columns]
    return df.drop(columns=present)


def mask_unscraped_details(df: pd.DataFrame) -> pd.DataFrame:
    """Restaura els booleans de detall a NaN quan l'anunci no es va visitar.

    Proxy: si `energy_cert` és NaN, no es va executar el mode `--with-details`
    a la P1, així que tots els `has_terrace`/`has_parking`/etc. eren `False`
    per defecte i no representen una observació real.
    """
    out = df.copy()
    unscraped = out["energy_cert"].isna()
    for col in DETAIL_BOOLEAN_COLS:
        if col in out.columns:
            # Convertim a object per permetre NaN dins booleans
            out[col] = out[col].astype("object")
            out.loc[unscraped, col] = pd.NA
    return out


def flag_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Afegeix columnes flag (no elimina) per anomalies identificades."""
    out = df.copy()
    price = pd.to_numeric(out["price_eur"], errors="coerce")
    area = pd.to_numeric(out["area_m2"], errors="coerce")

    out["is_price_anomaly"] = ((price <= 0) | (price < PRICE_MIN_EUR)).fillna(False)
    out["is_rural"] = (area > AREA_MAX_M2).fillna(False)
    return out
