"""
Feature engineering per al dataset de pisos.com.

Inclou:
- Derivació de província (Barcelona / Girona / Lleida / Tarragona) a partir
  del camp `locality`, usant un diccionari curat de municipis de Catalunya.
- Transformacions logarítmiques de variables monetàries i de superfície.
- One-hot / target encoding de categòriques.

El diccionari `MUNICIPI_TO_PROVINCIA` cobreix els municipis més freqüents al
dataset; els no trobats es mapegen a "Desconeguda" i s'inspeccionen
manualment al notebook EDA.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd


def _normalize(s: str) -> str:
    """Normalitza un nom de municipi: minúscules, sense accents, sense
    sufixos com 'Capital', 'Centro Urbano', etc.

    El normalitzador genera una clau canònica però NO és invertible. Si
    `L'Hospitalet de Llobregat` i `Hospitalet de Llobregat` apareixen
    barrejats al dataset, totes dues claus convergeixen a `hospitalet de
    llobregat`.
    """
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    # Treure sufixos típics del locality de pisos.com
    s = re.sub(r"\s+(capital|centro urbano|centre urba|provincia)\b.*$", "", s)
    # Treure articles inicials catalans, també amb apòstrof immediat (l'hospitalet)
    s = re.sub(r"^(l|el|la|els|les)['\s]+", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Diccionari curat municipi (normalitzat) → província.
# Cobreix els municipis més poblats i els que apareixen amb més freqüència
# al dataset de pisos.com per a Catalunya. Es completarà a mesura que es
# descobreixin nous municipis al notebook EDA.
MUNICIPI_TO_PROVINCIA: dict[str, str] = {
    # --- Barcelona (selecció: capital + comarques metropolitanes + Vallès, Maresme, Bages) ---
    "barcelona": "Barcelona",
    "hospitalet de llobregat": "Barcelona",
    "badalona": "Barcelona",
    "terrassa": "Barcelona",
    "sabadell": "Barcelona",
    "mataro": "Barcelona",
    "santa coloma de gramenet": "Barcelona",
    "cornella de llobregat": "Barcelona",
    "sant cugat del valles": "Barcelona",
    "sant boi de llobregat": "Barcelona",
    "rubi": "Barcelona",
    "manresa": "Barcelona",
    "vilanova i la geltru": "Barcelona",
    "viladecans": "Barcelona",
    "castelldefels": "Barcelona",
    "granollers": "Barcelona",
    "cerdanyola del valles": "Barcelona",
    "mollet del valles": "Barcelona",
    "esplugues de llobregat": "Barcelona",
    "gava": "Barcelona",
    "el prat de llobregat": "Barcelona",
    "prat de llobregat": "Barcelona",
    "ripollet": "Barcelona",
    "montcada i reixac": "Barcelona",
    "sant adria de besos": "Barcelona",
    "barbera del valles": "Barcelona",
    "sant feliu de llobregat": "Barcelona",
    "sant joan despi": "Barcelona",
    "sant pere de ribes": "Barcelona",
    "sitges": "Barcelona",
    "vic": "Barcelona",
    "igualada": "Barcelona",
    "vilafranca del penedes": "Barcelona",
    "premia de mar": "Barcelona",
    "el masnou": "Barcelona",
    "masnou": "Barcelona",
    "pineda de mar": "Barcelona",
    "calella": "Barcelona",
    "malgrat de mar": "Barcelona",
    "arenys de mar": "Barcelona",
    "vilassar de mar": "Barcelona",
    "canet de mar": "Barcelona",
    "tordera": "Barcelona",
    "palau solita i plegamans": "Barcelona",
    "cardedeu": "Barcelona",
    "la garriga": "Barcelona",
    "garriga": "Barcelona",
    "parets del valles": "Barcelona",
    "caldes de montbui": "Barcelona",
    "abrera": "Barcelona",
    "esparreguera": "Barcelona",
    "olesa de montserrat": "Barcelona",
    "martorell": "Barcelona",
    "sant andreu de la barca": "Barcelona",
    "molins de rei": "Barcelona",
    "pallejà": "Barcelona",
    "palleja": "Barcelona",
    "corbera de llobregat": "Barcelona",
    "vallirana": "Barcelona",
    "begues": "Barcelona",
    "torrelles de llobregat": "Barcelona",
    "sant vicenc dels horts": "Barcelona",
    "sant climent de llobregat": "Barcelona",
    "santa perpetua de mogoda": "Barcelona",
    "polinya": "Barcelona",
    "santa eulalia de roncana": "Barcelona",
    "lliça d amunt": "Barcelona",
    "lliçà d amunt": "Barcelona",
    "lliça de vall": "Barcelona",
    "berga": "Barcelona",
    "santpedor": "Barcelona",
    "sant fruitos de bages": "Barcelona",
    "navas": "Barcelona",
    "puig reig": "Barcelona",
    "gironella": "Barcelona",
    "moia": "Barcelona",
    "balsareny": "Barcelona",
    "olvan": "Barcelona",

    # --- Girona ---
    "girona": "Girona",
    "figueres": "Girona",
    "olot": "Girona",
    "blanes": "Girona",
    "lloret de mar": "Girona",
    "salt": "Girona",
    "palafrugell": "Girona",
    "sant feliu de guixols": "Girona",
    "ripoll": "Girona",
    "puigcerda": "Girona",
    "banyoles": "Girona",
    "roses": "Girona",
    "cassa de la selva": "Girona",
    "calonge": "Girona",
    "torroella de montgri": "Girona",
    "tossa de mar": "Girona",
    "castello d empuries": "Girona",
    "l escala": "Girona",
    "escala": "Girona",
    "palamos": "Girona",
    "pals": "Girona",
    "begur": "Girona",
    "cadaques": "Girona",
    "platja d aro": "Girona",
    "castell d aro": "Girona",
    "sant antoni de calonge": "Girona",
    "santa cristina d aro": "Girona",
    "sant pere pescador": "Girona",
    "empuriabrava": "Girona",
    "la bisbal d emporda": "Girona",
    "bisbal d emporda": "Girona",
    "anglesa": "Girona",

    # --- Lleida ---
    "lleida": "Lleida",
    "balaguer": "Lleida",
    "tarrega": "Lleida",
    "mollerussa": "Lleida",
    "la seu d urgell": "Lleida",
    "seu d urgell": "Lleida",
    "cervera": "Lleida",
    "les borges blanques": "Lleida",
    "borges blanques": "Lleida",
    "tremp": "Lleida",
    "agramunt": "Lleida",
    "alcarras": "Lleida",
    "almacelles": "Lleida",
    "vielha e mijaran": "Lleida",
    "vielha": "Lleida",
    "sort": "Lleida",
    "alpicat": "Lleida",
    "guissona": "Lleida",
    "ponts": "Lleida",
    "bellpuig": "Lleida",

    # --- Tarragona ---
    "tarragona": "Tarragona",
    "reus": "Tarragona",
    "tortosa": "Tarragona",
    "el vendrell": "Tarragona",
    "vendrell": "Tarragona",
    "valls": "Tarragona",
    "amposta": "Tarragona",
    "cambrils": "Tarragona",
    "salou": "Tarragona",
    "calafell": "Tarragona",
    "el morell": "Tarragona",
    "vila seca": "Tarragona",
    "vila-seca": "Tarragona",
    "constanti": "Tarragona",
    "torredembarra": "Tarragona",
    "altafulla": "Tarragona",
    "cunit": "Tarragona",
    "deltebre": "Tarragona",
    "sant carles de la rapita": "Tarragona",
    "la rapita": "Tarragona",
    "sant pere i sant pau": "Tarragona",
    "mont roig del camp": "Tarragona",
    "mont-roig del camp": "Tarragona",
    "miami platja": "Tarragona",
    "l ametlla de mar": "Tarragona",
    "ametlla de mar": "Tarragona",
    "l ampolla": "Tarragona",
    "ampolla": "Tarragona",
    "alcanar": "Tarragona",
    "ulldecona": "Tarragona",
    "mora d ebre": "Tarragona",
    "gandesa": "Tarragona",
    "falset": "Tarragona",
    "montblanc": "Tarragona",
    "santa coloma de queralt": "Tarragona",
    "creixell": "Tarragona",
    "roda de bera": "Tarragona",
    "el catllar": "Tarragona",
    "catllar": "Tarragona",

    # Girona (selva, empordà, baix empordà, cerdanya, garrotxa)
    "castell d aro platja d aro i s agaro": "Girona",
    "castell platja d aro": "Girona",
    "calonge i sant antoni": "Girona",
    "macanet de la selva": "Girona",
    "manyanet de la selva": "Girona",
    "llagostera": "Girona",
    "vidreres": "Girona",
    "llanca": "Girona",
    "caldes de malavella": "Girona",
    "riudarenes": "Girona",
    "sils": "Girona",
    "palau saverdera": "Girona",
    "riells i viabrea": "Girona",
    "hostalric": "Girona",
    "alp": "Girona",
    "amer": "Girona",
    "arbucies": "Girona",
    "bescano": "Girona",
    "celra": "Girona",
    "corca": "Girona",
    "fornells de la selva": "Girona",
    "fortia": "Girona",
    "garrigoles": "Girona",
    "la bisbal": "Girona",
    "la jonquera": "Girona",
    "jonquera": "Girona",
    "llivia": "Girona",
    "maca de la selva": "Girona",
    "navata": "Girona",
    "pau": "Girona",
    "peratallada": "Girona",
    "perelada": "Girona",
    "port de la selva": "Girona",
    "quart": "Girona",
    "ribes de freser": "Girona",
    "sant feliu de boada": "Girona",
    "sant gregori": "Girona",
    "sant joan de les abadesses": "Girona",
    "sant jordi desvalls": "Girona",
    "sant julia de ramis": "Girona",
    "santa coloma de farners": "Girona",
    "torrent": "Girona",
    "verges": "Girona",
    "vilablareix": "Girona",
    "vilamacolum": "Girona",
    "vilamalla": "Girona",
    "vilamaniscle": "Girona",
    "vilatenim": "Girona",
    "campllong": "Girona",
    "vall llobrega": "Girona",

    # Barcelona (vallès, maresme, baix llobregat, garraf, osona, anoia)
    "cubelles": "Barcelona",
    "castellar del valles": "Barcelona",
    "sant quirze del valles": "Barcelona",
    "tona": "Barcelona",
    "franqueses del valles": "Barcelona",
    "les franqueses del valles": "Barcelona",
    "alella": "Barcelona",
    "piera": "Barcelona",
    "canyelles": "Barcelona",
    "olivella": "Barcelona",
    "sant celoni": "Barcelona",
    "seva": "Barcelona",
    "sant sadurni d anoia": "Barcelona",
    "torello": "Barcelona",
    "aiguafreda": "Barcelona",
    "ametlla del valles": "Barcelona",
    "argentona": "Barcelona",
    "avinyo": "Barcelona",
    "calldetenes": "Barcelona",
    "callus": "Barcelona",
    "centelles": "Barcelona",
    "collbato": "Barcelona",
    "el papiol": "Barcelona",
    "papiol": "Barcelona",
    "el pla del penedes": "Barcelona",
    "pla del penedes": "Barcelona",
    "espinelves": "Barcelona",
    "folgueroles": "Barcelona",
    "gelida": "Barcelona",
    "manlleu": "Barcelona",
    "mediona": "Barcelona",
    "monistrol de montserrat": "Barcelona",
    "montgat": "Barcelona",
    "navarcles": "Barcelona",
    "olerdola": "Barcelona",
    "oristar": "Barcelona",
    "pacs del penedes": "Barcelona",
    "piera la roca": "Barcelona",
    "pontons": "Barcelona",
    "premia de dalt": "Barcelona",
    "rajadell": "Barcelona",
    "roca del valles": "Barcelona",
    "la roca del valles": "Barcelona",
    "sant antoni de vilamajor": "Barcelona",
    "sant esteve sesrovires": "Barcelona",
    "sant esteve de palautordera": "Barcelona",
    "sant fost de campsentelles": "Barcelona",
    "sant hipolit de voltrega": "Barcelona",
    "sant iscle de vallalta": "Barcelona",
    "sant llorenc savall": "Barcelona",
    "sant llorenc d hortons": "Barcelona",
    "sant marti sarroca": "Barcelona",
    "sant pere de torello": "Barcelona",
    "sant quintin de mediona": "Barcelona",
    "sant salvador de guardiola": "Barcelona",
    "sant vicenc de castellet": "Barcelona",
    "sant vicenc de montalt": "Barcelona",
    "santa eulalia de ronçana": "Barcelona",
    "santa maria de palautordera": "Barcelona",
    "santa margarida i els monjos": "Barcelona",
    "santa margarida de montbui": "Barcelona",
    "subirats": "Barcelona",
    "sentmenat": "Barcelona",
    "talamanca": "Barcelona",
    "tiana": "Barcelona",
    "torrelavit": "Barcelona",
    "vacarisses": "Barcelona",
    "vallgorguina": "Barcelona",
    "vallromanes": "Barcelona",
    "vilanova del cami": "Barcelona",
    "vilanova del valles": "Barcelona",
    "vilassar de dalt": "Barcelona",
    "viladrau": "Barcelona",
    "bigues i riells": "Barcelona",
    "el bruc": "Barcelona",
    "bruc": "Barcelona",
    "cabrera de mar": "Barcelona",
    "cabrils": "Barcelona",
    "castellbisbal": "Barcelona",
    "dosrius": "Barcelona",
    "matadepera": "Barcelona",

    # Tarragona
    "rapita": "Tarragona",
    "vandellos i l hospitalet de l infant": "Tarragona",
    "vandellos i hospitalet de l infant": "Tarragona",
    "vandellos": "Tarragona",
    "pobla de montornes": "Tarragona",
    "la pobla de montornes": "Tarragona",
    "santa barbara": "Tarragona",
    "santa oliva": "Tarragona",
    "roquetes": "Tarragona",
    "senia": "Tarragona",
    "la senia": "Tarragona",
    "perello": "Tarragona",
    "el perello": "Tarragona",
    "montmell": "Tarragona",
    "el montmell": "Tarragona",
    "albinyana": "Tarragona",
    "aiguamurcia": "Tarragona",
    "alcover": "Tarragona",
    "alforja": "Tarragona",
    "almoster": "Tarragona",
    "banyeres del penedes": "Tarragona",
    "bellvei": "Tarragona",
    "benissanet": "Tarragona",
    "bonastre": "Tarragona",
    "botarell": "Tarragona",
    "cabra del camp": "Tarragona",
    "calafell platja": "Tarragona",
    "camarles": "Tarragona",
    "capafonts": "Tarragona",
    "capçanes": "Tarragona",
    "castellvell del camp": "Tarragona",
    "cornudella de montsant": "Tarragona",
    "el lloar": "Tarragona",
    "lloar": "Tarragona",
    "el masroig": "Tarragona",
    "el pla de santa maria": "Tarragona",
    "pla de santa maria": "Tarragona",
    "el pont d armentera": "Tarragona",
    "pont d armentera": "Tarragona",
    "freginals": "Tarragona",
    "garidells": "Tarragona",
    "godall": "Tarragona",
    "horta de sant joan": "Tarragona",
    "l aldea": "Tarragona",
    "aldea": "Tarragona",
    "l arboc": "Tarragona",
    "arboc": "Tarragona",
    "la canonja": "Tarragona",
    "canonja": "Tarragona",
    "la fatarella": "Tarragona",
    "fatarella": "Tarragona",
    "la galera": "Tarragona",
    "galera": "Tarragona",
    "la palma d ebre": "Tarragona",
    "palma d ebre": "Tarragona",
    "la pobla de mafumet": "Tarragona",
    "pobla de mafumet": "Tarragona",
    "la riba": "Tarragona",
    "la secuita": "Tarragona",
    "secuita": "Tarragona",
    "la selva del camp": "Tarragona",
    "selva del camp": "Tarragona",
    "la torre de l espanyol": "Tarragona",
    "torre de l espanyol": "Tarragona",
    "les borges del camp": "Tarragona",
    "borges del camp": "Tarragona",
    "llorenc del penedes": "Tarragona",
    "marsa": "Tarragona",
    "masllorenc": "Tarragona",
    "masdenverge": "Tarragona",
    "masroig": "Tarragona",
    "mont roig": "Tarragona",
    "montbrio del camp": "Tarragona",
    "montferri": "Tarragona",
    "nulles": "Tarragona",
    "passanant i belltall": "Tarragona",
    "paüls": "Tarragona",
    "pauls": "Tarragona",
    "pinell de brai": "Tarragona",
    "pira": "Tarragona",
    "porrera": "Tarragona",
    "prades": "Tarragona",
    "pratdip": "Tarragona",
    "querol": "Tarragona",
    "rasquera": "Tarragona",
    "riba roja d ebre": "Tarragona",
    "riudecanyes": "Tarragona",
    "riudecols": "Tarragona",
    "riudoms": "Tarragona",
    "roda de bara": "Tarragona",
    "salomo": "Tarragona",
    "sant jaume d enveja": "Tarragona",
    "sant jaume dels domenys": "Tarragona",
    "sarral": "Tarragona",
    "tivissa": "Tarragona",
    "tivenys": "Tarragona",
    "torre de fontaubella": "Tarragona",
    "torre del compte": "Tarragona",
    "ulldemolins": "Tarragona",
    "vallclara": "Tarragona",
    "vallmoll": "Tarragona",
    "valls del riucorb": "Tarragona",
    "vandellos i l hospitalet": "Tarragona",
    "vespella de gaia": "Tarragona",
    "vilabella": "Tarragona",
    "vila rodona": "Tarragona",
    "vilanova d escornalbou": "Tarragona",
    "vilanova de prades": "Tarragona",
    "vilaplana": "Tarragona",
    "vilella alta": "Tarragona",
    "vilella baixa": "Tarragona",
    "vimbodi i poblet": "Tarragona",
    "xerta": "Tarragona",
    "el pla de manlleu": "Tarragona",
    "els pallaresos": "Tarragona",
    "pallaresos": "Tarragona",
    "el vilosell": "Tarragona",
    "vilosell": "Tarragona",
    "miravet": "Tarragona",

    # Lleida
    "artesa de segre": "Lleida",
    "alcoletge": "Lleida",
    "castelldans": "Lleida",
    "naut aran": "Lleida",
    "torrefarrera": "Lleida",
    "alfarras": "Lleida",
    "castellsera": "Lleida",
    "rossello": "Lleida",
    "linyola": "Lleida",
    "juneda": "Lleida",
    "bausen": "Lleida",
    "puigverd de lleida": "Lleida",
    "montferrer i castellbo": "Lleida",
    "abella de la conca": "Lleida",
    "ager": "Lleida",
    "alas i cerc": "Lleida",
    "albages": "Lleida",
    "albatarrec": "Lleida",
    "albesa": "Lleida",
    "alcanó": "Lleida",
    "alcano": "Lleida",
    "alfes": "Lleida",
    "almatret": "Lleida",
    "almenar": "Lleida",
    "alos de balaguer": "Lleida",
    "anglesola": "Lleida",
    "arbeca": "Lleida",
    "artesa de lleida": "Lleida",
    "aspa": "Lleida",
    "baix pallars": "Lleida",
    "baqueira": "Lleida",
    "barbens": "Lleida",
    "bellaguarda": "Lleida",
    "bellcaire d urgell": "Lleida",
    "bellmunt d urgell": "Lleida",
    "bell lloc d urgell": "Lleida",
    "belianes": "Lleida",
    "bellver de cerdanya": "Lleida",
    "biosca": "Lleida",
    "bossost": "Lleida",
    "cabanabona": "Lleida",
    "cervia de les garrigues": "Lleida",
    "ciutadilla": "Lleida",
    "es bordes": "Lleida",
    "estamariu": "Lleida",
    "esterri d aneu": "Lleida",
    "fondarella": "Lleida",
    "fulleda": "Lleida",
    "golmes": "Lleida",
    "gualta": "Lleida",
    "guimera": "Lleida",
    "ivars d urgell": "Lleida",
    "la granadella": "Lleida",
    "granadella": "Lleida",
    "la pobla de cervoles": "Lleida",
    "pobla de cervoles": "Lleida",
    "la pobla de segur": "Lleida",
    "pobla de segur": "Lleida",
    "lavansa i fornols": "Lleida",
    "les": "Lleida",
    "llardecans": "Lleida",
    "llavorsi": "Lleida",
    "llorenç de balafia": "Lleida",
    "maials": "Lleida",
    "menarguens": "Lleida",
    "miralcamp": "Lleida",
    "montgai": "Lleida",
    "montoliu de lleida": "Lleida",
    "naut": "Lleida",
    "oliana": "Lleida",
    "organya": "Lleida",
    "os de balaguer": "Lleida",
    "ossera": "Lleida",
    "penelles": "Lleida",
    "peramola": "Lleida",
    "pinell de solsones": "Lleida",
    "poal": "Lleida",
    "preixens": "Lleida",
    "puiggros": "Lleida",
    "rialp": "Lleida",
    "ribera d urgellet": "Lleida",
    "salas de pallars": "Lleida",
    "sallent": "Lleida",
    "sant esteve de la sarga": "Lleida",
    "sant llorenc de morunys": "Lleida",
    "sant marti de riucorb": "Lleida",
    "sant marti de tous": "Lleida",
    "sant ramon": "Lleida",
    "sarroca de lleida": "Lleida",
    "senterada": "Lleida",
    "seros": "Lleida",
    "soriguera": "Lleida",
    "soses": "Lleida",
    "sudanell": "Lleida",
    "talavera": "Lleida",
    "tarroja de segarra": "Lleida",
    "tarres": "Lleida",
    "termens": "Lleida",
    "tiurana": "Lleida",
    "torrebesses": "Lleida",
    "torre serona": "Lleida",
    "torrelameu": "Lleida",
    "torres de segre": "Lleida",
    "tornabous": "Lleida",
    "vallbona de les monges": "Lleida",
    "vall de boi": "Lleida",
    "verdu": "Lleida",
    "vila sana": "Lleida",
    "vilagrassa": "Lleida",
    "vilanova de bellpuig": "Lleida",
    "vilanova de la barca": "Lleida",
    "vilanova de meia": "Lleida",
    "vilanova de segria": "Lleida",
}


def add_provincia(df: pd.DataFrame, locality_col: str = "locality",
                  out_col: str = "provincia") -> pd.DataFrame:
    """Afegeix una columna `provincia` derivada de `locality`.

    Els municipis no presents al diccionari es mapegen a "Desconeguda" perquè
    es puguin inspeccionar i afegir manualment.
    """
    out = df.copy()
    out[out_col] = (
        out[locality_col]
        .map(_normalize)
        .map(MUNICIPI_TO_PROVINCIA)
        .fillna("Desconeguda")
    )
    return out


def add_log_target(df: pd.DataFrame, src_col: str = "price_per_m2",
                   out_col: str = "log_price_per_m2") -> pd.DataFrame:
    """Afegeix la variable objectiu log(price_per_m2). NaN si src_col ≤ 0."""
    out = df.copy()
    out[out_col] = np.where(out[src_col] > 0, np.log(out[src_col]), np.nan)
    return out


# Mapeig dels buckets de pisos.com a midpoint en anys (per a antiguitat numèrica).
# "Más de 50 años" → 75 és una estimació conservadora; el dataset té habitatges
# antics fins ~1900 (>120 anys) però la mediana del bucket es queda al voltant.
_CONSTRUCTION_YEAR_MAP: dict[str, float] = {
    "Menos de 5 años": 2.5,
    "Entre 5 y 10 años": 7.5,
    "Entre 10 y 20 años": 15.0,
    "Entre 20 y 30 años": 25.0,
    "Entre 30 y 50 años": 40.0,
    "Más de 50 años": 75.0,
}


def parse_construction_year(value) -> float:
    """`construction_year` (bucket text) → antiguitat en anys (midpoint)."""
    if not isinstance(value, str):
        return float("nan")
    return _CONSTRUCTION_YEAR_MAP.get(value.strip(), float("nan"))


# Mapeig de plantes especials a numèric. Els ordinals "1ª"-"20ª" es parsegen
# numèricament amb regex; els valors textuals es mapegen aquí.
_FLOOR_SPECIAL_MAP: dict[str, float] = {
    "bajo": 0.0,
    "entresuelo": 0.5,
    "principal": 0.7,
    "semisotano": -0.5,
    "semisótano": -0.5,
    "sotano": -1.0,
    "sótano": -1.0,
    "atico": 99.0,           # marcador "molt amunt"; el modelat l'usarà via flag is_atico
    "ático": 99.0,
}


def parse_floor(value) -> float:
    """`floor` text → planta numèrica.

    - "3ª" → 3
    - "Bajo" → 0, "Entresuelo" → 0.5, "Principal" → 0.7
    - "Semisótano" → -0.5, "Sótano" → -1
    - "Más de 20" → 21
    - Buit / desconegut → NaN
    """
    if not isinstance(value, str):
        return float("nan")
    s = value.strip().lower()
    if not s:
        return float("nan")
    # "Más de 20" / "Mas de 20"
    if "mas de" in s or "más de" in s:
        m = re.search(r"(\d+)", s)
        return float(m.group(1)) + 1 if m else float("nan")
    # Ordinal "Nª" / "N°" / "N"
    m = re.match(r"^(\d+)\s*[ª°]?$", s)
    if m:
        return float(m.group(1))
    # Especials
    return _FLOOR_SPECIAL_MAP.get(s, float("nan"))


# Mapeig de certificat energètic a ordinal. NaN → 0 ("sense certificat" — el
# 66% de missings d'aquest camp justifica tractar-lo com una categoria pròpia).
_ENERGY_CERT_MAP: dict[str, int] = {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}


def energy_cert_to_ord(value) -> int:
    """`energy_cert` (A-G) → ordinal 7-1. NaN/desconegut → 0."""
    if not isinstance(value, str):
        return 0
    return _ENERGY_CERT_MAP.get(value.strip().upper(), 0)


# Plaça Catalunya, BCN (referencia per a `dist_bcn_km`).
BCN_LAT, BCN_LON = 41.3870, 2.1701
EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Distància geodèsica (Haversine) en km. Accepta escalars o arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def province_by_spatial_join(df: pd.DataFrame, gadm_path: str,
                             out_col: str = "province") -> pd.DataFrame:
    """Assigna la província a cada anunci per spatial join contra GADM ESP nivell 2.

    Carrega el shapefile GADM (52 províncies espanyoles), construeix un
    GeoDataFrame amb les coordenades del CSV, i fa un sjoin "within".
    Per als anuncis amb coordenada NaN o que cauen fora de qualsevol polígon
    (rares — illes o errors), es fa fallback al diccionari de municipis.

    Requereix `latitude` i `longitude` ja recuperades (sense valors fora rang
    que farien que sjoin no trobi cap intersecció).
    """
    import geopandas as gpd  # import diferit per no obligar geopandas si no s'usa
    from shapely.geometry import Point

    gdf_admin = gpd.read_file(f"zip://{gadm_path}")
    cat = gdf_admin[gdf_admin["NAME_1"] == "Cataluña"][["NAME_2", "geometry"]].copy()
    cat = cat.rename(columns={"NAME_2": out_col})
    cat = cat.set_crs("EPSG:4326")

    out = df.copy()
    lat = pd.to_numeric(out["latitude"], errors="coerce")
    lon = pd.to_numeric(out["longitude"], errors="coerce")
    has_coord = lat.notna() & lon.notna()

    geometry = [Point(x, y) if not pd.isna(x) and not pd.isna(y) else None
                for x, y in zip(lon, lat)]
    pts = gpd.GeoDataFrame(out.assign(_idx=range(len(out))),
                           geometry=geometry, crs="EPSG:4326")
    pts_with = pts[has_coord]

    joined = gpd.sjoin(pts_with, cat, how="left", predicate="within")
    # En cas de múltiples polígons (no hauria de passar amb provinces), agafem el primer.
    joined = joined.drop_duplicates(subset="_idx", keep="first")

    province_map = pd.Series(joined.set_index("_idx")[out_col])
    out[out_col] = province_map.reindex(range(len(out))).values

    # Fallback al diccionari per als no-coberts (típicament 0 si tot va bé)
    missing = out[out_col].isna()
    if missing.any():
        dict_prov = (out.loc[missing, "locality"]
                        .map(_normalize)
                        .map(MUNICIPI_TO_PROVINCIA))
        out.loc[missing, out_col] = dict_prov
    out[out_col] = out[out_col].fillna("Fora Catalunya")
    return out


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Afegeix log(price_eur), log(price_per_m2), log(area_m2). NaN per a x ≤ 0."""
    out = df.copy()
    for src, dst in [("price_eur", "log_price_eur"),
                     ("price_per_m2", "log_price_per_m2"),
                     ("area_m2", "log_area_m2")]:
        s = pd.to_numeric(out[src], errors="coerce")
        log_vals = pd.Series(np.nan, index=out.index)
        pos = s > 0
        log_vals.loc[pos] = np.log(s.loc[pos])
        out[dst] = log_vals
    return out


def add_distance_to_bcn(df: pd.DataFrame, out_col: str = "dist_bcn_km") -> pd.DataFrame:
    """Distància geodèsica (Haversine) a Plaça Catalunya (BCN) en km."""
    out = df.copy()
    lat = pd.to_numeric(out["latitude"], errors="coerce")
    lon = pd.to_numeric(out["longitude"], errors="coerce")
    out[out_col] = haversine_km(lat, lon, BCN_LAT, BCN_LON)
    return out


def add_market_density(df: pd.DataFrame,
                       group_col: str = "locality",
                       out_col: str = "n_anuncis_locality") -> pd.DataFrame:
    """Per cada anunci, nombre total d'anuncis al mateix municipi.

    Proxy de mida i popularitat del mercat local. Es calcula sobre el dataset
    sencer (no es split entrenament/test) perquè és una característica del
    municipi, no de l'anunci individual.
    """
    out = df.copy()
    counts = out.groupby(group_col)[group_col].transform("count")
    out[out_col] = counts.astype("Int64")
    return out


def impute_by_property_type(df: pd.DataFrame, cols: list[str],
                            group_col: str = "property_type",
                            strategy: str = "median") -> pd.DataFrame:
    """Imputa columnes numèriques amb la mediana (o mitjana) per grup.

    L'estratificació per `property_type` és crítica: la mediana de `rooms`
    d'un Pis i d'una Casa divergeixen prou que una mediana global emmascararia
    variància estructural del dataset.

    Files amb `group_col` desconegut s'imputen amb la mediana global.
    """
    if strategy not in ("median", "mean"):
        raise ValueError(f"strategy ha de ser 'median' o 'mean', no {strategy!r}")
    out = df.copy()
    for col in cols:
        s = pd.to_numeric(out[col], errors="coerce")
        if strategy == "median":
            grouped = s.groupby(out[group_col]).transform("median")
            global_fill = s.median()
        else:
            grouped = s.groupby(out[group_col]).transform("mean")
            global_fill = s.mean()
        out[col] = s.fillna(grouped).fillna(global_fill)
    return out
