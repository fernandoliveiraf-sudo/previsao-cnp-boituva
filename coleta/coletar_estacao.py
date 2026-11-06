"""
coleta/coletar_estacao.py - coleta da estacao meteorologica automatica do CNP
(Prefeitura de Boituva + Centro Paula Souza; dados publicos do Klimadata em
https://klimadata.com/wx/boituva/) pelo GitHub Actions, de hora em hora, 24 h
por dia, sem depender do computador do autor.

O site so deixa no ar o vento de 5 min do DIA CORRENTE (json/day.json) e o
horario dos ultimos 7 dias (json/week.json): o que nao for guardado se perde.

Grava no diretorio dado (o branch "estacao-cnp" deste repositorio):
  dados/cnp_5min_AAAA-MM.csv  ts (UTC), vento_kmh, rajada_kmh, temp_c,
                              orvalho_c, umidade, chuva_mmh, radiacao_wm2,
                              pressao_hpa - um arquivo por mes
  dados/cnp_horario.csv       hora (UTC), ts, vento_kmh, rajada_kmh, direcao
  dados/agora.json            a ultima leitura (leitura_atual), que a pagina
                              le pelo raw.githubusercontent.com (o Klimadata
                              nao deixa o navegador ler direto: sem CORS)
O valor mais recente de cada instante vence, sem apagar o que ja havia.
So biblioteca padrao (roda no Python do runner, sem instalar nada).

Uso:  python coleta/coletar_estacao.py <diretorio do branch de dados>
"""
from __future__ import annotations

import csv
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

URL = "https://klimadata.com/wx/boituva/json/{}.json"
COLS_5MIN = ["vento_kmh", "rajada_kmh", "temp_c", "orvalho_c", "umidade", "chuva_mmh", "radiacao_wm2", "pressao_hpa"]
SERIES_DIA = [("chart4", "haysChart", ["vento_kmh", "rajada_kmh"]), ("chart3", "rainRate", ["chuva_mmh"]),
              ("chart1", "outTemp", ["temp_c"]), ("chart1", "dewpoint", ["orvalho_c"]),
              ("chart2", "outHumidity", ["umidade"]), ("chart6", "radiation", ["radiacao_wm2"]),
              ("chart7", "barometer", ["pressao_hpa"])]


def baixar(nome: str) -> dict:
    req = urllib.request.Request(URL.format(nome), headers={"User-Agent": "TCC-previsao-CNP (coleta horaria)"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def iso(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def ler(arq: Path, chave: str) -> dict:
    if not arq.exists():
        return {}
    with open(arq, newline="", encoding="utf-8") as f:
        return {r[chave]: r for r in csv.DictReader(f)}


def gravar(arq: Path, linhas: dict, cols: list[str]):
    arq.parent.mkdir(parents=True, exist_ok=True)
    with open(arq, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for k in sorted(linhas):
            w.writerow({c: linhas[k].get(c, "") for c in cols})


def juntar(antigo: dict, novo: dict) -> tuple[dict, int]:
    """O valor novo vence; vazio no novo nao apaga o antigo."""
    n = 0
    for k, r in novo.items():
        if k not in antigo:
            antigo[k], n = dict(r), n + 1
        else:
            for c, v in r.items():
                if v not in ("", None):
                    antigo[k][c] = v
    return antigo, n


def cinco_min() -> dict:
    j = baixar("day")
    linhas: dict[str, dict] = {}
    for grafico, nome, cols in SERIES_DIA:
        for ponto in j[grafico]["series"][nome]["data"]:
            vals = ponto[1:]
            if all(v is None for v in vals):
                continue
            r = linhas.setdefault(iso(ponto[0]), {"ts": iso(ponto[0])})
            for c, v in zip(cols, vals):
                r[c] = "" if v is None else round(float(v), 3)
    return linhas


def horario() -> dict:
    j = baixar("week")
    linhas: dict[str, dict] = {}
    for nome, col in (("windSpeed", "vento_kmh"), ("windGust", "rajada_kmh"), ("windDir", "direcao")):
        for t, v in j["chart5"]["series"][nome]["data"]:
            if v is None:
                continue
            hora = datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:00:00+00:00")
            r = linhas.setdefault(hora, {"hora": hora, "ts": iso(t)})
            r[col] = round(float(v), 3)
    return linhas


PONTOS = ["N", "NNE", "NE", "ENE", "L", "ESE", "SE", "SSE", "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]


def _num(txt) -> float | None:
    """'23,5' / '201' / 'N/A' -> numero (o weewx_data.json vem formatado em pt-BR)."""
    try:
        return float(str(txt).replace(".", "").replace(",", ".")) if str(txt).count(",") else float(txt)
    except (TypeError, ValueError):
        return None


def leitura_atual(agora_ms: float | None = None) -> dict:
    """A ultima leitura da estacao do CNP, para a pagina: vento e direcao
    (de onde vem) da leitura atual (json/weewx_data.json), rajada maxima dos
    ultimos 30 min e chuva do json/day.json, e a base de nuvem estimada pelo
    afastamento temperatura - orvalho (125 m por grau, a mesma conta da
    segunda etapa do Agora; so vale para nuvem de conveccao)."""
    j = baixar("weewx_data")
    c = j["current"]
    ts_ms = float(c["datetime_raw"]) * 1000
    t, td = _num(c.get("outTemp_formatted")), _num(str(c.get("dewpoint", "")).split(" ")[0])
    dirg = _num(c.get("winddir_formatted"))
    vento = _num(c.get("windspeed", "").split(" ")[0])
    out = {"status": "ok", "ts_utc": iso(ts_ms), "fonte": "estacao automatica do CNP (Klimadata)",
           "vento_kmh": vento, "rajada_kmh": _num(c.get("windGust_formatted")),
           "direcao_graus": None if dirg is None or not vento else dirg,
           "direcao_txt": None if dirg is None or not vento else PONTOS[int((dirg % 360) / 22.5 + 0.5) % 16],
           "temp_c": t, "orvalho_c": td, "umidade": _num(str(c.get("outHumidity", "")).rstrip("%")),
           "chuva_mmh": _num(str(c.get("rainRate", "")).split(" ")[0]),
           "base_estimada_m": None if t is None or td is None else round(max(0.0, 125 * (t - td)) / 10) * 10}
    try:                                    # rajada maxima e chuva dos ultimos 30 min (5 min)
        d = baixar("day")
        ini = ts_ms - 30 * 60 * 1000
        raj = [p[2] for p in d["chart4"]["series"]["haysChart"]["data"]
               if ini < p[0] <= ts_ms and p[2] is not None]
        chv = [p[1] for p in d["chart3"]["series"]["rainRate"]["data"] if ini < p[0] <= ts_ms and p[1] is not None]
        out["rajada_max_30_kmh"] = round(max(raj), 1) if raj else None
        out["choveu_30min"] = bool(chv and max(chv) > 0)
    except Exception:  # noqa: BLE001 - so a leitura atual
        out["rajada_max_30_kmh"], out["choveu_30min"] = None, None
    return out


def main():
    base = Path(sys.argv[1] if len(sys.argv) > 1 else ".") / "dados"
    try:
        base.mkdir(parents=True, exist_ok=True)
        (base / "agora.json").write_text(json.dumps(leitura_atual(), ensure_ascii=False), encoding="utf-8")
    except Exception as e:  # noqa: BLE001 - a coleta segue mesmo sem a leitura atual
        print(f"agora.json nao gravado: {type(e).__name__}: {e}")
    novos = cinco_min()
    por_mes: dict[str, dict] = {}
    for k, r in novos.items():
        por_mes.setdefault(k[:7], {})[k] = r
    total = 0
    for mes, linhas in por_mes.items():
        arq = base / f"cnp_5min_{mes}.csv"
        tudo, n = juntar(ler(arq, "ts"), linhas)
        gravar(arq, tudo, ["ts"] + COLS_5MIN)
        total += n
    arq_h = base / "cnp_horario.csv"
    tudo_h, n_h = juntar(ler(arq_h, "hora"), horario())
    gravar(arq_h, tudo_h, ["hora", "ts", "vento_kmh", "rajada_kmh", "direcao"])
    print(f"5 min: {total} instantes novos ({len(novos)} no day.json) | horario: {n_h} horas novas")


if __name__ == "__main__":
    main()
