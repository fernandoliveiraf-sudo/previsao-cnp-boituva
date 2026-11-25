"""
coleta/coletar_estacao.py - coleta da estacao meteorologica automatica do CNP
(Prefeitura de Boituva + Centro Paula Souza; dados publicos do Klimadata em
https://klimadata.com/pws/boituva) pelo GitHub Actions, 24 h por dia, sem
depender do computador do autor.

Desde 24/11/2026 o Klimadata serve os dados por uma API publica em JSON
(https://klimadata.com/pub/estacoes/boituva/...; o site antigo, wx/boituva/json,
saiu do ar as 16h10): serie?periodo=dia (ultimas 24 h em blocos de 10 min),
serie?periodo=mes (30 dias, por hora) e agora (a ultima leitura). Vento e
rajada vem em m/s; chuva em mm no bloco. O 10 min so fica no ar por 24 h: o que
nao for guardado se perde.

Grava no diretorio dado (o branch "estacao-cnp" deste repositorio), nas
unidades e na convencao da coleta antiga (as de modulo_rajada/estacao_cnp.py):
  dados/cnp_5min_AAAA-MM.csv  ts (UTC, fim do intervalo), vento_kmh, rajada_kmh,
                              temp_c, orvalho_c, umidade, chuva_mmh, radiacao_wm2,
                              pressao_hpa, qnh_hpa, uv, direcao - um arquivo por mes.
                              5 min ate 24/11/2026 16h10; depois 10 min, com
                              ts = carimbo da API + 5 min (o bloco T da API junta os
                              registros antigos T e T+5) e radiacao / 1,05 (a API
                              nova publica 5% acima da antiga)
  dados/cnp_horario.csv       hora (UTC, inicio da hora), ts, vento_kmh (media),
                              rajada_kmh (maximo), direcao, chuva_mm (soma)
  dados/agora.json            a ultima leitura (leitura_atual), que a pagina le
                              pelo raw.githubusercontent.com
O valor mais recente de cada instante vence, sem apagar o que ja havia. Campo
que sumir da API fica vazio; as partes (10 min, horario, leitura atual) sao
independentes. So biblioteca padrao (roda no Python do runner).

Uso:  python coleta/coletar_estacao.py <diretorio do branch de dados>
"""
from __future__ import annotations

import csv
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

URL = "https://klimadata.com/pub/estacoes/boituva/{}"
COLS_5MIN = ["vento_kmh", "rajada_kmh", "temp_c", "orvalho_c", "umidade", "chuva_mmh", "radiacao_wm2", "pressao_hpa",
             "qnh_hpa", "uv", "direcao"]
COLS_HORA = ["hora", "ts", "vento_kmh", "rajada_kmh", "direcao", "chuva_mm"]
# (campo da API, coluna, fator, essencial). A chuva fica em mm no bloco ("chuva") e vira mm/h nos 10 min.
CAMPOS = [("vento", "vento_kmh", 3.6, True), ("rajada", "rajada_kmh", 3.6, True),
          ("temperatura", "temp_c", 1, True), ("ponto_orvalho", "orvalho_c", 1, True),
          ("umidade", "umidade", 1, False), ("chuva", "chuva", 1, True),
          ("radiacao", "radiacao_wm2", 1 / 1.05, True), ("uv", "uv", 1, False),
          ("qnh", "qnh_hpa", 1, False), ("direcao", "direcao", 1, False)]
FIM_5MIN_MS = 1795547400000           # 24/11/2026 19h10 UTC: ultimo instante do site antigo (nao sobrescrever)
PONTOS = ["N", "NNE", "NE", "ENE", "L", "ESE", "SE", "SSE", "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]


def baixar(caminho: str):
    req = urllib.request.Request(URL.format(caminho), headers={"User-Agent": "TCC-previsao-CNP (coleta horaria)",
                                                               "Accept": "application/json"})
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


def _num(v) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def serie(periodo: str) -> tuple[list[int], dict[str, list], int, list[str]]:
    """(carimbos da API em ms, {coluna: valores ja convertidos}, intervalo em s, campos essenciais ausentes)."""
    j = baixar(f"serie?periodo={periodo}")
    t = [int(x) * 1000 for x in j["t"]]
    cols, ausentes = {}, []
    for campo, col, fator, essencial in CAMPOS:
        v = j.get(campo)
        if not isinstance(v, list) or len(v) != len(t):
            if essencial:
                ausentes.append(campo)
            continue
        cols[col] = [None if (x := _num(a)) is None else round(x * fator, 3) for a in v]
    return t, cols, int(j.get("intervalo_s") or 0), ausentes


def cinco_min() -> tuple[dict, list[str]]:
    """Ultimas 24 h (blocos de 10 min) -> {ts: linha} na convencao antiga, e os campos essenciais ausentes."""
    t, cols, passo, ausentes = serie("dia")
    if passo <= 0:
        raise ValueError("serie do dia sem intervalo_s")
    linhas: dict[str, dict] = {}
    for i, ms in enumerate(t):
        fim = ms + (passo - 300) * 1000
        if fim <= FIM_5MIN_MS:
            continue
        r = {"ts": iso(fim)}
        for c, v in cols.items():
            if c == "chuva":
                r["chuva_mmh"] = "" if v[i] is None else round(v[i] * 3600 / passo, 3)
            else:
                r[c] = "" if v[i] is None else v[i]
        if any(r.get(c) not in ("", None) for c in COLS_5MIN):
            linhas[r["ts"]] = r
    return linhas, ausentes


def horario() -> dict:
    """Por hora, ultimos 30 dias: hora = ts = inicio da hora. A hora corrente vem parcial e e regravada."""
    t, cols, _, _ = serie("mes")
    linhas: dict[str, dict] = {}
    for i, ms in enumerate(t):
        h = iso(ms)
        r = {"hora": h, "ts": h}
        for orig, col in (("vento_kmh", "vento_kmh"), ("rajada_kmh", "rajada_kmh"), ("direcao", "direcao"),
                          ("chuva", "chuva_mm")):
            v = cols.get(orig)
            r[col] = "" if v is None or v[i] is None else v[i]
        linhas[h] = r
    return linhas


def leitura_atual() -> dict:
    """A ultima leitura da estacao do CNP, para a pagina: vento, rajada e direcao (de onde vem),
    temperatura, orvalho e umidade da leitura atual (agora); rajada maxima e chuva dos ultimos
    30 min pelos blocos de 10 min; e a base de nuvem estimada pelo afastamento temperatura -
    orvalho (125 m por grau, a mesma conta da segunda etapa do Agora; so vale para nuvem de
    conveccao)."""
    c = baixar("agora")
    ts = datetime.fromisoformat(str(c["ultima_leitura"]).replace("Z", "+00:00"))
    ts_ms = ts.timestamp() * 1000
    t, td = _num(c.get("temperatura")), _num(c.get("ponto_orvalho"))
    vento, raj, dirg = _num(c.get("vento")), _num(c.get("rajada")), _num(c.get("direcao"))
    vento = None if vento is None else round(vento * 3.6, 1)
    raj = None if raj is None else round(raj * 3.6, 1)
    out = {"status": "ok" if c.get("no_ar", True) else "fora_do_ar", "ts_utc": iso(ts_ms),
           "fonte": "estacao automatica do CNP (Klimadata)",
           "vento_kmh": vento, "rajada_kmh": raj,
           "direcao_graus": None if dirg is None or not vento else dirg,
           "direcao_txt": None if dirg is None or not vento else PONTOS[int((dirg % 360) / 22.5 + 0.5) % 16],
           "temp_c": t, "orvalho_c": td, "umidade": _num(c.get("umidade")), "chuva_mmh": None,
           "base_estimada_m": None if t is None or td is None else round(max(0.0, 125 * (t - td)) / 10) * 10}
    try:                                    # rajada maxima e chuva dos ultimos 30 min (blocos de 10 min)
        tt, cols, passo, _ = serie("dia")
        fins = [ms + (passo - 300) * 1000 for ms in tt]
        sel = [i for i, f in enumerate(fins) if f > ts_ms - 30 * 60 * 1000]
        rajs = [cols["rajada_kmh"][i] for i in sel if cols.get("rajada_kmh") and cols["rajada_kmh"][i] is not None]
        if raj is not None:
            rajs.append(raj)
        chv = [cols["chuva"][i] for i in sel if cols.get("chuva") and cols["chuva"][i] is not None]
        out["rajada_max_30_kmh"] = round(max(rajs), 1) if rajs else None
        out["choveu_30min"] = bool(chv and max(chv) > 0)
        if chv and sel:
            ult = cols["chuva"][sel[-1]]
            out["chuva_mmh"] = None if ult is None else round(ult * 3600 / passo, 1)
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
    gravou = 0
    try:
        novos, ausentes = cinco_min()
        por_mes: dict[str, dict] = {}
        for k, r in novos.items():
            por_mes.setdefault(k[:7], {})[k] = r
        total = 0
        for mes, linhas in por_mes.items():
            arq = base / f"cnp_5min_{mes}.csv"
            tudo, n = juntar(ler(arq, "ts"), linhas)
            gravar(arq, tudo, ["ts"] + COLS_5MIN)
            total += n
        gravou += 1
        msg5 = f"10 min: {total} instantes novos ({len(novos)} na serie do dia)"
        if ausentes:
            msg5 += f"; CAMPOS AUSENTES: {', '.join(ausentes)}"
    except Exception as e:  # noqa: BLE001 - o horario segue mesmo sem os 10 min
        msg5 = f"10 min FALHOU: {type(e).__name__}: {e}"
    try:
        arq_h = base / "cnp_horario.csv"
        tudo_h, n_h = juntar(ler(arq_h, "hora"), horario())
        gravar(arq_h, tudo_h, COLS_HORA)
        gravou += 1
        msg_h = f"horario: {n_h} horas novas"
    except Exception as e:  # noqa: BLE001
        msg_h = f"horario FALHOU: {type(e).__name__}: {e}"
    print(f"{msg5} | {msg_h}")
    if not gravou:
        sys.exit(1)                     # nada gravado: a execucao aparece como falha no GitHub


if __name__ == "__main__":
    main()
