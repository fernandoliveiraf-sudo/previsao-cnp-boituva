"""
rodar_agora.py - o Agora rodando no GitHub Actions, sem o notebook.

Roda no repositorio do site, na pasta nuvem/, que o notebook mantem em dia
(publicar/nuvem/sincronizar.py, chamado por publicar/atualizar_pagina.py):
o mesmo codigo e os mesmos modelos de producao, mais as entradas da espinha
gravadas na rodada das 6h. A unica troca: a estacao do CNP vem direto do
day.json do Klimadata (5 min do dia, com temperatura e orvalho a cada 10 min),
porque aqui nao ha o historico local - e o Agora so olha os ultimos 60 min.

Disparado a cada 15 min, das 8h as 17h, por um agendador externo chamando a
API do GitHub (o cron do proprio GitHub atrasa horas em repositorio pequeno:
medido em 07/11/2026, ~5 execucoes por dia de 72 pedidas).

Uso: python nuvem/rodar_agora.py saida/agora.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent


def _carregar(nome: str, rel: str):
    sp = importlib.util.spec_from_file_location(nome, AQUI / rel)
    m = importlib.util.module_from_spec(sp)
    sys.modules[nome] = m
    sp.loader.exec_module(m)
    return m


def main(saida: str) -> int:
    sys.path.insert(0, str(AQUI))
    prod = _carregar("producao_satelite", "modulo_satelite/producao.py")
    prod.estacao_cnp.carregar_5min = prod.estacao_cnp.cinco_min
    r = prod.prever_agora()
    r["origem"] = "nuvem"
    Path(saida).parent.mkdir(parents=True, exist_ok=True)
    Path(saida).write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pc = r.get("prob_por_categoria") or {}
    print(r.get("status"), r.get("modelo", ""), r.get("ts_utc", ""), r.get("motivo") or "",
          " ".join(f"{k}={v:.2f}" for k, v in pc.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "agora.json"))
