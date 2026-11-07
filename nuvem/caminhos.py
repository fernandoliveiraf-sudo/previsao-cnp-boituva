"""
caminhos.py - onde cada arquivo do projeto mora, num lugar so.

Os testes (testes/Tn_*), os modulos e a pagina importam daqui em vez de
escrever caminho no codigo. Estrutura da DIRECAO_TCC.md (secao 3), aplicada
em 13/10/2026 por migrar_estrutura.sh (estado anterior: tag git
`antes-da-reestruturacao`).

    import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import caminhos as C
    df = pd.read_csv(C.achar(C.DATASET_ESTENDIDO))
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
ANTIGO = RAIZ.parent / "TCC_pos"          # pasta antiga, so leitura e so como ultimo recurso
FONTES = RAIZ / "FONTES_ORIGINAIS"

# --- codigo ------------------------------------------------------------------
PIPELINE = RAIZ / "pipeline"
ESPINHA = RAIZ / "espinha"
MODULO_DIAS = RAIZ / "modulo_dias"
MODULO_RAJADA = RAIZ / "modulo_rajada"
PUBLICAR = RAIZ / "publicar"
TESTES = RAIZ / "testes"
LEGADO = RAIZ / "legado"

# --- dados -------------------------------------------------------------------
DADOS = RAIZ / "dados"
RTAG = DADOS / "rtag"
LIVRO_EVENTOS = RTAG / "livro_rtag_eventos.csv"
LIVRO_DIAS = RTAG / "livro_rtag_dias.csv"
DIAS_EXCLUIDOS = RTAG / "dias_excluidos.csv"

ROTULOS = DADOS / "rotulos"
EVENTOS_WHATSAPP = ROTULOS / "eventos_status.csv"
ROTULOS_WHATSAPP = ROTULOS / "rotulos_15min.csv"
ROTULOS_LIVRO = ROTULOS / "rotulos_15min_livro.csv"
ROTULOS_UNIDO = ROTULOS / "rotulos_15min_unido.csv"           # fora do git
ROTULOS_UNIDO_MOTIVO = ROTULOS / "rotulos_15min_unido_motivo.csv"

METEO = DADOS / "meteo"                                       # 01_baixar_meteo.py
METEO_ESTENDIDO = DADOS / "meteo_estendido"                   # fora do git
DATASET = DADOS / "datasets" / "dataset.csv"
DATASET_ESTENDIDO = DADOS / "datasets" / "dataset_estendido.csv"   # fora do git
LEITURAS = DADOS / "leituras" / "leituras_camada.csv"
LEITURAS_UNIDO = DADOS / "leituras" / "leituras_camada_unido.csv"  # fora do git

ESTACOES_INMET = DADOS / "estacoes_inmet"
RAJADA = DADOS / "rajada"                                     # fora do git (T5)
# Tentativa 13, fora do OneDrive. No Colab: TCC_DADOS_RAIZ=/content/dados_tcc
DADOS_TCC = Path(os.environ.get("TCC_DADOS_RAIZ", "C:/dados_tcc"))

# Cortes do split temporal da espinha (os mesmos das tentativas 19-24 e do
# modelo em producao): treino < CORTE_VAL <= calibracao < CORTE_TESTE <= teste.
CORTE_VAL = "2026-03-01"
CORTE_TESTE = "2026-09-01"
# protocolo: 5 sementes. TCC_SEMENTES=42 roda um ensaio rapido (nao vale para
# conclusao: o criterio do projeto precisa do desvio entre sementes)
SEMENTES = [int(s) for s in os.environ.get("TCC_SEMENTES", "42,43,44,45,46").split(",")]

# Regra 5 da DIRECAO_TCC: texto fixo, na pagina publica e na monografia.
AVISO_RTA = ("A previsão é um apoio informativo. A decisão de abertura e fechamento da área "
             "é sempre e exclusivamente do RTA.")


def achar(p: Path) -> Path:
    """Devolve `p` se existir. Depois da migracao tudo deveria estar aqui; a
    pasta antiga so e consultada como ultimo recurso, com aviso (para nao
    esconder um arquivo que faltou copiar)."""
    p = Path(p)
    if p.exists():
        return p
    try:
        rel = p.relative_to(RAIZ)
    except ValueError:
        raise FileNotFoundError(p) from None
    alt = ANTIGO / rel
    if alt.exists():
        warnings.warn(f"{rel} lido da pasta antiga ({ANTIGO}); copie para {p.parent}", stacklevel=2)
        return alt
    raise FileNotFoundError(f"{p} (nem em {alt})")
