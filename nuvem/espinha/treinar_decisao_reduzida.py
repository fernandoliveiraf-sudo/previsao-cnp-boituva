"""
treinar_decisao_reduzida.py (Rede 5 - integracao previsao+decisao) - treina a
Rede 4 (decisao do RTA) usando SO as features que a Rede 2 (autorregressiva,
redes/02_espacotemporal/treinar_autorregressivo.py) consegue de fato PREVER
pra frente no tempo (nao as 252 colunas do pipeline principal, que incluem
niveis de pressao, previous_runs, satelite, etc. que a rede de previsao
multi-dia nao produz).

Por que isso e necessario: a Rede 4 original foi treinada com o dataset
completo (dado real/observado em cada instante). Pra usar a decisao numa
previsao de VERDADE (dias a frente, sem dado real disponivel), so podemos
alimentar o modelo com o que a Rede 2 realmente consegue prever: temperatura,
pressao, vento (kt, ja convertido de wind_x/wind_y), rajada, categoria de
chuva, categoria de teto, e as features de tempo (hora/estacao do ano, que
sao gratis - vem so da data, nao da meteorologia).

ACHADO IMPORTANTE (nao obvio): a primeira versao, sem calibracao, saiu com
skill NEGATIVO (-0.526 +/- 0.069 - pior que so usar climatologia!), mesmo o
modelo tendo ~46% de acerto de categoria. Diagnostico: o GBM ficava mal
calibrado sob o mesmo tipo de mudanca de distribuicao entre treino e teste
que ja atrapalhou outras redes deste projeto (o periodo de teste, dez/2025-
jan/2026, tem MUITO mais dias "todas categorias abertas" que o treino) -
sem as ~100 features extras do modelo completo pra ajudar a generalizar essa
mudanca, o GBM reduzido fazia previsoes confiantes e ERRADAS, o que o RPS
pune muito mais que uma previsao incerta. Corrigido com CALIBRACAO ISOTONICA
(ajustada na validacao, nunca no teste) - reconstroi a relacao entre a
probabilidade crua do modelo e a frequencia real observada. Resultado: skill
foi de -0.526 pra +0.116 (positivo, bate climatologia) so com essa correcao,
sem mudar nenhuma feature.

PROMOVIDO (tentativas/02_corn_ordinal/): classificacao ordinal agora via
CORN (Shi, Cao & Raschka, arXiv:2111.08851) em vez do metodo anterior
(K classificadores independentes + np.minimum.accumulate) - cada
classificador treina na subamostra CONDICIONAL (ja passou do limiar
anterior) e a combinacao final usa a regra da cadeia. Validado com 5 seeds:
skill 0.131+/-0.012 (era 0.106+/-0.014) - mesma direcao positiva do ganho
grande ja confirmado no modelo completo (0.290 vs 0.196).

Uso: python treinar_decisao_reduzida.py
Gera: modelo_reduzido.pkl (classificadores k-vs-rest + calibradores
isotonicos) e imprime a comparacao de skill contra o modelo completo.
"""
from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression  # noqa: F401 (pickles antigos)

from calibracao import CalibradorMedia

# A JANELA DE CALIBRACAO TEM DE SER LONGA, e isso saiu de medida. A versao
# anterior calibrava em 2 meses (jul-ago/2026), e dois meses herdam a
# sazonalidade deles: jul-ago e inverno e foi o periodo mais FECHADO do
# registro (26,8% fechado, 27,8% "todas"), enquanto set-out e primavera e o
# mais ABERTO (17,0% e 52,5%). O calibrador aprendia "desloca pra fechado" e
# era aplicado justamente onde a area mais abre - a pagina publica saia
# pessimista, prevendo "fechado" em dias com 69% de chance de D aberto.
# Testado contra set-out/2026, variando so a janela:
#   jul-ago/26  (2 meses, 795)    skill 0.373  P(fech) 28.1%  P(todas) 26.3%
#   mar-ago/26  (6 meses, 2230)   skill 0.492  P(fech) 23.8%  P(todas) 36.5%
#   set-out/25  (mesma estacao)   skill 0.390  P(fech) 29.4%  P(todas) 29.0%
#   um ano      (6520)            skill 0.487  P(fech) 19.3%  P(todas) 44.2%
#   (real no periodo:                          P(fech) 17.0%  P(todas) 52.5%)
# Casar a ESTACAO nao resolve - a janela de mesma estacao do ano anterior foi
# das piores. O que resolve e o TAMANHO: janela curta e amostra enviesada.
# ESCOLHIDO: seis meses. Seis meses e um ano empatam em skill (0.005 de
# diferenca contra 0.022 de soma dos desvios), e o ano tem P(fechado)
# marginal mais proximo do real - mas o ano come METADE DO TREINO (5196
# linhas contra 9486), e RPS ja e regra de pontuacao PROPRIA: ela premia
# probabilidade honesta integrando a distribuicao inteira, nao uma marginal.
# Seis meses ganha em RPS e em acerto (0.643 contra 0.627) com o dobro de
# dados de treino. Ganho sobre a janela de 2 meses: +0.119 de skill, com
# folga sobre a soma dos desvios (0.029).
#
# Tambem testada e DESCARTADA a calibracao fora-de-amostra (treinar o modelo
# em tudo e o calibrador num modelo parcial): skill 0.390, pior que as duas.
# O calibrador e especifico do modelo - ajustado aos erros de um, aplicado a
# outro, nao transfere. Fora-de-amostra de verdade exigiria K-fold com os
# mesmos hiperparametros e juncao das previsoes, que fica para depois.
#
# Cortes do split temporal. Sao do dataset original (jul/2025 a jan/2026);
# com o dataset estendido do livro de ocorrencias eles vem da linha de comando,
# porque usar os cortes antigos jogaria quase tudo em "treino" e deixaria
# validacao e teste com os mesmos poucos dias de sempre - o ganho de dados nao
# apareceria na metrica, que continuaria medida no mesmo punhado de dias.
CORTE_VAL = "2025-11-01"
CORTE_TESTE = "2025-12-01"
K = 4
# Hiperparametros do GBM, num dicionario para que experimentos possam
# sobrescrever sem copiar treinar_gbm_calibrado (ver o comentario la dentro).
# Estes valores foram BUSCADOS na Tentativa 23, em 27 configuracoes, e os
# originais venceram - a busca preferia menos capacidade (15 folhas, lr 0.03,
# l2 10) num recorte interno do treino, mas isso nao transferiu para o teste.
HP = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
          l2_regularization=1.0)

N_SEEDS = 5
SEEDS = [42, 43, 44, 45, 46]

CATS = ["AI", "A", "B", "C", "D"]

# Promovido da Tentativa 12 (05/10/2026): o conjunto original de 26 features
# nao tinha NENHUMA variavel de vento em altitude nem sinal continuo de
# nuvem/chuva - so vento e rajada de superficie, teto quase binario apos
# calibracao, e chuva binaria (tem_precip). Os dois grupos abaixo foram
# validados com 5 seeds (tentativas/12_limites_cnp/RESULTADO.md):
#   baseline (26 feat.)        skill +0.139 +/- 0.013   acerto 0.476
#   + vento de camada          skill +0.186 +/- 0.019   acerto 0.496
#   + nuvem/chuva continuas    skill +0.218 +/- 0.024   acerto 0.484
#   + ambos (esta config)      skill +0.252 +/- 0.012   acerto 0.532
# Os tres ganhos passam o criterio do projeto (diferenca > soma dos desvios),
# e o desvio CAI de 0.013 pra 0.012 - o modelo fica mais estavel.
#
# O vento de camada entrou porque o usuario contestou a analise anterior
# ("em diversas vezes que eu estive em Boituva ela estava fechada por causa
# do vento"): rajada > 40 km/h aparece em 19.6% dos fechamentos contra 1.8%
# das aberturas, e 850hPa > 60 km/h em 16.2% contra 0.8%. E tambem o que o
# RTA mais cita no podcast (podcast_terror/ACHADOS_CRITERIOS.md).
#
# RESSALVA: niveis de pressao e precipitation_probability NAO existem na
# Previous Runs API com defasagem (ver Tentativa 10), entao este conjunto
# serve pra previsao AO VIVO mas nao pode ser validado por horizonte com
# dados/previous_runs.csv.
VENTO_CAMADA = ["wind_speed_925hPa", "wind_speed_850hPa", "wind_speed_700hPa"]
NUVEM_CHUVA = ["cloud_cover", "cloud_cover_low", "precipitation",
               "precipitation_probability", "relative_humidity_2m"]

# UMA margem de vento e UMA folga de teto, nao cinco de cada (Tentativa 24).
# As cinco margens diferem entre si por uma CONSTANTE - margem_vento_X =
# limite_X - vento - e arvore e invariante a transformacao afim: partir em
# margem_vento_AI da exatamente a mesma particao que partir em qualquer
# outra. Conferido: a correlacao entre elas e 1.000000, nao 0.99. O mesmo
# vale para folga_teto_X = base_nuvem_ft - alt_abertura_X.
# A importancia por permutacao confirmou: as oito davam queda de skill
# EXATAMENTE 0.0000 com desvio 0.0000 - o modelo nunca as usava. Remove-las
# devolve resultado identico ao ultimo digito (skill 0.488+/-0.017, acerto
# 0.638 nos dois casos).
# Saem tambem base_nuvem_ft_d1h, base_nuvem_ft_d3h e cloud_cover_low_d1h,
# cuja contribuicao medida foi <= 0.001 (uma delas negativa).
# 34 -> 23 features, com skill 0.499+/-0.012 contra 0.488+/-0.017: a
# diferenca nao passa o criterio, entao o ganho e de SIMPLIFICACAO, nao de
# acuracia - menos feature para o pipeline ao vivo calcular e errar, que
# neste projeto ja custou um train/serve skew.
FEATURES_REDUZIDAS = (
    ["margem_vento_AI", "folga_teto_AI", "margem_rajada", "tem_precip"] +
    ["vento_kt_d1h", "vento_kt_d3h", "rajada_kt_d1h", "rajada_kt_d3h",
     "cloud_cover_low_d3h"] +
    # dia_semana saiu em 13/10/2026 (testes/T16_sem_dia_semana): empate em todas
    # as amostras (validacao cruzada, sem - com: +0,002, IC95 -0,004 a +0,008) e
    # e a unica entrada que nao e tempo nem sol - pela regra 4, fica o mais simples
    ["elev_solar", "sin_hora", "cos_hora", "sin_ano", "cos_ano"] +
    VENTO_CAMADA + NUVEM_CHUVA
)


def rps(prob, y):
    n, c = prob.shape
    obs = np.zeros_like(prob)
    obs[np.arange(n), y] = 1
    return float(((prob.cumsum(1) - obs.cumsum(1)) ** 2).sum(1).mean() / (c - 1))


def cumulativo_para_classes(pge):
    pge = np.minimum.accumulate(pge, axis=1)
    p = np.zeros((pge.shape[0], K + 1))
    p[:, 0] = 1 - pge[:, 0]
    for k in range(1, K):
        p[:, k] = pge[:, k - 1] - pge[:, k]
    p[:, K] = pge[:, K - 1]
    return np.clip(p, 1e-9, 1) / np.clip(p, 1e-9, 1).sum(1, keepdims=True)


def climatologia(tr, te):
    tr = tr.assign(m=tr.ts.dt.month, h=tr.ts.dt.hour)
    tab = (tr.groupby(["m", "h"]).alvo.value_counts(normalize=True)
             .unstack(fill_value=0).reindex(columns=range(K + 1), fill_value=0))
    geral = tr.alvo.value_counts(normalize=True).reindex(range(K + 1), fill_value=0).values
    out = []
    for _, r in te.iterrows():
        chave = (r.ts.month, r.ts.hour)
        out.append(tab.loc[chave].values if chave in tab.index else geral)
    return np.array(out)


def treinar_gbm_calibrado(tr, va, feats, seed):
    """
    GBM ordinal via CORN (Shi, Cao & Raschka - arXiv:2111.08851) - PROMOVIDO
    de tentativas/02_corn_ordinal/ apos validacao rigorosa (5 seeds): skill
    0.131+/-0.012 contra 0.106+/-0.014 do metodo anterior (K classificadores
    independentes + minimum.accumulate). Cada classificador k treina SO na
    subamostra que ja passou do limiar k-1 (probabilidade CONDICIONAL); a
    calibracao isotonica (ajustada em va, nunca no teste - ja usada aqui
    antes do CORN, pra corrigir a mudanca de distribuicao treino/teste
    achada nesta rede reduzida) tambem e feita na MESMA subamostra
    condicional da validacao. Combinar (prever(), abaixo) usa a regra da
    cadeia (produto das condicionais) em vez de minimum.accumulate.
    """
    modelos, calibradores = [], []
    subset_tr, subset_va = tr, va
    for k in range(1, K + 1):
        # Os hiperparametros saem de HP (no topo do arquivo) para que um
        # experimento possa varia-los sem COPIAR esta funcao. A copia foi
        # tentada e deu errado: ela omitiu a subamostragem condicional das
        # duas ultimas linhas do laco, que e o coracao do CORN, e o skill
        # medido caiu de 0.492 para 0.315 - o experimento acabou comparando
        # dois modelos quebrados entre si.
        m = HistGradientBoostingClassifier(early_stopping=True,
                                           random_state=seed, **HP)
        m.fit(subset_tr[feats].values, (subset_tr.alvo >= k).astype(int).values)

        proba_va = m.predict_proba(subset_va[feats].values)[:, 1]
        # Tentativa 14: media isotonica + Platt. A isotonica pura devolvia 0
        # exato pra escores baixos e zerava todas as categorias (regra da
        # cadeia); a media nunca chega a 0 e nao piora o skill (5 seeds).
        iso = CalibradorMedia().fit(proba_va, (subset_va.alvo >= k).astype(int).values)

        modelos.append(m)
        calibradores.append(iso)
        subset_tr = subset_tr[subset_tr.alvo >= k]
        subset_va = subset_va[subset_va.alvo >= k]
    return modelos, calibradores


def prever(modelos, calibradores, X, feats):
    """Regra da cadeia: P(y>=k) = produto das probabilidades condicionais
    calibradas ate k - monotonico por construcao (CORN), sem precisar de
    minimum.accumulate."""
    pge = np.zeros((len(X), K))
    prod = np.ones(len(X))
    for k in range(1, K + 1):
        proba = modelos[k - 1].predict_proba(X[feats].values)[:, 1]
        proba_cal = np.clip(calibradores[k - 1].predict(proba), 1e-6, 1 - 1e-6)
        prod = prod * proba_cal
        pge[:, k - 1] = prod
    return cumulativo_para_classes(pge)


def carregar(path="../dados/datasets/dataset.csv"):
    df = pd.read_csv(path, parse_dates=["ts", "ts_utc"])
    df = df.dropna(subset=["alvo"])
    # Linha sem meteorologia nao serve: com os rotulos do livro de ocorrencias
    # (que vao a 2022) e a meteorologia so desde 12/11/2024, 60% das linhas nao
    # tem tempo nenhum. Elas ainda trazem 6 das 34 features, porque hora,
    # estacao e dia da semana vem so da data, e o modelo aprenderia um prior
    # sazonal de dias dos quais nao se sabe o tempo. Ao servir, a meteorologia
    # existe sempre - treinar com elas cria uma assimetria que nao existe em
    # producao. No dataset antigo isto nao muda nada (cobertura 100%).
    if "vento_kt" in df.columns:
        antes = len(df)
        df = df.dropna(subset=["vento_kt"])
        if len(df) < antes:
            print(f"  {antes - len(df)} linhas sem meteorologia descartadas "
                  f"({len(df)} ficam)")
    df = df.sort_values("ts").reset_index(drop=True)
    faltando = [c for c in FEATURES_REDUZIDAS if c not in df.columns]
    if faltando:
        raise SystemExit(f"faltando no {path}: {faltando}")
    return df


def main():
    # python treinar_decisao_reduzida.py [dataset.csv] [corte_val] [corte_teste]
    global CORTE_VAL, CORTE_TESTE
    caminho = sys.argv[1] if len(sys.argv) > 1 else "../dados/datasets/dataset.csv"
    if len(sys.argv) > 3:
        CORTE_VAL, CORTE_TESTE = sys.argv[2], sys.argv[3]
    df = carregar(caminho)
    print(f"dataset: {caminho}   corte val={CORTE_VAL} teste={CORTE_TESTE}")
    tr = df[df.ts < CORTE_VAL].reset_index(drop=True)
    va = df[(df.ts >= CORTE_VAL) & (df.ts < CORTE_TESTE)].reset_index(drop=True)
    te = df[df.ts >= CORTE_TESTE].reset_index(drop=True)
    print(f"treino {len(tr)} | validacao {len(va)} | teste {len(te)}")
    print(f"features reduzidas ({len(FEATURES_REDUZIDAS)}): {FEATURES_REDUZIDAS}\n")

    r_clima = rps(climatologia(tr, te), te.alvo.values)

    resultados = []
    for seed in SEEDS[:N_SEEDS]:
        modelos, calibradores = treinar_gbm_calibrado(tr, va, FEATURES_REDUZIDAS, seed)
        p = prever(modelos, calibradores, te, FEATURES_REDUZIDAS)
        r = rps(p, te.alvo.values)
        resultados.append(dict(rps=r, skill=1 - r / r_clima, acerto=(p.argmax(1) == te.alvo.values).mean()))

    rps_vals = [r["rps"] for r in resultados]
    skill_vals = [r["skill"] for r in resultados]
    acerto_vals = [r["acerto"] for r in resultados]
    print(f"GBM reduzido+CORN+calibrado ({len(FEATURES_REDUZIDAS)} features previsiveis)  "
          f"RPS={np.mean(rps_vals):.4f}+/-{np.std(rps_vals):.4f}  "
          f"skill={np.mean(skill_vals):.3f}+/-{np.std(skill_vals):.3f}  "
          f"acerto={np.mean(acerto_vals):.3f}+/-{np.std(acerto_vals):.3f}")
    print("(comparar com o GBM completo de 130 features (tambem CORN) na avaliacao "
          "rigorosa: skill=0.290+/-0.016, acerto=0.490+/-0.017 - a diferenca mostra o "
          "custo real de restringir a decisao so ao que da pra prever pra frente no tempo)")

    # modelo final pra deploy: treina em tr, calibra em va (nao usa te) -
    # e o que sera usado pra prever decisao numa semana de verdade
    modelos, calibradores = treinar_gbm_calibrado(tr, va, FEATURES_REDUZIDAS, seed=42)
    with open("modelo_reduzido.pkl", "wb") as f:
        pickle.dump({"modelos": modelos, "calibradores": calibradores,
                     "features": FEATURES_REDUZIDAS, "K": K}, f)
    print("\nsalvo: modelo_reduzido.pkl (treinado em treino, calibrado em validacao, "
          "seed 42, pronto pra prever pra frente no tempo)")


if __name__ == "__main__":
    main()
