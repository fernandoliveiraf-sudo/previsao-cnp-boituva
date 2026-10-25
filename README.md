# Dados da estação meteorológica do CNP (Boituva)

Coleta automática (GitHub Actions, de hora em hora) dos dados públicos da estação
automática do Centro Nacional de Paraquedismo, instalada pela Prefeitura de Boituva
com o Centro Paula Souza e publicada pelo Klimadata (https://klimadata.com/wx/boituva/).

- `dados/cnp_5min_AAAA-MM.csv`: vento médio e rajada (km/h), temperatura, ponto de orvalho,
  umidade, chuva (mm/h), radiação e pressão, a cada 5–10 min (UTC).
- `dados/cnp_horario.csv`: vento, rajada e direção por hora (UTC).

O site só mantém o dia corrente em 5 min; esta coleta guarda o histórico para o TCC de
previsão de abertura da área de salto. Código: `coleta/coletar_estacao.py` no branch `main`.
