"""Gerado por publicar/nuvem/sincronizar.py - nao editar aqui."""
import numpy as np

LAT, LON = -23.2983, -47.6924


def elevacao_solar(ts_utc):
    j = ts_utc.dayofyear.to_numpy()
    h = (ts_utc.hour + ts_utc.minute / 60).to_numpy()
    g = 2 * np.pi / 365 * (j - 1 + (h - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = (h * 60 + eqtime + 4 * LON) % 1440
    ha = np.radians(tst / 4 - 180)
    la = np.radians(LAT)
    cosz = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    return np.degrees(np.arcsin(np.clip(cosz, -1, 1)))
