"""
Konwersja sRGB → CIE Lab (D65) oraz różnica barw CIEDE2000.

Wersja zwektoryzowana dla numpy. Wzór CIEDE2000 jest wiernym przeniesieniem
skalarnej implementacji z pliku ciede-2000.py w katalogu wtyczki (public domain),
przepisanym na operacje numpy by działał na całych tablicach (H, W, 3).

Funkcje:
  • rgb_to_lab(rgb)      — sRGB (0–255) → Lab; rgb: (..., 3) → (..., 3)
  • ciede2000(lab1, lab2) — różnica barw; broadcasting zgodnie z numpy
"""

import numpy as np


# ---------------------------------------------------------------------------
# sRGB (0–255) → CIE Lab (D65)
# ---------------------------------------------------------------------------

# sRGB → XYZ (D65), Bruce Lindbloom
_M_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

# Punkt biały D65
_XN, _YN, _ZN = 0.95047, 1.00000, 1.08883
_DELTA = 6.0 / 29.0
_DELTA3 = _DELTA ** 3
_INV3_DELTA2 = 1.0 / (3.0 * _DELTA ** 2)


def rgb_to_lab(rgb):
    """
    sRGB w zakresie 0–255 → CIE Lab (D65).
    rgb: numpy array (..., 3), dowolny dtype. Zakładamy że wartości są 0–255.
    Zwraca: float64 (..., 3) — kolejność (L*, a*, b*).
    """
    arr = np.asarray(rgb, dtype=np.float64) / 255.0
    # Linearyzacja sRGB
    lin = np.where(
        arr <= 0.04045,
        arr / 12.92,
        ((arr + 0.055) / 1.055) ** 2.4,
    )
    # sRGB linear → XYZ
    xyz = lin @ _M_SRGB_TO_XYZ.T
    # Normalizacja względem D65
    xyz[..., 0] /= _XN
    xyz[..., 1] /= _YN
    xyz[..., 2] /= _ZN
    # f(t)
    f = np.where(
        xyz > _DELTA3,
        np.cbrt(xyz),
        xyz * _INV3_DELTA2 + 4.0 / 29.0,
    )
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


# ---------------------------------------------------------------------------
# CIEDE2000 — wzór zgodny z ciede-2000.py (public domain w katalogu wtyczki)
# ---------------------------------------------------------------------------

_PI = np.pi
_K7 = 25.0 ** 7  # = 6103515625.0


def ciede2000(lab1, lab2):
    """
    Różnica barw CIEDE2000 (dE00).
    lab1, lab2: numpy arrays (..., 3) — broadcastowalne. Kolejność (L*, a*, b*).
    Zwraca: (...) float64 — wartości w zakresie ~[0, 185].

    Parametryczne k_l = k_c = k_h = 1.0 (warunki referencyjne).
    """
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)

    L1 = lab1[..., 0]; a1 = lab1[..., 1]; b1 = lab1[..., 2]
    L2 = lab2[..., 0]; a2 = lab2[..., 1]; b2 = lab2[..., 2]

    n = (np.sqrt(a1 * a1 + b1 * b1) + np.sqrt(a2 * a2 + b2 * b2)) * 0.5
    n = n ** 7
    n = 1.0 + 0.5 * (1.0 - np.sqrt(n / (n + _K7)))

    c1 = np.sqrt(a1 * a1 * n * n + b1 * b1)
    c2 = np.sqrt(a2 * a2 * n * n + b2 * b2)

    h1 = np.arctan2(b1, a1 * n)
    h2 = np.arctan2(b2, a2 * n)
    h1 = h1 + 2.0 * _PI * (h1 < 0.0)
    h2 = h2 + 2.0 * _PI * (h2 < 0.0)

    n_abs = np.abs(h2 - h1)
    # Zaokrąglenie zgodne ze skalarną implementacją (n ≈ π → n = π)
    n_abs = np.where(
        (_PI - 1e-14 < n_abs) & (n_abs < _PI + 1e-14),
        _PI,
        n_abs,
    )

    h_m = (h1 + h2) * 0.5
    h_d = (h2 - h1) * 0.5
    mask = _PI < n_abs
    h_d = np.where(mask, h_d + _PI, h_d)
    h_m = np.where(mask, h_m + _PI, h_m)

    p = 36.0 * h_m - 55.0 * _PI
    n_avg7 = ((c1 + c2) * 0.5) ** 7
    r_t = (-2.0 * np.sqrt(n_avg7 / (n_avg7 + _K7))
           * np.sin(_PI / 3.0 * np.exp(p * p / (-25.0 * _PI * _PI))))

    n_l = (L1 + L2) * 0.5 - 50.0
    n_l = n_l * n_l
    L = (L2 - L1) / (1.0 + 0.015 * n_l / np.sqrt(20.0 + n_l))

    t = (1.0
         + 0.24 * np.sin(2.0 * h_m + _PI * 0.5)
         + 0.32 * np.sin(3.0 * h_m + 8.0 * _PI / 15.0)
         - 0.17 * np.sin(h_m + _PI / 3.0)
         - 0.20 * np.sin(4.0 * h_m + 3.0 * _PI / 20.0))

    n_c = c1 + c2
    H = 2.0 * np.sqrt(c1 * c2) * np.sin(h_d) / (1.0 + 0.0075 * n_c * t)
    C = (c2 - c1) / (1.0 + 0.0225 * n_c)

    return np.sqrt(L * L + H * H + C * C + C * H * r_t)
