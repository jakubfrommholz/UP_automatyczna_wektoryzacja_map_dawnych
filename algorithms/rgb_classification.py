"""
RGB Pixel Classification — klasyfikacja pikseli metodą najbliższego koloru.

Algorytm oryginalny — brak odpowiednika w Deepness, GeoAI ani SAGA.
Dla każdego piksela rastra oblicza odległość do każdego zdefiniowanego
koloru modelowego i przypisuje klasę o najmniejszej odległości.

Dostępne metryki:
  • Euklidesowa (RGB)         — szybka, wprost na wartościach R, G, B.
  • CIEDE2000 (CIE Lab, D65)  — percepcyjna różnica barw; konwersja sRGB→Lab,
    wzór CIE ΔE2000 (zwektoryzowane przeniesienie kodu z ciede-2000.py).

Integracja: brak zewnętrznych zależności poza numpy (wbudowane w QGIS).
"""

import math

import numpy as np
from osgeo import gdal, osr

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterMatrix,
    QgsProcessingParameterEnum,
    QgsProcessingParameterRasterDestination,
    QgsRectangle,
)

from ._cie import rgb_to_lab, ciede2000

METRIC_EUCLIDEAN = 0
METRIC_CIEDE2000 = 1
TILE_SIZE = 2048


def _parse_colors(raw, feedback):
    """Parse flat matrix [Nazwa,R,G,B, ...] → list of (name, r, g, b)."""
    if not raw or len(raw) % 4 != 0:
        feedback.reportError(
            f'Tabela kolorów musi mieć 4 kolumny (Nazwa, R, G, B) '
            f'i co najmniej 1 wiersz. Podano {len(raw)} wartości.',
            fatalError=True)
        return None
    result = []
    for i in range(len(raw) // 4):
        off = i * 4
        name = str(raw[off])
        try:
            r, g, b = int(float(raw[off+1])), int(float(raw[off+2])), int(float(raw[off+3]))
        except (ValueError, TypeError) as e:
            feedback.reportError(
                f'Wiersz {i} ("{name}"): R, G, B muszą być liczbami. Błąd: {e}',
                fatalError=True)
            return None
        if not all(0 <= v <= 255 for v in (r, g, b)):
            feedback.reportError(
                f'Wiersz {i} ("{name}"): wartości RGB muszą być w zakresie 0–255.',
                fatalError=True)
            return None
        result.append((name, r, g, b))
    return result


def _classify_tile_euclidean(pixels_hw3: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """
    Euclidean distance classification.

    pixels_hw3 : float32 (H, W, 3)  — wartości RGB 0–255
    refs        : float32 (N, 3)
    Returns     : int16  (H, W)  — indeksy klas
    """
    H, W, _ = pixels_hw3.shape
    flat = pixels_hw3.reshape(-1, 3)                             # (H*W, 3)
    diffs = flat[:, np.newaxis, :] - refs[np.newaxis, :, :]     # (H*W, N, 3)
    dist = np.sqrt(np.sum(diffs ** 2, axis=2))                  # (H*W, N)
    return np.argmin(dist, axis=1).reshape(H, W).astype(np.int16)


def _classify_tile_ciede2000(pixels_hw3: np.ndarray,
                              refs_lab: np.ndarray) -> np.ndarray:
    """
    Klasyfikacja na podstawie różnicy barw CIEDE2000.

    pixels_hw3 : float32/float64 (H, W, 3) — RGB 0–255 (sRGB)
    refs_lab    : float64 (N, 3)            — Lab punktów referencyjnych
    Returns     : int16 (H, W)              — indeksy klas

    Iterujemy po klasach (zazwyczaj N małe), by uniknąć alokacji
    (H*W*N*3) — kluczowe dla dużych kafli.
    """
    H, W, _ = pixels_hw3.shape
    pix_lab = rgb_to_lab(pixels_hw3)  # (H, W, 3) float64
    N = refs_lab.shape[0]
    best_dist = np.full((H, W), np.inf, dtype=np.float64)
    best_idx = np.zeros((H, W), dtype=np.int16)
    for i in range(N):
        d = ciede2000(pix_lab, refs_lab[i])  # (H, W)
        better = d < best_dist
        best_dist = np.where(better, d, best_dist)
        best_idx = np.where(better, i, best_idx)
    return best_idx.astype(np.int16)


class RgbClassificationAlgorithm(QgsProcessingAlgorithm):

    INPUT = 'INPUT'
    COLORS = 'COLORS'
    METRIC = 'METRIC'
    OUTPUT = 'OUTPUT'

    METRICS = ['Euklidesowa (RGB)', 'CIEDE2000 (CIE Lab, D65)']

    def name(self):
        return 'rgb_classification'

    def displayName(self):
        return 'Klasyfikacja pikseli RGB'

    def group(self):
        return 'Klasyfikacja'

    def groupId(self):
        return 'classification'

    def shortHelpString(self):
        return (
            'Klasyfikuje każdy piksel rastra na podstawie odległości barw do '
            'zdefiniowanych kolorów modelowych.\n\n'
            'Dostępne metryki:\n'
            '  • Euklidesowa (RGB) — szybka, działa wprost na wartościach R, G, B.\n'
            '  • CIEDE2000 (CIE Lab, D65) — percepcyjnie poprawna różnica barw; '
            'sRGB jest najpierw konwertowane do Lab (D65), następnie liczona\n'
            '    jest odległość wg wzoru CIE ΔE2000 (zwektoryzowana wersja '
            'implementacji z pliku ciede-2000.py).\n\n'
            'Parametr "Kolory modelowe": tabela z kolumnami [Nazwa, R, G, B].\n'
            'Każdy wiersz = jedna klasa. Wartości R, G, B w zakresie 0–255.\n\n'
            'Wyjście: jednopasmowy GeoTIFF z indeksami klas (0, 1, 2, ...).\n\n'
            'Algorytm oryginalny — brak odpowiednika w Deepness, GeoAI, SAGA.\n'
            'Wymagane: numpy (wbudowane w QGIS).'
        )

    def createInstance(self):
        return RgbClassificationAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT, 'Wejściowy raster (min. 3 pasma RGB)'))

        self.addParameter(QgsProcessingParameterMatrix(
            self.COLORS,
            'Kolory modelowe (Nazwa, R, G, B)',
            numberRows=3,
            hasFixedNumberRows=False,
            headers=['Nazwa', 'R', 'G', 'B'],
            defaultValue=[
                'Tło',       0,   0,   0,
                'Droga',   128, 128, 128,
                'Budynek', 255,   0,   0,
            ]
        ))

        self.addParameter(QgsProcessingParameterEnum(
            self.METRIC, 'Metryka odległości',
            options=self.METRICS,
            defaultValue=METRIC_EUCLIDEAN))

        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT, 'Wyjściowy raster klas'))

    def checkParameterValues(self, parameters, context):
        raw = self.parameterAsMatrix(parameters, self.COLORS, context)
        if not raw or len(raw) % 4 != 0:
            return False, ('Tabela kolorów musi mieć 4 kolumny (Nazwa, R, G, B) '
                           'i co najmniej 1 wiersz.')
        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if layer and layer.dataProvider().bandCount() < 3:
            return False, 'Wejściowy raster musi mieć co najmniej 3 pasma (R, G, B).'
        return super().checkParameterValues(parameters, context)

    def processAlgorithm(self, parameters, context, feedback):
        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        raw_matrix = self.parameterAsMatrix(parameters, self.COLORS, context)
        metric = self.parameterAsEnum(parameters, self.METRIC, context)
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        colors = _parse_colors(raw_matrix, feedback)
        if colors is None:
            return {}

        n_colors = len(colors)
        metric_name = self.METRICS[metric] if metric < len(self.METRICS) else 'unknown'
        feedback.pushInfo(f'Metryka: {metric_name}')
        feedback.pushInfo(f'Liczba klas: {n_colors}')
        for i, (name, r, g, b) in enumerate(colors):
            feedback.pushInfo(f'  Klasa {i}: "{name}" = RGB({r}, {g}, {b})')

        color_refs_rgb = np.array([[r, g, b] for _, r, g, b in colors],
                                  dtype=np.float32)
        color_refs_lab = (rgb_to_lab(color_refs_rgb)
                          if metric == METRIC_CIEDE2000 else None)

        provider = layer.dataProvider()
        raster_w = provider.xSize()
        raster_h = provider.ySize()
        n_bands = provider.bandCount()
        extent = layer.extent()
        px = extent.width() / raster_w
        py = extent.height() / raster_h
        ox = extent.xMinimum()
        oy = extent.yMaximum()

        gdal_dtype = gdal.GDT_Byte if n_colors <= 255 else gdal.GDT_Int16

        srs = osr.SpatialReference()
        srs.ImportFromWkt(layer.crs().toWkt())
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_path, raster_w, raster_h, 1, gdal_dtype,
                               options=['COMPRESS=LZW'])
        out_ds.SetGeoTransform((ox, px, 0, oy, 0, -py))
        out_ds.SetProjection(srs.ExportToWkt())
        out_band = out_ds.GetRasterBand(1)

        dt_map = {1: np.uint8, 2: np.uint16, 3: np.int16,
                   4: np.uint32, 5: np.int32, 6: np.float32, 7: np.float64}

        x_bins = max(math.ceil(raster_w / TILE_SIZE), 1)
        y_bins = max(math.ceil(raster_h / TILE_SIZE), 1)
        total = x_bins * y_bins
        feedback.pushInfo(f'Przetwarzanie w {total} kaflach...')

        idx = 0
        for row in range(y_bins):
            for col in range(x_bins):
                if feedback.isCanceled():
                    out_ds.FlushCache()
                    out_ds = None
                    return {}

                cs = col * TILE_SIZE; rs = row * TILE_SIZE
                ce = min(cs + TILE_SIZE, raster_w); re = min(rs + TILE_SIZE, raster_h)
                aw = ce - cs; ah = re - rs

                tile_ext = QgsRectangle(
                    ox + cs * px, oy - re * py,
                    ox + ce * px, oy - rs * py)

                rgb_bands = []
                for b in range(1, 4):
                    if b > n_bands:
                        rgb_bands.append(np.zeros((ah, aw), dtype=np.float32))
                        continue
                    blk = provider.block(b, tile_ext, aw, ah)
                    if not blk.isValid():
                        rgb_bands.append(np.zeros((ah, aw), dtype=np.float32))
                    else:
                        dt = blk.dataType()
                        dtype = dt_map.get(dt, np.float32)
                        arr = np.frombuffer(bytes(blk.data()), dtype=dtype).reshape(ah, aw)
                        rgb_bands.append(arr.astype(np.float32))

                pixels = np.stack(rgb_bands, axis=2)  # (H, W, 3)
                if metric == METRIC_CIEDE2000:
                    classes = _classify_tile_ciede2000(pixels, color_refs_lab)
                else:
                    classes = _classify_tile_euclidean(pixels, color_refs_rgb)

                if gdal_dtype == gdal.GDT_Byte:
                    out_band.WriteArray(classes.astype(np.uint8), cs, rs)
                else:
                    out_band.WriteArray(classes, cs, rs)

                idx += 1
                feedback.setProgress(int(100 * idx / total))

        out_ds.FlushCache()
        out_ds = None
        feedback.pushInfo(f'Klasyfikacja zakończona → {output_path}')
        return {self.OUTPUT: output_path}
