"""
Watershed (Transformata wododziałowa) — segmentacja regionów oparta na cv2.watershed.

Pipeline (klasyczny OpenCV — auto-markery):
  1. Konwersja do skali szarości (średnia pasm, max 3).
  2. Opcjonalne wygładzenie GaussianBlur.
  3. Binarna maska obiektów (Otsu lub adaptacyjna).
  4. distanceTransform → mapa odległości od tła.
  5. Próg DIST_THRESHOLD * max(dist) → markery wewnętrzne.
  6. connectedComponents → numeracja markerów (1..N).
  7. cv2.watershed na obrazie 3-kanałowym uint8 → wynik z numerami regionów,
     granice oznaczone -1.

Wyjście: jednopasмowy GeoTIFF Int32 (nodata=-1).

Dla rastrów >5000×5000 px rozważyć cięcie — cv2.watershed trzyma cały obraz
w pamięci jako tablicę (H, W, 3) uint8.
"""

import numpy as np
from osgeo import gdal, osr

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsRectangle,
)


THRESH_OTSU = 0
THRESH_ADAPTIVE = 1


def _read_raster_to_uint8(layer, feedback):
    """Wczytuje raster do tablicy (H, W) uint8 (skala szarości) oraz (H, W, 3) uint8 RGB."""
    provider = layer.dataProvider()
    raster_w = provider.xSize()
    raster_h = provider.ySize()
    n_bands = provider.bandCount()
    extent = layer.extent()
    tile_ext = QgsRectangle(
        extent.xMinimum(), extent.yMinimum(),
        extent.xMaximum(), extent.yMaximum(),
    )

    dt_map = {1: np.uint8, 2: np.uint16, 3: np.int16,
               4: np.uint32, 5: np.int32, 6: np.float32, 7: np.float64}

    bands = []
    for b in range(1, min(n_bands, 3) + 1):
        blk = provider.block(b, tile_ext, raster_w, raster_h)
        if not blk.isValid():
            bands.append(np.zeros((raster_h, raster_w), dtype=np.float32))
        else:
            dt = blk.dataType()
            dtype = dt_map.get(dt, np.float32)
            arr = np.frombuffer(bytes(blk.data()), dtype=dtype).reshape(raster_h, raster_w)
            bands.append(arr.astype(np.float32))

    while len(bands) < 3:
        bands.append(bands[0].copy() if bands else np.zeros((raster_h, raster_w), dtype=np.float32))

    rgb = np.stack(bands[:3], axis=2)  # (H, W, 3)
    mn, mx = rgb.min(), rgb.max()
    if mx > mn:
        rgb_u8 = ((rgb - mn) / (mx - mn) * 255.0).astype(np.uint8)
    else:
        rgb_u8 = np.zeros_like(rgb, dtype=np.uint8)

    gray = np.mean(rgb_u8, axis=2).astype(np.uint8)
    feedback.pushInfo(f'Raster {raster_w}×{raster_h}, pasm={n_bands} → grayscale + RGB uint8')
    return rgb_u8, gray, raster_w, raster_h, extent, n_bands


class WatershedAlgorithm(QgsProcessingAlgorithm):

    INPUT = 'INPUT'
    BLUR_KSIZE = 'BLUR_KSIZE'
    THRESHOLD_METHOD = 'THRESHOLD_METHOD'
    DIST_THRESHOLD = 'DIST_THRESHOLD'
    OUTPUT = 'OUTPUT'

    THRESHOLD_OPTIONS = ['Otsu (globalny)', 'Adaptacyjny (lokalny)']

    def name(self):
        return 'watershed'

    def displayName(self):
        return 'Transformata wododziałowa (Watershed)'

    def group(self):
        return 'Przetwarzanie obrazu'

    def groupId(self):
        return 'image_processing'

    def shortHelpString(self):
        return (
            'Segmentacja regionów metodą wododziału (cv2.watershed) z auto-markerami.\n\n'
            'Algorytm:\n'
            '  1. Raster → grayscale → wygładzenie GaussianBlur (opcja).\n'
            '  2. Binaryzacja (Otsu lub adaptacyjna).\n'
            '  3. distanceTransform → próg DIST_THRESHOLD × max → markery.\n'
            '  4. connectedComponents → numeracja markerów.\n'
            '  5. cv2.watershed na obrazie RGB → wynik (granice = -1).\n\n'
            'Wyjście: GeoTIFF Int32, nodata = -1, każdy region ma własne ID.\n\n'
            'Dla rastrów > 5000×5000 px rozważ cięcie (cały obraz musi się '
            'zmieścić w pamięci jako uint8 RGB).\n\n'
            'Wymagane: opencv-python (cv2).'
        )

    def createInstance(self):
        return WatershedAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT, 'Wejściowy raster'))

        self.addParameter(QgsProcessingParameterNumber(
            self.BLUR_KSIZE, 'Rozmiar jądra GaussianBlur (0 = bez wygładzenia, nieparzysty)',
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=5, minValue=0, maxValue=99))

        self.addParameter(QgsProcessingParameterEnum(
            self.THRESHOLD_METHOD, 'Metoda binaryzacji',
            options=self.THRESHOLD_OPTIONS,
            defaultValue=THRESH_OTSU))

        self.addParameter(QgsProcessingParameterNumber(
            self.DIST_THRESHOLD, 'Próg distance transform (frakcja max)',
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.5, minValue=0.01, maxValue=0.99))

        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT, 'Wyjściowy raster regionów'))

    def checkParameterValues(self, parameters, context):
        try:
            import cv2  # noqa: F401
        except ImportError:
            return False, ('Brak opencv-python (cv2). '
                           'Zainstaluj: pip install opencv-python')
        ksize = self.parameterAsInt(parameters, self.BLUR_KSIZE, context)
        if ksize > 0 and ksize % 2 == 0:
            return False, 'Rozmiar jądra GaussianBlur musi być nieparzysty (lub 0).'
        return super().checkParameterValues(parameters, context)

    def processAlgorithm(self, parameters, context, feedback):
        try:
            import cv2
        except ImportError:
            feedback.reportError('Brak opencv-python.', fatalError=True)
            return {}

        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        ksize = self.parameterAsInt(parameters, self.BLUR_KSIZE, context)
        thresh_method = self.parameterAsEnum(parameters, self.THRESHOLD_METHOD, context)
        dist_thr = self.parameterAsDouble(parameters, self.DIST_THRESHOLD, context)
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        rgb, gray, w, h, extent, _ = _read_raster_to_uint8(layer, feedback)
        feedback.setProgress(10)

        if ksize > 0:
            gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)
            feedback.pushInfo(f'GaussianBlur k={ksize}')
        feedback.setProgress(20)

        if thresh_method == THRESH_OTSU:
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            feedback.pushInfo('Binaryzacja: Otsu')
        else:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2)
            feedback.pushInfo('Binaryzacja: adaptacyjna (Gaussian)')
        feedback.setProgress(35)

        # morphological opening — usuń szum
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # sure background — dylatacja
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # distance transform
        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        feedback.setProgress(55)

        # markery: piksele dist > threshold * max
        max_dist = float(dist.max())
        if max_dist <= 0:
            feedback.reportError(
                'Distance transform pusty — raster zawiera same tło. '
                'Sprawdź metodę binaryzacji.', fatalError=True)
            return {}
        _, sure_fg = cv2.threshold(dist, dist_thr * max_dist, 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        feedback.pushInfo(f'distanceTransform max={max_dist:.2f}, próg={dist_thr * max_dist:.2f}')

        unknown = cv2.subtract(sure_bg, sure_fg)
        n_markers, markers = cv2.connectedComponents(sure_fg)
        feedback.pushInfo(f'Liczba markerów (regionów): {n_markers - 1}')
        # +1 by tło miało wartość 1 (nie 0)
        markers = markers + 1
        markers[unknown == 255] = 0
        feedback.setProgress(70)

        # cv2.watershed wymaga obrazu 3-kanałowego uint8
        result = cv2.watershed(rgb, markers)
        feedback.setProgress(90)

        # Konwersja: 1 = tło, 2..N = regiony, -1 = granice
        # Zachowujemy -1 jako nodata; tło i regiony zostają.
        out = result.astype(np.int32)

        srs = osr.SpatialReference()
        srs.ImportFromWkt(layer.crs().toWkt())
        px = extent.width() / w
        py = extent.height() / h
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(output_path, w, h, 1, gdal.GDT_Int32,
                            options=['COMPRESS=LZW'])
        ds.SetGeoTransform((extent.xMinimum(), px, 0, extent.yMaximum(), 0, -py))
        ds.SetProjection(srs.ExportToWkt())
        band = ds.GetRasterBand(1)
        band.SetNoDataValue(-1)
        band.WriteArray(out)
        ds.FlushCache()
        ds = None

        feedback.setProgress(100)
        feedback.pushInfo(f'Watershed zakończony → {output_path}')
        return {self.OUTPUT: output_path}
