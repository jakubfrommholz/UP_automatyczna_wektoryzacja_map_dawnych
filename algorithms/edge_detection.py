"""
Edge Detection — wykrywanie krawędzi.

Integracja z SAGA Next Gen:
  • Jeśli SAGA NG Processing Provider jest zainstalowany, algorytm dynamicznie
    wykrywa dostępne algorytmy SAGA dot. krawędzi/gradientów i udostępnia je
    jako dodatkowe opcje metody.
  • Metody 0 (Canny) i 1 (Sobel) to zawsze własna implementacja (cv2).
  • Metody ≥ 2 są dynamicznie budowane z rejestracji SAGA — algorytm sprawdza
    registry przy każdym uruchomieniu.
  • Gdy SAGA jest niedostępny, dostępne są tylko metody cv2.
"""

import math

import numpy as np
from osgeo import gdal, osr

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsApplication,
    QgsRectangle,
)

from ._compat import saga_available, saga_find_edge_algorithm, cv2_available

METHOD_CANNY = 0
METHOD_SOBEL = 1
# Methods >= 2 are dynamically populated from SAGA


def _build_method_list():
    """Build the list of available methods at runtime."""
    methods = ['Canny (cv2)', 'Sobel — gradient natężenia (cv2)']
    saga_algo_id, saga_label = saga_find_edge_algorithm()
    if saga_algo_id:
        methods.append(f'SAGA: {saga_label}')
    return methods, saga_algo_id


LARGE_RASTER = 10_000 * 10_000


def _read_gray_full(provider, raster_w, raster_h, extent):
    """Read raster to uint8 grayscale (H, W). For small rasters."""
    n = provider.bandCount()
    ox, oy = extent.xMinimum(), extent.yMaximum()
    px = extent.width() / raster_w
    py = extent.height() / raster_h
    tile_ext = QgsRectangle(ox, oy - raster_h * py, ox + raster_w * px, oy)

    dt_map = {1: np.uint8, 2: np.uint16, 3: np.int16,
               4: np.uint32, 5: np.int32, 6: np.float32, 7: np.float64}
    bands = []
    for b in range(1, min(n + 1, 4)):
        blk = provider.block(b, tile_ext, raster_w, raster_h)
        if not blk.isValid():
            bands.append(np.zeros((raster_h, raster_w), dtype=np.float32))
        else:
            dt = blk.dataType()
            dtype = dt_map.get(dt, np.float32)
            arr = np.frombuffer(bytes(blk.data()), dtype=dtype).reshape(raster_h, raster_w)
            bands.append(arr.astype(np.float32))

    if not bands:
        return np.zeros((raster_h, raster_w), dtype=np.uint8)
    gray = np.mean(np.stack(bands, axis=0), axis=0)
    mn, mx = gray.min(), gray.max()
    if mx > mn:
        gray = (gray - mn) / (mx - mn) * 255.0
    return gray.astype(np.uint8)


def _read_gray_tiled(provider, raster_w, raster_h, extent, tile_size, feedback):
    """Tile-by-tile read for large rasters."""
    result = np.zeros((raster_h, raster_w), dtype=np.uint8)
    dt_map = {1: np.uint8, 2: np.uint16, 3: np.int16,
               4: np.uint32, 5: np.int32, 6: np.float32, 7: np.float64}
    n = provider.bandCount()
    px = extent.width() / raster_w
    py = extent.height() / raster_h
    ox = extent.xMinimum()
    oy = extent.yMaximum()

    x_b = max(math.ceil(raster_w / tile_size), 1)
    y_b = max(math.ceil(raster_h / tile_size), 1)
    total = x_b * y_b

    for row in range(y_b):
        for col in range(x_b):
            if feedback.isCanceled():
                return result
            cs = col * tile_size; rs = row * tile_size
            ce = min(cs + tile_size, raster_w); re = min(rs + tile_size, raster_h)
            aw = ce - cs; ah = re - rs
            te = QgsRectangle(ox + cs * px, oy - re * py, ox + ce * px, oy - rs * py)
            bands = []
            for b in range(1, min(n + 1, 4)):
                blk = provider.block(b, te, aw, ah)
                if not blk.isValid():
                    bands.append(np.zeros((ah, aw), dtype=np.float32))
                else:
                    dt = blk.dataType()
                    dtype = dt_map.get(dt, np.float32)
                    arr = np.frombuffer(bytes(blk.data()), dtype=dtype).reshape(ah, aw)
                    bands.append(arr.astype(np.float32))
            if bands:
                g = np.mean(np.stack(bands, axis=0), axis=0).astype(np.float32)
                mn, mx = g.min(), g.max()
                if mx > mn:
                    g = (g - mn) / (mx - mn) * 255.0
                result[rs:re, cs:ce] = g.astype(np.uint8)
            feedback.setProgress(int(50 * (row * x_b + col + 1) / total))
    return result


class EdgeDetectionAlgorithm(QgsProcessingAlgorithm):

    INPUT = 'INPUT'
    METHOD = 'METHOD'
    CANNY_LOW = 'CANNY_LOW'
    CANNY_HIGH = 'CANNY_HIGH'
    OUTPUT = 'OUTPUT'

    def name(self):
        return 'edge_detection'

    def displayName(self):
        return 'Wykrywanie krawędzi'

    def group(self):
        return 'Przetwarzanie obrazu'

    def groupId(self):
        return 'image_processing'

    def shortHelpString(self):
        methods, saga_id = _build_method_list()
        saga_status = (
            f'✓ SAGA NG wykryty — algorytm "{saga_id}" dostępny jako metoda.'
            if saga_id else
            '✗ SAGA NG niedostępny — dostępne tylko metody Canny i Sobel (cv2).')
        return (
            'Wykrywa krawędzie w rastrze.\n\n'
            'Dostępne metody:\n'
            '  • Canny — cienkie krawędzie; parametry: próg dolny i górny\n'
            '  • Sobel — mapa gradientu natężenia (float32)\n'
            '  • SAGA (gdy zainstalowany) — deleguje do algorytmu SAGA NG\n\n'
            f'Status SAGA: {saga_status}\n\n'
            'Raster jest konwertowany do skali szarości (średnia pasm RGB).\n'
            'Wymagane: opencv-python dla metod Canny i Sobel.'
        )

    def createInstance(self):
        return EdgeDetectionAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT, 'Wejściowy raster'))

        methods, _ = _build_method_list()
        self.addParameter(QgsProcessingParameterEnum(
            self.METHOD, 'Metoda wykrywania krawędzi',
            options=methods,
            defaultValue=METHOD_CANNY))

        self.addParameter(QgsProcessingParameterNumber(
            self.CANNY_LOW, 'Canny — próg dolny',
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=50, minValue=0, maxValue=255))

        self.addParameter(QgsProcessingParameterNumber(
            self.CANNY_HIGH, 'Canny — próg górny',
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=150, minValue=0, maxValue=255))

        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT, 'Wyjściowy raster krawędzi'))

    def checkParameterValues(self, parameters, context):
        method = self.parameterAsEnum(parameters, self.METHOD, context)
        if method in (METHOD_CANNY, METHOD_SOBEL):
            try:
                import cv2  # noqa: F401
            except ImportError:
                return False, ('Brak opencv-python (cv2). '
                               'Zainstaluj: pip install opencv-python')
        low = self.parameterAsInt(parameters, self.CANNY_LOW, context)
        high = self.parameterAsInt(parameters, self.CANNY_HIGH, context)
        if method == METHOD_CANNY and low >= high:
            return False, 'Próg dolny Canny musi być mniejszy niż próg górny.'
        return super().checkParameterValues(parameters, context)

    def processAlgorithm(self, parameters, context, feedback):
        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        method = self.parameterAsEnum(parameters, self.METHOD, context)
        canny_low = self.parameterAsInt(parameters, self.CANNY_LOW, context)
        canny_high = self.parameterAsInt(parameters, self.CANNY_HIGH, context)
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        methods, saga_algo_id = _build_method_list()
        method_name = methods[method] if method < len(methods) else 'unknown'
        feedback.pushInfo(f'Metoda: {method_name}')

        # --- SAGA delegation ---
        if method >= 2 and saga_algo_id:
            return self._run_saga(
                layer, saga_algo_id, output_path, parameters, context, feedback)

        # --- cv2 implementation ---
        try:
            import cv2
        except ImportError:
            feedback.reportError(
                'Brak opencv-python. Zainstaluj: pip install opencv-python',
                fatalError=True)
            return {}

        provider = layer.dataProvider()
        raster_w = provider.xSize()
        raster_h = provider.ySize()
        extent = layer.extent()
        px = extent.width() / raster_w
        py = extent.height() / raster_h

        feedback.pushInfo(f'Raster: {raster_w} × {raster_h} px')
        if raster_w * raster_h > LARGE_RASTER:
            feedback.pushInfo('Duży raster — tryb kaflowy...')
            gray = _read_gray_tiled(provider, raster_w, raster_h, extent, 4096, feedback)
        else:
            gray = _read_gray_full(provider, raster_w, raster_h, extent)
            feedback.setProgress(50)

        if feedback.isCanceled():
            return {}

        feedback.pushInfo('Obliczanie krawędzi...')
        if method == METHOD_CANNY:
            edges = cv2.Canny(gray, canny_low, canny_high)
            out_arr = edges.astype(np.uint8)
            gdal_dtype = gdal.GDT_Byte
        else:  # Sobel
            g = gray.astype(np.float32)
            sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
            out_arr = np.sqrt(sx ** 2 + sy ** 2).astype(np.float32)
            gdal_dtype = gdal.GDT_Float32

        feedback.setProgress(90)
        self._write_output(layer, output_path, out_arr, gdal_dtype, px, py, extent)
        feedback.setProgress(100)
        feedback.pushInfo(f'Wynik: {output_path}')
        return {self.OUTPUT: output_path}

    # ------------------------------------------------------------------
    def _run_saga(self, layer, saga_algo_id, output_path,
                  parameters, context, feedback):
        """Delegate to SAGA algorithm via processing.run()."""
        import processing  # QGIS processing module

        feedback.pushInfo(f'[SAGA NG] Delegowanie do algorytmu: {saga_algo_id}')

        algo = QgsApplication.processingRegistry().algorithmById(saga_algo_id)
        if algo is None:
            feedback.reportError(
                f'Algorytm SAGA "{saga_algo_id}" nie jest już dostępny.',
                fatalError=True)
            return {}

        # Build params — common input/output params for raster algorithms
        saga_params = {
            'INPUT': parameters.get(self.INPUT),
            'GRID': parameters.get(self.INPUT),
            'DEM': parameters.get(self.INPUT),
            'OUTPUT': output_path,
            'RESULT': output_path,
        }
        # Remove None values; SAGA algo will report missing required params itself
        saga_params = {k: v for k, v in saga_params.items() if v is not None}

        try:
            result = processing.run(saga_algo_id, saga_params,
                                    context=context, feedback=feedback)
            # SAGA may use different output key names
            out_key = next(
                (k for k in result if 'output' in k.lower() or 'result' in k.lower()),
                None)
            if out_key:
                return {self.OUTPUT: result[out_key]}
            feedback.pushWarning(
                'SAGA zakończył pracę, ale klucz wyjściowy nie został rozpoznany. '
                f'Dostępne klucze: {list(result.keys())}')
            return {self.OUTPUT: output_path}
        except Exception as e:
            feedback.reportError(
                f'Błąd algorytmu SAGA: {e}\nSprawdź parametry i spróbuj metody Canny/Sobel.',
                fatalError=True)
            return {}

    def _write_output(self, layer, path, arr, gdal_dtype, px, py, extent):
        srs = osr.SpatialReference()
        srs.ImportFromWkt(layer.crs().toWkt())
        h, w = arr.shape
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(path, w, h, 1, gdal_dtype, options=['COMPRESS=LZW'])
        ds.SetGeoTransform((extent.xMinimum(), px, 0, extent.yMaximum(), 0, -py))
        ds.SetProjection(srs.ExportToWkt())
        ds.GetRasterBand(1).WriteArray(arr)
        ds.FlushCache()
        ds = None
