"""
Seeded Region Growing — segmentacja przez ekspansję regionów od seedów wektorowych.

Klasyczny algorytm Adams & Bischof (1994) w uproszczonej wersji BFS:
  • Każdy seed (lub grupa seedów o tym samym CLASS_FIELD) inicjuje region.
  • Sąsiad piksela X jest dołączany do regionu R, jeśli różnica intensywności
    od średniej intensywności R nie przekracza TOLERANCE.
  • W razie konfliktu (piksel jest sąsiadem dwóch regionów jednocześnie) — wygrywa
    pierwszy region, który dotrze do piksela (kolejność BFS).
  • Piksele nigdy nie odwiedzone pozostają -1 (nodata).

Wejście:
  • Raster (grayscale; dla RGB liczona jest średnia pasm).
  • Warstwa wektorowa punktowa z seedami.
  • Opcjonalne pole CLASS_FIELD: gdy podane, seedy z tym samym class_id tworzą
    jeden region; gdy puste, każdy seed = osobny region (numerowane 1..N).

Wyjście: jednopasмowy GeoTIFF Int32 (nodata=-1) z indeksami regionów.

Wymagane: numpy (wbudowane). Bez cv2 / scipy / skimage.
"""

from collections import deque

import numpy as np
from osgeo import gdal, osr

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsCoordinateTransform,
    QgsProject,
    QgsRectangle,
    QgsWkbTypes,
)


def _read_gray(layer, feedback):
    """Wczytuje raster do (H, W) float32 (średnia pasm RGB lub pierwsze pasmo)."""
    provider = layer.dataProvider()
    raster_w = provider.xSize()
    raster_h = provider.ySize()
    n_bands = provider.bandCount()
    extent = layer.extent()

    dt_map = {1: np.uint8, 2: np.uint16, 3: np.int16,
               4: np.uint32, 5: np.int32, 6: np.float32, 7: np.float64}

    bands = []
    for b in range(1, min(n_bands, 3) + 1):
        blk = provider.block(b, extent, raster_w, raster_h)
        if not blk.isValid():
            bands.append(np.zeros((raster_h, raster_w), dtype=np.float32))
        else:
            dt = blk.dataType()
            dtype = dt_map.get(dt, np.float32)
            arr = np.frombuffer(bytes(blk.data()), dtype=dtype).reshape(raster_h, raster_w)
            bands.append(arr.astype(np.float32))

    if not bands:
        bands.append(np.zeros((raster_h, raster_w), dtype=np.float32))
    gray = np.mean(np.stack(bands, axis=0), axis=0)
    feedback.pushInfo(
        f'Raster {raster_w}×{raster_h}, pasm={n_bands}; '
        f'zakres intensywności: {gray.min():.1f}–{gray.max():.1f}')
    return gray, raster_w, raster_h, extent


def _collect_seeds(seed_source, raster_layer, class_field, feedback):
    """
    Zwraca listę krotek (row, col, class_id) — pixel coordinates seedów.
    Transformuje CRS warstwy seedów → CRS rastra.
    """
    raster_extent = raster_layer.extent()
    raster_crs = raster_layer.crs()
    rw = raster_layer.width()
    rh = raster_layer.height()
    px = raster_extent.width() / rw
    py = raster_extent.height() / rh
    ox = raster_extent.xMinimum()
    oy = raster_extent.yMaximum()

    src_crs = seed_source.sourceCrs()
    transform = None
    if src_crs.isValid() and src_crs != raster_crs:
        transform = QgsCoordinateTransform(src_crs, raster_crs, QgsProject.instance())

    use_field = bool(class_field)
    seeds = []
    next_auto_id = 1

    for feat in seed_source.getFeatures():
        geom = feat.geometry()
        if geom is None or geom.isEmpty():
            continue
        if QgsWkbTypes.geometryType(geom.wkbType()) != QgsWkbTypes.PointGeometry:
            continue
        if transform is not None:
            try:
                geom.transform(transform)
            except Exception as e:
                feedback.pushWarning(f'Pominięto seed (transform CRS): {e}')
                continue
        for pt in geom.vertices():
            x, y = pt.x(), pt.y()
            col = int((x - ox) / px)
            row = int((oy - y) / py)
            if 0 <= row < rh and 0 <= col < rw:
                if use_field:
                    try:
                        cid = int(feat[class_field])
                    except (TypeError, ValueError, KeyError):
                        feedback.pushWarning(
                            f'Pominięto seed (pole "{class_field}" nie jest int).')
                        continue
                else:
                    cid = next_auto_id
                    next_auto_id += 1
                seeds.append((row, col, cid))
            else:
                feedback.pushWarning(
                    f'Pominięto seed poza rastrem: ({x:.2f}, {y:.2f}).')
    return seeds


def _grow_regions(gray, seeds, tolerance, feedback):
    """
    BFS z 4-sąsiedztwem. Zwraca (H, W) int32 z labelami (-1 = nieprzypisany).
    """
    H, W = gray.shape
    labels = -np.ones((H, W), dtype=np.int32)
    sums = {}
    counts = {}
    queue = deque()

    for row, col, cid in seeds:
        if labels[row, col] != -1:
            continue  # seed na zajętym pikselu
        labels[row, col] = cid
        v = float(gray[row, col])
        sums[cid] = sums.get(cid, 0.0) + v
        counts[cid] = counts.get(cid, 0) + 1
        queue.append((row, col, cid))

    feedback.pushInfo(
        f'Inicjacja: {len(queue)} seedów, {len(sums)} regionów (klas).')

    visited = 0
    log_every = 50_000
    while queue:
        if feedback.isCanceled():
            return labels
        r, c, cid = queue.popleft()
        mean = sums[cid] / counts[cid]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and labels[nr, nc] == -1:
                v = float(gray[nr, nc])
                if abs(v - mean) <= tolerance:
                    labels[nr, nc] = cid
                    sums[cid] += v
                    counts[cid] += 1
                    queue.append((nr, nc, cid))
        visited += 1
        if visited % log_every == 0:
            feedback.pushInfo(f'  Przetworzono {visited} pikseli, kolejka={len(queue)}')

    feedback.pushInfo(
        f'Zakończono. Pikseli przypisanych: {int((labels != -1).sum())}/{H*W} '
        f'({100.0 * (labels != -1).mean():.1f}%).')
    return labels


class SeededRegionGrowingAlgorithm(QgsProcessingAlgorithm):

    INPUT = 'INPUT'
    SEED_LAYER = 'SEED_LAYER'
    CLASS_FIELD = 'CLASS_FIELD'
    TOLERANCE = 'TOLERANCE'
    OUTPUT = 'OUTPUT'

    def name(self):
        return 'region_growing'

    def displayName(self):
        return 'Seeded Region Growing'

    def group(self):
        return 'Przetwarzanie obrazu'

    def groupId(self):
        return 'image_processing'

    def shortHelpString(self):
        return (
            'Segmentacja przez ekspansję regionów od seedów (punktów wektorowych).\n\n'
            'Algorytm BFS w 4-sąsiedztwie:\n'
            '  1. Każdy seed (lub grupa seedów o tym samym CLASS_FIELD) inicjuje region.\n'
            '  2. Sąsiad jest dołączany jeśli |intensywność - średnia regionu| ≤ TOLERANCE.\n'
            '  3. Średnia regionu aktualizowana po każdym dołączeniu.\n'
            '  4. Konflikt → wygrywa pierwszy region (kolejność BFS).\n\n'
            'Pole CLASS_FIELD (opcjonalne):\n'
            '  • Puste → każdy seed = osobny region (numerowane 1..N).\n'
            '  • Podane → seedy z tym samym class_id tworzą jeden region.\n\n'
            'Złożoność: O(H × W) pikseli; liniowa, ale jednowątkowa.\n'
            'Dla rastrów > 4000×4000 px może trwać kilkadziesiąt sekund.\n\n'
            'Wyjście: GeoTIFF Int32, nodata = -1, każdy piksel = class_id lub -1.\n\n'
            'Wymagane: numpy (wbudowane).'
        )

    def createInstance(self):
        return SeededRegionGrowingAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT, 'Wejściowy raster'))

        self.addParameter(QgsProcessingParameterFeatureSource(
            self.SEED_LAYER, 'Warstwa seedów (punkty)',
            types=[QgsWkbTypes.PointGeometry]))

        self.addParameter(QgsProcessingParameterField(
            self.CLASS_FIELD, 'Pole z identyfikatorem klasy (opcjonalne, integer)',
            parentLayerParameterName=self.SEED_LAYER,
            type=QgsProcessingParameterField.Numeric,
            optional=True))

        self.addParameter(QgsProcessingParameterNumber(
            self.TOLERANCE, 'Tolerancja (różnica intensywności od średniej regionu)',
            type=QgsProcessingParameterNumber.Double,
            defaultValue=10.0, minValue=0.0))

        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT, 'Wyjściowy raster regionów'))

    def processAlgorithm(self, parameters, context, feedback):
        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        seed_source = self.parameterAsSource(parameters, self.SEED_LAYER, context)
        class_field = self.parameterAsString(parameters, self.CLASS_FIELD, context)
        tolerance = self.parameterAsDouble(parameters, self.TOLERANCE, context)
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        if seed_source is None:
            feedback.reportError('Warstwa seedów nieprawidłowa.', fatalError=True)
            return {}

        gray, w, h, extent = _read_gray(layer, feedback)
        feedback.setProgress(15)

        seeds = _collect_seeds(seed_source, layer, class_field, feedback)
        if not seeds:
            feedback.reportError(
                'Brak prawidłowych seedów (pustych, poza rastrem lub w niewłaściwym CRS).',
                fatalError=True)
            return {}
        feedback.setProgress(25)

        labels = _grow_regions(gray, seeds, tolerance, feedback)
        feedback.setProgress(85)

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
        band.WriteArray(labels)
        ds.FlushCache()
        ds = None

        feedback.setProgress(100)
        feedback.pushInfo(f'Region growing zakończony → {output_path}')
        return {self.OUTPUT: output_path}
