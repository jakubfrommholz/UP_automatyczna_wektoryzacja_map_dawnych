"""
Tile Export — cięcie rastra na kafle do uczenia maszynowego.

Wzorzec eksportu (naming, timestamp-subdir, toggle obrazów/masek) inspirowany
plikiem map_processor_training_data_export.py z wtyczki Deepness.

Integracja z Deepness:
  • Jeśli Deepness jest zainstalowany, extent wejściowego rastra jest zaokrąglany
    do siatki pikseli rastra przy użyciu deepness.processing.extent_utils.
    round_extent_to_rlayer_grid() — ta sama funkcja, której używa Deepness
    wewnętrznie, zapewniając spójność z jego workflow.
  • Reszta logiki (pętla po kaflach, zapis PNG + GeoTIFF + maski) to własna
    implementacja działająca niezależnie od Deepness.
  • Gdy Deepness jest niedostępny, stosowane jest własne zaokrąglenie do siatki.

Notatka o pełnej funkcjonalności wtyczki Vectorization Bridge:
  • SAGA Next Gen — WYMAGANA dla algorytmów klasyfikacji, k-means, edge detection.
  • GeoAI       — ZALECANA (opcjonalna) do trenowania modeli przed inferencją.
"""

import os
import math
from datetime import datetime

import numpy as np
from osgeo import gdal, ogr

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFolderDestination,
    QgsRectangle,
)

from ._compat import deepness_available, deepness_version, deepness_round_extent


# ---------------------------------------------------------------------------
# Tile grid helpers (own implementation, used when Deepness is unavailable
# and also as the core loop even when Deepness IS available)
# ---------------------------------------------------------------------------

def _compute_grid(raster_w, raster_h, tile_size, overlap):
    stride = max(tile_size - overlap, 1)
    x_bins = 1 if raster_w <= tile_size else math.ceil((raster_w - tile_size) / stride) + 1
    y_bins = 1 if raster_h <= tile_size else math.ceil((raster_h - tile_size) / stride) + 1
    return x_bins, y_bins, stride


def _round_extent_own(extent, raster_w, raster_h):
    """Own fallback: round extent to pixel grid (no-op here; extent is exact)."""
    return extent


def _read_tile_bands(provider, n_bands, tile_ext, actual_w, actual_h):
    dt_map = {1: np.uint8, 2: np.uint16, 3: np.int16,
               4: np.uint32, 5: np.int32, 6: np.float32, 7: np.float64}
    bands = []
    for b in range(1, n_bands + 1):
        blk = provider.block(b, tile_ext, actual_w, actual_h)
        if not blk.isValid():
            bands.append(np.zeros((actual_h, actual_w), dtype=np.float32))
        else:
            dt = blk.dataType()
            dtype = dt_map.get(dt, np.float32)
            raw = bytes(blk.data())
            arr = np.frombuffer(raw, dtype=dtype).reshape(actual_h, actual_w)
            bands.append(arr.astype(np.float32))
    return np.stack(bands, axis=0)  # (C, H, W)


def _rasterize_mask(vector_layer, width, height,
                    origin_x, origin_y, px, py, wkt_proj):
    import tempfile
    tmp_tif = os.path.join(tempfile.gettempdir(), '_vbridge_mask.tif')
    tmp_gpkg = os.path.join(tempfile.gettempdir(), '_vbridge_mask_vec.gpkg')

    from qgis.core import QgsVectorFileWriter, QgsCoordinateTransformContext
    opts = QgsVectorFileWriter.SaveVectorOptions()
    opts.driverName = 'GPKG'
    QgsVectorFileWriter.writeAsVectorFormatV3(
        vector_layer, tmp_gpkg, QgsCoordinateTransformContext(), opts)

    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(tmp_tif, width, height, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((origin_x, px, 0, origin_y, 0, -py))
    ds.SetProjection(wkt_proj)
    band = ds.GetRasterBand(1)
    band.Fill(0)

    ogr_ds = ogr.Open(tmp_gpkg)
    if ogr_ds:
        gdal.RasterizeLayer(ds, [1], ogr_ds.GetLayer(0), burn_values=[1])
        ogr_ds = None

    ds.FlushCache()
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    return arr  # (H, W) uint8


class TileExportAlgorithm(QgsProcessingAlgorithm):

    INPUT = 'INPUT'
    TILE_SIZE = 'TILE_SIZE'
    OVERLAP = 'OVERLAP'
    MASK_LAYER = 'MASK_LAYER'
    EXPORT_IMAGES = 'EXPORT_IMAGES'
    EXPORT_GEOTIFF = 'EXPORT_GEOTIFF'
    USE_TIMESTAMP_SUBDIR = 'USE_TIMESTAMP_SUBDIR'
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'

    def name(self):
        return 'tile_export'

    def displayName(self):
        return 'Cięcie rastra na kafle (Tile Export)'

    def group(self):
        return 'Deep learning'

    def groupId(self):
        return 'deeplearning'

    def shortHelpString(self):
        d_ver = deepness_version()
        d_status = (f'✓ Deepness {d_ver} wykryty — zaokrąglanie extent do siatki pikseli '
                    f'przy użyciu deepness.processing.extent_utils.'
                    if d_ver else
                    '✗ Deepness niedostępny — używana własna implementacja (pełna funkcjonalność).')
        return (
            'Dzieli wejściowy raster na kafelki do uczenia maszynowego.\n\n'
            'Dla każdego kafla mogą zostać zapisane:\n'
            '  • tile_img_{x}_{y}.png  — obraz RGB (do PyTorch / ONNX DataLoader)\n'
            '  • tile_img_{x}_{y}.tif  — GeoTIFF z georeferencją (opcja)\n'
            '  • tile_mask_{x}_{y}.png — maska binarna (gdy podana warstwa maski)\n\n'
            'Współrzędne {x}_{y} = {kolumna}_{wiersz} (konwencja Deepness).\n'
            'Domyślnie wyjście trafia do podkatalogu DDMMYYYY_HHMMSS — by każde '
            'uruchomienie miało własny folder.\n\n'
            f'Status integracji: {d_status}\n\n'
            'Wymagane: numpy (wbudowane), opencv-python (cv2).\n\n'
            'Notatka — pełna funkcjonalność wtyczki Vectorization Bridge:\n'
            '  • Wytrenowany model (PyTorch .pth lub ONNX .onnx) uruchomisz w\n'
            '    "Inferencja modelu" w tej samej wtyczce.\n'
            '  • SAGA Next Gen — WYMAGANA dla pełnej funkcjonalności wtyczki\n'
            '    (klasyfikacja, k-means, edge detection).\n'
            '  • GeoAI — ZALECANA (opcjonalna) do trenowania modeli.'
        )

    def createInstance(self):
        return TileExportAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT, 'Wejściowy raster'))

        self.addParameter(QgsProcessingParameterNumber(
            self.TILE_SIZE, 'Rozmiar kafla (piksele)',
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=256, minValue=64))

        self.addParameter(QgsProcessingParameterNumber(
            self.OVERLAP, 'Zakładka (piksele)',
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=0, minValue=0))

        self.addParameter(QgsProcessingParameterVectorLayer(
            self.MASK_LAYER, 'Warstwa maski wektorowej (opcjonalna)',
            optional=True))

        self.addParameter(QgsProcessingParameterBoolean(
            self.EXPORT_IMAGES, 'Zapisz kafle obrazów (PNG)',
            defaultValue=True))

        self.addParameter(QgsProcessingParameterBoolean(
            self.EXPORT_GEOTIFF, 'Zapisz kafle GeoTIFF (z georeferencją)',
            defaultValue=True))

        self.addParameter(QgsProcessingParameterBoolean(
            self.USE_TIMESTAMP_SUBDIR,
            'Utwórz podkatalog z timestampem (DDMMYYYY_HHMMSS)',
            defaultValue=True))

        self.addParameter(QgsProcessingParameterFolderDestination(
            self.OUTPUT_FOLDER, 'Folder wyjściowy'))

    def checkParameterValues(self, parameters, context):
        tile_size = self.parameterAsInt(parameters, self.TILE_SIZE, context)
        overlap = self.parameterAsInt(parameters, self.OVERLAP, context)
        if overlap >= tile_size:
            return False, 'Zakładka musi być mniejsza niż rozmiar kafla.'
        try:
            import cv2  # noqa: F401
        except ImportError:
            return False, ('Brak opencv-python (cv2). '
                           'Zainstaluj: pip install opencv-python')
        return super().checkParameterValues(parameters, context)

    def processAlgorithm(self, parameters, context, feedback):
        try:
            import cv2
        except ImportError:
            feedback.reportError(
                'Brak opencv-python. Zainstaluj: pip install opencv-python',
                fatalError=True)
            return {}

        from osgeo import osr

        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        tile_size = self.parameterAsInt(parameters, self.TILE_SIZE, context)
        overlap = self.parameterAsInt(parameters, self.OVERLAP, context)
        mask_layer = self.parameterAsVectorLayer(parameters, self.MASK_LAYER, context)
        export_images = self.parameterAsBoolean(parameters, self.EXPORT_IMAGES, context)
        export_geotiff = self.parameterAsBoolean(parameters, self.EXPORT_GEOTIFF, context)
        use_ts = self.parameterAsBoolean(parameters, self.USE_TIMESTAMP_SUBDIR, context)
        base_output_folder = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)

        if use_ts:
            ts = datetime.now().strftime('%d%m%Y_%H%M%S')
            output_folder = os.path.join(base_output_folder, ts)
        else:
            output_folder = base_output_folder
        os.makedirs(output_folder, exist_ok=True)
        feedback.pushInfo(f'Katalog wyjściowy: {output_folder}')

        if not export_images and not export_geotiff and mask_layer is None:
            feedback.reportError(
                'Wszystkie wyjścia wyłączone (obrazy=NIE, GeoTIFF=NIE, brak maski). '
                'Włącz przynajmniej jedno źródło danych.',
                fatalError=True)
            return {}

        # --- Deepness integration: extent rounding ---
        raw_extent = layer.extent()
        if deepness_available():
            extent = deepness_round_extent(raw_extent, layer)
            feedback.pushInfo(
                f'[Deepness {deepness_version()}] Extent zaokrąglony do siatki pikseli rastra.')
        else:
            extent = raw_extent
            feedback.pushInfo('[Własna implementacja] Deepness niedostępny.')

        provider = layer.dataProvider()
        raster_w = provider.xSize()
        raster_h = provider.ySize()
        n_bands = provider.bandCount()

        px = extent.width() / raster_w
        py = extent.height() / raster_h
        origin_x = extent.xMinimum()
        origin_y = extent.yMaximum()

        srs = osr.SpatialReference()
        srs.ImportFromWkt(layer.crs().toWkt())
        wkt_proj = srs.ExportToWkt()

        x_bins, y_bins, stride = _compute_grid(raster_w, raster_h, tile_size, overlap)
        total = x_bins * y_bins
        feedback.pushInfo(f'Siatka kafli: {x_bins} × {y_bins} = {total}')

        # Pre-rasterize mask
        mask_full = None
        if mask_layer is not None:
            feedback.pushInfo('Rasteryzacja maski wektorowej...')
            mask_full = _rasterize_mask(
                mask_layer, raster_w, raster_h,
                origin_x, origin_y, px, py, wkt_proj)

        idx = 0
        for row in range(y_bins):
            for col in range(x_bins):
                if feedback.isCanceled():
                    return {}

                col_s = col * stride
                row_s = row * stride
                col_e = min(col_s + tile_size, raster_w)
                row_e = min(row_s + tile_size, raster_h)
                aw = col_e - col_s
                ah = row_e - row_s

                tile_ext = QgsRectangle(
                    origin_x + col_s * px,
                    origin_y - row_e * py,
                    origin_x + col_e * px,
                    origin_y - row_s * py,
                )

                need_image = export_images or export_geotiff
                if need_image:
                    img = _read_tile_bands(provider, n_bands, tile_ext, aw, ah)
                    padded = np.zeros((n_bands, tile_size, tile_size), dtype=np.float32)
                    padded[:, :ah, :aw] = img
                else:
                    padded = None

                # PNG (for ML)
                if export_images and padded is not None:
                    img_u8 = np.clip(padded, 0, 255).astype(np.uint8)
                    png_path = os.path.join(output_folder, f'tile_img_{col}_{row}.png')
                    if n_bands == 1:
                        cv2.imwrite(png_path, img_u8[0])
                    elif n_bands >= 3:
                        cv2.imwrite(png_path, cv2.merge([img_u8[2], img_u8[1], img_u8[0]]))
                    else:
                        cv2.imwrite(png_path, img_u8[0])

                # GeoTIFF (georeferenced)
                if export_geotiff and padded is not None:
                    tif_path = os.path.join(output_folder, f'tile_img_{col}_{row}.tif')
                    gt = (origin_x + col_s * px, px, 0, origin_y - row_s * py, 0, -py)
                    driver = gdal.GetDriverByName('GTiff')
                    ds = driver.Create(tif_path, tile_size, tile_size, n_bands,
                                       gdal.GDT_Float32, options=['COMPRESS=LZW'])
                    ds.SetGeoTransform(gt)
                    ds.SetProjection(wkt_proj)
                    for b_i, band_data in enumerate(padded, start=1):
                        ds.GetRasterBand(b_i).WriteArray(band_data)
                    ds.FlushCache()
                    ds = None

                # Mask PNG
                if mask_full is not None:
                    mask_tile = np.zeros((tile_size, tile_size), dtype=np.uint8)
                    mask_tile[:ah, :aw] = mask_full[row_s:row_e, col_s:col_e]
                    cv2.imwrite(
                        os.path.join(output_folder, f'tile_mask_{col}_{row}.png'),
                        mask_tile)

                idx += 1
                feedback.setProgress(int(100 * idx / total))

        feedback.pushInfo(f'Zapisano {idx} kafli → {output_folder}')
        return {self.OUTPUT_FOLDER: output_folder}
