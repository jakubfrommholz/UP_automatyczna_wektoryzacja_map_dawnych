"""
Rozdziela raster wielopasmowy (typowo RGB) na osobne rastry jednopasmowe.

Cel: kompatybilność z algorytmami SAGA, które często wymagają pojedynczego
grida (jednego pasma) na wejściu. Zamiast ręcznego przygotowania w innym
narzędziu — można rozbić raster RGB na 3 osobne rastry R, G, B i podać
każdy z nich jako osobny grid (np. do SAGA: Random Forest, K-Means itd.).

Wejście: raster z min. 3 pasmami (pasma >3 są ignorowane).
Wyjście: 3 GeoTIFF (LZW), po jednym na pasmo. Typ danych, geotransform
i CRS są zachowane bez konwersji.
"""

import numpy as np
from osgeo import gdal, osr

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
)


_GDAL_DTYPE_MAP = {
    np.uint8: gdal.GDT_Byte,
    np.uint16: gdal.GDT_UInt16,
    np.int16: gdal.GDT_Int16,
    np.uint32: gdal.GDT_UInt32,
    np.int32: gdal.GDT_Int32,
    np.float32: gdal.GDT_Float32,
    np.float64: gdal.GDT_Float64,
}

_QGIS_DTYPE_MAP = {
    1: np.uint8, 2: np.uint16, 3: np.int16,
    4: np.uint32, 5: np.int32, 6: np.float32, 7: np.float64,
}


def _read_band(provider, band_index, w, h, extent):
    blk = provider.block(band_index, extent, w, h)
    if not blk.isValid():
        return np.zeros((h, w), dtype=np.float32)
    dtype = _QGIS_DTYPE_MAP.get(blk.dataType(), np.float32)
    return np.frombuffer(bytes(blk.data()), dtype=dtype).reshape(h, w).copy()


def _write_single_band(path, arr, extent, w, h, crs_wkt, nodata=None):
    gdal_dtype = _GDAL_DTYPE_MAP.get(arr.dtype.type, gdal.GDT_Float32)
    px = extent.width() / w
    py = extent.height() / h
    ds = gdal.GetDriverByName('GTiff').Create(
        path, w, h, 1, gdal_dtype, options=['COMPRESS=LZW'])
    ds.SetGeoTransform((extent.xMinimum(), px, 0, extent.yMaximum(), 0, -py))
    srs = osr.SpatialReference()
    srs.ImportFromWkt(crs_wkt)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(float(nodata))
    band.WriteArray(arr)
    ds.FlushCache()


class SplitRgbBandsAlgorithm(QgsProcessingAlgorithm):

    INPUT = 'INPUT'
    OUTPUT_R = 'OUTPUT_R'
    OUTPUT_G = 'OUTPUT_G'
    OUTPUT_B = 'OUTPUT_B'

    def name(self):
        return 'split_rgb_bands'

    def displayName(self):
        return 'Rozdziel raster RGB na pasma'

    def group(self):
        return 'Przetwarzanie obrazu'

    def groupId(self):
        return 'image_processing'

    def shortHelpString(self):
        return (
            'Rozdziela raster wielopasmowy (typowo RGB) na 3 osobne rastry '
            'jednopasmowe (R, G, B).\n\n'
            'Zastosowanie: niektóre algorytmy SAGA (np. klasyfikatory pikseli, '
            'k-means na rastrach) wymagają pojedynczego grida na wejściu. '
            'Aby użyć rastra RGB, rozdziel go tym narzędziem i podaj każde '
            'pasmo oddzielnie jako osobny grid.\n\n'
            'Działanie:\n'
            '  • Wejściowy raster musi mieć min. 3 pasma — pasma 1, 2, 3 '
            'są eksportowane odpowiednio jako R, G, B.\n'
            '  • Pasma >3 (np. alfa) są ignorowane.\n'
            '  • Typ danych, geotransform, CRS i nodata są zachowane.\n\n'
            'Wyjście: 3 GeoTIFF (kompresja LZW), po jednym na pasmo.\n\n'
            'Wymagane: numpy + GDAL (wbudowane w QGIS).'
        )

    def createInstance(self):
        return SplitRgbBandsAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT, 'Wejściowy raster (min. 3 pasma)'))
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT_R, 'Pasmo R (czerwone)'))
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT_G, 'Pasmo G (zielone)'))
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT_B, 'Pasmo B (niebieskie)'))

    def processAlgorithm(self, parameters, context, feedback):
        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        path_r = self.parameterAsOutputLayer(parameters, self.OUTPUT_R, context)
        path_g = self.parameterAsOutputLayer(parameters, self.OUTPUT_G, context)
        path_b = self.parameterAsOutputLayer(parameters, self.OUTPUT_B, context)

        provider = layer.dataProvider()
        n_bands = provider.bandCount()
        if n_bands < 3:
            feedback.reportError(
                f'Raster ma {n_bands} pasm — wymagane min. 3 (RGB).',
                fatalError=True)
            return {}

        w = provider.xSize()
        h = provider.ySize()
        extent = layer.extent()
        crs_wkt = layer.crs().toWkt()

        feedback.pushInfo(
            f'Raster {w}×{h}, pasm={n_bands} → eksport pasm 1, 2, 3 jako R, G, B.')

        outputs = [
            (1, path_r, 'R'),
            (2, path_g, 'G'),
            (3, path_b, 'B'),
        ]
        for i, (band_idx, out_path, label) in enumerate(outputs):
            if feedback.isCanceled():
                return {}
            arr = _read_band(provider, band_idx, w, h, extent)
            nodata = provider.sourceNoDataValue(band_idx) \
                if provider.sourceHasNoDataValue(band_idx) else None
            _write_single_band(out_path, arr, extent, w, h, crs_wkt, nodata=nodata)
            feedback.pushInfo(
                f'  Pasmo {label} (band {band_idx}) → {out_path} '
                f'[dtype={arr.dtype}, zakres={arr.min()}–{arr.max()}'
                + (f', nodata={nodata}]' if nodata is not None else ']'))
            feedback.setProgress(int((i + 1) / 3 * 100))

        return {
            self.OUTPUT_R: path_r,
            self.OUTPUT_G: path_g,
            self.OUTPUT_B: path_b,
        }
