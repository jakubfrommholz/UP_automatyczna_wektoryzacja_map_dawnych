"""
Inferencja modelu (PyTorch .pth/.pt + ONNX .onnx).

Algorytm uruchamia wytrenowany model segmentacji/klasyfikacji pikseli na rastrze
wejściowym. Obsługuje dwa formaty:
  • PyTorch (.pth, .pt) — wczytywany przez torch.load (lub torch.jit.load)
  • ONNX (.onnx)         — wczytywany przez onnxruntime.InferenceSession

Format wykrywany automatycznie po rozszerzeniu pliku, lub wybór ręczny.

Wzorzec gałęzi ONNX inspirowany Deepness'owym map_processor / model_base.py
(onnxruntime.InferenceSession + provider CPU). Nie używamy klas Deepness ani
nie wymagamy zainstalowanej wtyczki Deepness — implementacja jest własna,
minimalna i niezależna.

Notatka o pełnej funkcjonalności wtyczki Vectorization Bridge:
  • SAGA Next Gen — WYMAGANA dla algorytmów klasyfikacji, k-means, edge detection.
  • GeoAI       — ZALECANA (opcjonalna) do trenowania modeli przed inferencją.
"""

import math
import os

import numpy as np
from osgeo import gdal, osr

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterEnum,
    QgsProcessingParameterRasterDestination,
    QgsRectangle,
)

from ._compat import (
    deepness_available, deepness_version,
    torch_available, torch_version,
    onnxruntime_available, onnxruntime_version,
)


FORMAT_AUTO = 0
FORMAT_PYTORCH = 1
FORMAT_ONNX = 2

OUTPUT_CLASS_INDICES = 0
OUTPUT_SIGMOID_BINARY = 1
OUTPUT_PROBABILITY = 2


def _detect_format(path: str, format_enum: int) -> str:
    """Zwraca 'pytorch' lub 'onnx' na podstawie wyboru lub rozszerzenia."""
    if format_enum == FORMAT_PYTORCH:
        return 'pytorch'
    if format_enum == FORMAT_ONNX:
        return 'onnx'
    ext = os.path.splitext(path)[1].lower()
    return 'onnx' if ext == '.onnx' else 'pytorch'


def _compute_grid(raster_w, raster_h, tile_size, overlap):
    stride = max(tile_size - overlap, 1)
    x_bins = 1 if raster_w <= tile_size else math.ceil((raster_w - tile_size) / stride) + 1
    y_bins = 1 if raster_h <= tile_size else math.ceil((raster_h - tile_size) / stride) + 1
    return x_bins, y_bins, stride


def _read_tile(provider, n_channels, col_s, row_s, tile_size,
               raster_w, raster_h, extent, px, py):
    col_e = min(col_s + tile_size, raster_w)
    row_e = min(row_s + tile_size, raster_h)
    aw = col_e - col_s
    ah = row_e - row_s

    ox = extent.xMinimum()
    oy = extent.yMaximum()
    tile_ext = QgsRectangle(ox + col_s * px, oy - row_e * py,
                             ox + col_e * px, oy - row_s * py)

    dt_map = {1: np.uint8, 2: np.uint16, 3: np.int16,
               4: np.uint32, 5: np.int32, 6: np.float32, 7: np.float64}
    bands = []
    for b in range(1, n_channels + 1):
        blk = provider.block(b, tile_ext, aw, ah)
        if not blk.isValid():
            bands.append(np.zeros((ah, aw), dtype=np.float32))
        else:
            dt = blk.dataType()
            dtype = dt_map.get(dt, np.float32)
            arr = np.frombuffer(bytes(blk.data()), dtype=dtype).reshape(ah, aw)
            bands.append(arr.astype(np.float32))

    tile = np.stack(bands, axis=0)  # (C, ah, aw)
    if ah < tile_size or aw < tile_size:
        padded = np.zeros((n_channels, tile_size, tile_size), dtype=np.float32)
        padded[:, :ah, :aw] = tile
        return padded, aw, ah
    return tile, aw, ah


def _normalize_output(arr) -> np.ndarray:
    """
    Sprowadza wyjście modelu do tablicy (C, H, W) float32.
    Akceptuje shape: (1, C, H, W), (C, H, W), (1, H, W), (H, W).
    """
    if arr.ndim == 4:                    # (1, C, H, W)
        return arr[0].astype(np.float32)
    if arr.ndim == 3:                    # (C, H, W) lub (1, H, W)
        return arr.astype(np.float32)
    if arr.ndim == 2:                    # (H, W) — single channel logit/prob
        return arr[np.newaxis, ...].astype(np.float32)
    raise ValueError(f'Nieobsługiwany shape wyjścia: {arr.shape}')


class PthInferenceAlgorithm(QgsProcessingAlgorithm):

    INPUT = 'INPUT'
    MODEL_FILE = 'MODEL_FILE'
    MODEL_FORMAT = 'MODEL_FORMAT'
    TILE_SIZE = 'TILE_SIZE'
    OVERLAP = 'OVERLAP'
    INPUT_CHANNELS = 'INPUT_CHANNELS'
    OUTPUT_TYPE = 'OUTPUT_TYPE'
    OUTPUT = 'OUTPUT'

    FORMAT_OPTIONS = [
        'Auto (po rozszerzeniu pliku)',
        'PyTorch (.pth / .pt)',
        'ONNX (.onnx)',
    ]
    OUTPUT_TYPE_OPTIONS = [
        'Indeksy klas (argmax po pasmach, Int32)',
        'Sigmoid binary (próg 0.5, UInt8)',
        'Softmax probability (Float32)',
    ]

    def name(self):
        return 'pth_inference'

    def displayName(self):
        return 'Inferencja modelu (PyTorch / ONNX)'

    def group(self):
        return 'Deep learning'

    def groupId(self):
        return 'deeplearning'

    def shortHelpString(self):
        tv = torch_version()
        ov = onnxruntime_version()
        t_status = f'✓ PyTorch {tv}' if tv else '✗ PyTorch niedostępny'
        o_status = f'✓ onnxruntime {ov}' if ov else '✗ onnxruntime niedostępny'
        d_ver = deepness_version()
        d_note = (f'Deepness {d_ver} wykryty (komplementarny — obsługuje ONNX z własnym UI).'
                  if d_ver else 'Deepness niedostępny (opcjonalny).')
        return (
            'Uruchamia wytrenowany model segmentacji/klasyfikacji pikseli '
            'na wejściowym rastrze.\n\n'
            'Obsługiwane formaty modelu:\n'
            '  • PyTorch  .pth / .pt   — wymagany pakiet torch\n'
            '  • ONNX     .onnx        — wymagany pakiet onnxruntime\n'
            'Format wykrywany automatycznie po rozszerzeniu (lub wybór ręczny).\n\n'
            'PyTorch: model musi być zapisany jako pełny obiekt:\n'
            '    torch.save(model, "model.pth")\n'
            '(fallback: torch.jit.load dla TorchScript .pt).\n\n'
            'Typy wyjścia:\n'
            '  • Indeksy klas — argmax po wymiarze klasowym (Int32, nodata=-1)\n'
            '  • Sigmoid binary — model 1-kanałowy → próg 0.5 (UInt8 0/1)\n'
            '  • Softmax probability — Float32 z wartością prawdopodobieństwa\n\n'
            'OVERLAP > 0 — kafle nakładają się; logity są uśredniane przed\n'
            'argmax/threshold (lepsze wyniki na granicach kafli).\n\n'
            f'Status: {t_status}; {o_status}\n'
            f'{d_note}\n\n'
            'Notatka — pełna funkcjonalność wtyczki Vectorization Bridge:\n'
            '  • Dane treningowe (kafle + maski) przygotujesz w "Cięcie rastra na kafle".\n'
            '  • SAGA Next Gen — WYMAGANA dla pełnej funkcjonalności wtyczki.\n'
            '  • GeoAI — ZALECANA (opcjonalna) do trenowania modeli.'
        )

    def createInstance(self):
        return PthInferenceAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT, 'Wejściowy raster'))

        self.addParameter(QgsProcessingParameterFile(
            self.MODEL_FILE, 'Plik modelu (.pth, .pt lub .onnx)',
            fileFilter='Modele ML (*.pth *.pt *.onnx);;Wszystkie pliki (*.*)'))

        self.addParameter(QgsProcessingParameterEnum(
            self.MODEL_FORMAT, 'Format modelu',
            options=self.FORMAT_OPTIONS,
            defaultValue=FORMAT_AUTO))

        self.addParameter(QgsProcessingParameterNumber(
            self.TILE_SIZE, 'Rozmiar kafla (piksele)',
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=256, minValue=32))

        self.addParameter(QgsProcessingParameterNumber(
            self.OVERLAP, 'Zakładka kafli (piksele)',
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=0, minValue=0))

        self.addParameter(QgsProcessingParameterNumber(
            self.INPUT_CHANNELS, 'Liczba kanałów wejściowych',
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=3, minValue=1))

        self.addParameter(QgsProcessingParameterEnum(
            self.OUTPUT_TYPE, 'Typ wyjścia',
            options=self.OUTPUT_TYPE_OPTIONS,
            defaultValue=OUTPUT_CLASS_INDICES))

        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT, 'Wyjściowy raster'))

    def checkParameterValues(self, parameters, context):
        model_path = self.parameterAsString(parameters, self.MODEL_FILE, context)
        fmt_enum = self.parameterAsEnum(parameters, self.MODEL_FORMAT, context)
        tile_size = self.parameterAsInt(parameters, self.TILE_SIZE, context)
        overlap = self.parameterAsInt(parameters, self.OVERLAP, context)

        if overlap >= tile_size:
            return False, 'Zakładka musi być mniejsza niż rozmiar kafla.'

        if model_path:
            fmt = _detect_format(model_path, fmt_enum)
            if fmt == 'pytorch' and not torch_available():
                return False, ('Wybrany format PyTorch, ale brak biblioteki torch. '
                               'Zainstaluj torch lub przełącz format na ONNX.\n'
                               'https://pytorch.org/get-started/locally/')
            if fmt == 'onnx' and not onnxruntime_available():
                return False, ('Wybrany format ONNX, ale brak biblioteki onnxruntime. '
                               'Zainstaluj: pip install onnxruntime')
        return super().checkParameterValues(parameters, context)

    def processAlgorithm(self, parameters, context, feedback):
        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        model_path = self.parameterAsString(parameters, self.MODEL_FILE, context)
        fmt_enum = self.parameterAsEnum(parameters, self.MODEL_FORMAT, context)
        tile_size = self.parameterAsInt(parameters, self.TILE_SIZE, context)
        overlap = self.parameterAsInt(parameters, self.OVERLAP, context)
        n_channels = self.parameterAsInt(parameters, self.INPUT_CHANNELS, context)
        output_type = self.parameterAsEnum(parameters, self.OUTPUT_TYPE, context)
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        fmt = _detect_format(model_path, fmt_enum)
        feedback.pushInfo(f'Format modelu: {fmt.upper()}')

        if deepness_available():
            feedback.pushInfo(
                f'[Deepness {deepness_version()}] Wykryty (komplementarny).')

        # --- Załadowanie modelu ---
        runner = self._load_runner(fmt, model_path, feedback)
        if runner is None:
            return {}

        provider = layer.dataProvider()
        raster_w = provider.xSize()
        raster_h = provider.ySize()
        extent = layer.extent()
        px = extent.width() / raster_w
        py = extent.height() / raster_h

        x_bins, y_bins, stride = _compute_grid(raster_w, raster_h, tile_size, overlap)
        total = x_bins * y_bins
        feedback.pushInfo(
            f'Raster {raster_w}×{raster_h}; kafle {x_bins}×{y_bins} = {total} '
            f'(tile={tile_size}, overlap={overlap}, stride={stride})')

        # Akumulator dla overlap > 0; w przeciwnym razie piszemy bezpośrednio do GDAL.
        use_accum = overlap > 0
        accum = None        # (C, H, W) float32 — sumy logitów
        counter = None      # (H, W) float32 — licznik kafli

        srs = osr.SpatialReference()
        srs.ImportFromWkt(layer.crs().toWkt())
        driver = gdal.GetDriverByName('GTiff')

        # Wyjście otwierane lazily — po pierwszym kaflu wiemy ile pasm ma model
        out_ds = None
        out_band = None
        n_out_classes = None

        idx = 0
        for row in range(y_bins):
            for col in range(x_bins):
                if feedback.isCanceled():
                    if out_ds is not None:
                        out_ds.FlushCache()
                        out_ds = None
                    return {}

                col_s = col * stride
                row_s = row * stride

                tile_arr, aw, ah = _read_tile(
                    provider, n_channels, col_s, row_s,
                    tile_size, raster_w, raster_h, extent, px, py)

                try:
                    raw_out = runner(tile_arr)  # numpy array
                except Exception as e:
                    feedback.reportError(
                        f'Błąd inferencji na kaflu ({row},{col}): {e}',
                        fatalError=True)
                    if out_ds is not None:
                        out_ds.FlushCache()
                        out_ds = None
                    return {}

                model_out = _normalize_output(raw_out)  # (C, H, W) float32
                out_h, out_w = model_out.shape[1], model_out.shape[2]

                # Inicjalizacja akumulatora / pliku po pierwszym kaflu
                if use_accum and accum is None:
                    n_out_classes = model_out.shape[0]
                    accum = np.zeros((n_out_classes, raster_h, raster_w),
                                     dtype=np.float32)
                    counter = np.zeros((raster_h, raster_w), dtype=np.float32)
                if out_ds is None and not use_accum:
                    n_out_classes = model_out.shape[0]
                    out_ds, out_band = self._create_output(
                        driver, output_path, raster_w, raster_h,
                        srs, extent, px, py, output_type)

                if use_accum:
                    valid_h = min(out_h, ah, raster_h - row_s)
                    valid_w = min(out_w, aw, raster_w - col_s)
                    accum[:, row_s:row_s + valid_h, col_s:col_s + valid_w] += \
                        model_out[:, :valid_h, :valid_w]
                    counter[row_s:row_s + valid_h, col_s:col_s + valid_w] += 1.0
                else:
                    tile_pred = self._postprocess(model_out, output_type)
                    valid = tile_pred[:ah, :aw]
                    out_band.WriteArray(valid, col_s, row_s)

                idx += 1
                feedback.setProgress(int(100 * idx / total))

        # --- Stitching dla overlap > 0 ---
        if use_accum:
            feedback.pushInfo('Stitching kafli (uśrednianie nakładających się rejonów)...')
            counter = np.maximum(counter, 1.0)
            avg = accum / counter[np.newaxis, ...]
            full_pred = self._postprocess(avg, output_type)
            out_ds, out_band = self._create_output(
                driver, output_path, raster_w, raster_h,
                srs, extent, px, py, output_type)
            out_band.WriteArray(full_pred)

        if out_ds is not None:
            out_ds.FlushCache()
            out_ds = None

        feedback.pushInfo(f'Inferencja zakończona → {output_path}')
        return {self.OUTPUT: output_path}

    # ------------------------------------------------------------------
    def _load_runner(self, fmt, model_path, feedback):
        """
        Zwraca callable runner(tile_chw_float32) -> numpy array, lub None gdy błąd.
        Tile podawany w kształcie (C, H, W) float32 (znormalizowane 0..1).
        """
        if fmt == 'pytorch':
            try:
                import torch
            except ImportError:
                feedback.reportError(
                    'Brak PyTorch. Zainstaluj torch lub przełącz format na ONNX.',
                    fatalError=True)
                return None
            feedback.pushInfo(f'[PyTorch {torch.__version__}] Ładowanie modelu...')
            model = None
            try:
                model = torch.load(model_path, map_location='cpu', weights_only=False)
                feedback.pushInfo('Model załadowany przez torch.load().')
            except Exception as e1:
                feedback.pushInfo(f'torch.load nie powiodło się ({e1}), próba torch.jit.load...')
                try:
                    model = torch.jit.load(model_path, map_location='cpu')
                    feedback.pushInfo('Model załadowany przez torch.jit.load() (TorchScript).')
                except Exception as e2:
                    feedback.reportError(
                        f'Nie udało się załadować modelu PyTorch.\n'
                        f'  torch.load: {e1}\n  torch.jit.load: {e2}',
                        fatalError=True)
                    return None
            model.eval()

            def run(tile_chw):
                tensor = torch.from_numpy(tile_chw / 255.0).float().unsqueeze(0)
                with torch.no_grad():
                    out = model(tensor)
                if hasattr(out, 'cpu'):
                    return out.cpu().numpy()
                return np.asarray(out)
            return run

        # ONNX
        try:
            import onnxruntime as ort
        except ImportError:
            feedback.reportError(
                'Brak onnxruntime. Zainstaluj: pip install onnxruntime',
                fatalError=True)
            return None
        feedback.pushInfo(f'[onnxruntime {ort.__version__}] Ładowanie modelu...')
        try:
            session = ort.InferenceSession(
                model_path, providers=['CPUExecutionProvider'])
        except Exception as e:
            feedback.reportError(f'Nie udało się załadować modelu ONNX: {e}',
                                  fatalError=True)
            return None
        input_name = session.get_inputs()[0].name
        feedback.pushInfo(f'Wejście modelu: "{input_name}"')

        def run(tile_chw):
            x = (tile_chw / 255.0).astype(np.float32)[np.newaxis, ...]  # (1, C, H, W)
            out = session.run(None, {input_name: x})[0]
            return np.asarray(out)
        return run

    # ------------------------------------------------------------------
    def _postprocess(self, chw_logits, output_type):
        """
        chw_logits: (C, H, W) float32 — surowe wyjście modelu (logity / prob).
        Zwraca tablicę 2D zgodną z output_type.
        """
        if output_type == OUTPUT_CLASS_INDICES:
            # argmax po C; gdy C==1 → wszystko 0, ale to interpretowane jako 1 klasa
            return np.argmax(chw_logits, axis=0).astype(np.int32)
        if output_type == OUTPUT_SIGMOID_BINARY:
            # bierzemy pierwszy kanał (zakładamy model binary single-output)
            ch = chw_logits[0]
            # jeżeli wartości wyglądają na logity (poza [0,1]) — zastosuj sigmoid
            if ch.min() < 0.0 or ch.max() > 1.0:
                ch = 1.0 / (1.0 + np.exp(-ch))
            return (ch >= 0.5).astype(np.uint8)
        # OUTPUT_PROBABILITY
        if chw_logits.shape[0] == 1:
            ch = chw_logits[0]
            if ch.min() < 0.0 or ch.max() > 1.0:
                ch = 1.0 / (1.0 + np.exp(-ch))
            return ch.astype(np.float32)
        # softmax po C, weź max prob
        m = np.max(chw_logits, axis=0, keepdims=True)
        e = np.exp(chw_logits - m)
        s = e / np.sum(e, axis=0, keepdims=True)
        return np.max(s, axis=0).astype(np.float32)

    def _create_output(self, driver, path, w, h, srs, extent, px, py, output_type):
        if output_type == OUTPUT_CLASS_INDICES:
            gdal_dtype = gdal.GDT_Int32
        elif output_type == OUTPUT_SIGMOID_BINARY:
            gdal_dtype = gdal.GDT_Byte
        else:
            gdal_dtype = gdal.GDT_Float32
        ds = driver.Create(path, w, h, 1, gdal_dtype, options=['COMPRESS=LZW'])
        ds.SetGeoTransform((extent.xMinimum(), px, 0, extent.yMaximum(), 0, -py))
        ds.SetProjection(srs.ExportToWkt())
        band = ds.GetRasterBand(1)
        if output_type == OUTPUT_CLASS_INDICES:
            band.SetNoDataValue(-1)
        return ds, band
