"""
Compatibility helpers — detect installed plugins and expose their utilities.

All functions in this module are safe to call regardless of whether the
target plugin is installed. They return None / False when unavailable and
never raise ImportError to the caller.
"""

from qgis.core import QgsApplication


# ---------------------------------------------------------------------------
# Deepness
# ---------------------------------------------------------------------------

def deepness_version():
    """Return Deepness plugin version string, or None if not installed."""
    try:
        import deepness.metadata as _m  # noqa: F401
        # metadata.txt values are exposed via qgis plugin registry
        reg = QgsApplication.instance().pluginRegistry() if hasattr(
            QgsApplication.instance(), 'pluginRegistry') else None
        # Fallback: read metadata.txt directly
        import importlib.util, os
        spec = importlib.util.find_spec('deepness')
        if spec is None:
            return None
        pkg_dir = os.path.dirname(spec.origin)
        meta = os.path.join(pkg_dir, 'metadata.txt')
        if os.path.exists(meta):
            with open(meta, encoding='utf-8') as f:
                for line in f:
                    if line.startswith('version='):
                        return line.split('=', 1)[1].strip()
        return 'unknown'
    except Exception:
        return None


def deepness_available():
    return deepness_version() is not None


def deepness_round_extent(extent, rlayer):
    """
    Round a QgsRectangle to the raster layer's pixel grid using Deepness.
    Returns the rounded extent, or the original extent if Deepness is unavailable.
    """
    try:
        from deepness.processing.extent_utils import round_extent_to_rlayer_grid
        return round_extent_to_rlayer_grid(extent, rlayer)
    except Exception:
        return extent


def deepness_units_per_pixel(rlayer):
    """
    Return units-per-pixel for the raster layer's CRS using Deepness utility.
    Falls back to own calculation.
    """
    try:
        # Deepness calculates this in MapProcessor.__init__; replicate the logic
        extent = rlayer.extent()
        provider = rlayer.dataProvider()
        return extent.width() / provider.xSize()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GeoAI
# ---------------------------------------------------------------------------

def geoai_version():
    """Return GeoAI plugin version string, or None if not installed."""
    try:
        import importlib.util, os
        spec = importlib.util.find_spec('geoai')
        if spec is None:
            # Try the QGIS plugin package name
            spec = importlib.util.find_spec('geoai_plugin')
        if spec is None:
            return None
        pkg_dir = os.path.dirname(spec.origin)
        meta = os.path.join(pkg_dir, 'metadata.txt')
        if os.path.exists(meta):
            with open(meta, encoding='utf-8') as f:
                for line in f:
                    if line.startswith('version='):
                        return line.split('=', 1)[1].strip()
        return 'unknown'
    except Exception:
        return None


def geoai_available():
    return geoai_version() is not None


# ---------------------------------------------------------------------------
# SAGA Next Gen — Processing provider
# ---------------------------------------------------------------------------

def saga_available():
    """Return True if SAGA Next Gen Processing provider is loaded."""
    reg = QgsApplication.processingRegistry()
    return reg.providerById('sagang') is not None


def saga_find_rf_algorithm():
    """Find SAGA OpenCV Random Forest classification algorithm.
    Returns (algo_id, display_name) or (None, None).
    """
    reg = QgsApplication.processingRegistry()
    algo_id = 'sagang:imagery_opencv_randomforestclassification'
    if reg.algorithmById(algo_id) is not None:
        return algo_id, 'OpenCV Random Forest Classification'
    return None, None


def saga_find_edge_algorithm():
    """
    Find the best available SAGA edge-detection algorithm.
    Returns (algorithm_id, display_name) or (None, None).
    """
    reg = QgsApplication.processingRegistry()
    candidates = [
        ('sagang:edgedetectionfilter',          'SAGA Edge Detection Filter'),
        ('sagang:morphologicalfilter',           'SAGA Morphological Filter'),
        ('sagang:gradientvikvectorfield',        'SAGA Gradient (Vik)'),
        ('sagang:derivativesofarasterlayer',     'SAGA Raster Derivatives'),
        ('sagang:standarddeviationoverlap',      'SAGA Std Dev Overlap'),
    ]
    for algo_id, label in candidates:
        if reg.algorithmById(algo_id) is not None:
            return algo_id, label
    # Generic search: look for any sagang algo with 'edge' in name
    provider = reg.providerById('sagang')
    if provider:
        for algo in provider.algorithms():
            name_lower = algo.name().lower()
            if 'edge' in name_lower or 'gradient' in name_lower:
                return algo.id(), algo.displayName()
    return None, None


# ---------------------------------------------------------------------------
# Python library availability
# ---------------------------------------------------------------------------

def cv2_available():
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


def torch_available():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def torch_version():
    try:
        import torch
        return torch.__version__
    except ImportError:
        return None


def cv2_version():
    try:
        import cv2
        return cv2.__version__
    except ImportError:
        return None


def onnxruntime_available():
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def onnxruntime_version():
    try:
        import onnxruntime
        return onnxruntime.__version__
    except ImportError:
        return None
