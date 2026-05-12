"""
Algorytmy wykrywania krawędzi — proxy do SAGA Next Gen.
"""

from .saga_proxy import SagaProxyAlgorithm


_GROUP_NAME = 'SAGA — Wykrywanie krawędzi'
_GROUP_ID = 'saga_edge'


class SagaEdgeDetectionVigraAlgorithm(SagaProxyAlgorithm):
    SAGA_ALGO_ID = 'sagang:edgedetectionvigra'
    DISPLAY_NAME = 'Wykrywanie krawędzi: Vigra (SAGA)'
    GROUP_NAME = _GROUP_NAME
    GROUP_ID = _GROUP_ID


class SagaWomblingEdgeDetectionAlgorithm(SagaProxyAlgorithm):
    SAGA_ALGO_ID = 'sagang:womblingformultiplefeaturesedgedetection'
    DISPLAY_NAME = 'Wykrywanie krawędzi: Wombling (SAGA)'
    GROUP_NAME = _GROUP_NAME
    GROUP_ID = _GROUP_ID
