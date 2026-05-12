"""
Algorytmy klastrowania rastrów — proxy do SAGA Next Gen.
"""

from .saga_proxy import SagaProxyAlgorithm


class SagaKMeansClusteringAlgorithm(SagaProxyAlgorithm):
    SAGA_ALGO_ID = 'sagang:kmeansclusteringforrasters'
    DISPLAY_NAME = 'Klastrowanie K-Means dla rastrów (SAGA)'
    GROUP_NAME = 'Klasyfikacja'
    GROUP_ID = 'classification'
