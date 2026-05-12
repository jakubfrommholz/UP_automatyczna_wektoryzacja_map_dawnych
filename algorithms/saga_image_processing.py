"""
Algorytmy przetwarzania obrazu — proxy do SAGA Next Gen.
"""

from .saga_proxy import SagaProxyAlgorithm


class SagaRasterSkeletonizationAlgorithm(SagaProxyAlgorithm):
    SAGA_ALGO_ID = 'sagang:rasterskeletonization'
    DISPLAY_NAME = 'Szkieletyzacja rastra (SAGA)'
    GROUP_NAME = 'Przetwarzanie obrazu'
    GROUP_ID = 'image_processing'
