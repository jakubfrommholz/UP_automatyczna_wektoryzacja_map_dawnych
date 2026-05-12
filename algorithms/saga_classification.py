"""
Algorytmy klasyfikacji rastrów — proxy do SAGA Next Gen.

Każda klasa to cienki wrapper, który deleguje wykonanie do odpowiedniego
algorytmu SAGA poprzez processing.run() (patrz SagaProxyAlgorithm).
"""

from .saga_proxy import SagaProxyAlgorithm


_GROUP_NAME = 'Klasyfikacja'
_GROUP_ID = 'classification'


class SagaArtificialNeuralNetworkAlgorithm(SagaProxyAlgorithm):
    SAGA_ALGO_ID = 'sagang:artificialneuralnetworkclassification'
    DISPLAY_NAME = 'Klasyfikacja: Sieć neuronowa (SAGA)'
    GROUP_NAME = _GROUP_NAME
    GROUP_ID = _GROUP_ID


class SagaDecisionTreeAlgorithm(SagaProxyAlgorithm):
    SAGA_ALGO_ID = 'sagang:decisiontreeclassification'
    DISPLAY_NAME = 'Klasyfikacja: Drzewo decyzyjne (SAGA)'
    GROUP_NAME = _GROUP_NAME
    GROUP_ID = _GROUP_ID


class SagaLogisticRegressionAlgorithm(SagaProxyAlgorithm):
    SAGA_ALGO_ID = 'sagang:logisticregressionclassification'
    DISPLAY_NAME = 'Klasyfikacja: Regresja logistyczna (SAGA)'
    GROUP_NAME = _GROUP_NAME
    GROUP_ID = _GROUP_ID


class SagaNormalBayesAlgorithm(SagaProxyAlgorithm):
    SAGA_ALGO_ID = 'sagang:normalbayesclassification'
    DISPLAY_NAME = 'Klasyfikacja: Normal Bayes (SAGA)'
    GROUP_NAME = _GROUP_NAME
    GROUP_ID = _GROUP_ID


class SagaRandomForestAlgorithm(SagaProxyAlgorithm):
    SAGA_ALGO_ID = 'sagang:randomforestclassification'
    DISPLAY_NAME = 'Klasyfikacja: Random Forest (SAGA)'
    GROUP_NAME = _GROUP_NAME
    GROUP_ID = _GROUP_ID


class SagaSupportVectorMachineAlgorithm(SagaProxyAlgorithm):
    SAGA_ALGO_ID = 'sagang:supportvectormachineclassification'
    DISPLAY_NAME = 'Klasyfikacja: SVM (SAGA)'
    GROUP_NAME = _GROUP_NAME
    GROUP_ID = _GROUP_ID
