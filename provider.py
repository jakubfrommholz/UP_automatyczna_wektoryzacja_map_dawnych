import os

from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsProcessingProvider


ICON_PATH = os.path.join(os.path.dirname(__file__), 'icon.svg')


class VectorizationBridgeProvider(QgsProcessingProvider):

    def id(self):
        return 'vectorization_bridge'

    def name(self):
        return 'Vectorization Bridge'

    def longName(self):
        return 'Vectorization Bridge — wektoryzacja z integracją Deepness i SAGA'

    def icon(self):
        return QIcon(ICON_PATH)

    def loadAlgorithms(self):
        from .algorithms.tile_export import TileExportAlgorithm
        from .algorithms.pth_inference import PthInferenceAlgorithm
        from .algorithms.edge_detection import EdgeDetectionAlgorithm
        from .algorithms.rgb_classification import RgbClassificationAlgorithm
        from .algorithms.check_dependencies import CheckDependenciesAlgorithm
        from .algorithms.saga_classification import (
            SagaArtificialNeuralNetworkAlgorithm,
            SagaDecisionTreeAlgorithm,
            SagaLogisticRegressionAlgorithm,
            SagaNormalBayesAlgorithm,
            SagaRandomForestAlgorithm,
            SagaSupportVectorMachineAlgorithm,
        )
        from .algorithms.saga_clustering import SagaKMeansClusteringAlgorithm
        from .algorithms.saga_edge import (
            SagaEdgeDetectionVigraAlgorithm,
            SagaWomblingEdgeDetectionAlgorithm,
        )
        from .algorithms.saga_image_processing import (
            SagaRasterSkeletonizationAlgorithm,
        )
        from .algorithms.watershed import WatershedAlgorithm
        from .algorithms.region_growing import SeededRegionGrowingAlgorithm
        from .algorithms.split_rgb_bands import SplitRgbBandsAlgorithm

        for cls in (
            TileExportAlgorithm,
            PthInferenceAlgorithm,
            EdgeDetectionAlgorithm,
            RgbClassificationAlgorithm,
            CheckDependenciesAlgorithm,
            SagaArtificialNeuralNetworkAlgorithm,
            SagaDecisionTreeAlgorithm,
            SagaLogisticRegressionAlgorithm,
            SagaNormalBayesAlgorithm,
            SagaRandomForestAlgorithm,
            SagaSupportVectorMachineAlgorithm,
            SagaKMeansClusteringAlgorithm,
            SagaEdgeDetectionVigraAlgorithm,
            SagaWomblingEdgeDetectionAlgorithm,
            SagaRasterSkeletonizationAlgorithm,
            WatershedAlgorithm,
            SeededRegionGrowingAlgorithm,
            SplitRgbBandsAlgorithm,
        ):
            self.addAlgorithm(cls())
