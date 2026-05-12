import os

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMenu, QToolButton
from qgis.core import QgsApplication

from .provider import VectorizationBridgeProvider


ICON_PATH = os.path.join(os.path.dirname(__file__), 'icon.svg')

# (etykieta podgrupy, [(algo_id, etykieta), ...])
_MENU_GROUPS = [
    ('Diagnostyka', [
        ('vectorization_bridge:check_dependencies', 'Sprawdź zależności'),
    ]),
    ('Deep learning', [
        ('vectorization_bridge:tile_export',  'Cięcie rastra na kafle (ML)'),
        ('vectorization_bridge:pth_inference', 'Inferencja modelu (PyTorch / ONNX)'),
    ]),
    ('Klasyfikacja', [
        ('vectorization_bridge:rgb_classification',  'Klasyfikacja pikseli RGB (Euklidesowa / CIEDE2000)'),
        ('vectorization_bridge:artificialneuralnetworkclassification', 'SAGA: Sieć neuronowa'),
        ('vectorization_bridge:decisiontreeclassification',            'SAGA: Drzewo decyzyjne'),
        ('vectorization_bridge:logisticregressionclassification',      'SAGA: Regresja logistyczna'),
        ('vectorization_bridge:normalbayesclassification',             'SAGA: Normal Bayes'),
        ('vectorization_bridge:randomforestclassification',            'SAGA: Random Forest'),
        ('vectorization_bridge:supportvectormachineclassification',    'SAGA: SVM'),
        ('vectorization_bridge:kmeansclusteringforrasters',            'SAGA: K-Means dla rastrów'),
    ]),
    ('Przetwarzanie obrazu', [
        ('vectorization_bridge:edge_detection',          'Wykrywanie krawędzi (Canny/Sobel/SAGA)'),
        ('vectorization_bridge:rasterskeletonization',   'Szkieletyzacja rastra (SAGA)'),
        ('vectorization_bridge:watershed',               'Transformata wododziałowa'),
        ('vectorization_bridge:region_growing',          'Seeded Region Growing'),
        ('vectorization_bridge:split_rgb_bands',         'Rozdziel raster RGB na pasma'),
    ]),
    ('SAGA — Wykrywanie krawędzi', [
        ('vectorization_bridge:edgedetectionvigra',                       'Vigra'),
        ('vectorization_bridge:womblingformultiplefeaturesedgedetection', 'Wombling'),
    ]),
]


def _open_algo_dialog(algo_id):
    """Otwiera okno parametrów algorytmu Processing po jego ID."""
    try:
        from processing import execAlgorithmDialog
        execAlgorithmDialog(algo_id, {})
    except Exception:
        try:
            from qgis.utils import iface
            iface.actionShowProcessingToolbox().trigger()
        except Exception:
            pass


class VectorizationBridgePlugin:
    def __init__(self, iface):
        self.iface = iface
        self.provider = None
        self.menu = None
        self.tool_button = None
        self.toolbar = None            # dedykowany QToolBar wtyczki
        self.toolbar_action = None     # akcja widget'u w toolbarze (zwracana przez addWidget)
        self.plugin_menu_action = None # akcja w menu Wtyczki

    def initProcessing(self):
        self.provider = VectorizationBridgeProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()

        icon = QIcon(ICON_PATH)

        # Zbuduj menu z pogrupowanymi akcjami
        self.menu = QMenu('Vectorization Bridge', self.iface.mainWindow())
        for group_label, items in _MENU_GROUPS:
            sub = self.menu.addMenu(group_label)
            for algo_id, label in items:
                act = sub.addAction(label)
                act.triggered.connect(
                    lambda _checked=False, aid=algo_id: _open_algo_dialog(aid)
                )

        # Toolbar: QToolButton z trybem InstantPopup, by klik rozwijał menu
        self.tool_button = QToolButton(self.iface.mainWindow())
        self.tool_button.setIcon(icon)
        self.tool_button.setText('Vectorization Bridge')
        self.tool_button.setToolTip(
            'Vectorization Bridge — kliknij, aby wybrać algorytm'
        )
        self.tool_button.setPopupMode(QToolButton.InstantPopup)
        self.tool_button.setMenu(self.menu)

        # Dedykowany QToolBar wtyczki (osobna grupa w pasku narzędzi, nie "Wtyczki")
        self.toolbar = self.iface.addToolBar('Vectorization Bridge')
        self.toolbar.setObjectName('VectorizationBridgeToolbar')
        self.toolbar_action = self.toolbar.addWidget(self.tool_button)

        # Akcja w menu Wtyczki — pokazuje to samo menu
        self.plugin_menu_action = QAction(
            icon, 'Vectorization Bridge', self.iface.mainWindow()
        )
        self.plugin_menu_action.setMenu(self.menu)
        self.iface.addPluginToMenu('&Vectorization Bridge', self.plugin_menu_action)

    def unload(self):
        if self.plugin_menu_action is not None:
            try:
                self.iface.removePluginMenu(
                    '&Vectorization Bridge', self.plugin_menu_action
                )
            except Exception:
                pass
            self.plugin_menu_action = None

        if self.toolbar_action is not None:
            if self.toolbar is not None:
                try:
                    self.toolbar.removeAction(self.toolbar_action)
                except Exception:
                    pass
            self.toolbar_action = None

        if self.tool_button is not None:
            self.tool_button.deleteLater()
            self.tool_button = None

        if self.toolbar is not None:
            self.toolbar.deleteLater()
            self.toolbar = None

        if self.menu is not None:
            self.menu.deleteLater()
            self.menu = None

        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
