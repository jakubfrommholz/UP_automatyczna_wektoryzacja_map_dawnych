"""
Check Dependencies — diagnostyka zależności wtyczki.

Raportuje dostępność wszystkich wtyczek i bibliotek wymaganych lub
opcjonalnych dla algorytmów Vectorization Bridge:
  • Deepness       — integracja cięcia rastra (opcjonalne)
  • GeoAI          — trenowanie modeli (opcjonalne, informacyjnie)
  • SAGA NG        — wykrywanie krawędzi (opcjonalne)
  • opencv-python  — wymagane dla Canny/Sobel i eksportu PNG
  • PyTorch        — wymagane dla inferencji .pth
  • numpy          — zawsze dostępne w QGIS
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterBoolean,
    QgsProcessingOutputString,
    QgsApplication,
)

from ._compat import (
    deepness_available, deepness_version,
    geoai_available, geoai_version,
    saga_available, saga_find_edge_algorithm, saga_find_rf_algorithm,
    cv2_available, cv2_version,
    torch_available, torch_version,
    onnxruntime_available, onnxruntime_version,
)


def _status(ok: bool) -> str:
    return '✓' if ok else '✗'


class CheckDependenciesAlgorithm(QgsProcessingAlgorithm):

    VERBOSE = 'VERBOSE'
    REPORT = 'REPORT'

    def name(self):
        return 'check_dependencies'

    def displayName(self):
        return 'Sprawdź zależności (Check Dependencies)'

    def group(self):
        return 'Diagnostyka'

    def groupId(self):
        return 'diagnostics'

    def shortHelpString(self):
        return (
            'Sprawdza dostępność wszystkich wtyczek i bibliotek Python '
            'wymaganych lub opcjonalnych dla algorytmów Vectorization Bridge.\n\n'
            'Raport pojawi się w panelu wyników Processing oraz w logu.'
        )

    def createInstance(self):
        return CheckDependenciesAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterBoolean(
            self.VERBOSE, 'Szczegółowy raport (lista algorytmów SAGA)',
            defaultValue=False))

        self.addOutput(QgsProcessingOutputString(
            self.REPORT, 'Raport tekstowy'))

    def processAlgorithm(self, parameters, context, feedback):
        verbose = self.parameterAsBoolean(parameters, self.VERBOSE, context)
        lines = []

        def log(msg):
            feedback.pushInfo(msg)
            lines.append(msg)

        log('=' * 60)
        log('  Vectorization Bridge — Raport zależności')
        log('=' * 60)

        # --- Wymagane biblioteki Python ---
        log('')
        log('[ Biblioteki Python ]')

        import numpy as np
        log(f'  {_status(True)} numpy {np.__version__}  (wbudowane w QGIS)')

        cv2_ok = cv2_available()
        cv2_v = cv2_version() or 'niedostępne'
        log(f'  {_status(cv2_ok)} opencv-python (cv2)  {cv2_v}')
        if not cv2_ok:
            log('     → Wymagane dla: Cięcie rastra (PNG), Wykrywanie krawędzi')
            log('     → Instalacja: pip install opencv-python')

        torch_ok = torch_available()
        torch_v = torch_version() or 'niedostępne'
        log(f'  {_status(torch_ok)} torch (PyTorch)  {torch_v}')
        if not torch_ok:
            log('     → Wymagane dla: Inferencja modelu .pth / .pt')
            log('     → Instalacja: https://pytorch.org/get-started/locally/')

        ort_ok = onnxruntime_available()
        ort_v = onnxruntime_version() or 'niedostępne'
        log(f'  {_status(ort_ok)} onnxruntime  {ort_v}')
        if not ort_ok:
            log('     → Wymagane dla: Inferencja modelu w formacie ONNX (.onnx)')
            log('     → Instalacja: pip install onnxruntime')

        # --- Wtyczki QGIS ---
        log('')
        log('[ Wtyczki QGIS ]')

        saga_ok = saga_available()
        log(f'  {_status(saga_ok)} SAGA Next Gen Processing Provider  [WYMAGANA]')
        if saga_ok:
            saga_id, saga_label = saga_find_edge_algorithm()
            if saga_id:
                log(f'     → Aktywna integracja: algorytm "{saga_label}" ({saga_id})')
                log('       dostępny jako metoda w "Wykrywanie krawędzi"')
            log('     → 9 algorytmów proxy (klasyfikacja, k-means, edge detection)')
            log('       dostępnych w grupach "SAGA — *"')

            if verbose:
                log('')
                log('     Algorytmy SAGA zawierające "edge" lub "gradient":')
                provider = QgsApplication.processingRegistry().providerById('sagang')
                if provider:
                    found = [(a.id(), a.displayName()) for a in provider.algorithms()
                             if any(kw in a.name().lower()
                                    for kw in ('edge', 'gradient', 'filter', 'deriv'))]
                    for aid, aname in found[:20]:
                        log(f'       - {aid}: {aname}')
                    if not found:
                        log('       (brak)')
        else:
            log('     → WYMAGANA dla pełnej funkcjonalności wtyczki Vectorization Bridge')
            log('       (klasyfikacja pikseli, k-means, edge detection — łącznie 10 algo).')
            log('     → Instalacja: Plugins > Manage and Install > "SAGA Next Gen".')

        g_ok = geoai_available()
        g_ver = geoai_version() or 'niedostępna'
        log(f'  {_status(g_ok)} GeoAI  {g_ver}  [ZALECANA]')
        if g_ok:
            log('     → Dostępny do trenowania modeli segmentacji')
        else:
            log('     → ZALECANA (opcjonalna). Użyj do trenowania modeli przed inferencją.')

        d_ok = deepness_available()
        d_ver = deepness_version() or 'niedostępna'
        log(f'  {_status(d_ok)} Deepness  {d_ver}  [opcjonalna]')
        if d_ok:
            log('     → Aktywna integracja: zaokrąglanie extent w "Cięcie rastra na kafle"')
            log('     → Komplementarna — Deepness ma własne UI dla modeli ONNX')
        else:
            log('     → Opcjonalna. Bez niej algorytmy działają we własnym trybie.')

        # --- Podsumowanie algorytmów ---
        log('')
        log('[ Dostępność algorytmów ]')

        def algo_status(name, reqs, ok):
            s = _status(ok)
            req_str = ', '.join(reqs) if reqs else 'brak'
            log(f'  {s} {name}  (wymaga: {req_str})')

        algo_status('Cięcie rastra na kafle (ML)', ['cv2'], cv2_ok)
        algo_status('Inferencja modelu — PyTorch (.pth/.pt)', ['torch'], torch_ok)
        algo_status('Inferencja modelu — ONNX (.onnx)', ['onnxruntime'], ort_ok)
        algo_status('Wykrywanie krawędzi — Canny/Sobel', ['cv2'], cv2_ok)
        algo_status('Wykrywanie krawędzi — SAGA',
                    ['SAGA NG'],
                    saga_ok and saga_find_edge_algorithm()[0] is not None)
        algo_status('Klasyfikacja pikseli RGB (Euklidesowa / CIEDE2000)', ['numpy'], True)
        rf_algo_id, _ = saga_find_rf_algorithm()
        algo_status('Klasyfikacja Random Forest (SAGA)', ['SAGA NG (OpenCV RF)'], rf_algo_id is not None)
        algo_status('Algorytmy proxy SAGA (9 szt. — klasyfikacja, k-means, edge)',
                    ['SAGA NG'], saga_ok)

        log('')
        log('=' * 60)

        report = '\n'.join(lines)
        return {self.REPORT: report}
