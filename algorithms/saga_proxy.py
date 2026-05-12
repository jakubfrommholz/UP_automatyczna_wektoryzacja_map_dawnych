"""
SagaProxyAlgorithm — wspólna klasa bazowa dla algorytmów, które wyłącznie
delegują wykonanie do konkretnego algorytmu z providera "sagang"
(SAGA Next Gen).

Każda podklasa ustawia trzy atrybuty klasowe:
  • SAGA_ALGO_ID  — pełny identyfikator algorytmu SAGA (np. 'sagang:randomforestclassification')
  • DISPLAY_NAME  — nazwa wyświetlana w Processing Toolbox
  • GROUP_NAME / GROUP_ID — grupa tematyczna w naszym providerze

Mechanizm:
  • initAlgorithm() klonuje definicje parametrów z algorytmu SAGA tak, by
    okno parametrów było identyczne jak natywne okno SAGA.
  • processAlgorithm() przekazuje 1:1 słownik parametrów do processing.run().
  • Gdy SAGA NG nie jest zainstalowane, algorytmy nadal pojawiają się w
    toolboxie, ale checkParameterValues zwraca jasny komunikat błędu.
"""

from qgis.core import (
    QgsApplication,
    QgsProcessingAlgorithm,
)


def _get_saga_algo(algo_id):
    """Zwraca instancję algorytmu SAGA o podanym ID lub None."""
    return QgsApplication.processingRegistry().algorithmById(algo_id)


class SagaProxyAlgorithm(QgsProcessingAlgorithm):

    SAGA_ALGO_ID = ''
    DISPLAY_NAME = ''
    GROUP_NAME = ''
    GROUP_ID = ''

    def name(self):
        return self.SAGA_ALGO_ID.split(':', 1)[-1]

    def displayName(self):
        return self.DISPLAY_NAME

    def group(self):
        return self.GROUP_NAME

    def groupId(self):
        return self.GROUP_ID

    def createInstance(self):
        return type(self)()

    def shortHelpString(self):
        algo = _get_saga_algo(self.SAGA_ALGO_ID)
        if algo is None:
            return (
                f'Cienki proxy algorytmu SAGA: <b>{self.SAGA_ALGO_ID}</b>.<br><br>'
                'Wymaga zainstalowanej wtyczki <b>SAGA Next Gen</b>.<br>'
                'Po instalacji ten algorytm wystawi parametry identyczne jak '
                'oryginalny algorytm SAGA i przekaże je przez processing.run().'
            )
        try:
            base_help = algo.shortHelpString() or ''
        except Exception:
            base_help = ''
        return (
            f'Deleguje do algorytmu SAGA: <b>{self.SAGA_ALGO_ID}</b><br><br>'
            f'{base_help}'
        )

    def initAlgorithm(self, config=None):
        algo = _get_saga_algo(self.SAGA_ALGO_ID)
        if algo is None:
            return
        for p in algo.parameterDefinitions():
            try:
                self.addParameter(p.clone())
            except Exception:
                pass
        # Output definitions z destination params są rejestrowane automatycznie
        # przez addParameter(p.clone()) — nie wolno ponownie dodawać outputów
        # z cudzej instancji algorytmu (use-after-free → crash QGIS).

    def checkParameterValues(self, parameters, context):
        if _get_saga_algo(self.SAGA_ALGO_ID) is None:
            return False, (
                'Wymagana wtyczka "SAGA Next Gen" — nie jest zainstalowana lub '
                'provider "sagang" nie został załadowany.\n'
                'Zainstaluj: Plugins > Manage and Install Plugins > "SAGA Next Gen".'
            )
        return super().checkParameterValues(parameters, context)

    def processAlgorithm(self, parameters, context, feedback):
        if _get_saga_algo(self.SAGA_ALGO_ID) is None:
            feedback.reportError(
                'Wymagana wtyczka "SAGA Next Gen" nie jest zainstalowana.',
                fatalError=True,
            )
            return {}

        import processing  # QGIS processing module

        feedback.pushInfo(
            f'[Vectorization Bridge → SAGA] Deleguję do: {self.SAGA_ALGO_ID}'
        )
        forward = {k: v for k, v in parameters.items() if v is not None}
        result = processing.run(
            self.SAGA_ALGO_ID, forward,
            context=context, feedback=feedback,
        )
        return result
