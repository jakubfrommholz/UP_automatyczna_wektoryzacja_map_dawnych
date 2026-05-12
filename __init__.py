def classFactory(iface):
    from .plugin import VectorizationBridgePlugin
    return VectorizationBridgePlugin(iface)
