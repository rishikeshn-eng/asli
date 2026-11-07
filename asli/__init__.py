"""asli: detect AI-generated / cloned speech in Indian languages."""


def Detector(*a, **k):
    from .detect import Detector as _D
    return _D(*a, **k)


__version__ = "0.1.0"
