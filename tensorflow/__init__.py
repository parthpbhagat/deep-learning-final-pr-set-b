"""
Local TensorFlow compatibility bridge to bypass Windows 11 Smart App Control
blocking of _pywrap_tensorflow_internal.pyd when Python is installed in User directory.
"""
from . import keras

__version__ = "2.16.1"
__all__ = ["keras", "__version__"]
