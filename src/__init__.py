"""
POD vs EDR Storm — density inversion package.

Initializes the Orekit VM and loads orekit-data on first import.
"""

import os as _os

import orekit as _orekit
_orekit.initVM()

from orekit.pyhelpers import setup_orekit_curdir as _setup

# Resolve orekit-data.zip relative to this package
_data_path = _os.path.join(_os.path.dirname(__file__), '..', 'misc', 'orekit-data.zip')
_data_path = _os.path.abspath(_data_path)
_setup(_data_path)
