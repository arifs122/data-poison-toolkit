"""
Data Poisoning Toolkit - A Comprehensive Framework for ML Security Research

This toolkit provides a complete framework for evaluating the robustness of machine 
learning models against adversarial data poisoning attacks and testing various defense 
mechanisms against such attacks.

Authors:
    - Muhammet Anil Yagiz (Lead Researcher)
    - Muhammed Arif Dayi (Co-Researcher)
    - Dr. Polat Goktas (Principal Investigator)

Presented at BlackHat Europe 2025 | Submitted to IEEE Conference
"""

__version__ = "1.0.0"
__author__ = "Muhammet Anil Yagiz, Muhammed Arif Dayi, Dr. Polat Goktas"
__email__ = ""
__license__ = "MIT"
__status__ = "Production"

# Core modules
from .demo_runner import ExperimentConfig, DatasetConfig, get_config

# Attack modules
from .attacks.base_attack import Attack, BaseAttackConfig, DSPOConfig

# Defense modules  
from .detectors.base_defense import Defense

# Make key classes available at package level
__all__ = [
    "ExperimentConfig",
    "DatasetConfig", 
    "get_config",
    "Attack",
    "BaseAttackConfig",
    "DSPOConfig",
    "Defense",
    "__version__",
    "__author__",
]

# Package metadata
PACKAGE_NAME = "datapoisontoolkit"
DESCRIPTION = "A Comprehensive Framework for ML Security Research"
URL = "https://github.com/[username]/data-poison-toolkit"
KEYWORDS = [
    "machine learning",
    "security",
    "data poisoning", 
    "adversarial attacks",
    "defense mechanisms",
    "research",
]

# Academic information
CONFERENCES = {
    "blackhat": "BlackHat Europe 2025",
    "ieee": "IEEE Conference Submission",
}

CITATION = """
@software{yagiz2025datapoisontoolkit,
  title={Data Poisoning Toolkit: A Comprehensive Framework for ML Security Research},
  author={Yagiz, Muhammet Anil and Dayi, Muhammed Arif and Goktas, Polat},
  year={2025},
  url={https://github.com/[username]/data-poison-toolkit},
  note={Presented at BlackHat Europe 2025, Submitted to IEEE Conference}
}
"""

def get_version():
    """Return the current version of the package."""
    return __version__

def get_citation():
    """Return the BibTeX citation for academic use."""
    return CITATION.strip()

def print_info():
    """Print package information."""
    print(f"""
Data Poisoning Toolkit v{__version__}
{DESCRIPTION}

Authors: {__author__}
License: {__license__}
Status: {__status__}

Conference Presentations:
- {CONFERENCES['blackhat']}
- {CONFERENCES['ieee']}

For academic use, please cite:
{get_citation()}
    """.strip())

# Ensure critical dependencies are available
try:
    import numpy
    import pandas
    import torch
    import sklearn
    import matplotlib
    import seaborn
except ImportError as e:
    raise ImportError(
        f"Missing required dependency: {e}. "
        "Please install all requirements with: pip install datapoisontoolkit"
    ) from e

# Version compatibility checks
import sys
if sys.version_info < (3, 8):
    raise RuntimeError(
        "Data Poisoning Toolkit requires Python 3.8 or higher. "
        f"Current version: {sys.version_info.major}.{sys.version_info.minor}"
    )

# Security notice
import warnings
warnings.warn(
    "⚠️ SECURITY NOTICE: This toolkit is designed for academic research and "
    "authorized security testing only. Users are responsible for ensuring "
    "compliance with applicable laws and ethical guidelines. "
    "The authors do not support malicious use of this software.",
    UserWarning,
    stacklevel=2
)