"""
Isotropic Metal Electromagnetic Skin Effect Analysis
====================================================

A comprehensive Python toolkit for analyzing electromagnetic skin effect 
response in isotropic metals using analytic Boltzmann transport theory.

These are the modules needed to run the skin effect calculation and analysis notebook.
"""

__version__ = '0.1.0'
__author__ = 'Timothy W. Branch'

# Import main modules
from . import electronic_transport
from . import numerical_fermi_surface_utilities
from . import figure_helper
from . import common_plot_labels
from . import figure_templates

__all__ = [
    'electronic_transport',
    'numerical_fermi_surface_utilities',
    'figure_helper', 
    'common_plot_labels',
    'figure_templates'
]