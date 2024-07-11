"""
Twpasolver package.

This module provides to the basic tools and utilities for the analysis and simulation of
traveling wave parametric amplifiers (TWPAs). It includes classes and functions
for performing analyses, handling files, and working with matrices and arrays
specific to TWPA models.

Classes:
    * TWPAnalysis: Runner for standard analysis routines of 3WM twpa models.
    * ABCDArray: Represents ABCD matrices used in the analysis of two-port networks.
    * TwoPortCell: Wraps an ABCDArray with additional data and conversion functions to other representations.
    * Frequencies: respresents a frequency span with variable measurement unit.

Functions:
    * read_file: Reads data from a specified ```json``` or ```hdf5``` file.
    * save_to_file: Saves data encoded in a dictionary to a specified ```json``` or ```hdf5``` file.
"""

import importlib.metadata as im

__version__ = im.version(__package__)

from twpasolver.analysis import TWPAnalysis
from twpasolver.file_utils import read_file, save_to_file
from twpasolver.frequency import Frequencies
from twpasolver.matrices_arrays import ABCDArray
from twpasolver.twoport import TwoPortCell
