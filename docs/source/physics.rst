Models and Physics Notes
========================

This page summarises what the gain models solve, their assumptions, and the solver settings that matter in
practice. The tutorials show them at work; the module docstrings of :mod:`twpasolver.cmes.bloch` and
:mod:`twpasolver.cmes.bloch_fb` hold the derivations.

Linear response
---------------

A TWPA is a chain of supercells (one period of the line, possibly made of several unit cells). From the ABCD
matrix ``T`` of a supercell, :attr:`TWPAnalysis.linear <twpasolver.analysis.TWPAnalysis.linear>` gives the Bloch
wavenumber ``k`` and attenuation ``alpha`` per unit cell, the Bloch impedance ``Zb``, the stopbands
(``|tr T / 2| > 1`` for the lossless supercell) and the transmission of the whole device. ``k`` follows the winding
of the propagating mode through the elements of the supercell, so it is continuous across stopbands and resonator
notches and negative in left-handed bands. A segmented device (a chain of uniform segments, see notebook 06) has
these quantities per segment, with arrays of shape ``(n_segments, n_freq)``.

Gain models
-----------

Both models integrate coupled mode equations (CMEs) for the slowly varying amplitudes of a set of modes
(:class:`~twpasolver.modes_rwa.ModeSet`), keeping the three- and four-wave-mixing terms allowed by the rotating-wave
approximation for that set.

``simple``
    The traditional CMEs of a homogeneous line, forward waves only. The supercell enters only through ``k`` and
    ``alpha``; the couplings are those of a uniform medium. Fast, and the standard reference in the literature, but
    approximate for strongly loaded lines and near stopbands. It ignores reflections and the connectors in
    :class:`~twpasolver.analysis.Ports` (a warning is logged when connectors are set): its gain is the output over
    the input current inside the line. Segmented devices are treated as a smooth taper, with ``k`` interpolated
    linearly between segment midpoints.

``bloch``
    Forward and backward waves in the exact Bloch-mode basis of each segment, with the couplings computed from the
    current profiles inside the supercell. Reflections at the device ends (including connectors) and at junctions
    between segments are exact, and the forward and backward amplitudes are found by iterating forward and
    backward integrations (Anderson-accelerated) until they are consistent. Use it for loaded or dispersion-
    engineered lines, devices operated near stopbands, mismatched or connectorised devices, and segmented devices.

When the line is matched and the tones are far from stopbands the two models agree, which is a useful check.

Ports and port-resolved gains
-----------------------------

:class:`~twpasolver.analysis.Ports` holds the connectors at both ends (any two-port model, port 1 facing the
external line) and the impedance ``Z_ext`` of the external lines. A :class:`~twpasolver.analysis.GainResult`
reports the currents incident from the external lines (``a_in`` at x=0, ``a_out`` at x=L) and going out into them
(``b_in``, ``b_out``), including the prompt reflection of the incident waves, and the signal gains ``gain_db``
(S21), ``gain_db_12``, ``gain_db_11`` and ``gain_db_22``. A gain whose reference input is zero is NaN. With inputs at
both ends the outputs contain the response to both, so inject from one side at a time to measure S-parameters.

Stopband edges and the edge basis
---------------------------------

At a stopband edge the two Bloch modes of the supercell coalesce (an exceptional point of ``T``), and every
coupling projected on them diverges. The ``bloch`` model therefore expands a mode without input that lies inside a
stopband or close to its edge on a *fixed* pair of Bloch vectors taken on the passband side, where it is well
defined, and solves the coupling between the two directions exactly. ``BlochSolverConfig(edge_threshold=...)`` sets
"close": the normalised splitting ``|sinh(gamma d)|`` of the two modes (``d`` the supercell length) below which the
fixed basis is used. The default 0.1 works for the devices in the tutorials; results should not depend on it, which
can be checked by changing it. ``simulate_stopband_modes=False`` instead switches off the processes of those modes
smoothly, which is faster but drops their physics. Modes with an input (pump, signal) are never changed; a warning
says when one of them lies near an edge, where its description is unreliable.

Mixing terms and the reciprocal lattice
---------------------------------------

Inside a supercell the nonlinear couplings vary with position. Their Fourier components ``G_m = 2 pi m / d`` can
quasi-phase-match processes whose wavenumbers do not add up. The ``bloch`` model keeps a term when its residual
mismatch ``|dk + G_m|`` is small compared with the wavenumbers of the process (``kappa_cutoff``), up to the order
``m_max``. ``m_max="auto"`` uses order 1 everywhere and raises it, only where needed, for modes inside or near a
stopband. Note that ``m_max`` counts reciprocal vectors of the *supercell*, so for long supercells (many unit cells
per period) the same physical harmonic needs a larger ``m``.

Convergence
-----------

The forward-backward iteration stops when the largest change of any mode between passes, relative to that mode's
largest forward plus backward envelope, falls below ``convergence_threshold`` (default 0.01). This residual is
dominated by nearly empty backward envelopes, so values of 1e-3 to 1e-1 are common while the gains are already
stable; results are determined to within that tolerance. ``GainResult.converged`` flags the points that reached
it, and the solver warns about the others. Raise ``passes`` for those points, and lower
``convergence_threshold`` when comparing results that differ by less than the tolerance.

Mode sets
---------

The CMEs only describe the modes in the :class:`~twpasolver.modes_rwa.ModeSet`. Pump harmonics and frequency
conversion terms take power from the pump and the signal when they are phase matched, so a result should be
checked by adding them (``ModeSet.extended_3wm``, or a custom set) until the gain stops changing. Lines without
dispersion at high frequency, such as Josephson lines far below their plasma frequency, need particular care (see
notebook 05).

Custom nonlinear cells
----------------------

The ``bloch`` model needs to know which series inductors of a cell are nonlinear. The package cells describe
theirs (``nl_scale`` sets the nonlinear fraction of the line inductance); a custom cell defines
``bloch_nl_elements()``. See :doc:`implementing_models`.
