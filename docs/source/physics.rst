Models and Physics Notes
========================

This page summarises what the gain models solve, their assumptions, and the solver settings that matter in
practice. The :doc:`tutorials` show them at work; the module docstrings of :mod:`twpasolver.cmes.bloch` and
:mod:`twpasolver.cmes.bloch_fb` hold the derivations.

Linear response
---------------

A TWPA is a chain of supercells, each one period of the line made of one or more unit cells. From the ABCD matrix
``T`` of a supercell, :attr:`TWPAnalysis.linear <twpasolver.analysis.TWPAnalysis.linear>` gives the Bloch
wavenumber ``k`` and attenuation ``alpha`` per unit cell, the Bloch impedance ``Zb``, the stopbands
(``|tr T / 2| > 1`` for the lossless supercell) and the transmission of the whole device. ``k`` follows the winding
of the propagating mode through the elements of the supercell, so it is continuous across stopbands and resonator
notches and negative in left-handed bands. A segmented device (a chain of uniform segments) has these quantities
per segment, with arrays of shape ``(n_segments, n_freq)``.

Nonlinearity
------------

The nonlinearity is a property of the :class:`~twpasolver.models.TWPA`, applied to the series line inductance of
its cells. The kinetic inductance depends on the current through it as ``L(I) = Lk [1 + (I/Istar)^2 +
(I/Istar2)^4]``; with a dc bias ``Idc`` and a small rf current ``i``, ``L = L(Idc) (1 + epsilon i + xi i^2)``.
``epsilon`` (1/A) sets the three-wave-mixing (3WM) and ``xi`` (1/A^2) the four-wave-mixing (4WM) couplings. Without
``Istar2``, ``xi = 1 / (Istar^2 + Idc^2)`` and ``epsilon = 2 Idc xi``: an unbiased line mixes four waves only. The
cell inductance ``L`` is the biased one, ``L(Idc)``, and sets the dispersion as given. ``Ip0`` and ``Is0`` are
peak amplitudes, in A, of the pump and signal waves incident at the input. In a segmented device every segment has
its own ``Istar`` and ``Idc``.

Gain models
-----------

Both models integrate coupled mode equations (CMEs) for the slowly varying amplitudes of a set of modes
(:class:`~twpasolver.modes_rwa.ModeSet`), with the 3WM and 4WM terms that the rotating-wave approximation keeps for
that set.

``simple``
    The textbook CMEs of a homogeneous line, forward waves only. The supercell enters only through ``k`` and
    ``alpha``; the couplings are those of a uniform medium. It is fast and matches the literature, but is
    approximate for strongly loaded lines and near stopbands. It ignores reflections and the connectors in
    :class:`~twpasolver.analysis.Ports` (and warns when connectors are set): its gain is the ratio of output to
    input current inside the line. A segmented device is treated as a smooth taper, with ``k`` interpolated
    linearly between segment midpoints.

``bloch``
    Forward and backward waves in the exact Bloch-mode basis of each segment, with couplings computed from the
    current profiles inside the supercell. Reflections at the device ends (including connectors) and at the
    junctions between segments are exact, and the two directions are iterated (with Anderson acceleration) until
    they are consistent. Use it for loaded or dispersion-engineered lines, devices operated near stopbands,
    mismatched or packaged devices, and segmented devices.

The two models agree when the line is matched and the tones are far from stopbands, which is a useful check.

Ports and port-resolved gains
-----------------------------

:class:`~twpasolver.analysis.Ports` holds the connectors at both ends (any two-port model, port 1 facing the
external line) and the impedance ``Z_ext`` of the external lines. A :class:`~twpasolver.analysis.GainResult`
reports the currents incident from the external lines (``a_in`` at x=0, ``a_out`` at x=L) and leaving into them
(``b_in``, ``b_out``), including the prompt reflection of the incident waves, and the signal gains ``gain_db``
(S21), ``gain_db_12``, ``gain_db_11`` and ``gain_db_22``. A gain whose reference input is zero is NaN. With inputs at
both ends the outputs contain the response to both, so inject from one end at a time to obtain S-parameters.
:meth:`TWPAnalysis.response() <twpasolver.analysis.TWPAnalysis.response>` gives the linear response including the
connectors, for comparison with measurements of a packaged device; without pump the ``bloch`` gains reduce to it.

Stopband edges and the edge basis
---------------------------------

At a stopband edge the two Bloch modes of the supercell coalesce (an exceptional point of ``T``), and every coupling
projected on them diverges. The ``bloch`` model therefore expands a mode without input that lies inside a stopband
or close to its edge on a *fixed* pair of Bloch vectors taken on the passband side, where it is well defined, and
solves the coupling between its two directions exactly. ``BlochSolverConfig(edge_threshold=...)`` sets "close": the
normalised splitting ``|sinh(gamma d)|`` of the two modes (``d`` the supercell length) below which the fixed basis
is used. The default 0.1 works for the devices in the tutorials; results should not depend on it, which can be
checked by changing it. ``simulate_stopband_modes=False`` instead switches off the processes of those modes
smoothly, which is faster but drops their physics. Modes with an input (pump, signal) are never changed; a warning
says when one of them lies near an edge, where its description is unreliable.

Mixing terms and the reciprocal lattice
---------------------------------------

Inside a supercell the nonlinear couplings vary with position. Their Fourier components ``G_m = 2 pi m / d`` can
quasi-phase-match processes whose wavenumbers do not add up. The ``bloch`` model keeps a term when its residual
mismatch ``|dk + G_m|`` is small compared with the wavenumbers of the process (``kappa_cutoff``, default 0.2), up to
the order ``m_max``. ``m_max="auto"`` uses order 1 everywhere and raises it, only where needed, for modes inside or
near a stopband. ``m_max`` counts reciprocal vectors of the *supercell*, so in a long supercell (many unit cells per
period) the same physical harmonic needs a larger ``m``.

Convergence
-----------

The forward-backward iteration stops when the largest change of any mode between passes, relative to that mode's
largest forward plus backward envelope, falls below ``convergence_threshold`` (default 0.01). Nearly empty backward
envelopes dominate this residual, so values of 1e-3 to 1e-1 are common while the gains are already stable; results
are determined to within that tolerance. ``GainResult.converged`` flags the points that reached it, and the solver
warns about the others. For those, raise ``passes`` together with ``anderson_depth`` (for example to
``passes - 10``, so that the acceleration uses the longer history); lower ``convergence_threshold`` when comparing
results that differ by less than the tolerance.

Mode sets
---------

The CMEs describe only the modes in the :class:`~twpasolver.modes_rwa.ModeSet`. Pump harmonics and
frequency-conversion products take power from the pump and the signal when they are phase matched, so check a
result by adding them (``ModeSet.extended_3wm``, or a custom set) until the gain stops changing. Sidebands of two or
more signal photons (``2 fs``, ``fp - 2 fs`` and their conversion products, ``n_signal_harmonics`` and
``n_sideband_conversion``) are second order in the signal: they matter in compression, not for the small-signal
gain. The idler ``fp - m fs`` of order ``m`` crosses 0 Hz at ``fs = fp / m``, so the signals must stay on one side
of it. Lines with little
dispersion at high frequency, such as kinetic-inductance lines far below their cutoff, need particular care: their
pump harmonics and up-conversion products stay nearly phase matched (see :doc:`notebooks/05_other_twpas`).

Custom nonlinear cells
----------------------

The ``bloch`` model needs to know which series inductors of a cell are nonlinear. The package cells describe
theirs (``nl_scale`` sets the nonlinear fraction of the line inductance); a custom cell defines
``bloch_nl_elements()``. See :doc:`implementing_models`.
