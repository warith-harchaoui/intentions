"""Fast guard for the confusion-matrix ink convention.

The heatmaps paint each cell from white (count 0) to the engine's full hue
(the largest count). The number printed inside must stay legible, so the ink
follows one rule the user asked for explicitly: **black on a light cell, white
on a dark one**. These dependency-free checks pin that behaviour so a future
palette or threshold tweak cannot silently make a cell's number unreadable.

Author
------
Project maintainers.
"""

from __future__ import annotations

from .confusion import _INK_DARK, _INK_LIGHT, _cell_ink, _relative_luminance


def test_empty_cell_uses_dark_ink() -> None:
    """A white (count 0) cell reads black, whatever the engine hue."""
    # t = 0 → pure white background → the darkest, most contrasted ink wins.
    for hue in ("#007AFF", "#28CD41", "#FFCC00", "#AF52DE"):
        assert _cell_ink(0, 10, hue) == _INK_DARK


def test_saturated_dark_hue_flips_to_white() -> None:
    """The densest cell of a dark hue (blue, purple) takes white ink."""
    # Blue (#007AFF) and purple (#AF52DE) sit near luminance 0.2, well under
    # the 0.5 split, so their full-strength cells must invert to white.
    assert _cell_ink(10, 10, "#007AFF") == _INK_LIGHT
    assert _cell_ink(10, 10, "#AF52DE") == _INK_LIGHT


def test_light_hue_keeps_black_ink() -> None:
    """A light hue (yellow) never gets dark enough to warrant white ink."""
    # Yellow (#FFCC00) stays above the split even at full strength, so every
    # cell of a yellow matrix keeps black ink.
    for count in range(0, 11):
        assert _cell_ink(count, 10, "#FFCC00") == _INK_DARK


def test_ink_switches_once_and_stays() -> None:
    """Ink goes light exactly when the cell crosses the darkness split."""
    # As the count climbs the background only darkens, so once white ink kicks
    # in it must not revert to black — a monotone light→dark luminance ramp.
    hue = "#007AFF"
    inks = [_cell_ink(c, 10, hue) for c in range(0, 11)]
    first_light = inks.index(_INK_LIGHT)
    assert all(ink == _INK_LIGHT for ink in inks[first_light:])


def test_relative_luminance_bounds() -> None:
    """Luminance is 0 for black and 1 for white (the WCAG anchors)."""
    assert _relative_luminance((0, 0, 0)) == 0.0
    assert _relative_luminance((255, 255, 255)) == 1.0
