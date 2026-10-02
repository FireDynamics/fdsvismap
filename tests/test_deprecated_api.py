"""Tests for the deprecated 0.2 API: every old name warns and forwards to its replacement."""

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fdsvismap import Sign, VisMap
from fdsvismap._deprecation import deprecated_attribute, warn_deprecated

matplotlib.use("Agg")


class TestDeprecationHelpers:
    """The helpers that every alias is built on."""

    def test_the_message_names_old_and_new(self):
        with pytest.warns(
            DeprecationWarning,
            match=r"^old\(\) was replaced by new\(\) in fdsvismap 0\.3 and will be removed in 1\.0\.$",
        ):
            warn_deprecated("old()", "new()", stacklevel=2)

    def test_the_warning_points_at_the_caller(self):
        def caller():
            warn_deprecated("old()", "new()", stacklevel=2)

        with pytest.warns(DeprecationWarning) as record:
            caller()
        assert record[0].filename == __file__
        assert record[0].lineno == caller.__code__.co_firstlineno + 1

    def test_attribute_alias_reads_and_writes_the_new_name(self):
        class Holder:
            def __init__(self):
                self.new = 1

            old = deprecated_attribute("old", "new")

        holder = Holder()
        with pytest.warns(
            DeprecationWarning, match="The attribute old was replaced by new"
        ) as record:
            assert holder.old == 1
        assert record[0].filename == __file__
        with pytest.warns(DeprecationWarning):
            holder.old = 2
        assert holder.new == 2
        assert "old" not in holder.__dict__
