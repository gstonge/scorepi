#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the score functions

Author: Guillaume St-Onge <stongeg1@gmail.com>
"""

import pytest
import numpy as np
from scorepi import *

class TestIntervalScore:
    def test_raise_error_bad_shape(self):
        observation = [0.1]*3
        lower = [0.1]*3
        upper = [0.1]*2
        interval_range = 90

        with pytest.raises(ValueError):
            interval_score(observation,lower,upper,interval_range)

    def test_raise_error_bad_alpha(self):
        observation = [0.1]*3
        lower = [0.1]*3
        upper = [0.1]*3
        interval_range = 110

        with pytest.raises(ValueError):
            interval_score(observation,lower,upper,interval_range)


    def test_score(self):
        observation = [0.5]*3
        lower = [0.4,0.5,0.6]
        upper = [0.45,0.55,0.65]
        interval_range = 50
        alpha = 1-interval_range/100
        out = interval_score(observation,lower,upper,interval_range)

        assert np.allclose(out['dispersion'],[0.05,0.05,0.05])
        assert np.allclose(out['underprediction'],[0.,0.,0.1*2/alpha])
        assert np.allclose(out['overprediction'],[0.05*2/alpha,0.,0.])
        assert np.allclose(out['interval_score'],[0.05*(1+2/alpha),0.05,0.05+0.2/alpha])

class TestCoverage:
    def test_raise_error_bad_shape(self):
        observation = [0.1]*3
        lower = [0.1]*3
        upper = [0.1]*2
        with pytest.raises(ValueError):
            coverage(observation,lower,upper)

    def test_score(self):
        observation = [0.5]*3
        lower = [0.4,0.5,0.6]
        upper = [0.45,0.55,0.65]

        assert np.isclose(coverage(observation,lower,upper),1/3)

