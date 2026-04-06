"""Tests for temperature grid construction."""

from averyml.config.search import SearchConfig
from averyml.search.temperature import TemperaturePoint, build_grid, filter_diagonal_band


class TestTemperaturePoint:
    def test_t_eff(self):
        point = TemperaturePoint(
            t_train=1.5,
            t_eval=0.8,
            rho_train=None,
            rho_eval=None,
        )
        assert point.t_eff == 1.5 * 0.8

    def test_repr(self):
        point = TemperaturePoint(t_train=1.0, t_eval=1.0, rho_train=None, rho_eval=None)
        assert "T_train=1.0" in repr(point)


class TestBuildGrid:
    def test_grid_size(self):
        config = SearchConfig(
            base_model_id="test/model",
            t_train_values=[0.5, 1.0, 2.0],
            t_eval_values=[0.6, 1.0],
        )
        grid = build_grid(config)
        assert len(grid) == 3 * 2  # Cartesian product

    def test_grid_values(self):
        config = SearchConfig(
            base_model_id="test/model",
            t_train_values=[1.0],
            t_eval_values=[0.8],
        )
        grid = build_grid(config)
        assert len(grid) == 1
        assert grid[0].t_train == 1.0
        assert grid[0].t_eval == 0.8


class TestFilterDiagonalBand:
    def test_filters_correctly(self):
        config = SearchConfig(
            base_model_id="test/model",
            t_train_values=[0.5, 1.0, 2.0, 3.0],
            t_eval_values=[0.3, 0.5, 1.0, 2.0],
        )
        grid = build_grid(config)
        filtered = filter_diagonal_band(grid, t_eff_min=0.8, t_eff_max=1.6)

        for point in filtered:
            assert 0.8 <= point.t_eff <= 1.6

        # Some points should be filtered out
        assert len(filtered) < len(grid)
