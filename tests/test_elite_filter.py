import os
import sys
import types
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import validation
import config as _config


def _set_elite(symbol, cfg):
    if not hasattr(_config, "ELITE_SYMBOLS"):
        setattr(_config, "ELITE_SYMBOLS", {})
    _config.ELITE_SYMBOLS[symbol] = cfg


def test_approval_all_ok(monkeypatch):
    _set_elite(
        "AAA3",
        {
            "min_pf": 1.4,
            "min_sharpe": 0.6,
            "min_win_rate": 0.35,
            "max_drawdown": 0.30,
            "adx_threshold": 15.0,
            "mom_min": 0.01,
            "priority": {"max_drawdown": 5},
        },
    )
    metrics = {
        "profit_factor": 1.6,
        "sharpe": 0.8,
        "win_rate": 0.40,
        "max_drawdown": 0.20,
    }
    approved, result = validation.evaluate_elite_entry("AAA3", metrics, ttl_seconds=0)
    assert approved is True
    assert result["failures"] == []


def test_partial_rejection(monkeypatch):
    _set_elite("BBB3", {"min_pf": 1.8, "min_sharpe": 0.6, "max_drawdown": 0.25})
    metrics = {"profit_factor": 1.5, "sharpe": 0.7, "max_drawdown": 0.20}
    approved, result = validation.evaluate_elite_entry("BBB3", metrics, ttl_seconds=0)
    assert approved is False
    fails = result["failures"]
    assert any(f["metric"] == "min_pf" for f in fails)


def test_total_rejection(monkeypatch):
    _set_elite("CCC3", {"min_pf": 2.0, "min_sharpe": 1.0, "max_drawdown": 0.15})
    metrics = {"profit_factor": 1.2, "sharpe": 0.5, "max_drawdown": 0.30}
    approved, result = validation.evaluate_elite_entry("CCC3", metrics, ttl_seconds=0)
    assert approved is False
    fails = [f["metric"] for f in result["failures"]]
    assert "min_pf" in fails and "min_sharpe" in fails and "max_drawdown" in fails
