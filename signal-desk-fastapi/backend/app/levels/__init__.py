"""levels - an OPTIONAL, INFORMATIONAL layer: probable support/resistance zones with their touch history.

  config.py      models/levels_config.json (default enabled: false), same pattern as quant15_live_policy.json
  data.py        timeframes / spans, candle loading, resampling
  candidates.py  swing pivots, prior-period extremes, volume nodes, optional round numbers, clustering into zones
  touches.py     the touch log: every entry into a zone, approach direction, outcome, role flips
  service.py     one (timeframe, span) cell -> zones with evidence; several cells -> a response
  routes.py      FastAPI router under /api/levels

Nothing in strategy.py, paper.py, hub.py or quant15/live.py may import this package (tests/test_levels.py fails if they do).
Levels never trigger, gate, size or alert on anything. Everything is computed only from candles closed before `as_of`.
"""
