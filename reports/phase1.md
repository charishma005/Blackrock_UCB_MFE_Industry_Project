# Phase-1 analyst diagnostics

- source: **fred**, horizon: **63d**, meetings: **313** (2019-01-04 → 2024-12-27)
- LLM agents: **yes**

## 1. Faithfulness

_Input isolation is structural — the base class hands each analyst an AsOf gate, so it can only read its declared series. `isolation_ok` and `no_lookahead` should be True for all._

```
                declared_inputs  accessed  isolation_ok undeclared_reads  no_lookahead latest_index
analyst                                                                                            
inflation              CPIAUCSL  CPIAUCSL          True                -          True   2024-12-15
labor_tightness          UNRATE    UNRATE          True                -          True   2024-12-08
balance_sheet             WALCL     WALCL          True                -          True   2024-12-25
term_premium              DGS10     DGS10          True                -          True   2024-12-27
```

### Responsiveness vs contamination (view vs each driver's honest measurement)

_own_corr high + cross_corr low ⇒ faithful. Deterministic own_corr = 1 by construction; the LLM column is where drift shows up._

**Deterministic Phase-1**

```
                 own_corr  cross_corr_max  cross_corr_mean most_contaminating  faithfulness
driver                                                                                     
inflation             1.0           0.432            0.383    labor_tightness         0.568
labor_tightness       1.0           0.432            0.223          inflation         0.568
balance_sheet         1.0           0.416            0.196          inflation         0.584
term_premium          1.0           0.302            0.151          inflation         0.698
```

**LLM Phase-2**

```
                 own_corr  cross_corr_max  cross_corr_mean most_contaminating  faithfulness
driver                                                                                     
inflation           0.966           0.459            0.410      balance_sheet         0.507
labor_tightness     0.901           0.477            0.282          inflation         0.424
balance_sheet       0.879           0.253            0.130          inflation         0.626
term_premium        0.969           0.298            0.161          inflation         0.671
```

### Reasoning stays on-topic (lexicon proxy)

_on_topic_rate high, contamination_rate low ⇒ the text stays about the driver._

**Deterministic Phase-1**

```
                 on_topic_rate  contamination_rate    n
driver                                                 
inflation                  1.0                 0.0  313
labor_tightness            1.0                 0.0  313
balance_sheet              1.0                 0.0  313
term_premium               1.0                 0.0  313
```

**LLM Phase-2**

```
                 on_topic_rate  contamination_rate    n
driver                                                 
inflation                  1.0               0.026  313
labor_tightness            1.0               0.006  313
balance_sheet              1.0               0.412  313
term_premium               1.0               0.438  313
```

## 2. Correctness
### Directional accuracy over the horizon, vs a persistence baseline

_edge_vs_persistence > 0 means the analyst beats 'the last move continues'; edge_vs_random > 0 means it beats a coin flip._

**Deterministic Phase-1**

```
                   n  hit_rate  info_score  persistence_hit  edge_vs_persistence  edge_vs_random
driver                                                                                          
inflation        304     0.562       0.157            0.135                0.428           0.062
labor_tightness  242     0.401       0.004            0.103                0.298          -0.099
balance_sheet    304     0.812       0.603            0.717                0.095           0.312
term_premium     300     0.483      -0.034            0.510               -0.027          -0.017
```

**LLM Phase-2**

```
                   n  hit_rate  info_score  persistence_hit  edge_vs_persistence  edge_vs_random
driver                                                                                          
inflation        304     0.526       0.123            0.135                0.391           0.026
labor_tightness  242     0.393      -0.068            0.103                0.289          -0.107
balance_sheet    304     0.822       0.585            0.717                0.105           0.322
term_premium     300     0.493      -0.000            0.510               -0.017          -0.007
```

## 3. Lookahead

_Data-slice lookahead is covered by input isolation above (no_lookahead column) and the AsOf unit test. Below is the LLM training-cutoff test._

```
                 det_hit  llm_hit  information_gain  override_n  override_hit                                                                          interpretation
driver                                                                                                                                                               
inflation          0.562    0.526            -0.036          25         0.040  REAL history — large positive gain / high override_hit ⇒ possible training-cutoff leak
labor_tightness    0.401    0.393            -0.008          20         0.150  REAL history — large positive gain / high override_hit ⇒ possible training-cutoff leak
balance_sheet      0.812    0.822             0.010          28         0.536  REAL history — large positive gain / high override_hit ⇒ possible training-cutoff leak
term_premium       0.483    0.493             0.010           5         0.800  REAL history — large positive gain / high override_hit ⇒ possible training-cutoff leak
```

## 4. Correlation between agents

Average |off-diagonal| — deterministic: **0.238**, LLM: **0.216**.  Lower = more independent (the isolation the thesis buys).

**Deterministic**

```
                 inflation  labor_tightness  balance_sheet  term_premium
inflation            1.000            0.432          0.416         0.302
labor_tightness      0.432            1.000          0.128         0.108
balance_sheet        0.416            0.128          1.000         0.043
term_premium         0.302            0.108          0.043         1.000
```

**LLM**

```
                 inflation  labor_tightness  balance_sheet  term_premium
inflation            1.000            0.465          0.289         0.347
labor_tightness      0.465            1.000          0.065         0.083
balance_sheet        0.289            0.065          1.000        -0.046
term_premium         0.347            0.083         -0.046         1.000
```

