# DemandMesh - evaluation report

Model `v20261008-202627` · data 2023-10-01 → 2026-09-30 · 240 series · holdout 2026-09-03 → 2026-09-30.

All data is synthetic (see `src/demandmesh/data_generator.py`); numbers show the method works end-to-end, not what a real deployment would achieve.

## Forecast accuracy (28-day holdout, 7-day-ahead features)

| Method | WAPE | MAE | Bias |
|---|---|---|---|
| **DemandMesh P50** | **26.9%** | 2.85 | -3.8% |
| seasonal naive 7d | 42.0% | 4.45 | -1.0% |
| same weekday mean 4w | 35.6% | 3.78 | -2.0% |
| moving average 28d | 40.2% | 4.26 | -2.2% |

Improvement over the best baseline: **24.5%** lower WAPE.

| Aggregation level | WAPE |
|---|---|
| series day | 26.9% |
| zone day | 12.2% |
| city day | 6.4% |

## Uncertainty calibration

P10-P90 nominal coverage 80% · raw model 77.9% · after conformal calibration **79.6%** (interval widened by 0.13 orders).

## Capacity booking: P50 vs P90 (city-day freight, holdout)

Shares are of actual freight. Idle capacity costs money; unplanned freight means spot-hiring trucks at a premium.

| capacity_booked_at | city_days_short_pct | unplanned_freight_pct | idle_capacity_pct |
|---|---|---|---|
| P50 sum | 63.4 | 5.5 | 2.1 |
| naive P90 sum | 0.0 | 0.0 | 50.0 |
| pooled P90 (used) | 14.7 | 0.9 | 9.8 |

## Routing: ALNS vs nearest-neighbour dispatch (next-day plan, P90)

| city | drops | freight_kg | alns_vehicles | alns_km | alns_cost_inr | nn_vehicles | nn_km | nn_cost_inr | km_saved_pct | cost_saved_pct | unserved |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Delhi NCR | 25 | 9,001.5 | 4 | 157.3 | 10,090.9 | 4 | 160.8 | 10,181.9 | 2.2 | 0.9 | 0 |
| Mumbai MMR | 20 | 7,086.0 | 3 | 265.6 | 11,405.3 | 3 | 270.0 | 11,518.9 | 1.6 | 1.0 | 0 |
| Ahmedabad | 18 | 6,166.2 | 3 | 179.6 | 9,168.7 | 3 | 224.1 | 10,325.9 | 19.9 | 11.2 | 0 |
| Ludhiana | 17 | 5,334.4 | 3 | 67.7 | 6,260.1 | 3 | 108.3 | 7,315.1 | 37.5 | 14.4 | 0 |
| Surat | 16 | 5,162.8 | 3 | 132.8 | 7,951.6 | 3 | 142.1 | 8,194.3 | 6.6 | 3.0 | 0 |
| Coimbatore | 12 | 4,250.5 | 2 | 73.1 | 4,901.6 | 2 | 96.2 | 5,500.1 | 23.9 | 10.9 | 0 |
| Kolkata | 16 | 5,285.0 | 3 | 105.9 | 7,254.3 | 3 | 148.3 | 8,355.1 | 28.6 | 13.2 | 0 |
| Pune | 24 | 8,577.8 | 4 | 152.2 | 9,956.8 | 4 | 191.8 | 10,988.0 | 20.7 | 9.4 | 0 |

Network total: 1134 km vs 1342 km (**15.5% less distance**), ₹66,989 vs ₹72,379 per day (**7.4% lower cost**).

## Top drivers (permutation importance)

| Feature | MAE increase |
|---|---|
| orders_roll_mean_28 | 4.073 |
| dow | 1.075 |
| enq_momentum | 1.027 |
| enq_roll_mean_7 | 0.111 |
| enq_to_order_conv_28 | 0.090 |
| days_to_festival | 0.047 |
| zone_id | 0.038 |
| is_sunday | 0.035 |
| is_festival_day | 0.015 |
| category | 0.012 |
