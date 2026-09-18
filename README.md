# German Day-Ahead Electricity Price & Negative Price Risk Forecasting (`german-power-price-forecasting`)

## Objective
Forecast next-day hourly wholesale electricity spot prices (€/MWh) in the German/Luxembourg (DE-LU) bidding zone and assess the risk of negative price events (`day_ahead_price_eur_mwh < 0.00`). The project operates under strict operational market cutoff discipline: every feature used for forecasting delivery day $D$ must have been publicly available prior to the Day-Ahead auction closure on day $D-1$ (12:00 CET).

## Primary Data Source
- **SMARD.de** — The official open electricity market data platform of the German Federal Network Agency (*Bundesnetzagentur*).
- **Target Market**: Germany/Luxembourg (DE-LU) bidding zone.
- **Analysis Period**: 2023-01-01 through 2024-12-31 (hourly resolution).

## Current Status
- **Milestone 1 Completed**: Workspace environment and repository initialized (Python 3.11 virtual environment, initial directory structure, dependency definitions, and Git tracking).
- **Next Milestone**: Milestone 2 — Data Ingestion Pipeline (retrieving and structuring 2023–2024 SMARD data).
