"""DemandMesh - hyper-local B2B demand forecasting and dynamic route optimisation.

The package is organised as a small production pipeline:

    data_generator      -> realistic synthetic B2B marketplace data (India)
    data_preprocessing  -> validation, cleaning and leakage-safe feature engineering
    model_training      -> quantile gradient-boosted forecaster + evaluation + registry
    forecaster          -> inference wrapper used by the API
    route_optimizer     -> CVRPTW solver (Adaptive Large Neighbourhood Search)
    planner             -> forecast -> freight load -> optimised dispatch plan
    drift               -> Population Stability Index monitoring
    app                 -> FastAPI service
"""

__version__ = "1.0.0"
