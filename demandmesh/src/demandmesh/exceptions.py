"""Typed exception hierarchy.

Every failure the service can anticipate maps to one of these classes, which the
API layer translates into a precise HTTP status. Anything outside the hierarchy
is a genuine bug and surfaces as a 500 with a request id for tracing.
"""


class DemandMeshError(Exception):
    """Base class for all domain errors raised by DemandMesh."""


class DataValidationError(DemandMeshError):
    """Input data violates the expected schema or integrity constraints."""


class ModelArtifactError(DemandMeshError):
    """A model artefact is missing, corrupted, or incompatible."""


class ModelNotReadyError(DemandMeshError):
    """The service was asked to forecast before a model was loaded."""


class ForecastRequestError(DemandMeshError):
    """A forecast request is semantically invalid (unknown zone, bad horizon, ...)."""


class RoutingError(DemandMeshError):
    """A routing problem is malformed or cannot be built."""
