from __future__ import annotations


class VfError(Exception):
    """Base exception for VisFlow errors."""


class InvalidConfigurationError(VfError):
    """Exception raised for invalid configuration settings."""


class RequiredModuleNotFoundError(VfError):
    """Exception raised when a required module is not found."""
