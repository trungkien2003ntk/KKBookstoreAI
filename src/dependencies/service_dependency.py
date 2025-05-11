"""
This module provides a dependency injection function to create and retrieve a Service instance.
"""

from src.services.service import ServiceManager

service = ServiceManager()


def get_service() -> ServiceManager:
    """
    Dependency injection function to provide a Service instance.

    Returns:
        Service: An instance of the Service class.
    """
    return service
