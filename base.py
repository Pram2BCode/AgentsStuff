from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict
from typing import Any, Dict

class Agent(BaseModel, ABC):
    """Abstract base class for all agents."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Runs the agent's task."""
        pass


