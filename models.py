from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field
from typing import Dict, Any


class DetraffAction(Action):
    # 0: North-South Green, 1: East-West Green
    phase: int = Field(..., description="0 for North-South Green, 1 for East-West Green")


class DetraffObservation(Observation):
    # Traffic State
    lane_queues: Dict[str, int] = Field(..., description="Number of vehicles in each lane")
    emergency_waiting: Dict[str, bool] = Field(..., description="True if an emergency vehicle is in the lane")
    current_phase: int = Field(..., description="The currently active traffic light phase")
    
    # Required Hackathon Fields (Flat Style)
    reward: float = Field(..., description="Step reward normalized between 0.0 and 1.0")
    done: bool = Field(False, description="Whether the episode has reached its step limit")
    
    # Debugging/Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
