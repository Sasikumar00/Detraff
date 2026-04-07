import uuid
import random
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import DetraffAction, DetraffObservation


class DetraffEnvironment(Environment):
    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_name="normal_traffic"):
        self.task_name = task_name
        self.lanes = ["north", "south", "east", "west"]
        self.max_steps = 100
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)

        # Adjust difficulty based on task
        if task_name == "low_traffic":
            self.spawn_rate, self.ev_rate = 0.1, 0.0
        elif task_name == "emergency_peak":
            self.spawn_rate, self.ev_rate = 0.5, 0.2
        else: # normal_traffic
            self.spawn_rate, self.ev_rate = 0.3, 0.05

        self.reset()

    def reset(self) -> DetraffObservation:
        """Initializes the intersection to a clean state."""
        self.queues = {lane: 0 for lane in self.lanes}
        self.ev_present = {lane: False for lane in self.lanes}
        self.current_step = 0
        self.phase = 0

        # Reset the state for a new episode
        self._state.episode_id = str(uuid.uuid4())
        self._state.step_count = 0

        return self._get_obs(reward=0.0, done=False)

    def _get_obs(self, reward: float, done: bool) -> DetraffObservation:
        """Helper to package the current state into the flat model."""
        return DetraffObservation(
            lane_queues=self.queues.copy(),
            emergency_waiting=self.ev_present.copy(),
            current_phase=self.phase,
            reward=reward,
            done=done,
            metadata={
                "step": self.current_step,
                "total_vehicles": sum(self.queues.values()),
                "ev_count": sum(1 for v in self.ev_present.values() if v)
            }
        )

    def step(self, action: DetraffAction) -> DetraffObservation:
        # Increment step count in the state object
        self._state.step_count += 1

        self.current_step += 1
        self.phase = action.phase
        ev_before = self.ev_present.copy()
        
        # --- 1. Physics: Vehicles move through Green lights ---
        green_lanes = ["north", "south"] if self.phase == 0 else ["east", "west"]
        for lane in green_lanes:
            if self.queues[lane] > 0:
                # Clear 2 vehicles per step
                cleared = 2
                self.queues[lane] = max(0, self.queues[lane] - cleared)
                # If lane is empty, emergency vehicle has passed
                if self.queues[lane] == 0:
                    self.ev_present[lane] = False

        # --- 2. Simulation: New vehicles and EVs arrive ---
        for lane in self.lanes:
            # 30% chance of a standard car arriving
            if random.random() < self.spawn_rate:
                self.queues[lane] += 1
            
            if not self.ev_present[lane] and random.random() < self.ev_rate:
                self.ev_present[lane] = True
                self.queues[lane] += 1

        # --- 3. Scoring: Calculate Reward ---
        total_cars = sum(self.queues.values())
        ev_waiting_count = sum(1 for present in self.ev_present.values() if present)

        if not hasattr(self, "prev_total"):
            self.prev_total = total_cars
            self.prev_ev = ev_waiting_count
            delta_cars = 0
            delta_ev = 0
        else:
            delta_cars = self.prev_total - total_cars
            delta_ev = self.prev_ev - ev_waiting_count

        reward = 0.0

        reward += delta_cars * 0.05
        reward += delta_ev * 0.3

        reward -= total_cars * 0.002
        reward -= ev_waiting_count * 0.03
        reward -= 0.01

        green_lanes = ["north", "south"] if self.phase == 0 else ["east", "west"]
        if any(ev_before[lane] for lane in green_lanes):
            reward += 0.15

        reward = max(-1.0, min(1.0, reward))

        self.prev_total = total_cars
        self.prev_ev = ev_waiting_count

        # --- 4. Termination ---
        done = self.current_step >= self.max_steps
        
        return self._get_obs(reward=reward, done=done)

    @property
    def state(self) -> State:
        return self._state
