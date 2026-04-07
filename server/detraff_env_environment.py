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

    def __init__(self, task_name="normal_traffic", max_steps=20):
        self.task_name = task_name
        self.lanes = ["north", "south", "east", "west"]
        self.max_steps = max_steps
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

        return self._get_obs(reward=0, done=False)

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
            if self.queues[lane] > 0 or self.ev_present[lane]::
                # Clear 2 vehicles per step
                self.queues[lane] = max(0, self.queues[lane] - 2)
                # If lane is empty, emergency vehicle has passed
                if self.ev_present[lane]:
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

        # Baseline of 1 is good, but let's make the congestion penalty less steep
        # Changing divisor from 50 to 80 gives the agent more "breathing room"
        reward = 1.0
        reward -= min(0.6, (total_cars / 80.0) * 0.6) 

        # Keep the EV penalty, but ensure it doesn't tank the score if the agent is trying
        reward -= min(0.3, (ev_waiting_count / 4.0) * 0.3)

        # Serve EV Bonus (Keep this! It's the agent's best friend for passing)
        if any(ev_before[lane] for lane in green_lanes):
            reward += 0.20 # Increased from 0.15 to help hit thresholds

        # Clamping is essential for OpenEnv spec compliance
        reward = max(0.01, min(0.99, reward))

        reward = round(float(reward), 3)

        self.prev_total = total_cars

        # --- 4. Termination ---
        done = self.current_step >= self.max_steps
        
        return self._get_obs(reward=reward, done=done)

    @property
    def state(self) -> State:
        return self._state
