# 🚦 Detraff: Autonomous Traffic Control RL Environment

**Detraff** is a Reinforcement Learning (RL) environment built using Meta's **OpenEnv** framework. It simulates a 4-way intersection where an AI agent must manage traffic lights to optimize vehicle throughput while maintaining **strict prioritization** for emergency vehicles (Ambulances, Fire Trucks).

This project was developed for **Round 1 of the Scalar OpenEnv Hackathon (2026)**.

-----

## 🏗️ Environment Specification

### 1\. Action Space

The agent controls the traffic light phases.

  * **`0`**: North-South Green (East-West Red)
  * **`1`**: East-West Green (North-South Red)

### 2\. Observation Space

The environment provides a "Flat Observation" containing:

  * `lane_queues`: Current vehicle count for North, South, East, and West lanes.
  * `emergency_waiting`: Boolean flags indicating if an Emergency Vehicle (EV) is stuck in a specific lane.
  * `current_phase`: The currently active light phase.
  * `reward`: The immediate feedback for the previous action.
  * `done`: Boolean indicating if the episode (100 steps) has concluded.

### 3\. Reward Function

To enforce strict priority, the reward is calculated using a weighted penalty system:

$$Reward = \max\left(0, \frac{100 - (\text{TotalCars} \times 1) - (\text{TotalEVs} \times 25)}{100}\right)$$

  * **Standard Penalty:** -1 per vehicle waiting in any lane.
  * **Priority Penalty:** -25 per Emergency Vehicle waiting in any lane.
  * **Normalization:** The score is clamped between $0.0$ and $1.0$.

-----

## 🚀 Getting Started

### Prerequisites

  * Python 3.10+
  * Docker (for local validation)
  * [uv](https://github.com/astral-sh/uv) (recommended for dependency management)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/detraff_env.git
    cd detraff_env
    ```

2.  **Set up the environment:**

    ```bash
    uv sync
    # OR
    pip install -r requirements.txt
    ```

3.  **Set Environment Variables:**

    ```bash
    export HF_TOKEN="your_huggingface_token"
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
    ```

-----

## 🛠️ Usage

### Running the Server Locally

To start the FastAPI environment server:

```bash
export PYTHONPATH=$PYTHONPATH:.
python server/app.py
```

The server will be available at `http://localhost:8000`. You can view the API documentation at `/docs`.

### Running the Inference Baseline

To run the AI agent and generate the mandatory hackathon logs:

```bash
python inference.py
```

-----

## 🧪 Validation & Submission

This environment is fully compliant with the **OpenEnv Spec**.

### 1\. Local Validation

To run the pre-submission validator:

```bash
bash validator.py http://localhost:8000 --repo_path .
```

### 2\. Hugging Face Deployment

The environment is deployed as a Hugging Face Space.

  * **Direct API URL:** `https://your-username-detraff-env.hf.space/web`
  * **Entry Point:** `server.app:app`

### 3\. Logging Format

The `inference.py` script emits structured logs required for automated grading:

  * `[START]`: Episode initialization.
  * `[STEP]`: Per-step action, reward, and state.
  * `[END]`: Final cumulative score and success status.

-----

## 📂 Project Structure

```text
detraff_env/
├── server/
│   ├── app.py              # FastAPI Server Entry Point
│   └── environment.py      # Core Traffic Logic & MDP
├── models.py               # Pydantic Action/Observation Schemas
├── inference.py            # AI Agent Baseline & Logging
├── openenv.yaml            # Hackathon Metadata & Task Definitions
├── pyproject.toml          # Dependency Management
└── Dockerfile              # Container Configuration
```

-----

**Collaborator:** Gemini 3 Flash (AI Plus Tier)  
**Hackathon:** Scalar OpenEnv Round 1 (2026)
