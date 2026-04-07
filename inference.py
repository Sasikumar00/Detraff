import asyncio
import os
import textwrap
from typing import List, Optional
from openai import OpenAI

# Import your local models - Ensure PYTHONPATH=. is set
from models import DetraffAction
from server.detraff_env_environment import DetraffEnvironment

# Env Vars (Mandatory)
API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK_NAME", "normal_traffic")
BENCHMARK = "detraff_env"

MAX_STEPS = 10 
SYSTEM_PROMPT = """
You are an AI Traffic Controller for a 4-way intersection.
Your goal: Maximize traffic flow and prioritize emergency vehicles (EVs).
Actions: 
- Respond with "0" for North-South Green.
- Respond with "1" for East-West Green.
Strict Priority: If an emergency vehicle is waiting, you MUST turn that lane green immediately.
Reply ONLY with the number (0 or 1).
"""

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_action_from_llm(client: OpenAI, obs) -> int:
    prompt = f"Current Queues: {obs.lane_queues}. Emergency Waiting: {obs.emergency_waiting}. Current Phase: {obs.current_phase}. Choose 0 or 1:"
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0
        )
        content = res.choices[0].message.content.strip()
        if "1" in content: return 1
        if "0" in content: return 0
        return 0
    except: return 0

def get_success_threshold(task_name):
    thresholds = {
        "low_traffic": 0.8,
        "normal_traffic": 0.5,
        "emergency_peak": 0.3
    }
    return thresholds.get(task_name, 0.5)

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = DetraffEnvironment(task_name=TASK_NAME, max_steps=MAX_STEPS) # Local instantiation for validator
    
    rewards = []
    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)
    
    try:
        obs = env.reset()
        for step in range(1, MAX_STEPS + 1):
            action_int = get_action_from_llm(client, obs)
            obs = env.step(DetraffAction(phase=action_int))
            
            rewards.append(obs.reward)
            log_step(step, str(action_int), obs.reward, obs.done, None)
            if obs.done: break
            
        final_score = sum(rewards) / len(rewards) if rewards else 0.01
        final_score = max(0.01, min(0.99, final_score))
        threshold = get_success_threshold(TASK_NAME)
        success = final_score >= threshold
        log_end(success=success, steps=len(rewards), score=final_score, rewards=rewards)
    except Exception as e:
        log_end(success=False, steps=0, score=0.01, rewards=[])
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())