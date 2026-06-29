import os
import re
import json
import asyncio
import httpx
from typing import List, Dict, Any
from openai import AsyncOpenAI


class SwarmOrchestrator:
    def __init__(self, agent_configs: Dict[str, Any], nvidia_api_key: str = None):
        self.agent_configs = agent_configs
        self.api_key = nvidia_api_key or os.getenv("NVIDIA_API_KEY")

        if not self.api_key:
            raise ValueError(
                "NVIDIA_API_KEY is missing. Pass it directly or set the "
                "NVIDIA_API_KEY environment variable."
            )

        # OPTIMIZATION: High-performance connection pooling for concurrent agent streams
        self.http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            timeout=httpx.Timeout(10.0, connect=2.0),
        )

        self.client = AsyncOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key,
            http_client=self.http_client,
        )

        # OPTIMIZATION: Non-greedy regex to eliminate catastrophic CPU backtracking
        self.json_extractor = re.compile(r"(\{.*?\})", re.DOTALL)

    # ------------------------------------------------------------------
    # LLM-backed agent thought generation
    # ------------------------------------------------------------------
    async def generate_agent_thought(self, agent_name: str, environment_state: str) -> str:
        """Call the NVIDIA-hosted LLM to produce an agent action payload."""
        system_prompt = f"You are {agent_name}, operating in a coordinated robotic mesh."
        user_prompt = (
            f"Environment Telemetry:\n{environment_state}\n"
            f"Respond strictly in this JSON template:\n"
            f'{{\n  "agent": "{agent_name}",\n  "action": "update_grid",\n'
            f'  "target_cell": [X, Y, Z],\n  "velocity": 1.25\n}}'
        )

        try:
            response = await self.client.chat.completions.create(
                model="meta/llama-3.1-8b-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=128,
                temperature=0.0,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return json.dumps({
                "agent": agent_name,
                "error": str(e),
                "action": "noop",
            })

    # ------------------------------------------------------------------
    # JSON response parser
    # ------------------------------------------------------------------
    async def parse_agent_response(self, agent_name: str, raw_text: str) -> Dict[str, Any]:
        """Extract and validate JSON from raw LLM output."""
        match = self.json_extractor.search(raw_text)
        if not match:
            return {"status": "fallback", "agent": agent_name, "content": raw_text}

        cleaned_payload = match.group(0)

        # Safe exception boundary — prevent serialization failures from crashing the loop
        try:
            parsed_data = json.loads(cleaned_payload)
            return {"status": "success", "agent": agent_name, "data": parsed_data}
        except (json.JSONDecodeError, TypeError) as e:
            return {
                "status": "parse_error",
                "agent": agent_name,
                "error": str(e),
                "raw_captured": cleaned_payload,
            }

    # ------------------------------------------------------------------
    # Orchestration step — fan-out to all agents, fan-in parsed results
    # ------------------------------------------------------------------
    async def orchestrate_step(self, payload_context: str) -> List[Dict[str, Any]]:
        """Run all agents in parallel and return their parsed responses."""
        # Fan-out: generate thoughts concurrently for every agent
        raw_tasks = [
            self.generate_agent_thought(name, payload_context)
            for name in self.agent_configs.keys()
        ]
        raw_results: List[str] = await asyncio.gather(*raw_tasks)

        # Fan-in: parse each raw LLM response
        parse_tasks = [
            self.parse_agent_response(name, raw)
            for name, raw in zip(self.agent_configs.keys(), raw_results)
        ]
        parsed_results: List[Dict[str, Any]] = await asyncio.gather(*parse_tasks)
        return parsed_results

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------
    async def close(self):
        """Close the underlying HTTP connection pool."""
        await self.http_client.aclose()


# ======================================================================
# Standalone test harness
# ======================================================================
async def _main():
    agent_configs = {
        "DroneAlpha": {"role": "scout", "priority": 1},
        "DroneBeta": {"role": "collector", "priority": 2},
        "DroneGamma": {"role": "defender", "priority": 3},
    }

    orchestrator = SwarmOrchestrator(agent_configs=agent_configs)
    try:
        env_state = (
            "Grid: 100x100  |  Obstacle count: 12  |  "
            "Ally positions: [(10,20,0), (55,42,0)]  |  Threat level: LOW"
        )
        print(">> Running orchestration step ...")
        results = await orchestrator.orchestrate_step(env_state)
        print(json.dumps(results, indent=2))
    finally:
        await orchestrator.close()


if __name__ == "__main__":
    asyncio.run(_main())
