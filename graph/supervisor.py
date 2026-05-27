from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List
import httpx

from agents.attacker  import AttackerAgent
from agents.defender  import DefenderAgent
from agents.evaluator import ResponseEvaluator
from memory.exploit_store import ExploitStore
from memory.elo_tracker   import ELOTracker

# The State is the shared memory passed between every node in the graph.
# LangGraph passes this dict from node to node, updating it at each step.
class RedTeamState(TypedDict):
    round_num:       int
    system_prompt:   str          # Target's rules (fetched once, reused)
    attack_prompt:   Optional[str]
    attack_type:     Optional[str]
    target_response: Optional[str]
    eval_verdict:    Optional[str]   # 'EXPLOIT' or 'DEFENDED'
    eval_severity:   Optional[float]
    violated_rule:   Optional[str]
    defender_caught: Optional[bool]
    attacker_elo:    float
    defender_elo:    float
    past_exploits:   List[dict]      # Retrieved from Qdrant for context
    total_exploits:  int             # Running count
    stop_flag:       bool            # Set to True to end the loop

