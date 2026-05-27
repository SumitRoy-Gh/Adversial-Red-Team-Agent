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

# Each node is a Python function that:
# - receives the full RedTeamState
# - does one unit of work
# - returns a dict of only the keys it changed

class RedTeamSupervisor:
    def __init__(self, max_rounds=500):
        self.attacker   = AttackerAgent()
        self.defender   = DefenderAgent()
        self.evaluator  = ResponseEvaluator()
        self.exploit_store = ExploitStore()
        self.elo_tracker   = ELOTracker()
        self.max_rounds    = max_rounds
        self.graph         = self._build_graph()

    # ── NODE 1: fetch the target's system prompt ──────────────────────
    def node_fetch_target_info(self, state: RedTeamState) -> dict:
        """Called once at the start. Fetches target rules."""
        r = httpx.get('http://localhost:8000/system-prompt')
        att_elo, def_elo = self.elo_tracker.get_current_elos()
        return {
            'system_prompt': r.json()['system_prompt'],
            'attacker_elo':  att_elo,
            'defender_elo':  def_elo,
            'total_exploits': self.exploit_store.count(),
        }

    # ── NODE 2: attacker generates an attack ──────────────────────────
    def node_attacker_generate(self, state: RedTeamState) -> dict:
        # Get similar past exploits from Qdrant for novelty context
        past = self.exploit_store.get_similar_exploits(
            query_text=state['system_prompt'], top_k=5
        )
        attack = self.attacker.generate_attack(
            system_prompt=state['system_prompt'],
            past_attacks=past
        )
        print(f"[Round {state['round_num']}] ATTACK ({attack['attack_type']}): "
              f"{attack['attack_prompt'][:80]}...")
        return {
            'attack_prompt': attack['attack_prompt'],
            'attack_type':   attack['attack_type'],
            'past_exploits': past,
        }

    # ── NODE 3: send attack to target app ─────────────────────────────
    def node_query_target(self, state: RedTeamState) -> dict:
        r = httpx.post(
            'http://localhost:8000/query',
            json={'prompt': state['attack_prompt']},
            timeout=30.0
        )
        response = r.json()['response']
        print(f"[Round {state['round_num']}] RESPONSE: {response[:80]}...")
        return {'target_response': response}

    # ── NODE 4: evaluate whether the attack succeeded ─────────────────
    def node_evaluate(self, state: RedTeamState) -> dict:
        eval_result = self.evaluator.evaluate(
            system_prompt=state['system_prompt'],
            attack_prompt=state['attack_prompt'],
            response=state['target_response'],
        )
        print(f"[Round {state['round_num']}] VERDICT: {eval_result['verdict']} ",
              f"(severity={eval_result['severity']:.2f})")
        return {
            'eval_verdict':  eval_result['verdict'],
            'eval_severity': eval_result['severity'],
            'violated_rule': eval_result.get('violated_rule'),
        }

    # ── NODE 5: update ELO, store exploit if successful ───────────────
    def node_update_scores(self, state: RedTeamState) -> dict:
        attacker_won = (state['eval_verdict'] == 'EXPLOIT')

        new_att, new_def = self.elo_tracker.record_round(
            round_num=state['round_num'],
            attacker_won=attacker_won,
            attack_type=state['attack_type']
        )

        total = state['total_exploits']
        if attacker_won:
            self.exploit_store.store_exploit(
                attack_text=state['attack_prompt'],
                response_text=state['target_response'],
                attack_type=state['attack_type'],
                severity=state['eval_severity'],
                round_num=state['round_num']
            )
            total += 1
            print(f"[Round {state['round_num']}] *** EXPLOIT STORED *** ELO: ",
                  f"ATT={new_att:.0f} DEF={new_def:.0f}")

        from learning.fine_tune import DetectorFineTuner

        # Every 50 rounds, trigger fine-tuning if we have enough exploits
        if state['round_num'] % 50 == 0 and total >= 10:
            print(f'[Round {state["round_num"]}] Starting fine-tuning on {total} exploits...')
            tuner = DetectorFineTuner()
            metrics = tuner.fine_tune()
            print(f'Fine-tune complete. Eval accuracy: {metrics["eval_accuracy"]:.3f}')

        return {
            'attacker_elo':   new_att,
            'defender_elo':   new_def,
            'total_exploits': total,
        }

    # ── NODE 6: update defender strategy ─────────────────────────────
    def node_defender_update(self, state: RedTeamState) -> dict:
        self.defender.update(
            attack_prompt=state['attack_prompt'],
            attack_type=state['attack_type'],
            verdict=state['eval_verdict'],
            violated_rule=state['violated_rule'],
        )
        next_round = state['round_num'] + 1
        stop = next_round > self.max_rounds
        return {'round_num': next_round, 'stop_flag': stop}

    # ── Build and compile the LangGraph ──────────────────────────────
    def _build_graph(self) -> StateGraph:
        g = StateGraph(RedTeamState)

        # Add all nodes
        g.add_node('fetch_target',     self.node_fetch_target_info)
        g.add_node('attacker_generate',self.node_attacker_generate)
        g.add_node('query_target',     self.node_query_target)
        g.add_node('evaluate',         self.node_evaluate)
        g.add_node('update_scores',    self.node_update_scores)
        g.add_node('defender_update',  self.node_defender_update)

        # Set entry point — the first node that runs
        g.set_entry_point('fetch_target')

        # Linear edges: fetch → attack → query → evaluate → score → defend
        g.add_edge('fetch_target',      'attacker_generate')
        g.add_edge('attacker_generate', 'query_target')
        g.add_edge('query_target',      'evaluate')
        g.add_edge('evaluate',          'update_scores')
        g.add_edge('update_scores',     'defender_update')

        # Conditional edge: after defender update, either loop back or stop
        g.add_conditional_edges(
            'defender_update',
            lambda state: END if state['stop_flag'] else 'attacker_generate',
            {'attacker_generate': 'attacker_generate', END: END}
        )

        return g.compile()

    def run(self):
        """Start the self-play loop from the initial state."""
        initial_state = RedTeamState(
            round_num=1, system_prompt='',
            attack_prompt=None, attack_type=None, target_response=None,
            eval_verdict=None, eval_severity=None, violated_rule=None,
            defender_caught=None, attacker_elo=1200.0, defender_elo=1200.0,
            past_exploits=[], total_exploits=0, stop_flag=False
        )
        self.graph.invoke(initial_state)
