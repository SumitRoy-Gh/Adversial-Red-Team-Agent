import torch  # Must be imported first on Windows to avoid DLL load/initialization conflicts with CUDA/sentence_transformers
from dotenv import load_dotenv
load_dotenv(override=True)

from graph.supervisor import RedTeamSupervisor

if __name__ == '__main__':
    print('Starting Adversarial Red-Team Self-Play Loop...')
    print('Open http://localhost:8501 for the live dashboard')
    supervisor = RedTeamSupervisor(max_rounds=500)
    supervisor.run()
    print('Done. Check elo_scores.db and qdrant_storage for results.')

