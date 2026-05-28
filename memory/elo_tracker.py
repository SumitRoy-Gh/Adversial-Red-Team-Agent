import sqlite3, math
from typing import Tuple, List
from datetime import datetime

class ELOTracker:
    K_FACTOR   = 32    # How much a single round shifts the ELO score
    INITIAL_ELO = 1200  # Starting ELO for both agents (chess standard)

    def __init__(self, db_path='elo_scores.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._setup()

    def _setup(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS elo_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                round_num   INTEGER,
                timestamp   TEXT,
                attacker_elo REAL,
                defender_elo REAL,
                attacker_won INTEGER,   -- 1 if attacker succeeded, 0 if caught
                attack_type TEXT
            )
        ''')
        self.conn.commit()

    def get_current_elos(self) -> Tuple[float, float]:
        """Returns (attacker_elo, defender_elo)."""
        row = self.conn.execute(
            'SELECT attacker_elo, defender_elo FROM elo_history ORDER BY id DESC LIMIT 1'
        ).fetchone()
        return row if row else (self.INITIAL_ELO, self.INITIAL_ELO)

    def get_last_round_num(self) -> int:
        """Returns the last recorded round number, or 0 if none exist."""
        row = self.conn.execute(
            'SELECT round_num FROM elo_history ORDER BY id DESC LIMIT 1'
        ).fetchone()
        return row[0] if row else 0

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Standard ELO expected score formula."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def record_round(self, round_num: int, attacker_won: bool, attack_type: str):
        """
        Update ELO scores after one round.
        attacker_won=True means the attack bypassed the defender (attacker point).
        attacker_won=False means the defender caught the attack (defender point).
        """
        att_elo, def_elo = self.get_current_elos()

        # Actual scores: 1 = win, 0 = loss
        att_actual = 1.0 if attacker_won else 0.0
        def_actual = 1.0 - att_actual

        # Expected scores based on current ratings
        att_expected = self._expected_score(att_elo, def_elo)
        def_expected = self._expected_score(def_elo, att_elo)

        # New ratings
        new_att = att_elo + self.K_FACTOR * (att_actual - att_expected)
        new_def = def_elo + self.K_FACTOR * (def_actual - def_expected)

        self.conn.execute(
            '''INSERT INTO elo_history
               (round_num, timestamp, attacker_elo, defender_elo, attacker_won, attack_type)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (round_num, datetime.utcnow().isoformat(),
             round(new_att, 2), round(new_def, 2), int(attacker_won), attack_type)
        )
        self.conn.commit()
        return new_att, new_def

    def get_history(self) -> List[dict]:
        rows = self.conn.execute(
            'SELECT * FROM elo_history ORDER BY round_num'
        ).fetchall()
        cols = ['id','round_num','timestamp','attacker_elo','defender_elo','attacker_won','attack_type']
        return [dict(zip(cols, r)) for r in rows]

