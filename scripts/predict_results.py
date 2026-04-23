from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

from scripts.common import probabilities_from_row, rank_outcomes, read_csv_semicolon, write_csv_semicolon, result_from_row, save_json

LOGGER = logging.getLogger(__name__)

BLACKLIST = {
    "PALMEIRAS/SP", "FLUMINENSE/RJ", "SAO PAULO/SP", "BAHIA/BA", "ATHLETICO/PR",
    "CORITIBA/PR", "BOTAFOGO/RJ", "VASCO DA GAMA/RJ", "ATLETICO/MG", "CRUZEIRO/MG",
}

TARGET_RANK_COUNTS = (9, 6, 5)  # top1, top2, top3
TARGET_SIGN_COUNTS = (9, 5, 6)  # 1, X, 2
TARGET_TRIPLES = 3


class Option(NamedTuple):
    palpite: str
    rank_delta: Tuple[int, int, int]
    sign_delta: Tuple[int, int, int]
    triple_delta: int
    score: float


def load_model() -> Dict:
    with Path("models/model.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _team_bonus(row: Dict[str, str], mark: str) -> float:
    bonus = 0.0
    if row["Mandante"] in BLACKLIST and mark == "2":
        bonus += 0.15
    if row["Visitante"] in BLACKLIST and mark == "1":
        bonus += 0.15
    if row["Mandante"] == "FLAMENGO/RJ" and mark == "1":
        bonus += 0.25
    if row["Visitante"] == "FLAMENGO/RJ" and mark == "2":
        bonus += 0.25
    return bonus


def _build_options(rows: List[Dict[str, str]], model: Dict) -> List[List[Option]]:
    options_per_game: List[List[Option]] = []
    for idx, row in enumerate(rows):
        probs = probabilities_from_row(row)
        ranks = rank_outcomes(probs)
        pos_top = {
            ranks.top1: 0,
            ranks.top2: 1,
            ranks.top3: 2,
        }

        game_options: List[Option] = []
        for mark in ("1", "X", "2"):
            rdelta = [0, 0, 0]
            sdelta = [0, 0, 0]
            rdelta[pos_top[mark]] += 1
            sdelta[(0 if mark == "1" else 1 if mark == "X" else 2)] += 1

            top_name = f"top{pos_top[mark] + 1}"
            score = float(model[top_name].get(str(idx), 0.0))
            score += _team_bonus(row, mark)

            game_options.append(Option(mark, tuple(rdelta), tuple(sdelta), 0, score))

        # triple 1X2
        game_options.append(
            Option(
                "1X2",
                (1, 1, 1),
                (1, 1, 1),
                1,
                0.9 + _team_bonus(row, "1") + _team_bonus(row, "2"),
            )
        )

        # force Flamengo favor by filtering options where Flamengo exists and mark against Flamengo.
        if row["Mandante"] == "FLAMENGO/RJ":
            game_options = [op for op in game_options if op.palpite in ("1", "1X2")]
        if row["Visitante"] == "FLAMENGO/RJ":
            game_options = [op for op in game_options if op.palpite in ("2", "1X2")]

        options_per_game.append(game_options)
    return options_per_game


def _add3(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def generate_predictions() -> None:
    model = load_model()
    rows = read_csv_semicolon("data/proximo_concurso.preprocessed.csv")
    rows = sorted(rows, key=lambda x: int(x["Jogo"]))
    options_per_game = _build_options(rows, model)

    @lru_cache(maxsize=None)
    def solve(i: int, r1: int, r2: int, r3: int, s1: int, sx: int, s2: int, tr: int) -> Optional[Tuple[float, Tuple[str, ...]]]:
        if i == len(rows):
            if (r1, r2, r3) == TARGET_RANK_COUNTS and (s1, sx, s2) == TARGET_SIGN_COUNTS and tr == TARGET_TRIPLES:
                return 0.0, tuple()
            return None

        # pruning upper/lower bounds
        rem = len(rows) - i
        if tr > TARGET_TRIPLES or tr + rem < TARGET_TRIPLES:
            return None
        if any(v > t for v, t in zip((r1, r2, r3), TARGET_RANK_COUNTS)):
            return None
        if any(v > t for v, t in zip((s1, sx, s2), TARGET_SIGN_COUNTS)):
            return None

        best: Optional[Tuple[float, Tuple[str, ...]]] = None
        for op in options_per_game[i]:
            nr = _add3((r1, r2, r3), op.rank_delta)
            ns = _add3((s1, sx, s2), op.sign_delta)
            nxt = solve(i + 1, nr[0], nr[1], nr[2], ns[0], ns[1], ns[2], tr + op.triple_delta)
            if nxt is None:
                continue
            total = op.score + nxt[0]
            seq = (op.palpite,) + nxt[1]
            if best is None or total > best[0]:
                best = (total, seq)
        return best

    solution = solve(0, 0, 0, 0, 0, 0, 0, 0)
    if solution is None:
        raise RuntimeError("Não foi encontrada solução viável para as hard constraints.")

    _, picks = solution
    output_rows: List[Dict[str, str]] = []
    debug_rows: List[Dict[str, str]] = []
    for row, pick in zip(rows, picks):
        out = dict(row)
        out["Palpite"] = pick
        output_rows.append(out)

        probs = probabilities_from_row(row)
        ranks = rank_outcomes(probs)
        debug_rows.append(
            {
                "Concurso": row["Concurso"],
                "Jogo": row["Jogo"],
                "Mandante": row["Mandante"],
                "Visitante": row["Visitante"],
                "Palpite": pick,
                "top1": ranks.top1,
                "top2": ranks.top2,
                "top3": ranks.top3,
                "p(top1)": f"{probs[ranks.top1]:.6f}",
                "p(top2)": f"{probs[ranks.top2]:.6f}",
                "p(top3)": f"{probs[ranks.top3]:.6f}",
            }
        )

    write_csv_semicolon("output/predictions.csv", output_rows, output_rows[0].keys())

    summary = {
        "hard_constraints": {
            "target_rank_counts_top1_top2_top3": TARGET_RANK_COUNTS,
            "target_sign_counts_1_x_2": TARGET_SIGN_COUNTS,
            "target_triples": TARGET_TRIPLES,
            "target_doubles": 0,
            "target_secos": len(rows) - TARGET_TRIPLES,
        },
        "solution": {
            "triples": sum(1 for p in picks if p == "1X2"),
            "doubles": sum(1 for p in picks if len(p) == 2),
            "secos": sum(1 for p in picks if len(p) == 1),
            "palpites": list(picks),
        },
        "debug_rows": debug_rows,
    }
    save_json("output/debug_metrics.json", summary)
    LOGGER.info("Predições geradas em output/predictions.csv")
    LOGGER.info("Debug salvo em output/debug_metrics.json")


if __name__ == "__main__":
    generate_predictions()
