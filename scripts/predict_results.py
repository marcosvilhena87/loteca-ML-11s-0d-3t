from __future__ import annotations

import json
import logging
from math import log
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

from scripts.common import probabilities_from_row, rank_outcomes, read_csv_semicolon, write_csv_semicolon, save_json

LOGGER = logging.getLogger(__name__)

BLACKLIST = {
    "PALMEIRAS/SP", "FLUMINENSE/RJ", "SAO PAULO/SP", "BAHIA/BA", "ATHLETICO/PR",
    "CORITIBA/PR", "BOTAFOGO/RJ", "VASCO DA GAMA/RJ", "ATLETICO/MG", "CRUZEIRO/MG",
}

TARGET_RANK_COUNTS = (9, 6, 5)  # top1, top2, top3
TARGET_SIGN_COUNTS = (9, 5, 6)  # 1, X, 2
TARGET_TRIPLES = 3
TOP_K_CANDIDATES = 10


class Option(NamedTuple):
    palpite: str
    rank_delta: Tuple[int, int, int]
    sign_delta: Tuple[int, int, int]
    triple_delta: int
    score: float
    is_top1: bool
    entropy: float


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


def _team_rate_bonus(model: Dict, row: Dict[str, str], mark: str) -> float:
    team_rates = model.get("team_rates", {})
    mandante = row["Mandante"]
    visitante = row["Visitante"]
    if mark == "1":
        return 0.10 * team_rates.get(mandante, {}).get("1", 0.0)
    if mark == "X":
        return 0.10 * (
            team_rates.get(mandante, {}).get("X", 0.0)
            + team_rates.get(visitante, {}).get("X", 0.0)
        ) / 2.0
    if mark == "2":
        return 0.10 * team_rates.get(visitante, {}).get("2", 0.0)
    return 0.0


def _entropy(probs: Dict[str, float]) -> float:
    return -sum(p * log(p) for p in probs.values() if p > 0)


def _build_options(rows: List[Dict[str, str]], model: Dict) -> List[List[Option]]:
    options_per_game: List[List[Option]] = []
    for idx, row in enumerate(rows):
        probs = probabilities_from_row(row)
        ranks = rank_outcomes(probs)
        entropy = _entropy(probs)
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
            score += _team_rate_bonus(model, row, mark)

            game_options.append(
                Option(mark, tuple(rdelta), tuple(sdelta), 0, score, mark == ranks.top1, entropy)
            )

        # triple 1X2
        game_options.append(
            Option(
                "1X2",
                (1, 1, 1),
                (1, 1, 1),
                1,
                0.60 + 0.40 * entropy + _team_bonus(row, "1") + _team_bonus(row, "2"),
                False,
                entropy,
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


def _top_k(candidates: List[Tuple[float, Tuple[str, ...]]], k: int) -> List[Tuple[float, Tuple[str, ...]]]:
    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[:k]


def generate_predictions() -> None:
    model = load_model()
    rows = read_csv_semicolon("data/proximo_concurso.preprocessed.csv")
    rows = sorted(rows, key=lambda x: int(x["Jogo"]))
    options_per_game = _build_options(rows, model)

    avg_run_top1 = float(model.get("top1", {}).get("avg_run", 0.0))

    @lru_cache(maxsize=None)
    def solve(
        i: int, r1: int, r2: int, r3: int, s1: int, sx: int, s2: int, tr: int, run_top1: int
    ) -> Tuple[Tuple[float, Tuple[str, ...]], ...]:
        if i == len(rows):
            if (r1, r2, r3) == TARGET_RANK_COUNTS and (s1, sx, s2) == TARGET_SIGN_COUNTS and tr == TARGET_TRIPLES:
                final_penalty = abs(run_top1 - avg_run_top1) * 0.15 if run_top1 else 0.0
                return ((-final_penalty, tuple()),)
            return tuple()

        # pruning upper/lower bounds
        rem = len(rows) - i
        if tr > TARGET_TRIPLES or tr + rem < TARGET_TRIPLES:
            return tuple()
        if any(v > t for v, t in zip((r1, r2, r3), TARGET_RANK_COUNTS)):
            return tuple()
        if any(v > t for v, t in zip((s1, sx, s2), TARGET_SIGN_COUNTS)):
            return tuple()

        candidates: List[Tuple[float, Tuple[str, ...]]] = []
        for op in options_per_game[i]:
            nr = _add3((r1, r2, r3), op.rank_delta)
            ns = _add3((s1, sx, s2), op.sign_delta)
            next_run_top1 = run_top1 + 1 if op.is_top1 else 0
            run_penalty = abs(next_run_top1 - avg_run_top1) * 0.15 if op.is_top1 else (abs(run_top1 - avg_run_top1) * 0.15 if run_top1 else 0.0)
            nxt_candidates = solve(i + 1, nr[0], nr[1], nr[2], ns[0], ns[1], ns[2], tr + op.triple_delta, next_run_top1)
            for nxt_score, nxt_seq in nxt_candidates:
                total = op.score - run_penalty + nxt_score
                seq = (op.palpite,) + nxt_seq
                candidates.append((total, seq))
        return tuple(_top_k(candidates, TOP_K_CANDIDATES))

    solutions = list(solve(0, 0, 0, 0, 0, 0, 0, 0, 0))
    if not solutions:
        raise RuntimeError("Não foi encontrada solução viável para as hard constraints.")

    solutions.sort(key=lambda x: x[0], reverse=True)
    best_score, picks = solutions[0]
    output_rows: List[Dict[str, str]] = []
    debug_rows: List[Dict[str, str]] = []
    triple_entropy_rows: List[Dict[str, str]] = []
    picked_option_by_game = {
        (i, op.palpite): op for i, game_options in enumerate(options_per_game) for op in game_options
    }
    for idx, (row, pick) in enumerate(zip(rows, picks)):
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
        if pick == "1X2":
            triple_entropy_rows.append(
                {
                    "Jogo": row["Jogo"],
                    "Mandante": row["Mandante"],
                    "Visitante": row["Visitante"],
                    "entropia": f"{picked_option_by_game[(idx, pick)].entropy:.6f}",
                }
            )

    write_csv_semicolon("output/predictions.csv", output_rows, output_rows[0].keys())

    count_signs = {
        "1": sum(1 for p in picks if p == "1"),
        "X": sum(1 for p in picks if p == "X"),
        "2": sum(1 for p in picks if p == "2"),
    }
    rank_counts = {"top1": 0, "top2": 0, "top3": 0}
    blacklist_conflicts = 0
    flamengo_favor = 0
    risks: List[Tuple[float, Dict[str, str]]] = []
    for row, pick in zip(rows, picks):
        probs = probabilities_from_row(row)
        ranks = rank_outcomes(probs)
        if pick in ("1", "X", "2"):
            if pick == ranks.top1:
                rank_counts["top1"] += 1
            elif pick == ranks.top2:
                rank_counts["top2"] += 1
            else:
                rank_counts["top3"] += 1
            sorted_probs = sorted(probs.values(), reverse=True)
            margin = sorted_probs[0] - sorted_probs[1]
            risks.append((margin, {"Jogo": row["Jogo"], "Mandante": row["Mandante"], "Visitante": row["Visitante"], "Palpite": pick, "margem_probabilidade": f"{margin:.6f}"}))
        if row["Mandante"] in BLACKLIST and pick == "1":
            blacklist_conflicts += 1
        if row["Visitante"] in BLACKLIST and pick == "2":
            blacklist_conflicts += 1
        if row["Mandante"] == "FLAMENGO/RJ" and pick in ("1", "1X2"):
            flamengo_favor += 1
        if row["Visitante"] == "FLAMENGO/RJ" and pick in ("2", "1X2"):
            flamengo_favor += 1

    summary = {
        "hard_constraints": {
            "target_rank_counts_top1_top2_top3": TARGET_RANK_COUNTS,
            "target_sign_counts_1_x_2": TARGET_SIGN_COUNTS,
            "target_triples": TARGET_TRIPLES,
            "target_doubles": 0,
            "target_secos": len(rows) - TARGET_TRIPLES,
        },
        "solution": {
            "score_total": round(best_score, 6),
            "triples": sum(1 for p in picks if p == "1X2"),
            "doubles": sum(1 for p in picks if len(p) == 2),
            "secos": sum(1 for p in picks if len(p) == 1),
            "palpites": list(picks),
        },
        "post_solver_report": {
            "quantidade_top1_top2_top3": rank_counts,
            "quantidade_1_x_2": count_signs,
            "jogos_contra_blacklist": blacklist_conflicts,
            "jogos_pro_flamengo": flamengo_favor,
            "triplos_escolhidos_por_entropia": triple_entropy_rows,
            "maiores_riscos_da_aposta": [r[1] for r in sorted(risks, key=lambda x: x[0])[:5]],
        },
        "candidate_bets": [
            {"rank": idx + 1, "score": round(score, 6), "palpites": list(candidate)}
            for idx, (score, candidate) in enumerate(solutions[:TOP_K_CANDIDATES])
        ],
        "debug_rows": debug_rows,
    }
    save_json("output/debug_metrics.json", summary)
    LOGGER.info("Predições geradas em output/predictions.csv")
    LOGGER.info("Debug salvo em output/debug_metrics.json")


if __name__ == "__main__":
    generate_predictions()
