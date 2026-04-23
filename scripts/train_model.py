from __future__ import annotations

import logging
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Tuple

from scripts.common import probabilities_from_row, rank_outcomes, read_csv_semicolon, result_from_row, save_json

LOGGER = logging.getLogger(__name__)


def _compute_run_lengths(bits: List[int]) -> List[int]:
    runs: List[int] = []
    curr = 0
    for b in bits:
        if b == 1:
            curr += 1
        elif curr:
            runs.append(curr)
            curr = 0
    if curr:
        runs.append(curr)
    return runs


def _success_by_position(rows: List[Dict[str, str]], top_name: str) -> Dict[str, float]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["Concurso"]].append(row)

    pos_hits = defaultdict(list)
    all_runs: List[int] = []

    for concurso_rows in grouped.values():
        ordered = sorted(concurso_rows, key=lambda r: probabilities_from_row(r)[rank_outcomes(probabilities_from_row(r)).__getattribute__(top_name)], reverse=True)
        bits = []
        for idx, row in enumerate(ordered):
            hit = 1 if row.get(top_name, "0") == "1" else 0
            pos_hits[idx].append(hit)
            bits.append(hit)
        all_runs.extend(_compute_run_lengths(bits))

    out = {str(pos): mean(values) for pos, values in pos_hits.items() if values}
    out["avg_run"] = mean(all_runs) if all_runs else 0.0
    out["max_run"] = float(max(all_runs) if all_runs else 0)
    return out


def _team_outcome_rates(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: Dict[str, int] = defaultdict(int)
    for row in rows:
        res = result_from_row(row)
        if not res:
            continue
        for team_key in ("Mandante", "Visitante"):
            team = row[team_key]
            counts[team][res] += 1
            totals[team] += 1

    rates: Dict[str, Dict[str, float]] = {}
    for team, team_counts in counts.items():
        total = max(totals[team], 1)
        rates[team] = {k: team_counts.get(k, 0) / total for k in ("1", "X", "2")}
    return rates


def train() -> None:
    rows = read_csv_semicolon("data/concursos_anteriores.preprocessed.csv")
    model = {
        "top1": _success_by_position(rows, "top1"),
        "top2": _success_by_position(rows, "top2"),
        "top3": _success_by_position(rows, "top3"),
        "team_rates": _team_outcome_rates(rows),
        "metadata": {
            "rows": len(rows),
            "concursos": len({r['Concurso'] for r in rows}),
        },
    }
    save_json("models/model.json", model)

    LOGGER.info("Modelo treinado com %s linhas históricas.", len(rows))
    LOGGER.info("Média de runs top1/top2/top3: %.3f / %.3f / %.3f", model["top1"]["avg_run"], model["top2"]["avg_run"], model["top3"]["avg_run"])


if __name__ == "__main__":
    train()
