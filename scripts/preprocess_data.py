from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

from scripts.common import (
    format_decimal,
    probabilities_from_row,
    rank_outcomes,
    read_csv_semicolon,
    write_csv_semicolon,
)

LOGGER = logging.getLogger(__name__)


def enrich_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    enriched: List[Dict[str, str]] = []
    for row in rows:
        probs = probabilities_from_row(row)
        ranking = rank_outcomes(probs)

        row = dict(row)
        row["p(top1)"] = format_decimal(probs[ranking.top1], 9)
        row["p(top2)"] = format_decimal(probs[ranking.top2], 9)
        row["p(top3)"] = format_decimal(probs[ranking.top3], 9)

        for top, outcome in (("top1", ranking.top1), ("top2", ranking.top2), ("top3", ranking.top3)):
            row[top] = "1" if row.get(outcome, "0") == "1" else "0"

        enriched.append(row)
    return enriched


def preprocess() -> None:
    hist = read_csv_semicolon("data/concursos_anteriores.csv")
    upcoming = read_csv_semicolon("data/proximo_concurso.csv")

    hist_enriched = enrich_rows(hist)
    upcoming_enriched = enrich_rows(upcoming)

    write_csv_semicolon("data/concursos_anteriores.preprocessed.csv", hist_enriched, hist_enriched[0].keys())
    write_csv_semicolon("data/proximo_concurso.preprocessed.csv", upcoming_enriched, upcoming_enriched[0].keys())

    by_concurso = defaultdict(int)
    for row in hist_enriched:
        by_concurso[row["Concurso"]] += 1

    LOGGER.info("Pré-processamento concluído | concursos=%s | jogos=%s", len(by_concurso), len(hist_enriched))
    LOGGER.info("Arquivo salvo: data/concursos_anteriores.preprocessed.csv")
    LOGGER.info("Arquivo salvo: data/proximo_concurso.preprocessed.csv")


if __name__ == "__main__":
    preprocess()
