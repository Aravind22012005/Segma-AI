"""Grounds the synthetic customer generator in real data from the UCI Bank
Marketing dataset (https://archive.ics.uci.edu/dataset/222/bank+marketing),
downloaded live into `data/uci_raw/bank-full.csv` (45,211 real rows from a
Portuguese bank's term-deposit marketing campaign).

What this gives us that pure synthesis can't: real joint correlations between
age, job, marital status, education, housing/personal loan uptake, and prior
marketing-campaign response -- for the ~1,500 rows we do carve from it.

What it deliberately does NOT give us: the UCI dataset has no income,
credit score, city, or account-balance-in-INR fields, and its "balance" is a
2008-2010 Portuguese current-account balance in EUR -- not comparable in
currency, era, or context to an Indian retail bank's rupee balances. Rather
than fabricate a fake EUR->INR conversion, we only borrow the *shape*: each
sampled UCI row's balance percentile *within its job group* is carried over
and used to quantile-map into our own archetype-conditioned synthetic
balance distribution (see `uci_balance_quantile` below), so someone who
ranks high for their job in the real data also ranks high in ours -- without
pretending the raw numbers are interchangeable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

UCI_PATH = Path(__file__).resolve().parent / "uci_raw" / "bank-full.csv"

# Which real UCI `job` categories plausibly feed each persona archetype in
# generate_data.py. An archetype draws real (age, marital, education, loan,
# campaign-history) tuples only from rows whose job matches its list, so the
# real joint correlations between those fields are preserved per persona.
ARCHETYPE_TO_UCI_JOBS = {
    "Priority Affluent": ["management", "entrepreneur", "self-employed"],
    "Mass Salaried Regular": ["admin.", "technician", "services", "blue-collar"],
    "Young Digital": ["student", "admin.", "technician"],
    "Dormant Inactive": ["unemployed", "housemaid", "unknown", "blue-collar"],
    "Retiree Conservative": ["retired"],
    "Credit Stressed": ["blue-collar", "services", "unemployed"],
    "Business Growth": ["entrepreneur", "self-employed", "management"],
}

POUTCOME_LABELS = {
    "success": "Success", "failure": "Failure", "other": "Other", "unknown": "Never Contacted",
}


class UCICalibrationSource:
    def __init__(self, path: Path = UCI_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"UCI Bank Marketing data not found at {path}. Download it with:\n"
                f'  curl -L -o data/uci_raw/bank_marketing.zip '
                f'"https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"\n'
                f"  unzip -o data/uci_raw/bank_marketing.zip -d data/uci_raw && unzip -o data/uci_raw/bank.zip -d data/uci_raw"
            )
        df = pd.read_csv(path, sep=";", quotechar='"')
        # rank of each row's balance within its own job group -> used to
        # quantile-map into our synthetic INR balance distribution
        df["balance_quantile_in_job"] = df.groupby("job")["balance"].rank(pct=True)
        self.df = df
        self._by_job = {job: sub.reset_index(drop=True) for job, sub in df.groupby("job")}

    def sample_profile(self, archetype: str, rng: np.random.Generator) -> dict:
        jobs = ARCHETYPE_TO_UCI_JOBS.get(archetype, list(self._by_job.keys()))
        jobs = [j for j in jobs if j in self._by_job]
        job = jobs[rng.integers(0, len(jobs))]
        pool = self._by_job[job]
        row = pool.iloc[int(rng.integers(0, len(pool)))]

        prior_contacts = int(row["previous"])
        pdays = int(row["pdays"])
        return {
            "Age": int(row["age"]),
            "MaritalStatus": str(row["marital"]).capitalize(),
            "EducationLevel": str(row["education"]).capitalize(),
            "HousingLoanFlag": row["housing"] == "yes",
            "PersonalLoanFlag": row["loan"] == "yes",
            "PriorCampaignContacts": prior_contacts,
            "DaysSinceLastCampaignContact": None if pdays == -1 else pdays,
            "PriorCampaignOutcome": POUTCOME_LABELS.get(str(row["poutcome"]), "Never Contacted"),
            "uci_balance_quantile": float(row["balance_quantile_in_job"]),
            "uci_job": job,
        }


_source: UCICalibrationSource | None = None


def get_source() -> UCICalibrationSource:
    global _source
    if _source is None:
        _source = UCICalibrationSource()
    return _source
