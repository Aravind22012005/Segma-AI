"""Unit tests for backend.tools.segmentation_tool.

rule_based_tiering():
    balance = view["AvgMonthlyBalance"], freq = view["MonthlyTxnFrequency"]
    dormant_freq_threshold = max(1.0, freq.quantile(0.15))   (unless overridden)
    score = 0.5 * zscore(balance) + 0.5 * zscore(freq)
    is_dormant  = (freq <= dormant_freq_threshold) | (RecencyDays >= dormant_recency_days [90])
    is_priority = (~is_dormant) & (score >= score.quantile(0.75)) & (balance >= balance.median())

ml_clustering():
    StandardScaler -> KMeans (auto k via silhouette over k_range if k is None) -> PCA(2D).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.tools.segmentation_tool import DEFAULT_TIER_LABELS, ml_clustering, rule_based_tiering


class TestRuleBasedTiering:
    def test_every_customer_gets_exactly_one_tier(self, tiering_view):
        tier, _rules = rule_based_tiering(tiering_view)

        assert len(tier) == len(tiering_view)
        assert tier.notna().all()
        assert set(tier.unique()) <= set(DEFAULT_TIER_LABELS)

    def test_priority_requires_and_not_or(self, tiering_view):
        """Row 0 is engineered with an extreme MonthlyTxnFrequency so its
        composite score alone clears the Priority score threshold, but its
        AvgMonthlyBalance is the lowest in the set (below the median). If
        the implementation used OR instead of AND for the score/balance
        conditions, this row would wrongly become Priority.
        """
        tier, rules = rule_based_tiering(tiering_view)

        balance = tiering_view["AvgMonthlyBalance"].astype(float)
        freq = tiering_view["MonthlyTxnFrequency"].astype(float)
        balance_z = (balance - balance.mean()) / balance.std()
        freq_z = (freq - freq.mean()) / freq.std()
        score = 0.5 * balance_z + 0.5 * freq_z

        score_threshold = rules["thresholds"]["priority_score_threshold"]
        balance_floor = rules["thresholds"]["priority_balance_floor"]

        # Confirm this is a genuine "OR would wrongly pass" case: the score
        # condition is satisfied on its own, but the balance condition is not.
        assert score.iloc[0] >= score_threshold
        assert balance.iloc[0] < balance_floor

        # And the AND-based implementation correctly withholds Priority.
        assert tier.iloc[0] != "Priority"
        assert tier.iloc[0] == "Regular"

    def test_priority_requires_both_high_score_and_high_balance(self, tiering_view):
        """Rows 19-23 have both high balance and high frequency, so they
        should legitimately earn Priority."""
        tier, _rules = rule_based_tiering(tiering_view)

        assert (tier.iloc[19:24] == "Priority").all()

    def test_dormant_for_low_frequency(self, tiering_view):
        """Rows 1-4 sit at/below the 15th-percentile frequency threshold and
        should be flagged Dormant even though RecencyDays is low (10)."""
        tier, rules = rule_based_tiering(tiering_view)

        freq = tiering_view["MonthlyTxnFrequency"].astype(float)
        dormant_freq_threshold = rules["thresholds"]["dormant_freq_threshold"]

        low_freq_mask = freq <= dormant_freq_threshold
        assert low_freq_mask.iloc[1:5].all()  # sanity check on the fixture itself
        assert (tier[low_freq_mask] == "Dormant").all()

    def test_dormant_for_high_recency_even_with_healthy_frequency_and_balance(self):
        """A customer with normal balance/frequency but RecencyDays past the
        dormant_recency_days threshold (default 90) must still be Dormant --
        the recency condition is an OR with the frequency condition."""
        view = pd.DataFrame(
            {
                "CustomerID": [f"CUST{i:03d}" for i in range(10)],
                "AvgMonthlyBalance": np.linspace(10000, 30000, 10),
                "MonthlyTxnFrequency": np.linspace(20, 30, 10),  # comfortably above the 15th pct
                "RecencyDays": [10.0] * 9 + [95.0],  # last customer: stale
            }
        )

        tier, rules = rule_based_tiering(view)

        dormant_freq_threshold = rules["thresholds"]["dormant_freq_threshold"]
        assert view["MonthlyTxnFrequency"].iloc[-1] > dormant_freq_threshold

        # The stale customer is Dormant purely via the recency OR-branch...
        assert tier.iloc[-1] == "Dormant"
        # ...while every other row, none of which crosses the recency
        # threshold, is Dormant if and only if its own frequency is at/below
        # the frequency threshold (the OR's other branch) -- proving recency
        # is evaluated independently rather than being ignored.
        freq_dormant = view["MonthlyTxnFrequency"].iloc[:9] <= dormant_freq_threshold
        assert ((tier.iloc[:9] == "Dormant") == freq_dormant).all()

    def test_zero_transaction_customer_does_not_crash_and_is_dormant(self, tiering_view, zero_txn_row):
        """A customer with zero transactions has MonthlyTxnFrequency == 0 and
        RecencyDays == 9999 (per build_unified_customer_view's fill logic).
        It should be classified without error and land in Dormant."""
        view = pd.concat([tiering_view, pd.DataFrame([zero_txn_row])], ignore_index=True)

        tier, _rules = rule_based_tiering(view)

        assert len(tier) == len(view)
        assert tier.notna().all()
        assert tier.iloc[-1] == "Dormant"


class TestMlClustering:
    def test_labels_every_row(self, clustering_view):
        features = ["AnnualIncome", "MonthlyTxnFrequency", "CreditScore"]

        result = ml_clustering(clustering_view, features=features)

        labels = result["labels"]
        assert len(labels) == len(clustering_view)
        assert not pd.isna(labels).any()

    def test_produces_documented_outputs(self, clustering_view):
        features = ["AnnualIncome", "MonthlyTxnFrequency", "CreditScore"]

        result = ml_clustering(clustering_view, features=features)

        k = result["k"]
        assert isinstance(k, int)
        assert k in range(3, 7)

        # centroids: one row per cluster, one column per feature
        assert len(result["centroids"]) == k
        for centroid in result["centroids"].values():
            assert set(centroid.keys()) == set(features)

        # PCA(2D) projection: one 2D coordinate per row
        pca_coords = np.asarray(result["pca_coords"])
        assert pca_coords.shape == (len(clustering_view), 2)

        assert len(result["pca_explained_variance"]) == 2
        assert set(np.unique(result["labels"])) == set(range(k))

    def test_fixed_k_skips_auto_selection(self, clustering_view):
        features = ["AnnualIncome", "MonthlyTxnFrequency", "CreditScore"]

        result = ml_clustering(clustering_view, features=features, k=3)

        assert result["k"] == 3
        assert result["k_selection_scores"] == {}
        assert len(result["centroids"]) == 3

    def test_recovers_the_three_well_separated_groups(self, clustering_view):
        """The fixture's three groups are separated by thousands of units in
        AnnualIncome, so silhouette-based k selection should land on k=3 and
        every row within one of the original 10-row groups should land in a
        single cluster."""
        features = ["AnnualIncome", "MonthlyTxnFrequency", "CreditScore"]

        result = ml_clustering(clustering_view, features=features)

        labels = np.asarray(result["labels"])
        assert result["k"] == 3
        for start in (0, 10, 20):
            group_labels = labels[start : start + 10]
            assert len(set(group_labels)) == 1
