"""
Synthetic retail-banking data generator, grounded in real data where real
data actually exists (see `uci_calibration.py` and DATA_SCHEMA.md's "Data
provenance" section for exactly which fields are real vs. calibrated vs.
synthetic).

Produces three linked tables that together let a segmentation/personalization
agent answer almost any business question a judge could throw at it:

    customers.csv     -- static profile + account + credit snapshot
    products.csv       -- product ownership flags per customer
    transactions.csv   -- 24 months of transaction history per customer

Design note
-----------
Rows are generated from 7 hidden "persona archetypes" (see ARCHETYPES below)
that control the joint distribution of income, balance behaviour, spending
patterns and product ownership. This is what makes clustering/EDA on the
resulting data actually find something -- a purely independent-random CSV
clusters into noise.

Age, marital status, education, housing/personal loan uptake, and prior
marketing-campaign response are bootstrap-sampled from real rows of the UCI
Bank Marketing dataset (job-matched per archetype -- see
`ARCHETYPE_TO_UCI_JOBS` in uci_calibration.py), not invented. Balance keeps
its own archetype-conditioned synthetic distribution (UCI's EUR/2008-Portugal
balance isn't comparable to INR/today), but is quantile-mapped against each
sampled row's real balance rank within its job group, so real relative
standing carries over even though the currency doesn't.

The archetype label is written ONLY to `customers_ground_truth.csv`
(a superset of customers.csv with one extra `_persona_archetype` column).
The agent must never read that file for segmentation -- it exists purely so
we can sanity-check the agent's discovered segments against a known answer
during development/demo ("did the unsupervised clustering roughly recover
these personas?").

Everything is seeded (RANDOM_SEED) for reproducibility.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from uci_calibration import get_source as get_uci_source
from datetime import datetime, timedelta

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

N_CUSTOMERS = 1500
HISTORY_MONTHS = 24
REFERENCE_DATE = datetime(2026, 7, 25)  # "today" -- end of transaction window
START_DATE = REFERENCE_DATE - timedelta(days=30 * HISTORY_MONTHS)

CITIES = [
    ("Mumbai", "Tier1"), ("Delhi", "Tier1"), ("Bengaluru", "Tier1"),
    ("Chennai", "Tier1"), ("Kolkata", "Tier1"), ("Hyderabad", "Tier1"),
    ("Pune", "Tier1"), ("Ahmedabad", "Tier2"), ("Jaipur", "Tier2"),
    ("Lucknow", "Tier2"), ("Coimbatore", "Tier2"), ("Chandigarh", "Tier2"),
    ("Indore", "Tier2"), ("Nagpur", "Tier2"), ("Kochi", "Tier2"),
    ("Bhopal", "Tier3"), ("Patna", "Tier3"), ("Ranchi", "Tier3"),
]

MERCHANTS_BY_CATEGORY = {
    "Groceries": ["BigBasket", "DMart", "Reliance Fresh", "Local Kirana Store", "Nature's Basket"],
    "Dining": ["Zomato", "Swiggy", "Barbeque Nation", "Cafe Coffee Day", "Domino's"],
    "Fuel": ["Indian Oil", "HP Petrol Pump", "Shell", "Bharat Petroleum"],
    "Utilities": ["BSES Electricity", "Airtel Broadband", "Jio Fiber", "Municipal Water Board"],
    "Shopping": ["Amazon", "Flipkart", "Myntra", "Reliance Trends", "Croma"],
    "Entertainment": ["BookMyShow", "Netflix", "Amazon Prime", "PVR Cinemas", "Spotify"],
    "Travel": ["MakeMyTrip", "IRCTC", "IndiGo Airlines", "Ola", "Uber"],
    "Healthcare": ["Apollo Pharmacy", "Practo", "Max Healthcare", "1mg"],
    "Education": ["BYJU'S", "Coursera", "University Fee Portal", "Udemy"],
    "Rent": ["NoBroker Rent Pay", "Landlord Transfer"],
    "Salary Credit": ["Employer Payroll"],
    "Transfer": ["UPI Transfer", "NEFT Transfer", "IMPS Transfer"],
    "ATM Withdrawal": ["Self ATM Withdrawal"],
    "Investment": ["Zerodha", "Groww", "Mutual Fund SIP", "NPS Contribution"],
    "Insurance Premium": ["LIC Premium", "HDFC Life", "Star Health Insurance"],
    "EMI Payment": ["Loan EMI Auto-Debit"],
    "Online Services": ["Google Play", "Apple", "SaaS Subscription"],
    "Interest Credit": ["Savings Interest Credit"],
    "Business Revenue": ["POS Settlement", "Client Payment"],
}

# categories that are money coming IN
CREDIT_CATEGORIES = {"Salary Credit", "Interest Credit", "Business Revenue", "Transfer"}

CHANNELS = ["UPI", "Debit Card", "Credit Card", "NetBanking", "ATM", "Branch"]

OCCUPATIONS = ["Salaried", "Self-Employed", "Business Owner", "Professional",
               "Retired", "Student", "Homemaker"]

ACCOUNT_TYPES = ["Savings", "Current", "Salary", "Senior Citizen", "Student"]

LOAN_TYPES = ["Home Loan", "Personal Loan", "Auto Loan", "Education Loan"]

GENDERS = ["Male", "Female", "Other"]
GENDER_P = [0.49, 0.49, 0.02]

# ---------------------------------------------------------------------------
# Persona archetypes -- each is a bundle of distributions used to sample a
# realistic, internally-consistent customer. Weights must sum to 1.
# ---------------------------------------------------------------------------
ARCHETYPES = {
    "Priority Affluent": dict(
        weight=0.12, age=(35, 60), occupation_p={"Salaried": .35, "Business Owner": .3, "Professional": .35},
        income_lognorm=(14.6, 0.45),          # ~ INR 2.2M-8M/yr
        balance_lognorm=(13.3, 0.55),          # avg maintained balance
        balance_trend=(0.002, 0.01),           # slow steady growth
        txn_per_month=(18, 35), avg_txn_value_lognorm=(8.3, 0.7),
        credit_score=(750, 900), card_util=(5, 30),
        product_p=dict(SavingsAccount=.98, CreditCard=.9, FixedDeposit=.75,
                        Insurance=.8, Investment=.85, Loan=.35),
        spend_profile=dict(Groceries=.08, Dining=.1, Fuel=.05, Utilities=.05,
                            Shopping=.12, Entertainment=.06, Travel=.12,
                            Healthcare=.05, Investment=.12, InsurancePremium=.05,
                            Transfer=.1, Business=.0, EMI=.1),
    ),
    "Mass Salaried Regular": dict(
        weight=0.30, age=(25, 50), occupation_p={"Salaried": .85, "Professional": .15},
        income_lognorm=(13.2, 0.35),
        balance_lognorm=(11.2, 0.6),
        balance_trend=(-0.001, 0.008),
        txn_per_month=(8, 16), avg_txn_value_lognorm=(7.2, 0.6),
        credit_score=(650, 780), card_util=(20, 55),
        product_p=dict(SavingsAccount=.99, CreditCard=.55, FixedDeposit=.35,
                        Insurance=.45, Investment=.3, Loan=.3),
        spend_profile=dict(Groceries=.14, Dining=.1, Fuel=.08, Utilities=.08,
                            Shopping=.12, Entertainment=.06, Travel=.05,
                            Healthcare=.05, Investment=.04, InsurancePremium=.04,
                            Transfer=.1, Business=.0, EMI=.14),
    ),
    "Young Digital": dict(
        weight=0.15, age=(21, 32), occupation_p={"Salaried": .55, "Student": .3, "Self-Employed": .15},
        income_lognorm=(12.6, 0.4),
        balance_lognorm=(9.8, 0.7),
        balance_trend=(0.004, 0.015),
        txn_per_month=(14, 28), avg_txn_value_lognorm=(6.3, 0.7),
        credit_score=(600, 720), card_util=(15, 60),
        product_p=dict(SavingsAccount=.97, CreditCard=.4, FixedDeposit=.1,
                        Insurance=.15, Investment=.35, Loan=.1),
        spend_profile=dict(Groceries=.08, Dining=.18, Fuel=.04, Utilities=.05,
                            Shopping=.2, Entertainment=.14, Travel=.08,
                            Healthcare=.03, Investment=.06, InsurancePremium=.01,
                            Transfer=.12, Business=.0, EMI=.01),
    ),
    "Dormant Inactive": dict(
        weight=0.15, age=(22, 70), occupation_p={"Salaried": .4, "Homemaker": .25, "Retired": .2, "Self-Employed": .15},
        income_lognorm=(12.2, 0.5),
        balance_lognorm=(8.6, 0.8),
        balance_trend=(-0.01, 0.01),
        txn_per_month=(0, 4), avg_txn_value_lognorm=(6.5, 0.8),
        credit_score=(550, 750), card_util=(0, 40),
        product_p=dict(SavingsAccount=.99, CreditCard=.15, FixedDeposit=.15,
                        Insurance=.1, Investment=.05, Loan=.1),
        spend_profile=dict(Groceries=.15, Dining=.05, Fuel=.05, Utilities=.15,
                            Shopping=.08, Entertainment=.03, Travel=.02,
                            Healthcare=.08, Investment=.0, InsurancePremium=.02,
                            Transfer=.35, Business=.0, EMI=.02),
    ),
    "Retiree Conservative": dict(
        weight=0.10, age=(60, 85), occupation_p={"Retired": .9, "Homemaker": .1},
        income_lognorm=(12.6, 0.35),
        balance_lognorm=(12.4, 0.5),
        balance_trend=(-0.003, 0.005),
        txn_per_month=(3, 10), avg_txn_value_lognorm=(7.8, 0.6),
        credit_score=(700, 830), card_util=(0, 20),
        product_p=dict(SavingsAccount=.99, CreditCard=.2, FixedDeposit=.8,
                        Insurance=.65, Investment=.25, Loan=.05),
        spend_profile=dict(Groceries=.2, Dining=.05, Fuel=.03, Utilities=.12,
                            Shopping=.05, Entertainment=.02, Travel=.05,
                            Healthcare=.18, Investment=.05, InsurancePremium=.1,
                            Transfer=.15, Business=.0, EMI=.0),
    ),
    "Credit Stressed": dict(
        weight=0.10, age=(28, 50), occupation_p={"Salaried": .7, "Self-Employed": .3},
        income_lognorm=(12.9, 0.4),
        balance_lognorm=(9.0, 0.9),
        balance_trend=(-0.015, 0.02),
        txn_per_month=(10, 22), avg_txn_value_lognorm=(6.8, 0.7),
        credit_score=(300, 620), card_util=(65, 100),
        product_p=dict(SavingsAccount=.98, CreditCard=.7, FixedDeposit=.1,
                        Insurance=.2, Investment=.05, Loan=.65),
        spend_profile=dict(Groceries=.1, Dining=.08, Fuel=.06, Utilities=.08,
                            Shopping=.1, Entertainment=.05, Travel=.02,
                            Healthcare=.06, Investment=.0, InsurancePremium=.02,
                            Transfer=.13, Business=.0, EMI=.3),
    ),
    "Business Growth": dict(
        weight=0.08, age=(30, 55), occupation_p={"Business Owner": .7, "Self-Employed": .3},
        income_lognorm=(14.0, 0.6),
        balance_lognorm=(12.6, 0.8),
        balance_trend=(0.006, 0.03),
        txn_per_month=(20, 40), avg_txn_value_lognorm=(8.6, 0.9),
        credit_score=(680, 800), card_util=(30, 70),
        product_p=dict(SavingsAccount=.9, CreditCard=.6, FixedDeposit=.4,
                        Insurance=.5, Investment=.55, Loan=.55),
        spend_profile=dict(Groceries=.03, Dining=.05, Fuel=.06, Utilities=.06,
                            Shopping=.05, Entertainment=.02, Travel=.08,
                            Healthcare=.03, Investment=.08, InsurancePremium=.03,
                            Transfer=.16, Business=.3, EMI=.05),
    ),
}

SPEND_KEY_TO_CATEGORY = {
    "Groceries": "Groceries", "Dining": "Dining", "Fuel": "Fuel", "Utilities": "Utilities",
    "Shopping": "Shopping", "Entertainment": "Entertainment", "Travel": "Travel",
    "Healthcare": "Healthcare", "Investment": "Investment", "InsurancePremium": "Insurance Premium",
    "Transfer": "Transfer", "Business": "Business Revenue", "EMI": "EMI Payment",
}


def sample_archetypes(n):
    names = list(ARCHETYPES.keys())
    weights = [ARCHETYPES[k]["weight"] for k in names]
    return rng.choice(names, size=n, p=weights)


def build_customers():
    uci = get_uci_source()
    archetype_labels = sample_archetypes(N_CUSTOMERS)
    rows = []
    for i in range(N_CUSTOMERS):
        arc_name = archetype_labels[i]
        arc = ARCHETYPES[arc_name]

        # Real, job-matched bootstrap sample from the UCI Bank Marketing dataset --
        # age, marital status, education, housing/personal loan uptake, and prior
        # marketing-campaign response all come from an actual respondent row, not
        # a hand-picked distribution.
        uci_profile = uci.sample_profile(arc_name, rng)

        age = uci_profile["Age"]
        gender = rng.choice(GENDERS, p=GENDER_P)
        occ_names = list(arc["occupation_p"].keys())
        occ_p = list(arc["occupation_p"].values())
        occupation = rng.choice(occ_names, p=occ_p)
        city, tier = CITIES[rng.integers(0, len(CITIES))]

        income = float(rng.lognormal(arc["income_lognorm"][0], arc["income_lognorm"][1]))
        income = round(np.clip(income, 120_000, 15_000_000), 2)

        # Balance keeps our own archetype-conditioned INR distribution (UCI's EUR/2008
        # balance isn't comparable), but is quantile-mapped through the sampled row's
        # real balance rank within its job group -- so relative standing is real even
        # though the currency/scale isn't.
        q = float(np.clip(uci_profile["uci_balance_quantile"], 0.001, 0.999))
        mu, sigma = arc["balance_lognorm"]
        avg_balance = float(np.exp(mu + sigma * norm.ppf(q)))
        avg_balance = round(np.clip(avg_balance, 500, 20_000_000), 2)

        account_type = "Senior Citizen" if age >= 60 else (
            "Student" if occupation == "Student" else rng.choice(["Savings", "Current", "Salary"], p=[.55, .15, .3]))
        account_age_months = int(np.clip(rng.integers(1, 241), 1, 240))

        credit_score = int(np.clip(rng.normal((arc["credit_score"][0] + arc["credit_score"][1]) / 2,
                                               (arc["credit_score"][1] - arc["credit_score"][0]) / 4), 300, 900))

        # Housing/personal loan flags are real (UCI); Auto/Education loans, which UCI
        # doesn't capture, are still archetype-synthetic but only apply as a residual
        # on top of the real flags (not double-counted).
        if uci_profile["HousingLoanFlag"] and uci_profile["PersonalLoanFlag"]:
            loan_type, has_loan = rng.choice(["Home Loan", "Personal Loan"]), True
        elif uci_profile["HousingLoanFlag"]:
            loan_type, has_loan = "Home Loan", True
        elif uci_profile["PersonalLoanFlag"]:
            loan_type, has_loan = "Personal Loan", True
        elif rng.random() < arc["product_p"]["Loan"] * 0.35:
            loan_type, has_loan = rng.choice(["Auto Loan", "Education Loan"]), True
        else:
            loan_type, has_loan = "None", False
        loan_status = rng.choice(["Active", "Closed", "Defaulted"], p=[.75, .2, .05]) if has_loan else "None"
        emi = round(float(income) / 12 * rng.uniform(0.08, 0.35), 2) if (has_loan and loan_status == "Active") else 0.0

        has_card = rng.random() < arc["product_p"]["CreditCard"]
        # Beta(2,5) skews utilization toward the low end with a long tail -- shape
        # inspired by the published ~27.5% mean, right-skewed utilization distribution
        # in Kaggle's "Credit Card Customers" dataset (calibrated from documented
        # summary statistics, no raw file -- see DATA_SCHEMA.md).
        lo, hi = arc["card_util"]
        card_util = round(float(np.clip(lo + rng.beta(2, 5) * (hi - lo), 0, 100)), 1) if has_card else 0.0

        rows.append(dict(
            CustomerID=f"CUST{i+1:05d}",
            Age=age, Gender=gender, Occupation=occupation, AnnualIncome=income,
            MaritalStatus=uci_profile["MaritalStatus"], EducationLevel=uci_profile["EducationLevel"],
            City=city, CityTier=tier,
            AccountType=account_type, AccountAgeMonths=account_age_months,
            AvgMonthlyBalance=avg_balance,
            CreditScore=credit_score,
            LoanType=loan_type, LoanStatus=loan_status, EMI=emi,
            HasCreditCard=int(has_card), CardUtilizationPct=card_util,
            PriorCampaignContacts=uci_profile["PriorCampaignContacts"],
            DaysSinceLastCampaignContact=uci_profile["DaysSinceLastCampaignContact"],
            PriorCampaignOutcome=uci_profile["PriorCampaignOutcome"],
            _persona_archetype=arc_name,
            _balance_trend=rng.uniform(*arc["balance_trend"]),
            _txn_per_month_range=arc["txn_per_month"],
            _avg_txn_value_lognorm=arc["avg_txn_value_lognorm"],
            _spend_profile=arc["spend_profile"],
        ))
    return pd.DataFrame(rows)


def build_products(customers_df):
    rows = []
    for _, c in customers_df.iterrows():
        arc = ARCHETYPES[c["_persona_archetype"]]
        p = arc["product_p"]
        rows.append(dict(
            CustomerID=c["CustomerID"],
            SavingsAccount=int(rng.random() < p["SavingsAccount"]),
            CreditCard=int(c["HasCreditCard"]),
            FixedDeposit=int(rng.random() < p["FixedDeposit"]),
            Insurance=int(rng.random() < p["Insurance"]),
            Investment=int(rng.random() < p["Investment"]),
            Loan=int(c["LoanStatus"] != "None"),
        ))
    df = pd.DataFrame(rows)
    df["NumProductsOwned"] = df[["SavingsAccount", "CreditCard", "FixedDeposit",
                                  "Insurance", "Investment", "Loan"]].sum(axis=1)
    return df


def build_transactions(customers_df):
    all_rows = []
    txn_counter = 1
    for _, c in customers_df.iterrows():
        cust_id = c["CustomerID"]
        city = c["City"]
        lo, hi = c["_txn_per_month_range"]
        spend_profile = c["_spend_profile"]
        salary_eligible = c["Occupation"] in ("Salaried", "Professional")
        biz_eligible = c["Occupation"] in ("Business Owner", "Self-Employed")
        has_loan = c["LoanStatus"] == "Active"
        has_investment = rng.random() < ARCHETYPES[c["_persona_archetype"]]["product_p"]["Investment"]
        has_insurance = rng.random() < ARCHETYPES[c["_persona_archetype"]]["product_p"]["Insurance"]

        cat_names = list(spend_profile.keys())
        cat_p = np.array(list(spend_profile.values()))
        cat_p = cat_p / cat_p.sum()

        mu, sigma = c["_avg_txn_value_lognorm"]

        for m in range(HISTORY_MONTHS):
            month_start = START_DATE + timedelta(days=30 * m)
            n_txn = max(0, int(rng.poisson(rng.uniform(lo, hi))))

            # recurring salary credit
            if salary_eligible:
                all_rows.append((txn_counter, cust_id, month_start + timedelta(days=int(rng.integers(0, 5))),
                                  "Employer Payroll", "Salary Credit",
                                  round(float(c["AnnualIncome"]) / 12 * rng.uniform(0.95, 1.02), 2),
                                  "Credit", city, "NetBanking"))
                txn_counter += 1
            if biz_eligible:
                for _ in range(int(rng.integers(2, 8))):
                    all_rows.append((txn_counter, cust_id, month_start + timedelta(days=int(rng.integers(0, 29))),
                                      "Client Payment", "Business Revenue",
                                      round(float(rng.lognormal(mu, sigma)) * rng.uniform(1.5, 4), 2),
                                      "Credit", city, "NetBanking"))
                    txn_counter += 1
            if has_loan:
                all_rows.append((txn_counter, cust_id, month_start + timedelta(days=int(rng.integers(0, 10))),
                                  "Loan EMI Auto-Debit", "EMI Payment", round(float(c["EMI"]), 2),
                                  "Debit", city, "NetBanking"))
                txn_counter += 1
            if has_investment and rng.random() < 0.8:
                all_rows.append((txn_counter, cust_id, month_start + timedelta(days=int(rng.integers(0, 10))),
                                  "Mutual Fund SIP", "Investment",
                                  round(float(rng.lognormal(mu * 0.8, 0.5)), 2),
                                  "Debit", city, "NetBanking"))
                txn_counter += 1
            if has_insurance and rng.random() < 0.3:
                all_rows.append((txn_counter, cust_id, month_start + timedelta(days=int(rng.integers(0, 20))),
                                  "LIC Premium", "Insurance Premium",
                                  round(float(rng.lognormal(mu * 0.7, 0.4)), 2),
                                  "Debit", city, "NetBanking"))
                txn_counter += 1

            if n_txn == 0:
                continue

            categories = rng.choice(cat_names, size=n_txn, p=cat_p)
            days_offset = rng.integers(0, 29, size=n_txn)
            amounts = rng.lognormal(mu, sigma, size=n_txn)
            for cat_key, day_off, amt in zip(categories, days_offset, amounts):
                category = SPEND_KEY_TO_CATEGORY[cat_key]
                merchant = MERCHANTS_BY_CATEGORY[category][rng.integers(0, len(MERCHANTS_BY_CATEGORY[category]))]
                debit_credit = "Credit" if category in CREDIT_CATEGORIES else "Debit"
                txn_city = city if rng.random() > 0.08 else CITIES[rng.integers(0, len(CITIES))][0]
                channel = "UPI" if category in ("Groceries", "Dining", "Transfer") and rng.random() < 0.6 \
                    else CHANNELS[rng.integers(0, len(CHANNELS))]
                all_rows.append((txn_counter, cust_id, month_start + timedelta(days=int(day_off)),
                                  merchant, category, round(float(amt), 2), debit_credit, txn_city, channel))
                txn_counter += 1

    cols = ["TransactionID", "CustomerID", "Date", "Merchant", "Category",
            "Amount", "DebitCredit", "Location", "Channel"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["TransactionID"] = [f"TXN{n:07d}" for n in df["TransactionID"]]
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values(["CustomerID", "Date"]).reset_index(drop=True)
    return df


def main():
    print("Generating customers...")
    customers_df = build_customers()

    print("Generating products...")
    products_df = build_products(customers_df)

    print("Generating transactions (this is the slow part)...")
    transactions_df = build_transactions(customers_df)

    # ground-truth (dev-only) export, then strip internal columns for the
    # agent-facing dataset
    internal_cols = [c for c in customers_df.columns if c.startswith("_")]
    customers_df.drop(columns=[c for c in internal_cols if c != "_persona_archetype"]) \
        .to_csv("customers_ground_truth.csv", index=False)

    customers_clean = customers_df.drop(columns=internal_cols)
    customers_clean.to_csv("customers.csv", index=False)
    products_df.to_csv("products.csv", index=False)
    transactions_df.to_csv("transactions.csv", index=False)

    print(f"customers.csv        : {len(customers_clean):,} rows")
    print(f"products.csv          : {len(products_df):,} rows")
    print(f"transactions.csv      : {len(transactions_df):,} rows")
    print(f"customers_ground_truth.csv (dev only): {len(customers_df):,} rows")


if __name__ == "__main__":
    main()
