"""
Synthetic Vendor Dataset Generator
-----------------------------------
Generates a realistic oil & gas procurement vendor dataset for the
Vendor Reliability Scoring project.

Design idea (the psych/econ bridge):
Each vendor has a hidden ("latent") underlying reliability trait —
similar to how psychometrics treats traits like conscientiousness as
unobservable but expressed through multiple measurable indicators.
We never give the model this latent trait directly. Instead, we let
it drive noisy, imperfect observable signals across three domains:

  1. Operational  (on-time delivery, defect rate, lead time variance)
  2. Financial     (price competitiveness, payment terms compliance)
  3. Behavioral    (RFQ response speed, communication responsiveness,
                    price-quote volatility, dispute frequency)

This mirrors real procurement practice: you never see "reliability"
directly, you infer it from a mix of hard operational data and softer
behavioral signals — exactly the kind of measurement problem
psychology is built to handle, applied to an economic decision.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_VENDORS = 450

rng = np.random.default_rng(RNG_SEED)

CATEGORIES = [
    "Valves & Actuators", "Pipes & Fittings", "Instrumentation",
    "Electrical Equipment", "Safety Equipment (PPE)", "Drilling Tools",
    "Industrial Chemicals", "Rotating Equipment Spares", "Structural Steel",
    "Subsea Hardware",
]

COUNTRIES = [
    "China", "United Kingdom", "United States", "Nigeria", "UAE",
    "India", "Germany", "South Korea", "Italy", "Singapore",
]
# Rough sourcing-country weighting (mirrors typical vendor pool skew)
COUNTRY_WEIGHTS = [0.28, 0.10, 0.10, 0.14, 0.09, 0.09, 0.07, 0.05, 0.04, 0.04]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def clip01(x):
    return np.clip(x, 0, 1)


def generate_vendors(n=N_VENDORS, seed=RNG_SEED):
    rng = np.random.default_rng(seed)

    vendor_id = [f"V{str(i).zfill(4)}" for i in range(1, n + 1)]
    category = rng.choice(CATEGORIES, size=n)
    country = rng.choice(COUNTRIES, size=n, p=COUNTRY_WEIGHTS)

    # --- Latent reliability trait (standard normal) ---
    # This is the "ground truth" psychological/operational trait we
    # are trying to recover. It is NOT included in the final dataset.
    latent_reliability = rng.normal(loc=0, scale=1, size=n)

    # Vendor tenure and scale (partly independent of reliability,
    # but slightly correlated — older/bigger vendors trend more reliable)
    years_as_vendor = np.clip(
        rng.gamma(shape=2.0, scale=2.2, size=n) + 0.4 * latent_reliability, 0.3, 20
    ).round(1)

    num_orders_completed = np.clip(
        (years_as_vendor * rng.uniform(3, 9, size=n)) + rng.normal(0, 5, size=n),
        1, None
    ).round().astype(int)

    avg_order_value_usd = np.round(
        np.exp(rng.normal(loc=9.5, scale=0.9, size=n)), -2
    )  # roughly $2k - $200k, log-normal spread

    # --- Operational features (strong signal of latent trait) ---
    on_time_delivery_rate = clip01(
        sigmoid(1.4 * latent_reliability + rng.normal(0, 0.6, size=n)) * 0.9 + 0.05
    ).round(3)

    defect_rejection_rate = clip01(
        sigmoid(-1.3 * latent_reliability + rng.normal(0, 0.6, size=n)) * 0.25
    ).round(3)

    lead_time_variance_days = np.clip(
        6 - 3.2 * latent_reliability + rng.normal(0, 2.5, size=n), 0.2, None
    ).round(1)

    # --- Financial features (moderate signal) ---
    # price_competitiveness: ratio to market average (1.0 = at market,
    # <1 = cheaper, >1 = pricier). Weak/no link to reliability on its
    # own (cheap != reliable) but very unreliable vendors are somewhat
    # more likely to lowball then renegotiate.
    price_competitiveness = np.round(
        1.0 + rng.normal(0, 0.15, size=n) - 0.05 * latent_reliability, 3
    )

    payment_terms_compliance = clip01(
        sigmoid(1.0 * latent_reliability + rng.normal(0, 0.7, size=n)) * 0.95 + 0.03
    ).round(3)

    # --- Behavioral features (light touch, moderate signal) ---
    # RFQ response time: faster response weakly associated with
    # reliability (conscientiousness-like proxy)
    rfq_response_time_hours = np.clip(
        30 - 9 * latent_reliability + rng.normal(0, 12, size=n), 1, 120
    ).round(1)

    communication_responsiveness_score = clip01(
        sigmoid(1.1 * latent_reliability + rng.normal(0, 0.8, size=n))
    ).round(3) * 100
    communication_responsiveness_score = communication_responsiveness_score.round(1)

    # Price volatility across quotes: how much a vendor's quoted price
    # for similar items swings across RFQ rounds. Higher volatility is
    # a mild behavioral red flag (anchoring/opportunistic pricing),
    # weakly associated with lower reliability.
    price_volatility_across_quotes = np.clip(
        0.12 - 0.045 * latent_reliability + rng.normal(0, 0.05, size=n), 0.01, 0.6
    ).round(3)

    dispute_frequency_per_year = np.clip(
        1.6 - 1.1 * latent_reliability + rng.normal(0, 0.9, size=n), 0, None
    ).round(2)

    has_iso_certification = (
        sigmoid(0.8 * latent_reliability + rng.normal(0, 1.0, size=n)) > 0.5
    ).astype(int)

    df = pd.DataFrame({
        "vendor_id": vendor_id,
        "category": category,
        "country": country,
        "years_as_vendor": years_as_vendor,
        "num_orders_completed": num_orders_completed,
        "avg_order_value_usd": avg_order_value_usd,
        "on_time_delivery_rate": on_time_delivery_rate,
        "defect_rejection_rate": defect_rejection_rate,
        "lead_time_variance_days": lead_time_variance_days,
        "price_competitiveness": price_competitiveness,
        "payment_terms_compliance": payment_terms_compliance,
        "rfq_response_time_hours": rfq_response_time_hours,
        "communication_responsiveness_score": communication_responsiveness_score,
        "price_volatility_across_quotes": price_volatility_across_quotes,
        "dispute_frequency_per_year": dispute_frequency_per_year,
        "has_iso_certification": has_iso_certification,
    })

    # --- Composite reliability score (the modeling TARGET) ---
    # Built transparently from observable fields (not the latent trait
    # directly) so the scoring logic is explainable to a non-technical
    # stakeholder. Weights reflect what actually matters in oil & gas
    # procurement: on-time delivery and quality dominate; behavioral
    # signals contribute a smaller, secondary share.
    def minmax(s):
        return (s - s.min()) / (s.max() - s.min())

    score = (
        0.28 * minmax(df.on_time_delivery_rate)
        + 0.20 * minmax(-df.defect_rejection_rate)
        + 0.14 * minmax(-df.lead_time_variance_days)
        + 0.12 * minmax(df.payment_terms_compliance)
        + 0.10 * minmax(-df.dispute_frequency_per_year)
        + 0.06 * minmax(df.communication_responsiveness_score)
        + 0.05 * minmax(-df.rfq_response_time_hours)
        + 0.05 * minmax(-df.price_volatility_across_quotes)
    )

    # --- Non-linear "quality shock" penalty ---
    # Mirrors prospect theory's kinked value function: losses (here,
    # defect rates crossing a tolerance threshold) are weighted more
    # steeply than an equivalent gain would be rewarded. Once a
    # vendor's defect rate crosses ~15%, procurement teams don't
    # discount them a little more — trust drops off a cliff. This
    # gives the dataset a genuine non-linearity that a plain linear
    # model cannot capture but tree-based models can.
    defect_threshold = 0.15
    shock = np.where(
        df.defect_rejection_rate > defect_threshold,
        (df.defect_rejection_rate - defect_threshold) * 1.8,  # steep extra penalty
        0.0,
    )

    # Realistic measurement noise: two vendors with identical stats
    # rarely get identical scores in practice (one-off events, audit
    # timing, reviewer variation).
    noise = rng.normal(0, 0.035, size=n)

    score_adjusted = np.clip(score - shock + noise, 0, None)
    df["reliability_score"] = np.round(
        (score_adjusted / score_adjusted.max()) * 100, 1
    )

    def tier(s):
        if s >= 75:
            return "Preferred"
        elif s >= 55:
            return "Approved"
        elif s >= 35:
            return "Watchlist"
        else:
            return "High Risk"

    df["risk_tier"] = df["reliability_score"].apply(tier)

    return df


if __name__ == "__main__":
    df = generate_vendors()
    out_path = "vendor_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} vendors -> {out_path}")
    print(df["risk_tier"].value_counts())
    print(df.describe(include="all").T)
