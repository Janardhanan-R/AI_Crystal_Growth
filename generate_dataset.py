"""
Synthetic dataset generator for AI-powered Bridgman crystal-growth decision support.

Anchor / reference values below were extracted from the uploaded reference document
("014 2.pdf" - a Bridgman crystal-growth feature spec sheet filled in with one
real/example experimental record, material system = CdTe, 15 g charge).

The generator:
  1. Samples raw process parameters (furnace, growth, environment, material) from
     distributions centered on the anchor record with physically plausible spread.
  2. Derives engineered stability features (variance, dT/dt, stability index) from
     those raw parameters, not independently at random.
  3. Derives target variables (quality score/category, crack probability,
     dislocation density, oxygen concentration, diameter uniformity) from a shared
     latent "process instability" signal plus independent noise per target, so the
     required qualitative relationships hold without exact algebraic leakage.
  4. Appends the anchor row, tagged Data_Source = "Experimental" (values preserved
     as given in the source document; fields not present in the source are filled
     using the same physically-consistent formulas used for the synthetic rows, or
     a typical default - documented in the methodology report).
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_SYNTHETIC = 5000

# ---------------------------------------------------------------------------
# 1. Anchor (experimental) record - from the uploaded reference document
# ---------------------------------------------------------------------------
TORR_PER_MBAR = 0.750062
PA_PER_TORR = 133.322

anchor_vacuum_torr = 1e-5 * TORR_PER_MBAR  # source: "10^-5 mbar"

ANCHOR = dict(
    Furnace_Set_Temperature_C=1000.0,
    Actual_Furnace_Temperature_C=1062.0,
    Temperature_Gradient_C_per_cm=4.0,          # source range 2-6, midpoint used
    Cooling_Rate_C_per_min=round(20.0 / 60.0, 4),  # source: 20 deg/hr -> deg/min
    Translation_Speed_mm_per_hr=10.0,
    Growth_Time_hr=12.0,
    Stage_Position_mm=210.0,                    # source: 21 cm from top
    Crystal_Length_Grown_mm=60.0,                # source: 6 cm
    Solidification_Percentage=100.0,
    Vacuum_Level_Torr=anchor_vacuum_torr,
    Initial_Silicon_Mass_g=15.0,                 # source material = CdTe, not Si (see notes)
    Dopant_Concentration_ppm=0.0,                # source: NA -> treated as undoped
    Dopant_Type="Undoped",
    Crucible_Material="Quartz",                  # source: not given -> typical default
    Ambient_Temperature_C=24.0,                  # source: NA -> typical lab default
    Crack_Probability=0.08,                      # source value, preserved as-is
    Dislocation_Density_cm2=2.0e5,               # source value, preserved as-is
)
ANCHOR["Heating_Power_W"] = 1200 + 6.0 * ANCHOR["Furnace_Set_Temperature_C"] + 250 * ANCHOR["Temperature_Gradient_C_per_cm"]
ANCHOR["Chamber_Pressure_Pa"] = ANCHOR["Vacuum_Level_Torr"] * PA_PER_TORR

DOPANT_TYPES = ["Indium (In)", "Chlorine (Cl)", "Aluminum (Al)", "Antimony (Sb)"]
CRUCIBLE_MATERIALS = ["Quartz", "Graphite-coated Quartz", "Pyrolytic BN (pBN)", "Carbon-coated Quartz"]
CRUCIBLE_PROBS = [0.40, 0.25, 0.20, 0.15]
QUARTZ_LIKE = {"Quartz", "Graphite-coated Quartz", "Carbon-coated Quartz"}


def generate_synthetic(n, rng):
    # ---- Furnace parameters -------------------------------------------------
    set_temp = np.clip(rng.normal(1000, 45, n), 880, 1150)

    comp = rng.choice([0, 1, 2], size=n, p=[0.65, 0.28, 0.07])
    sign = rng.choice([-1.0, 1.0], size=n)
    delta = np.where(
        comp == 0, rng.normal(0, 8, n),
        np.where(comp == 1, sign * rng.normal(28, 14, n), sign * rng.normal(60, 18, n)),
    )
    actual_temp = set_temp + delta
    temp_error = set_temp - actual_temp  # required formula: Set - Actual

    gradient = np.clip(rng.normal(4.0, 1.1, n), 1.2, 7.5)
    heating_power = np.clip(
        1200 + 6.0 * set_temp + 250 * gradient + rng.normal(0, 300, n), 1500, 12000
    )
    cooling_rate = np.clip(rng.lognormal(np.log(0.45), 0.75, n), 0.03, 5.0)

    # ---- Growth parameters ---------------------------------------------------
    # Translation speed and growth time are sampled as (roughly) independent
    # process choices; crystal length grown is the physical product of the two
    # (length ~= speed * time * eff), capped by the melt capacity implied by
    # the initial charge mass (a run that reaches 100% solidification stops
    # growing further regardless of remaining nominal time/speed). This keeps
    # length positively correlated with BOTH growth time and translation speed,
    # as required, while still respecting a physical capacity limit.
    translation_speed = np.clip(rng.lognormal(np.log(7.0), 0.45, n), 1.0, 22.0)
    growth_time = np.clip(rng.lognormal(np.log(9.0), 0.55, n), 1.0, 48.0)
    efficiency = rng.uniform(0.82, 0.99, n)
    raw_length = translation_speed * growth_time * efficiency

    initial_mass = np.clip(rng.normal(16.0, 4.5, n), 6.0, 35.0)
    total_target_length = np.clip(initial_mass * 4.0 * (1 + rng.normal(0, 0.10, n)), 20.0, 220.0)

    crystal_length = np.clip(
        np.minimum(raw_length, total_target_length * rng.uniform(0.97, 1.05, n)), 0.5, None
    )

    solidification_pct = np.clip(
        crystal_length / total_target_length * 100 + rng.normal(0, 1.5, n), 0, 100
    )

    stage_position = np.clip(
        rng.uniform(60, 150, n) + crystal_length * rng.uniform(0.85, 1.05, n) + rng.normal(0, 3, n),
        0, 450,
    )

    # ---- Environment parameters ----------------------------------------------
    vacuum_torr = np.clip(rng.lognormal(np.log(7.5e-6), 0.8, n), 1e-7, 5e-4)
    chamber_pa = np.clip(vacuum_torr * PA_PER_TORR * (1 + rng.normal(0, 0.05, n)), 1e-5, None)
    ambient_temp = np.clip(rng.normal(24.0, 2.5, n), 15.0, 33.0)

    # ---- Material parameters ---------------------------------------------------
    is_undoped = rng.random(n) < 0.30
    dopant_conc = np.where(
        is_undoped, 0.0, np.clip(rng.lognormal(np.log(60.0), 1.0, n), 0.1, 800.0)
    )
    dopant_type = np.where(
        is_undoped, "Undoped", rng.choice(DOPANT_TYPES, size=n)
    )
    crucible_material = rng.choice(CRUCIBLE_MATERIALS, size=n, p=CRUCIBLE_PROBS)

    # ---- Engineered stability features ----------------------------------------
    abs_err = np.abs(temp_error)
    temp_variance = np.clip(
        0.05 + 0.004 * abs_err ** 1.3 + rng.gamma(2.0, 0.3, n), 0.02, 45.0
    )
    speed_variance = (translation_speed * (0.015 + 0.08 * rng.random(n))) ** 2
    avg_temp = (set_temp + actual_temp) / 2.0

    cooling_phase = rng.random(n) < 0.25
    dT_dt = (
        -0.15 * cooling_rate * cooling_phase
        + rng.normal(0, 0.05 + 0.05 * np.sqrt(temp_variance), n)
    )

    norm_temp_var = np.clip(temp_variance / 12.0, 0, 1.2)
    norm_speed_var = np.clip(speed_variance / 0.6, 0, 1.2)
    norm_err = np.clip(abs_err / 60.0, 0, 1.2)
    norm_grad_dev = np.clip(np.abs(gradient - 4.0) / 4.0, 0, 1.2)

    raw_instability = 0.30 * norm_temp_var + 0.20 * norm_speed_var + 0.30 * norm_err + 0.20 * norm_grad_dev
    stability_index = np.clip(1 - raw_instability + rng.normal(0, 0.05, n), 0, 1)

    # ---- Targets ---------------------------------------------------------------
    vacuum_penalty = np.clip((vacuum_torr - 1e-5) / 1e-4, 0, 1)
    speed_ideal_penalty = np.clip(np.abs(translation_speed - 8.0) / 12.0, 0, 1)
    cooling_penalty = np.clip((cooling_rate - 0.6), 0, None) / 2.5
    cooling_penalty = np.clip(cooling_penalty, 0, 1)
    material_noise = rng.normal(0, 0.06, n)

    total_penalty = np.clip(
        0.70 * raw_instability
        + 0.20 * vacuum_penalty
        + 0.20 * speed_ideal_penalty
        + 0.25 * cooling_penalty
        + material_noise,
        0, None,
    )

    quality_score = np.clip(100 * np.exp(-1.4 * total_penalty) + rng.normal(0, 5.0, n), 0, 100)

    def bucket(score):
        return np.select(
            [score >= 85, score >= 70, score >= 50],
            ["Excellent", "Good", "Average"],
            default="Poor",
        )

    quality_category = bucket(quality_score)

    z = (
        -4.3
        + 5.6 * raw_instability
        + 2.3 * cooling_penalty
        + 2.6 * vacuum_penalty
        + rng.normal(0, 0.5, n)
    )
    crack_prob = np.clip(1 / (1 + np.exp(-z)), 0, 1)

    dislocation_median = 7000 * np.exp(3.3 * raw_instability)
    dislocation_density = np.clip(
        dislocation_median * rng.lognormal(0, 0.35, n), 500, 5_000_000
    )

    quartz_factor = np.where(np.isin(crucible_material, list(QUARTZ_LIKE)), 1.0, 0.15)
    oxygen_conc = np.clip(
        quartz_factor * (5 + rng.gamma(2.0, 2.0, n)) + vacuum_penalty * 1.0 + rng.normal(0, 1, n),
        0.2, 30.0,
    )

    diameter_uniformity = np.clip(
        70 + 28 * stability_index - 5 * speed_ideal_penalty + rng.normal(0, 3, n), 40, 100
    )

    df = pd.DataFrame({
        "Furnace_Set_Temperature_C": set_temp,
        "Actual_Furnace_Temperature_C": actual_temp,
        "Temperature_Error_C": temp_error,
        "Temperature_Gradient_C_per_cm": gradient,
        "Heating_Power_W": heating_power,
        "Cooling_Rate_C_per_min": cooling_rate,
        "Translation_Speed_mm_per_hr": translation_speed,
        "Growth_Time_hr": growth_time,
        "Stage_Position_mm": stage_position,
        "Crystal_Length_Grown_mm": crystal_length,
        "Solidification_Percentage": solidification_pct,
        "Chamber_Pressure_Pa": chamber_pa,
        "Vacuum_Level_Torr": vacuum_torr,
        "Ambient_Temperature_C": ambient_temp,
        "Initial_Silicon_Mass_g": initial_mass,
        "Dopant_Concentration_ppm": dopant_conc,
        "Dopant_Type": dopant_type,
        "Crucible_Material": crucible_material,
        "Temperature_Variance_Last_10min": temp_variance,
        "Translation_Speed_Variance": speed_variance,
        "Average_Temperature_C": avg_temp,
        "dT_dt_C_per_min": dT_dt,
        "Stability_Index": stability_index,
        "Crystal_Quality_Score": quality_score,
        "Crystal_Quality_Category": quality_category,
        "Crack_Probability": crack_prob,
        "Dislocation_Density_cm2": dislocation_density,
        "Oxygen_Concentration_ppm": oxygen_conc,
        "Crystal_Diameter_Uniformity_Percent": diameter_uniformity,
        "Data_Source": "Synthetic",
    })
    return df


def build_anchor_row():
    a = ANCHOR
    abs_err = abs(a["Temperature_Error_C"] if "Temperature_Error_C" in a else (a["Furnace_Set_Temperature_C"] - a["Actual_Furnace_Temperature_C"]))
    temp_error = a["Furnace_Set_Temperature_C"] - a["Actual_Furnace_Temperature_C"]
    abs_err = abs(temp_error)
    temp_variance = 0.05 + 0.004 * abs_err ** 1.3 + 0.6  # deterministic (no noise) for the single real record
    speed_variance = (a["Translation_Speed_mm_per_hr"] * 0.03) ** 2
    avg_temp = (a["Furnace_Set_Temperature_C"] + a["Actual_Furnace_Temperature_C"]) / 2.0
    dT_dt = 0.0  # steady-state snapshot, not captured in source
    norm_temp_var = min(temp_variance / 12.0, 1.2)
    norm_speed_var = min(speed_variance / 0.6, 1.2)
    norm_err = min(abs_err / 60.0, 1.2)
    norm_grad_dev = min(abs(a["Temperature_Gradient_C_per_cm"] - 4.0) / 4.0, 1.2)
    raw_instability = 0.30 * norm_temp_var + 0.20 * norm_speed_var + 0.30 * norm_err + 0.20 * norm_grad_dev
    stability_index = max(0.0, min(1.0, 1 - raw_instability))

    quartz_factor = 1.0 if a["Crucible_Material"] in QUARTZ_LIKE else 0.15
    oxygen_conc = quartz_factor * 9.0
    diameter_uniformity = 70 + 28 * stability_index

    row = {
        "Furnace_Set_Temperature_C": a["Furnace_Set_Temperature_C"],
        "Actual_Furnace_Temperature_C": a["Actual_Furnace_Temperature_C"],
        "Temperature_Error_C": temp_error,
        "Temperature_Gradient_C_per_cm": a["Temperature_Gradient_C_per_cm"],
        "Heating_Power_W": a["Heating_Power_W"],
        "Cooling_Rate_C_per_min": a["Cooling_Rate_C_per_min"],
        "Translation_Speed_mm_per_hr": a["Translation_Speed_mm_per_hr"],
        "Growth_Time_hr": a["Growth_Time_hr"],
        "Stage_Position_mm": a["Stage_Position_mm"],
        "Crystal_Length_Grown_mm": a["Crystal_Length_Grown_mm"],
        "Solidification_Percentage": a["Solidification_Percentage"],
        "Chamber_Pressure_Pa": a["Chamber_Pressure_Pa"],
        "Vacuum_Level_Torr": a["Vacuum_Level_Torr"],
        "Ambient_Temperature_C": a["Ambient_Temperature_C"],
        "Initial_Silicon_Mass_g": a["Initial_Silicon_Mass_g"],
        "Dopant_Concentration_ppm": a["Dopant_Concentration_ppm"],
        "Dopant_Type": a["Dopant_Type"],
        "Crucible_Material": a["Crucible_Material"],
        "Temperature_Variance_Last_10min": temp_variance,
        "Translation_Speed_Variance": speed_variance,
        "Average_Temperature_C": avg_temp,
        "dT_dt_C_per_min": dT_dt,
        "Stability_Index": stability_index,
        "Crystal_Quality_Score": np.nan,   # filled below via bucket-consistent placeholder
        "Crystal_Quality_Category": np.nan,
        "Crack_Probability": a["Crack_Probability"],
        "Dislocation_Density_cm2": a["Dislocation_Density_cm2"],
        "Oxygen_Concentration_ppm": oxygen_conc,
        "Crystal_Diameter_Uniformity_Percent": diameter_uniformity,
        "Data_Source": "Experimental",
    }
    # The source document did not give an explicit 0-100 quality score for the
    # anchor record itself (only category anchors + example crack/dislocation
    # values). We derive a score from the same stability signal used for the
    # synthetic rows so the anchor stays internally consistent with the rest of
    # the dataset, and flag it in the methodology notes.
    penalty = raw_instability * 0.70
    score = max(0, min(100, 100 * np.exp(-1.4 * penalty)))
    row["Crystal_Quality_Score"] = score
    if score >= 85:
        row["Crystal_Quality_Category"] = "Excellent"
    elif score >= 70:
        row["Crystal_Quality_Category"] = "Good"
    elif score >= 50:
        row["Crystal_Quality_Category"] = "Average"
    else:
        row["Crystal_Quality_Category"] = "Poor"
    return row


if __name__ == "__main__":
    synth = generate_synthetic(N_SYNTHETIC, rng)
    anchor_row = pd.DataFrame([build_anchor_row()])

    df = pd.concat([anchor_row, synth], ignore_index=True)

    # dedupe safety check (numeric rounding to avoid float-noise false negatives)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    rounded = df.copy()
    rounded[numeric_cols] = rounded[numeric_cols].round(6)
    dup_mask = rounded.duplicated(keep="first")
    n_dupes = int(dup_mask.sum())
    if n_dupes:
        df = df[~dup_mask].reset_index(drop=True)
        extra = generate_synthetic(n_dupes + 5, rng)
        extra.insert(0, "Data_Source", extra.pop("Data_Source"))
        df = pd.concat([df, extra.iloc[:n_dupes]], ignore_index=True)

    df.insert(0, "Record_ID", ["EXP-0001"] + [f"SYN-{i:05d}" for i in range(1, len(df))])

    # rounding for realistic sensor/reporting precision
    round_map = {
        "Furnace_Set_Temperature_C": 1, "Actual_Furnace_Temperature_C": 1, "Temperature_Error_C": 1,
        "Temperature_Gradient_C_per_cm": 2, "Heating_Power_W": 0, "Cooling_Rate_C_per_min": 3,
        "Translation_Speed_mm_per_hr": 2, "Growth_Time_hr": 2, "Stage_Position_mm": 1,
        "Crystal_Length_Grown_mm": 2, "Solidification_Percentage": 2, "Chamber_Pressure_Pa": 6,
        "Vacuum_Level_Torr": 8, "Ambient_Temperature_C": 1, "Initial_Silicon_Mass_g": 2,
        "Dopant_Concentration_ppm": 2, "Temperature_Variance_Last_10min": 3,
        "Translation_Speed_Variance": 4, "Average_Temperature_C": 1, "dT_dt_C_per_min": 4,
        "Stability_Index": 4, "Crystal_Quality_Score": 2, "Crack_Probability": 4,
        "Dislocation_Density_cm2": 0, "Oxygen_Concentration_ppm": 2,
        "Crystal_Diameter_Uniformity_Percent": 2,
    }
    for col, dec in round_map.items():
        df[col] = df[col].round(dec)

    # Recompute exact-consistency derived columns from the now-rounded source
    # columns, so the values stored in the CSV satisfy the required formulas
    # exactly (not just before rounding).
    df["Temperature_Error_C"] = (df["Furnace_Set_Temperature_C"] - df["Actual_Furnace_Temperature_C"]).round(1)
    df["Average_Temperature_C"] = ((df["Furnace_Set_Temperature_C"] + df["Actual_Furnace_Temperature_C"]) / 2).round(1)
    df["Crystal_Quality_Category"] = np.select(
        [df["Crystal_Quality_Score"] >= 85, df["Crystal_Quality_Score"] >= 70, df["Crystal_Quality_Score"] >= 50],
        ["Excellent", "Good", "Average"],
        default="Poor",
    )

    out_path = "crystal_growth_synthetic_dataset.csv"
    df.to_csv(out_path, index=False)
    print("Saved", out_path, "shape:", df.shape)
    print(df["Data_Source"].value_counts())
    print(df["Crystal_Quality_Category"].value_counts())
