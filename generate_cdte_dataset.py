"""
Anchor-constrained scenario generator for the CdTe Bridgman crystal-growth dataset.

CRITICAL CONTEXT (read before touching the numbers below):
  - The "original CdTe dataset" is a single real experimental record (n=1), extracted
    from the uploaded reference sheet "014 2.pdf". It is NOT a multi-record dataset.
  - With n=1, no statistical distribution can be legitimately fit (no bootstrap, KDE,
    copula, or correlation-preserving sampling is possible - all require multiple real
    observations). No governing physics equations connecting CdTe Bridgman process
    parameters to crystal quality were supplied in either source document, so a
    physics-based digital twin is also not possible here.
  - The only defensible technique available is CONTROLLED STATISTICAL PERTURBATION
    (explicitly one of the user-approved methods): each raw process parameter is
    perturbed independently around the single anchor value using a modest, symmetric,
    DOCUMENTED noise width. No cross-variable "physical relationship" is invented
    between raw parameters (e.g. heating power is NOT modelled as a function of
    temperature - that would be fabricating physics).
  - Target variables (Crystal_Quality_Score, Crystal_Quality_Category, Crack_Probability,
    Dislocation_Density_cm2, Oxygen_Concentration_ppm, Crystal_Diameter_Uniformity_Percent)
    have NO statistical or physical basis to derive from varied inputs. Per the user's
    explicit, confirmed choice, they are generated using a transparent, single-mechanism
    "distance from the anchor operating point" scenario heuristic (deviation_score below),
    clearly labelled throughout as engineering judgement / illustrative placeholder logic,
    NOT a scientific prediction. Crack_Probability and Dislocation_Density_cm2 are anchored
    at the anchor row's real values (0.08 and 2e5 cm^-2) when deviation_score ~ 0; the other
    four targets have NO real anchor value at all and are pure scenario placeholders.
  - Five raw features have NO anchor value and NO literature support at all (Heating_Power_W,
    Ambient_Temperature_C, Dopant_Concentration_ppm, Dopant_Type, Crucible_Material). Rather
    than fabricate numbers for these, they are left as "Not available in source dataset" for
    EVERY row (experimental and synthetic) and excluded from the recommended ML feature
    matrix X. This keeps the schema matching the original 29-feature spec sheet without
    inventing data merely to fill cells.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
N_SYNTHETIC = 5000

TORR_PER_MBAR = 0.750062
PA_PER_TORR = 133.322

NOT_AVAILABLE = "Not available in source dataset"

# ---------------------------------------------------------------------------
# The single real anchor record (from "014 2.pdf")
# ---------------------------------------------------------------------------
ANCHOR = dict(
    Furnace_Set_Temperature_C=1000.0,
    Actual_Furnace_Temperature_C=1062.0,          # -> Temperature_Error_C = -62.0
    Temperature_Gradient_C_per_cm_range=(2.0, 6.0),  # source gives a range, not a point
    Cooling_Rate_C_per_min=20.0 / 60.0,            # source: 20 deg/hr -> deg/min
    Translation_Speed_mm_per_hr=10.0,
    Growth_Time_hr=12.0,
    Stage_Position_mm=210.0,                       # 21 cm from top
    Crystal_Length_Grown_mm=60.0,                  # 6 cm
    Solidification_Percentage=100.0,
    Vacuum_Level_Torr=1e-5 * TORR_PER_MBAR,
    Initial_Charge_Mass_g=15.0,                    # material = CdTe (source struck "silicon")
    Crack_Probability=0.08,                        # given directly in source
    Dislocation_Density_cm2=2.0e5,                 # given directly in source
)
ANCHOR["Temperature_Error_C"] = ANCHOR["Furnace_Set_Temperature_C"] - ANCHOR["Actual_Furnace_Temperature_C"]
ANCHOR["Chamber_Pressure_Pa"] = ANCHOR["Vacuum_Level_Torr"] * PA_PER_TORR  # unit conversion, not a new measurement
ANCHOR["Average_Temperature_C"] = (ANCHOR["Furnace_Set_Temperature_C"] + ANCHOR["Actual_Furnace_Temperature_C"]) / 2.0
ANCHOR["Temperature_Gradient_C_per_cm"] = float(np.mean(ANCHOR["Temperature_Gradient_C_per_cm_range"]))  # midpoint, for display only

# Documented perturbation widths (engineering judgement; NOT statistically fit, NOT
# literature-validated - the choice of magnitude itself is arbitrary and is flagged as such).
SIGMA_REL = dict(
    Furnace_Set_Temperature_C=0.05,
    Cooling_Rate_C_per_min=0.20,
    Translation_Speed_mm_per_hr=0.20,
    Growth_Time_hr=0.25,
    Stage_Position_mm=0.20,
    Crystal_Length_Grown_mm=0.20,
    Initial_Charge_Mass_g=0.20,
)
SIGMA_TEMP_ERROR_ABS = 15.0  # degC, applied additively to the anchor's -62 degC error
SIGMA_VACUUM_LOGNORMAL = 0.6
SIGMA_SOLIDIFICATION_ABS = 20.0  # one-sided perturbation below the anchor's 100% ceiling


def generate_synthetic(n, rng):
    def pert_rel(anchor_val, key):
        return anchor_val * (1 + rng.normal(0, SIGMA_REL[key], n))

    set_temp = np.clip(pert_rel(ANCHOR["Furnace_Set_Temperature_C"], "Furnace_Set_Temperature_C"), 1, None)
    temp_error = ANCHOR["Temperature_Error_C"] + rng.normal(0, SIGMA_TEMP_ERROR_ABS, n)
    actual_temp = set_temp - temp_error

    lo, hi = ANCHOR["Temperature_Gradient_C_per_cm_range"]
    gradient = rng.uniform(lo, hi, n)  # sampling within the source-given range itself

    cooling_rate = np.clip(pert_rel(ANCHOR["Cooling_Rate_C_per_min"], "Cooling_Rate_C_per_min"), 1e-3, None)
    translation_speed = np.clip(pert_rel(ANCHOR["Translation_Speed_mm_per_hr"], "Translation_Speed_mm_per_hr"), 1e-3, None)
    growth_time = np.clip(pert_rel(ANCHOR["Growth_Time_hr"], "Growth_Time_hr"), 1e-3, None)
    stage_position = np.clip(pert_rel(ANCHOR["Stage_Position_mm"], "Stage_Position_mm"), 0, None)
    crystal_length = np.clip(pert_rel(ANCHOR["Crystal_Length_Grown_mm"], "Crystal_Length_Grown_mm"), 0, None)
    solidification = np.clip(100 - np.abs(rng.normal(0, SIGMA_SOLIDIFICATION_ABS, n)), 0, 100)  # 100% is a hard ceiling by definition
    charge_mass = np.clip(pert_rel(ANCHOR["Initial_Charge_Mass_g"], "Initial_Charge_Mass_g"), 1e-3, None)

    vacuum_torr = np.clip(ANCHOR["Vacuum_Level_Torr"] * rng.lognormal(0, SIGMA_VACUUM_LOGNORMAL, n), 1e-9, 1.0)
    chamber_pa = vacuum_torr * PA_PER_TORR  # unit conversion only

    avg_temp = (set_temp + actual_temp) / 2.0

    # ---- deviation_score: a single, transparent summary of how far this row's raw
    # process parameters sit from the anchor, in units of the perturbation width used
    # to generate them. This is NOT a physical stability metric - it is literally a
    # standardized distance computed from the random draws above, reused consistently
    # below for the engineered "stability" features and for the target scenario logic,
    # so there is exactly one place this heuristic is defined.
    z_temp = np.abs(set_temp - ANCHOR["Furnace_Set_Temperature_C"]) / (ANCHOR["Furnace_Set_Temperature_C"] * SIGMA_REL["Furnace_Set_Temperature_C"])
    z_err = np.abs(temp_error - ANCHOR["Temperature_Error_C"]) / SIGMA_TEMP_ERROR_ABS
    z_grad = np.abs(gradient - ANCHOR["Temperature_Gradient_C_per_cm"]) / ((hi - lo) / 2)
    z_cool = np.abs(cooling_rate - ANCHOR["Cooling_Rate_C_per_min"]) / (ANCHOR["Cooling_Rate_C_per_min"] * SIGMA_REL["Cooling_Rate_C_per_min"])
    z_speed = np.abs(translation_speed - ANCHOR["Translation_Speed_mm_per_hr"]) / (ANCHOR["Translation_Speed_mm_per_hr"] * SIGMA_REL["Translation_Speed_mm_per_hr"])
    z_vac = np.abs(np.log(vacuum_torr / ANCHOR["Vacuum_Level_Torr"])) / SIGMA_VACUUM_LOGNORMAL
    deviation_score = np.mean([z_temp, z_err, z_grad, z_cool, z_speed, z_vac], axis=0)
    deviation_score = np.clip(deviation_score, 0, 6)  # cap for numerical sanity only

    # ---- Engineered features: proxies derived FROM the perturbation draws above
    # (schema completeness, matching the original spec sheet's "Section E" features).
    # These are NOT measured sensor time-series - no such data exists for either the
    # anchor row or the synthetic rows. Documented as proxies, not measurements.
    temperature_variance = np.round((z_temp * 0.6 + rng.normal(0, 0.1, n).clip(min=0)) ** 2, 4)
    speed_variance = np.round((z_speed * 0.08 + rng.normal(0, 0.02, n).clip(min=0)) ** 2, 6)
    dT_dt = np.round(np.sign(temp_error - ANCHOR["Temperature_Error_C"]) * z_temp * 0.05 + rng.normal(0, 0.03, n), 4)
    stability_index = np.round(np.clip(1 - deviation_score / 4, 0, 1), 4)

    # ---- Targets: transparent "distance from anchor" scenario heuristic (user-approved).
    # Crack_Probability and Dislocation_Density_cm2 are anchored at the source's real
    # values (0.08, 2e5 cm^-2) at deviation_score ~ 0. The remaining four targets have
    # NO real anchor value at all - their center values (80, 8 ppm, 95%) are arbitrary
    # illustrative constants, not sourced from data or literature.
    crack_probability = np.clip(0.08 * np.exp(0.5 * deviation_score) * rng.lognormal(0, 0.25, n), 0, 1)
    dislocation_density = np.clip(2.0e5 * np.exp(0.4 * deviation_score) * rng.lognormal(0, 0.3, n), 1e3, 5e6)
    quality_score = np.clip(80 - 13 * deviation_score + rng.normal(0, 8, n), 0, 100)
    oxygen_conc = np.clip(8 + 4 * deviation_score + rng.gamma(2.0, 1.2, n), 0.1, 40)
    diameter_uniformity = np.clip(95 - 15 * deviation_score + rng.normal(0, 4, n), 40, 100)

    quality_category = np.select(
        [quality_score >= 85, quality_score >= 70, quality_score >= 50],
        ["Excellent", "Good", "Average"],
        default="Poor",
    )  # thresholds are the ones literally given in the source sheet

    df = pd.DataFrame({
        "Furnace_Set_Temperature_C": set_temp,
        "Actual_Furnace_Temperature_C": actual_temp,
        "Temperature_Error_C": temp_error,
        "Temperature_Gradient_C_per_cm": gradient,
        "Heating_Power_W": NOT_AVAILABLE,
        "Cooling_Rate_C_per_min": cooling_rate,
        "Translation_Speed_mm_per_hr": translation_speed,
        "Growth_Time_hr": growth_time,
        "Stage_Position_mm": stage_position,
        "Crystal_Length_Grown_mm": crystal_length,
        "Solidification_Percentage": solidification,
        "Chamber_Pressure_Pa": chamber_pa,
        "Vacuum_Level_Torr": vacuum_torr,
        "Ambient_Temperature_C": NOT_AVAILABLE,
        "Initial_Charge_Mass_g": charge_mass,
        "Dopant_Concentration_ppm": NOT_AVAILABLE,
        "Dopant_Type": NOT_AVAILABLE,
        "Crucible_Material": NOT_AVAILABLE,
        "Temperature_Variance_Last_10min": temperature_variance,
        "Translation_Speed_Variance": speed_variance,
        "Average_Temperature_C": avg_temp,
        "dT_dt_C_per_min": dT_dt,
        "Stability_Index": stability_index,
        "Crystal_Quality_Score": quality_score,
        "Crystal_Quality_Category": quality_category,
        "Crack_Probability": crack_probability,
        "Dislocation_Density_cm2": dislocation_density,
        "Oxygen_Concentration_ppm": oxygen_conc,
        "Crystal_Diameter_Uniformity_Percent": diameter_uniformity,
        "Data_Source": "Synthetic-ScenarioSampling",
        "Original_Record_ID": "EXP-0001",
        "Extrapolated": True,
        "Synthetic_Generation_Method": (
            "Controlled statistical perturbation around the single anchor record "
            "(independent per-variable noise, documented widths; no cross-variable "
            "physics assumed). Targets generated from a transparent 'distance from "
            "anchor' scenario heuristic, engineering judgement only - not statistically "
            "fit, not physics-derived, not literature-validated."
        ),
        "_deviation_score": deviation_score,  # kept for validation/report use, dropped before final CSV
    })
    return df


def build_anchor_row():
    a = ANCHOR
    return {
        "Furnace_Set_Temperature_C": a["Furnace_Set_Temperature_C"],
        "Actual_Furnace_Temperature_C": a["Actual_Furnace_Temperature_C"],
        "Temperature_Error_C": a["Temperature_Error_C"],
        "Temperature_Gradient_C_per_cm": a["Temperature_Gradient_C_per_cm"],
        "Heating_Power_W": NOT_AVAILABLE,
        "Cooling_Rate_C_per_min": round(a["Cooling_Rate_C_per_min"], 4),
        "Translation_Speed_mm_per_hr": a["Translation_Speed_mm_per_hr"],
        "Growth_Time_hr": a["Growth_Time_hr"],
        "Stage_Position_mm": a["Stage_Position_mm"],
        "Crystal_Length_Grown_mm": a["Crystal_Length_Grown_mm"],
        "Solidification_Percentage": a["Solidification_Percentage"],
        "Chamber_Pressure_Pa": a["Chamber_Pressure_Pa"],
        "Vacuum_Level_Torr": a["Vacuum_Level_Torr"],
        "Ambient_Temperature_C": NOT_AVAILABLE,
        "Initial_Charge_Mass_g": a["Initial_Charge_Mass_g"],
        "Dopant_Concentration_ppm": NOT_AVAILABLE,
        "Dopant_Type": NOT_AVAILABLE,
        "Crucible_Material": NOT_AVAILABLE,
        "Temperature_Variance_Last_10min": NOT_AVAILABLE,
        "Translation_Speed_Variance": NOT_AVAILABLE,
        "Average_Temperature_C": a["Average_Temperature_C"],
        "dT_dt_C_per_min": NOT_AVAILABLE,
        "Stability_Index": NOT_AVAILABLE,
        "Crystal_Quality_Score": NOT_AVAILABLE,
        "Crystal_Quality_Category": NOT_AVAILABLE,
        "Crack_Probability": a["Crack_Probability"],
        "Dislocation_Density_cm2": a["Dislocation_Density_cm2"],
        "Oxygen_Concentration_ppm": NOT_AVAILABLE,
        "Crystal_Diameter_Uniformity_Percent": NOT_AVAILABLE,
        "Data_Source": "Experimental",
        "Original_Record_ID": "EXP-0001",
        "Extrapolated": False,
        "Synthetic_Generation_Method": "N/A - real experimental record",
        "_deviation_score": 0.0,
    }


if __name__ == "__main__":
    synth = generate_synthetic(N_SYNTHETIC, rng)
    anchor_row = pd.DataFrame([build_anchor_row()])
    df = pd.concat([anchor_row, synth], ignore_index=True)

    # rounding for readability on numeric columns only (strings like NOT_AVAILABLE untouched)
    round_map = {
        "Furnace_Set_Temperature_C": 1, "Actual_Furnace_Temperature_C": 1, "Temperature_Error_C": 1,
        "Temperature_Gradient_C_per_cm": 2, "Cooling_Rate_C_per_min": 3, "Translation_Speed_mm_per_hr": 2,
        "Growth_Time_hr": 2, "Stage_Position_mm": 1, "Crystal_Length_Grown_mm": 2, "Solidification_Percentage": 2,
        "Chamber_Pressure_Pa": 6, "Vacuum_Level_Torr": 9, "Initial_Charge_Mass_g": 2, "Average_Temperature_C": 1,
        "Crystal_Quality_Score": 2, "Crack_Probability": 4, "Dislocation_Density_cm2": 0,
        "Oxygen_Concentration_ppm": 2, "Crystal_Diameter_Uniformity_Percent": 2,
    }
    for col, dec in round_map.items():
        df[col] = df[col].apply(lambda v, d=dec: v if v == NOT_AVAILABLE else round(float(v), d))

    # exact-consistency recompute after rounding
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")
    df["Temperature_Error_C"] = (to_num(df["Furnace_Set_Temperature_C"]) - to_num(df["Actual_Furnace_Temperature_C"])).round(1)
    df["Average_Temperature_C"] = ((to_num(df["Furnace_Set_Temperature_C"]) + to_num(df["Actual_Furnace_Temperature_C"])) / 2).round(1)
    score_num = to_num(df["Crystal_Quality_Score"])
    cat = np.select([score_num >= 85, score_num >= 70, score_num >= 50], ["Excellent", "Good", "Average"], default="Poor")
    df["Crystal_Quality_Category"] = np.where(df["Crystal_Quality_Score"] == NOT_AVAILABLE, NOT_AVAILABLE, cat)

    deviation_score = df.pop("_deviation_score")

    df.insert(0, "Record_ID", ["EXP-0001"] + [f"SYN-{i:05d}" for i in range(1, len(df))])

    df.to_csv("synthetic_CdTe_crystal_growth_5000.csv", index=False)
    df.assign(_deviation_score=deviation_score).to_csv("_working_with_deviation_score.csv", index=False)
    print("Saved. Shape:", df.shape)
    print(df["Data_Source"].value_counts())
    print(df.loc[df["Data_Source"] != "Experimental", "Crystal_Quality_Category"].value_counts())
