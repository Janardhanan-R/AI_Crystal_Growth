from src.surrogate import FastSurrogate
import pandas as pd

surrogate = FastSurrogate.load("models/fast_surrogate.joblib")

X_new = pd.DataFrame([{
    "T1": 1265, "T2": 1240, "T3": 1210, "T4": 1080, "T5": 1045,
    "dT_12": 25, "dT_23": 30, "dT_34": 130, "dT_45": 35,
    "dT_15": 220, "T_mean": 1168
}])

result = surrogate.predict(X_new)
print("Predictions:\n", result["predictions"])
print("\nUncertainty (GPR std):\n", result["uncertainty"])
print("\nInference time (ms):", result["inference_time_per_sample_ms"])
print("Warnings:", result["warnings"])
