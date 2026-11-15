from typing import List, Optional
import io
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

app = FastAPI(
    title="Experimentation and Causal Inference Platform",
    description="An API for A/B testing and statistical analysis",
    version="1.0.0",
)


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://nexvia-ds04-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SampleSizeRequest(BaseModel):
    conversion_rate_a: float = Field(gt=0, lt=1)
    conversion_rate_b: float = Field(gt=0, lt=1)
    significance_level: float = Field(default=0.05, gt=0, lt=1)
    statistical_power: float = Field(default=0.80, gt=0, lt=1)


class ABTestRequest(BaseModel):
    visitors_a: int = Field(gt=0)
    conversions_a: int = Field(ge=0)
    visitors_b: int = Field(gt=0)
    conversions_b: int = Field(ge=0)


class MultipleTestRequest(BaseModel):
    p_values: List[float]


class TTestRequest(BaseModel):
    group_a: List[float]
    group_b: List[float]


class SimulationRequest(BaseModel):
    visitors_per_group: int = Field(gt=0)
    conversion_rate_a: float = Field(ge=0, le=1)
    conversion_rate_b: float = Field(ge=0, le=1)
    random_seed: Optional[int] = 42


class SequentialTestRequest(ABTestRequest):
    current_look: int = Field(ge=1)
    max_looks: int = Field(ge=1)
    significance_level: float = Field(default=0.05, gt=0, lt=1)


class BayesianTestRequest(ABTestRequest):
    simulations: int = Field(default=50000, ge=1000, le=200000)


class PSMRequest(BaseModel):
    data: List[dict]
    treatment_column: str
    outcome_column: str
    covariate_columns: List[str]


class DiDRequest(BaseModel):
    data: List[dict]
    outcome_column: str
    treatment_group_column: str
    post_period_column: str


class SyntheticControlRequest(BaseModel):
    data: List[dict]
    outcome_column: str
    treated_unit_column: str
    time_column: str
    treated_unit: str


class DoWhyRequest(BaseModel):
    data: List[dict]
    treatment_column: str
    outcome_column: str
    covariate_columns: List[str]


class EconMLRequest(BaseModel):
    data: List[dict]
    treatment_column: str
    outcome_column: str
    covariate_columns: List[str]


def _validate_ab_counts(req):
    if req.conversions_a > req.visitors_a or req.conversions_b > req.visitors_b:
        raise HTTPException(status_code=400, detail="Conversions cannot exceed visitors.")


def _read_csv(file_bytes: bytes) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.BytesIO(file_bytes))
        if frame.empty:
            raise HTTPException(status_code=400, detail="The uploaded CSV is empty.")
        return frame
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}")


def _validate_columns(df, treatment, outcome, covariates):
    missing = [c for c in [treatment, outcome, *covariates] if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Columns not found: {missing}")
    if not covariates:
        raise HTTPException(status_code=400, detail="Provide at least one covariate column.")
    clean = df[[treatment, outcome, *covariates]].dropna().copy()
    if clean.empty:
        raise HTTPException(status_code=400, detail="No complete rows remain after removing missing values.")
    return clean


@app.get("/")
def home():
    return {"message": "Welcome to the Experimentation Platform!", "status": "Running successfully"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/calculate-sample-size")
def calculate_sample_size(req: SampleSizeRequest):
    effect = abs(proportion_effectsize(req.conversion_rate_a, req.conversion_rate_b))
    if effect == 0:
        raise HTTPException(status_code=400, detail="Conversion rates must be different.")
    analysis = NormalIndPower()
    n = analysis.solve_power(
        effect_size=effect,
        alpha=req.significance_level,
        power=req.statistical_power,
        ratio=1.0,
        alternative="two-sided",
    )
    per_group = int(np.ceil(n))
    return {
        "sample_size_per_group": per_group,
        "total_sample_size": per_group * 2,
        "effect_size": float(effect),
        "significance_level": req.significance_level,
        "statistical_power": req.statistical_power,
    }


@app.post("/analyze-ab-test")
def analyze_ab_test(req: ABTestRequest):
    _validate_ab_counts(req)
    table = [[req.conversions_a, req.visitors_a - req.conversions_a],
             [req.conversions_b, req.visitors_b - req.conversions_b]]
    chi2, p_value, dof, _ = stats.chi2_contingency(table, correction=False)
    rate_a = req.conversions_a / req.visitors_a
    rate_b = req.conversions_b / req.visitors_b
    return {
        "conversion_rate_a": rate_a,
        "conversion_rate_b": rate_b,
        "absolute_lift": rate_b - rate_a,
        "relative_lift": (rate_b - rate_a) / rate_a if rate_a else None,
        "chi_square_statistic": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "is_statistically_significant": bool(p_value < 0.05),
    }


@app.post("/correct-multiple-tests")
def correct_multiple_tests(req: MultipleTestRequest):
    if not req.p_values or any((p < 0 or p > 1) for p in req.p_values):
        raise HTTPException(status_code=400, detail="Provide p-values between 0 and 1.")
    reject_bonf, p_bonf, _, _ = multipletests(req.p_values, method="bonferroni")
    reject_bh, p_bh, _, _ = multipletests(req.p_values, method="fdr_bh")
    return {
        "original_p_values": req.p_values,
        "bonferroni_adjusted": p_bonf.tolist(),
        "bonferroni_reject": reject_bonf.tolist(),
        "bh_adjusted": p_bh.tolist(),
        "bh_reject": reject_bh.tolist(),
    }


@app.post("/analyze-ttest")
def analyze_ttest(req: TTestRequest):
    if len(req.group_a) < 2 or len(req.group_b) < 2:
        raise HTTPException(status_code=400, detail="Each group must contain at least two observations.")
    result = stats.ttest_ind(req.group_a, req.group_b, equal_var=False)
    return {
        "test": "Welch's independent t-test",
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "mean_group_a": float(np.mean(req.group_a)),
        "mean_group_b": float(np.mean(req.group_b)),
        "is_statistically_significant": bool(result.pvalue < 0.05),
        "degrees_of_freedom": float(result.df) if hasattr(result, "df") else None,
    }


@app.post("/simulate-ab-test")
def simulate_ab_test(req: SimulationRequest):
    rng = np.random.default_rng(req.random_seed)
    a = rng.binomial(1, req.conversion_rate_a, req.visitors_per_group)
    b = rng.binomial(1, req.conversion_rate_b, req.visitors_per_group)
    return {
        "visitors_per_group": req.visitors_per_group,
        "simulated_conversions_a": int(a.sum()),
        "simulated_conversions_b": int(b.sum()),
        "simulated_rate_a": float(a.mean()),
        "simulated_rate_b": float(b.mean()),
        "random_seed": req.random_seed,
    }


@app.post("/sequential-test")
def sequential_test(req: SequentialTestRequest):
    _validate_ab_counts(req)
    if req.current_look > req.max_looks:
        raise HTTPException(status_code=400, detail="Current look cannot exceed maximum planned looks.")
    table = [[req.conversions_a, req.visitors_a - req.conversions_a],
             [req.conversions_b, req.visitors_b - req.conversions_b]]
    chi2, p_value, _, _ = stats.chi2_contingency(table, correction=False)
    adjusted_alpha = req.significance_level / req.max_looks
    return {
        "p_value": float(p_value),
        "adjusted_significance_threshold": adjusted_alpha,
        "current_look": req.current_look,
        "max_looks": req.max_looks,
        "significant_at_current_look": bool(p_value < adjusted_alpha),
        "chi_square_statistic": float(chi2),
        "method": "Bonferroni-adjusted threshold across planned looks",
    }


@app.post("/bayesian-ab-test")
def bayesian_ab_test(req: BayesianTestRequest):
    _validate_ab_counts(req)
    rng = np.random.default_rng(42)
    a_draws = rng.beta(1 + req.conversions_a, 1 + req.visitors_a - req.conversions_a, req.simulations)
    b_draws = rng.beta(1 + req.conversions_b, 1 + req.visitors_b - req.conversions_b, req.simulations)
    return {
        "probability_b_better_than_a": float(np.mean(b_draws > a_draws)),
        "expected_conversion_rate_a": float(np.mean(a_draws)),
        "expected_conversion_rate_b": float(np.mean(b_draws)),
        "expected_lift": float(np.mean(b_draws - a_draws)),
        "simulations": req.simulations,
        "prior": "Beta(1, 1) for each group",
    }


@app.post("/propensity-score-matching")
def propensity_score_matching(req: PSMRequest):
    df = pd.DataFrame(req.data)
    df = _validate_columns(df, req.treatment_column, req.outcome_column, req.covariate_columns)
    treatment = df[req.treatment_column].astype(int).to_numpy()
    if len(np.unique(treatment)) != 2:
        raise HTTPException(status_code=400, detail="Treatment column must contain two groups coded 0 and 1.")
    x = pd.get_dummies(df[req.covariate_columns], drop_first=True)
    scores = LogisticRegression(max_iter=2000).fit(x, treatment).predict_proba(x)[:, 1]
    treated, control = np.where(treatment == 1)[0], np.where(treatment == 0)[0]
    if not len(treated) or not len(control):
        raise HTTPException(status_code=400, detail="Both treatment groups are required.")
    nn = NearestNeighbors(n_neighbors=1).fit(scores[control].reshape(-1, 1))
    distances, indices = nn.kneighbors(scores[treated].reshape(-1, 1))
    y = pd.to_numeric(df[req.outcome_column], errors="coerce").to_numpy()
    valid = np.isfinite(y[treated]) & np.isfinite(y[control[indices[:, 0]]])
    if not valid.any():
        raise HTTPException(status_code=400, detail="Outcome column must be numeric and non-missing.")
    effect = float(np.mean(y[treated[valid]] - y[control[indices[valid, 0]]]))
    return {"average_treatment_effect": effect, "matched_pairs": int(valid.sum()),
            "mean_propensity_score_distance": float(np.mean(distances[valid])),
            "method": "1:1 nearest-neighbour propensity-score matching"}


@app.post("/difference-in-differences")
def difference_in_differences(req: DiDRequest):
    df = pd.DataFrame(req.data)
    required = [req.outcome_column, req.treatment_group_column, req.post_period_column]
    if any(c not in df.columns for c in required):
        raise HTTPException(status_code=400, detail="One or more requested columns are missing.")
    df = df[required].dropna()
    treated_pre = df[(df[req.treatment_group_column] == 1) & (df[req.post_period_column] == 0)][req.outcome_column].mean()
    treated_post = df[(df[req.treatment_group_column] == 1) & (df[req.post_period_column] == 1)][req.outcome_column].mean()
    control_pre = df[(df[req.treatment_group_column] == 0) & (df[req.post_period_column] == 0)][req.outcome_column].mean()
    control_post = df[(df[req.treatment_group_column] == 0) & (df[req.post_period_column] == 1)][req.outcome_column].mean()
    vals = [treated_pre, treated_post, control_pre, control_post]
    if any(pd.isna(v) for v in vals):
        raise HTTPException(status_code=400, detail="Each group must have observations in both periods.")
    effect = float((treated_post - treated_pre) - (control_post - control_pre))
    return {"average_treatment_effect": effect, "treated_pre": float(treated_pre),
            "treated_post": float(treated_post), "control_pre": float(control_pre),
            "control_post": float(control_post), "method": "Difference-in-Differences"}


@app.post("/synthetic-control")
def synthetic_control(req: SyntheticControlRequest):
    df = pd.DataFrame(req.data)
    required = [req.outcome_column, req.treated_unit_column, req.time_column]
    if any(c not in df.columns for c in required):
        raise HTTPException(status_code=400, detail="One or more requested columns are missing.")
    treated = df[df[req.treated_unit_column].astype(str) == str(req.treated_unit)]
    controls = df[df[req.treated_unit_column].astype(str) != str(req.treated_unit)]
    if treated.empty or controls.empty:
        raise HTTPException(status_code=400, detail="Treated unit and control units are both required.")
    control_mean = controls.groupby(req.time_column)[req.outcome_column].mean()
    comparison = treated.set_index(req.time_column)[req.outcome_column].to_frame("treated").join(control_mean.rename("synthetic"), how="inner")
    if comparison.empty:
        raise HTTPException(status_code=400, detail="No overlapping time periods found.")
    comparison["gap"] = comparison["treated"] - comparison["synthetic"]
    return {"average_treatment_effect": float(comparison["gap"].mean()),
            "periods_used": int(len(comparison)), "results": comparison.reset_index().to_dict(orient="records"),
            "method": "Unweighted mean of control units (baseline synthetic-control approximation)"}


@app.post("/dowhy-causal-analysis")
def dowhy_causal_analysis(req: DoWhyRequest):
    df = _validate_columns(pd.DataFrame(req.data), req.treatment_column, req.outcome_column, req.covariate_columns)
    try:
        from dowhy import CausalModel
        model = CausalModel(data=df, treatment=req.treatment_column, outcome=req.outcome_column,
                            common_causes=req.covariate_columns)
        estimand = model.identify_effect()
        estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
        value = float(estimate.value)
        return {"average_treatment_effect": value, "method": "DoWhy Backdoor Linear Regression",
                "observations_used": int(len(df)), "treatment_column": req.treatment_column,
                "outcome_column": req.outcome_column, "covariate_columns": req.covariate_columns,
                "interpretation": "Positive estimated treatment effect." if value > 0 else "Negative or zero estimated treatment effect."}
    except ImportError:
        raise HTTPException(status_code=501, detail="DoWhy is not installed in this environment.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"DoWhy analysis failed: {exc}")


@app.post("/econml-causal-analysis")
def econml_causal_analysis(req: EconMLRequest):
    df = _validate_columns(pd.DataFrame(req.data), req.treatment_column, req.outcome_column, req.covariate_columns)
    try:
        from econml.dml import LinearDML
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        y = pd.to_numeric(df[req.outcome_column], errors="coerce").to_numpy()
        t = pd.to_numeric(df[req.treatment_column], errors="coerce").to_numpy()
        x = pd.get_dummies(df[req.covariate_columns], drop_first=True)
        valid = np.isfinite(y) & np.isfinite(t)
        model = LinearDML(model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42),
                          model_t=RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42),
                          random_state=42)
        model.fit(y[valid], t[valid], X=x.to_numpy()[valid])
        effect = float(np.mean(model.effect(x.to_numpy()[valid])))
        return {"average_treatment_effect": effect, "method": "EconML LinearDML",
                "observations_used": int(valid.sum()), "treatment_column": req.treatment_column,
                "outcome_column": req.outcome_column, "covariate_columns": req.covariate_columns,
                "interpretation": "Positive estimated treatment effect." if effect > 0 else "Negative or zero estimated treatment effect."}
    except ImportError:
        raise HTTPException(status_code=501, detail="EconML is not installed in this environment.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"EconML analysis failed: {exc}")


@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    df = _read_csv(await file.read())
    return {
        "filename": file.filename,
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "data_types": {str(k): str(v) for k, v in df.dtypes.items()},
        "missing_values": {str(k): int(v) for k, v in df.isna().sum().items()},
        "preview": df.head(5).replace({np.nan: None}).to_dict(orient="records"),
    }


@app.post("/analyze-uploaded-dataset")
async def analyze_uploaded_dataset(
    file: UploadFile = File(...),
    treatment_column: str = Form(...),
    outcome_column: str = Form(...),
    covariate_columns: str = Form(""),
):
    df = _read_csv(await file.read())
    covariates = [c.strip() for c in covariate_columns.split(",") if c.strip()]
    req = DoWhyRequest(data=df.to_dict(orient="records"), treatment_column=treatment_column,
                       outcome_column=outcome_column, covariate_columns=covariates)
    result = dowhy_causal_analysis(req)
    result.update({"filename": file.filename, "message": "Causal analysis completed successfully.",
                   "covariate_columns": covariates})
    return result


@app.post("/analyze-uploaded-dataset-econml")
async def analyze_uploaded_dataset_econml(
    file: UploadFile = File(...),
    treatment_column: str = Form(...),
    outcome_column: str = Form(...),
    covariate_columns: str = Form(""),
):
    df = _read_csv(await file.read())
    covariates = [c.strip() for c in covariate_columns.split(",") if c.strip()]
    req = EconMLRequest(data=df.to_dict(orient="records"), treatment_column=treatment_column,
                        outcome_column=outcome_column, covariate_columns=covariates)
    result = econml_causal_analysis(req)
    result.update({"filename": file.filename, "message": "Causal analysis completed successfully.",
                   "covariate_columns": covariates})
    return result
