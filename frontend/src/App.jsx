import { useState } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function postJSON(endpoint, payload) {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json();

  if (!response.ok || data.error || data.detail) {
    throw new Error(data.detail || data.error || "Request failed.");
  }

  return data;
}

function ResultJSON({ title, data }) {
  if (!data) return null;

  return (
    <div className="result-card">
      <h2>{title}</h2>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState("Overview");

  // Shared dataset upload and causal inference state
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [treatmentColumn, setTreatmentColumn] = useState("treatment");
  const [outcomeColumn, setOutcomeColumn] = useState("score");
  const [covariateColumns, setCovariateColumns] = useState("study_hours,attendance");
  const [causalMethod, setCausalMethod] = useState("dowhy");
  const [causalResult, setCausalResult] = useState(null);
  const [causalLoading, setCausalLoading] = useState(false);
  const [causalError, setCausalError] = useState("");

  // Common result and loading state for the other analysis modules
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Sample size calculator
  const [conversionRateA, setConversionRateA] = useState("10");
  const [conversionRateB, setConversionRateB] = useState("13");
  const [significanceLevel, setSignificanceLevel] = useState("0.05");
  const [statisticalPower, setStatisticalPower] = useState("0.80");

  // A/B testing and simulation
  const [visitorsA, setVisitorsA] = useState("1000");
  const [conversionsA, setConversionsA] = useState("100");
  const [visitorsB, setVisitorsB] = useState("1000");
  const [conversionsB, setConversionsB] = useState("130");
  const [simulationVisitors, setSimulationVisitors] = useState("1000");
  const [simulationRateA, setSimulationRateA] = useState("0.10");
  const [simulationRateB, setSimulationRateB] = useState("0.13");

  // Multiple testing
  const [pValues, setPValues] = useState("0.01, 0.03, 0.04, 0.20");

  // Sequential testing
  const [currentLook, setCurrentLook] = useState("1");
  const [maxLooks, setMaxLooks] = useState("5");

  // T-test sample data
  const [groupA, setGroupA] = useState("12, 14, 15, 13, 16");
  const [groupB, setGroupB] = useState("15, 17, 18, 16, 19");

  const menuItems = [
    "Overview",
    "A/B Testing",
    "T-Test",
    "Sample Size Calculator",
    "Multiple Testing",
    "Sequential Testing",
    "Bayesian Analysis",
    "Causal Inference",
    "Dataset Analysis",
  ];

  const resetMessage = () => {
    setResult(null);
    setError("");
  };

  const runRequest = async (endpoint, payload) => {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const data = await postJSON(endpoint, payload);
      setResult(data);
    } catch (err) {
      setError(err.message || "Could not connect to the backend. Check that FastAPI is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleSampleSizeCalculation = () =>
    runRequest("/calculate-sample-size", {
      conversion_rate_a: Number(conversionRateA) / 100,
      conversion_rate_b: Number(conversionRateB) / 100,
      significance_level: Number(significanceLevel),
      statistical_power: Number(statisticalPower),
    });

  const handleABTest = () =>
    runRequest("/analyze-ab-test", {
      visitors_a: Number(visitorsA),
      conversions_a: Number(conversionsA),
      visitors_b: Number(visitorsB),
      conversions_b: Number(conversionsB),
    });

  const handleSimulation = () =>
    runRequest("/simulate-ab-test", {
      visitors_per_group: Number(simulationVisitors),
      conversion_rate_a: Number(simulationRateA),
      conversion_rate_b: Number(simulationRateB),
      random_seed: 42,
    });

  const handleMultipleTesting = () => {
    const values = pValues.split(",").map((value) => Number(value.trim()));
    if (!values.length || values.some((value) => !Number.isFinite(value))) {
      setError("Enter valid comma-separated p-values, for example: 0.01, 0.03, 0.04");
      setResult(null);
      return;
    }
    runRequest("/correct-multiple-tests", { p_values: values });
  };

  const handleSequentialTest = () =>
    runRequest("/sequential-test", {
      visitors_a: Number(visitorsA),
      conversions_a: Number(conversionsA),
      visitors_b: Number(visitorsB),
      conversions_b: Number(conversionsB),
      current_look: Number(currentLook),
      max_looks: Number(maxLooks),
      significance_level: Number(significanceLevel),
    });

  const handleBayesianTest = () =>
    runRequest("/bayesian-ab-test", {
      visitors_a: Number(visitorsA),
      conversions_a: Number(conversionsA),
      visitors_b: Number(visitorsB),
      conversions_b: Number(conversionsB),
    });

  const parseNumberList = (value) =>
    value.split(",").map((item) => Number(item.trim()));

  const handleTTest = () => {
    const valuesA = parseNumberList(groupA);
    const valuesB = parseNumberList(groupB);

    if (
      valuesA.length < 2 ||
      valuesB.length < 2 ||
      [...valuesA, ...valuesB].some((value) => !Number.isFinite(value))
    ) {
      setError("Enter at least two valid numeric observations in each group.");
      setResult(null);
      return;
    }

    runRequest("/analyze-ttest", { group_a: valuesA, group_b: valuesB });
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadError("Please select a CSV file first.");
      return;
    }

    setUploadLoading(true);
    setUploadError("");
    setUploadResult(null);
    setCausalResult(null);
    setCausalError("");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${API_BASE}/upload-dataset`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (!response.ok || data.error || data.detail) {
        throw new Error(data.detail || data.error || "Dataset upload failed.");
      }

      setUploadResult(data);
    } catch (err) {
      setUploadError(err.message || "Could not connect to the backend. Check that FastAPI is running.");
    } finally {
      setUploadLoading(false);
    }
  };

  const handleCausalAnalysis = async () => {
    if (!selectedFile) {
      setCausalError("Please upload a CSV file in Dataset Analysis first.");
      return;
    }

    if (!treatmentColumn.trim() || !outcomeColumn.trim()) {
      setCausalError("Enter both the treatment and outcome column names.");
      return;
    }

    setCausalLoading(true);
    setCausalError("");
    setCausalResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("treatment_column", treatmentColumn.trim());
    formData.append("outcome_column", outcomeColumn.trim());
    formData.append("covariate_columns", covariateColumns.trim());

    const endpoint =
      causalMethod === "dowhy"
        ? "/analyze-uploaded-dataset"
        : "/analyze-uploaded-dataset-econml";

    try {
      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (!response.ok || data.error || data.detail) {
        throw new Error(data.detail || data.error || "Causal analysis failed.");
      }

      setCausalResult(data);
    } catch (err) {
      setCausalError(err.message || "Could not connect to the backend. Check that FastAPI is running.");
    } finally {
      setCausalLoading(false);
    }
  };

  const renderOverview = () => (
    <>
      <div className="section-heading">
        <h2>Platform Overview</h2>
        <p>Your experimentation and causal inference toolkit</p>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <span className="stat-icon">🧪</span>
          <p>Analysis Methods</p>
          <h2>8+</h2>
          <small>Statistical and causal tools</small>
        </div>
        <div className="stat-card">
          <span className="stat-icon">📊</span>
          <p>Experiment Analysis</p>
          <h2>A/B</h2>
          <small>Compare treatment groups</small>
        </div>
        <div className="stat-card">
          <span className="stat-icon">🎯</span>
          <p>Causal Methods</p>
          <h2>4</h2>
          <small>Estimate treatment effects</small>
        </div>
        <div className="stat-card">
          <span className="stat-icon">📁</span>
          <p>Data Analysis</p>
          <h2>CSV</h2>
          <small>Upload and inspect your data</small>
        </div>
      </div>

      <div className="section-heading">
        <h2>Explore Analysis Tools</h2>
        <p>Select a tool from the sidebar to get started.</p>
      </div>

      <div className="tools-grid">
        {menuItems.slice(1).map((item, index) => (
          <button className="tool-card" key={item} onClick={() => setActiveTab(item)}>
            <span className="tool-number">{String(index + 1).padStart(2, "0")}</span>
            <h3>{item}</h3>
            <p>Open this analysis tool and explore your data.</p>
            <span className="tool-link">Explore tool →</span>
          </button>
        ))}
      </div>
    </>
  );

  const renderDatasetAnalysis = () => (
    <div className="upload-section">
      <div className="upload-card">
        <h2>Upload Your Dataset</h2>
        <p>Select a CSV file to inspect its columns, missing values and preview.</p>

        <div className="file-input-area">
          <input
            type="file"
            accept=".csv,text/csv"
            onChange={(event) => {
              setSelectedFile(event.target.files?.[0] || null);
              setUploadResult(null);
              setUploadError("");
              setCausalResult(null);
              setCausalError("");
            }}
          />
          {selectedFile && <p className="selected-file">Selected: {selectedFile.name}</p>}
          <button className="upload-button" onClick={handleUpload} disabled={uploadLoading || !selectedFile}>
            {uploadLoading ? "Uploading..." : "Upload Dataset"}
          </button>
        </div>

        {uploadError && <p className="error-message">{uploadError}</p>}
      </div>

      {uploadResult && (
        <div className="result-card">
          <h2>Dataset Upload Successful!</h2>
          <p>File: {uploadResult.filename}</p>
          <h3>Dataset Details</h3>
          <div className="dataset-stats">
            <div><strong>{uploadResult.total_rows}</strong><span>Total Rows</span></div>
            <div><strong>{uploadResult.total_columns}</strong><span>Total Columns</span></div>
          </div>

          <h3>Column Information</h3>
          <div className="table-scroll">
            <table className="dataset-table">
              <thead><tr><th>Column Name</th><th>Data Type</th><th>Missing Values</th></tr></thead>
              <tbody>
                {Object.entries(uploadResult.data_types || {}).map(([column, dtype]) => (
                  <tr key={column}>
                    <td>{column}</td><td>{dtype}</td><td>{uploadResult.missing_values?.[column] ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h3>Data Preview (First 5 Rows)</h3>
          <div className="table-scroll">
            <table className="dataset-table">
              <thead><tr>{(uploadResult.column_names || []).map((column) => <th key={column}>{column}</th>)}</tr></thead>
              <tbody>
                {(uploadResult.preview || []).map((row, index) => (
                  <tr key={index}>
                    {(uploadResult.column_names || []).map((column) => (
                      <td key={column}>{String(row[column] ?? "")}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <button className="upload-button" onClick={() => setActiveTab("Causal Inference")}>
            Continue to Causal Inference
          </button>
        </div>
      )}
    </div>
  );

  const renderCausalInference = () => (
    <div className="upload-section">
      <div className="upload-card">
        <h2>Causal Inference Analysis</h2>
        <p>Upload your CSV in Dataset Analysis first. Then enter the column names and choose an estimation method.</p>

        {!selectedFile && <p className="error-message">Please upload a CSV file in Dataset Analysis first.</p>}
        {selectedFile && <p className="selected-file">Selected dataset: {selectedFile.name}</p>}

        <label htmlFor="treatment-column">Treatment Column</label>
        <input id="treatment-column" type="text" value={treatmentColumn} onChange={(event) => setTreatmentColumn(event.target.value)} placeholder="Example: treatment" />

        <label htmlFor="outcome-column">Outcome Column</label>
        <input id="outcome-column" type="text" value={outcomeColumn} onChange={(event) => setOutcomeColumn(event.target.value)} placeholder="Example: score" />

        <label htmlFor="covariate-columns">Covariate Columns (comma-separated)</label>
        <input id="covariate-columns" type="text" value={covariateColumns} onChange={(event) => setCovariateColumns(event.target.value)} placeholder="Example: study_hours,attendance" />

        <label htmlFor="causal-method">Analysis Method</label>
        <select id="causal-method" value={causalMethod} onChange={(event) => setCausalMethod(event.target.value)}>
          <option value="dowhy">DoWhy</option>
          <option value="econml">EconML</option>
        </select>

        <button className="upload-button" onClick={handleCausalAnalysis} disabled={causalLoading || !selectedFile}>
          {causalLoading ? "Analyzing..." : "Run Causal Analysis"}
        </button>
        {causalError && <p className="error-message">{causalError}</p>}
      </div>

      {causalResult && (
        <div className="result-card">
          <h2>Analysis Results</h2>
          <div className="causal-result-highlight">
            <p>Estimated Average Treatment Effect</p>
            <h1>{Number(causalResult.average_treatment_effect).toFixed(4)}</h1>
            <p>{causalResult.interpretation}</p>
          </div>
          <h3>Analysis Details</h3>
          <p><strong>Method:</strong> {causalResult.method}</p>
          <p><strong>Dataset:</strong> {causalResult.filename}</p>
          <p><strong>Observations:</strong> {causalResult.observations_used}</p>
          <p><strong>Treatment:</strong> {causalResult.treatment_column}</p>
          <p><strong>Outcome:</strong> {causalResult.outcome_column}</p>
          <p><strong>Covariates:</strong> {causalResult.covariate_columns?.join(", ")}</p>
          <p>{causalResult.message}</p>
          <p><small>Interpret causal estimates cautiously; results depend on data quality and assumptions.</small></p>
        </div>
      )}
    </div>
  );

  const renderSampleSize = () => (
    <div className="upload-section">
      <div className="upload-card">
        <h2>Sample Size Calculator</h2>
        <p>Estimate the number of observations required in each group for an A/B experiment.</p>
        <label>Group A conversion rate (%)</label>
        <input type="number" min="0.01" max="99.99" step="0.01" value={conversionRateA} onChange={(e) => setConversionRateA(e.target.value)} />
        <label>Group B expected conversion rate (%)</label>
        <input type="number" min="0.01" max="99.99" step="0.01" value={conversionRateB} onChange={(e) => setConversionRateB(e.target.value)} />
        <label>Significance level (alpha)</label>
        <input type="number" min="0.001" max="0.999" step="0.01" value={significanceLevel} onChange={(e) => setSignificanceLevel(e.target.value)} />
        <label>Statistical power</label>
        <input type="number" min="0.01" max="0.999" step="0.01" value={statisticalPower} onChange={(e) => setStatisticalPower(e.target.value)} />
        <button className="upload-button" onClick={handleSampleSizeCalculation} disabled={loading}> 
          {loading ? "Calculating..." : "Calculate Required Sample"}
        </button>
        {error && <p className="error-message">{error}</p>}
      </div>
      <ResultJSON title="Sample Size Results" data={result} />
    </div>
  );

  const renderABTesting = () => (
    <div className="upload-section">
      <div className="upload-card">
        <h2>A/B Test Analysis</h2>
        <p>Compare conversion results for control group A and variant group B using a chi-square test.</p>
        <label>Group A visitors</label>
        <input type="number" min="1" value={visitorsA} onChange={(e) => setVisitorsA(e.target.value)} />
        <label>Group A conversions</label>
        <input type="number" min="0" value={conversionsA} onChange={(e) => setConversionsA(e.target.value)} />
        <label>Group B visitors</label>
        <input type="number" min="1" value={visitorsB} onChange={(e) => setVisitorsB(e.target.value)} />
        <label>Group B conversions</label>
        <input type="number" min="0" value={conversionsB} onChange={(e) => setConversionsB(e.target.value)} />
        <button className="upload-button" onClick={handleABTest} disabled={loading}>{loading ? "Analyzing..." : "Analyze A/B Test"}</button>
        <button className="upload-button" onClick={handleSimulation} disabled={loading} style={{ marginTop: 10 }}>Simulate Experiment Data</button>
        <div style={{ marginTop: 22 }}>
          <h3>Simulation Settings</h3>
          <label>Visitors per group</label>
          <input type="number" min="1" value={simulationVisitors} onChange={(e) => setSimulationVisitors(e.target.value)} />
          <label>Group A conversion probability (0–1)</label>
          <input type="number" min="0.001" max="0.999" step="0.01" value={simulationRateA} onChange={(e) => setSimulationRateA(e.target.value)} />
          <label>Group B conversion probability (0–1)</label>
          <input type="number" min="0.001" max="0.999" step="0.01" value={simulationRateB} onChange={(e) => setSimulationRateB(e.target.value)} />
        </div>
        {error && <p className="error-message">{error}</p>}
      </div>
      <ResultJSON title="A/B Testing Results" data={result} />
    </div>
  );

  const renderMultipleTesting = () => (
    <div className="upload-section">
      <div className="upload-card">
        <h2>Multiple Testing Correction</h2>
        <p>Enter p-values separated by commas. The API calculates Bonferroni and Benjamini–Hochberg corrections.</p>
        <label>Original p-values</label>
        <textarea rows="4" value={pValues} onChange={(e) => setPValues(e.target.value)} />
        <button className="upload-button" onClick={handleMultipleTesting} disabled={loading}>{loading ? "Calculating..." : "Correct Multiple Tests"}</button>
        {error && <p className="error-message">{error}</p>}
      </div>
      <ResultJSON title="Multiple Testing Results" data={result} />
    </div>
  );

  const renderSequentialTesting = () => (
    <div className="upload-section">
      <div className="upload-card">
        <h2>Sequential Testing</h2>
        <p>Evaluate experiment results at a planned analysis look. This backend uses a conservative Bonferroni-adjusted significance threshold across the planned looks.</p>
        <label>Group A visitors</label>
        <input type="number" min="1" value={visitorsA} onChange={(e) => setVisitorsA(e.target.value)} />
        <label>Group A conversions</label>
        <input type="number" min="0" value={conversionsA} onChange={(e) => setConversionsA(e.target.value)} />
        <label>Group B visitors</label>
        <input type="number" min="1" value={visitorsB} onChange={(e) => setVisitorsB(e.target.value)} />
        <label>Group B conversions</label>
        <input type="number" min="0" value={conversionsB} onChange={(e) => setConversionsB(e.target.value)} />
        <label>Current analysis look</label>
        <input type="number" min="1" value={currentLook} onChange={(e) => setCurrentLook(e.target.value)} />
        <label>Maximum planned looks</label>
        <input type="number" min="1" value={maxLooks} onChange={(e) => setMaxLooks(e.target.value)} />
        <button className="upload-button" onClick={handleSequentialTest} disabled={loading}>{loading ? "Analyzing..." : "Run Sequential Test"}</button>
        {error && <p className="error-message">{error}</p>}
      </div>
      <ResultJSON title="Sequential Testing Results" data={result} />
    </div>
  );

  const renderBayesian = () => (
    <div className="upload-section">
      <div className="upload-card">
        <h2>Bayesian A/B Analysis</h2>
        <p>Estimate the posterior probability that group B has a higher conversion rate than group A.</p>
        <label>Group A visitors</label>
        <input type="number" min="1" value={visitorsA} onChange={(e) => setVisitorsA(e.target.value)} />
        <label>Group A conversions</label>
        <input type="number" min="0" value={conversionsA} onChange={(e) => setConversionsA(e.target.value)} />
        <label>Group B visitors</label>
        <input type="number" min="1" value={visitorsB} onChange={(e) => setVisitorsB(e.target.value)} />
        <label>Group B conversions</label>
        <input type="number" min="0" value={conversionsB} onChange={(e) => setConversionsB(e.target.value)} />
        <button className="upload-button" onClick={handleBayesianTest} disabled={loading}>{loading ? "Sampling posterior..." : "Run Bayesian Analysis"}</button>
        {error && <p className="error-message">{error}</p>}
      </div>
      <ResultJSON title="Bayesian Analysis Results" data={result} />
    </div>
  );

  const renderTTest = () => (
    <div className="upload-section">
      <div className="upload-card">
        <h2>Welch's Independent T-Test</h2>
        <p>Enter numeric observations separated by commas for each group.</p>
        <label>Group A observations</label>
        <textarea rows="3" value={groupA} onChange={(e) => setGroupA(e.target.value)} />
        <label>Group B observations</label>
        <textarea rows="3" value={groupB} onChange={(e) => setGroupB(e.target.value)} />
        <button className="upload-button" onClick={handleTTest} disabled={loading}>{loading ? "Analyzing..." : "Run T-Test"}</button>
        {error && <p className="error-message">{error}</p>}
      </div>
      <ResultJSON title="T-Test Results" data={result} />
    </div>
  );

  const renderPlaceholder = () => (
    <div className="placeholder-card">
      <span className="placeholder-icon">⚙️</span>
      <h2>{activeTab}</h2>
      <p>This module is part of the platform. Its frontend controls and results view are not included in this version.</p>
      <span className="coming-soon">MODULE IN DEVELOPMENT</span>
    </div>
  );

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-icon">C</span>
          <div>
            <h2>CausalLab</h2>
            <p>Experimentation Platform</p>
          </div>
        </div>

        <p className="menu-label">WORKSPACE</p>
        <nav>
          {menuItems.map((item) => (
            <button
              key={item}
              className={`nav-item ${activeTab === item ? "active" : ""}`}
              onClick={() => {
                setActiveTab(item);
                resetMessage();
              }}
            >
              {item === "Overview" ? "▦" : "◈"} {item}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <p>DS-04 Capstone Project</p>
          <span>Powered by FastAPI + React</span>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <span>Experimentation &amp; Causal Inference</span>
          <span className="status"><span className="status-dot"></span> Platform</span>
        </header>

        <section className="page-content">
          <div className="welcome">
            <div>
              <p className="eyebrow">ANALYTICS WORKSPACE</p>
              <h1>{activeTab}</h1>
              <p>Design experiments, analyze results, and discover causal relationships in your data.</p>
            </div>
          </div>

          {activeTab === "Overview" ? renderOverview()
            : activeTab === "A/B Testing" ? renderABTesting()
            : activeTab === "Sample Size Calculator" ? renderSampleSize()
            : activeTab === "Multiple Testing" ? renderMultipleTesting()
            : activeTab === "Sequential Testing" ? renderSequentialTesting()
            : activeTab === "Bayesian Analysis" ? renderBayesian()
            : activeTab === "Causal Inference" ? renderCausalInference()
            : activeTab === "Dataset Analysis" ? renderDatasetAnalysis()
            : activeTab === "T-Test" ? renderTTest()
            : renderPlaceholder()}
        </section>
      </main>
    </div>
  );
}

export default App;
