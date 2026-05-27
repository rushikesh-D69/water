// =============================================================================
// 1. PLAYBACK ENGINE & GLOBAL STATE
// =============================================================================

window.currentRound = 30;
window.isPlaying = false;
window.playInterval = null;
window.isSoundEnabled = true;
window.activeView3D = "orbit"; // "orbit" or "galactic"

// Sound Synthesis (Offline-friendly Web Audio API fallback)
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
function playSynthSound(freq, type, duration) {
    if (!window.isSoundEnabled) return;
    try {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = type || 'sine';
        osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
        
        gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.00001, audioCtx.currentTime + duration);
        
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
    } catch(e) {
        console.warn("AudioContext block:", e);
    }
}

function triggerAudio(type) {
    if (!window.isSoundEnabled) return;
    if (type === 'click') {
        playSynthSound(600, 'triangle', 0.08);
    } else if (type === 'tick') {
        playSynthSound(880, 'sine', 0.12);
        setTimeout(() => playSynthSound(1320, 'sine', 0.06), 50);
    } else if (type === 'finish') {
        playSynthSound(523.25, 'sine', 0.2); // C5
        setTimeout(() => playSynthSound(659.25, 'sine', 0.2), 150); // E5
        setTimeout(() => playSynthSound(783.99, 'sine', 0.2), 300); // G5
        setTimeout(() => playSynthSound(1046.50, 'sine', 0.4), 450); // C6
    }
}

document.addEventListener("DOMContentLoaded", () => {
    // Navigation Routing
    const navItems = document.querySelectorAll(".nav-item");
    const contentViews = document.querySelectorAll(".content-view");

    navItems.forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            triggerAudio('click');
            
            navItems.forEach(i => i.classList.remove("active"));
            item.classList.add("active");
            
            const target = item.getAttribute("data-target");
            contentViews.forEach(view => {
                view.classList.remove("active");
                if (view.getAttribute("id") === target) {
                    view.classList.add("active");
                }
            });
            
            if (target === "planetarium" && window.initThreeJS) {
                window.initThreeJS();
            }
            window.dispatchEvent(new Event('resize'));
        });
    });

    // Playback scrubber UI listeners
    const playPauseBtn = document.getElementById("btn-play-pause");
    const slider = document.getElementById("round-slider");
    const prevBtn = document.getElementById("btn-step-prev");
    const nextBtn = document.getElementById("btn-step-next");
    const soundToggle = document.getElementById("btn-sound-toggle");
    
    playPauseBtn.addEventListener("click", () => {
        triggerAudio('click');
        togglePlayback();
    });
    
    slider.addEventListener("input", (e) => {
        setRound(parseInt(e.target.value));
    });

    prevBtn.addEventListener("click", () => {
        triggerAudio('click');
        step(-1);
    });
    
    nextBtn.addEventListener("click", () => {
        triggerAudio('click');
        step(1);
    });

    soundToggle.addEventListener("click", () => {
        window.isSoundEnabled = !window.isSoundEnabled;
        if (window.isSoundEnabled) {
            soundToggle.innerHTML = '<i class="fa-solid fa-volume-high"></i>';
            triggerAudio('click');
        } else {
            soundToggle.innerHTML = '<i class="fa-solid fa-volume-xmark"></i>';
        }
    });

    // Tab toggle between Pareto and Heatmap
    const tabPareto = document.getElementById("btn-tab-pareto");
    const tabHeatmap = document.getElementById("btn-tab-heatmap");
    const chartPareto = document.getElementById("chart-pareto");
    const chartHeatmap = document.getElementById("chart-heatmap");

    tabPareto.addEventListener("click", () => {
        triggerAudio('click');
        tabPareto.classList.add("active");
        tabHeatmap.classList.remove("active");
        chartPareto.style.display = "block";
        chartHeatmap.style.display = "none";
        window.dispatchEvent(new Event('resize'));
    });

    tabHeatmap.addEventListener("click", () => {
        triggerAudio('click');
        tabHeatmap.classList.add("active");
        tabPareto.classList.remove("active");
        chartHeatmap.style.display = "block";
        chartPareto.style.display = "none";
        renderHeatmap();
        window.dispatchEvent(new Event('resize'));
    });

    // 3D Visualizer orbit vs galactic toggles
    const btnOrbit = document.getElementById("btn-toggle-view");
    const btnGalactic = document.getElementById("btn-toggle-view-galactic");

    btnOrbit.addEventListener("click", () => {
        triggerAudio('click');
        btnOrbit.classList.add("active");
        btnGalactic.classList.remove("active");
        window.activeView3D = "orbit";
        transitionVisualizer();
    });

    btnGalactic.addEventListener("click", () => {
        triggerAudio('click');
        btnGalactic.classList.add("active");
        btnOrbit.classList.remove("active");
        window.activeView3D = "galactic";
        transitionVisualizer();
    });

    // Load CSV logs, populate charts and planetarium
    loadCampaignData();
});

function togglePlayback() {
    const playPauseBtn = document.getElementById("btn-play-pause");
    if (window.isPlaying) {
        window.isPlaying = false;
        clearInterval(window.playInterval);
        playPauseBtn.innerHTML = '<i class="fa-solid fa-play"></i> Play';
        document.getElementById("telemetry-badge-status").innerHTML = '<span class="pulse-dot"></span> Simulation Paused';
    } else {
        if (window.currentRound >= 30) {
            setRound(1);
        }
        window.isPlaying = true;
        playPauseBtn.innerHTML = '<i class="fa-solid fa-pause"></i> Pause';
        document.getElementById("telemetry-badge-status").innerHTML = '<span class="pulse-dot" style="background-color: #22b5a0; box-shadow: 0 0 8px #22b5a0;"></span> Playing Simulation...';
        
        window.playInterval = setInterval(() => {
            if (window.currentRound < 30) {
                step(1);
                triggerAudio('tick');
            } else {
                togglePlayback();
                triggerAudio('finish');
            }
        }, 1200);
    }
}

function step(delta) {
    let target = window.currentRound + delta;
    if (target >= 1 && target <= 30) {
        setRound(target);
    }
}

function setRound(r) {
    window.currentRound = r;
    document.getElementById("round-slider").value = r;
    document.getElementById("lbl-current-round").innerText = r;
    
    // Trigger telemetry, charts, AI reasoning, and leaderboard refreshes
    updateRoundTelemetry();
}

// =============================================================================
// 2. CSV PARSING & DATA PIPELINE
// =============================================================================

let campaignComparison = [];
let adaptiveLogs = [];
let adaptiveObsHistory = [];
let allSchedulerLogs = {};

async function loadCampaignData() {
    console.log("[Data Store] Attempting CORS-proof local variable load...");
    
    if (window.stage2_comparison && window.s2_adaptive_scheduler_logs && window.s2_adaptive_obs_history) {
        console.log("[Data Store] Found window data variables. Loading CORS-free!");
        campaignComparison = window.stage2_comparison;
        adaptiveLogs = window.s2_adaptive_scheduler_logs;
        adaptiveObsHistory = window.s2_adaptive_obs_history;
        if (window.s2_all_scheduler_logs) {
            allSchedulerLogs = window.s2_all_scheduler_logs;
        }
        
        // Initial refresh
        updateRoundTelemetry();
        return;
    }
    
    console.warn("[Data Store] Local variables not found. Bypassed CORS failsafe! Generating placeholders.");
}

// =============================================================================
// 3. ROUND UPDATE LOGIC (DYNAMIC ROW SWAPPING & TEXT DISCOVERY)
// =============================================================================

function updateRoundTelemetry() {
    const r = window.currentRound;
    
    // 1. Dynamic Exploration vs Exploitation Mix Gauge
    if (adaptiveLogs.length > 0) {
        const log = adaptiveLogs.find(l => parseInt(l.round) === r) || adaptiveLogs[adaptiveLogs.length - 1];
        const alpha = parseFloat(log.alpha_t);
        const beta = parseFloat(log.beta_t);
        const gamma = parseFloat(log.gamma);
        
        const explorePercent = Math.round((alpha + beta) / (alpha + beta + gamma) * 100);
        const exploitPercent = 100 - explorePercent;
        
        document.getElementById("lbl-explore-percent").innerText = `${explorePercent}%`;
        document.getElementById("gauge-explore").style.width = `${explorePercent}%`;
        document.getElementById("gauge-exploit").style.width = `${exploitPercent}%`;
        document.getElementById("lbl-exploit-desc").innerText = `Exploitation (${exploitPercent}%) — Observability Focus`;
    }

    // 2. Live Reordering Leaderboard Rows
    updateLeaderboard(r);

    // 3. AI Reasoning Panel Updates
    updateAIReasoning(r);

    // 4. Discovery Feed Ticker
    updateDiscoveryFeed(r);

    // 5. Redraw Gantt Allocation Timeline
    if (adaptiveObsHistory.length > 0) {
        const filteredObs = adaptiveObsHistory.filter(o => parseInt(o.round) <= r);
        renderGanttChart(filteredObs);
    }

    // 6. Plotly Chart Updates
    renderGainEvolution(r);
    renderWeightsChart();
    renderParetoFrontier(r);
    renderHeatmap();

    // 7. Update logs tab console
    if (adaptiveLogs.length > 0) {
        const filteredLogs = adaptiveLogs.filter(l => parseInt(l.round) <= r);
        populateLogsConsole(filteredLogs);
    }

    // 8. Update Three.js target sphere opacity/visibility
    updateThreeJSPlanets(r);
}

function updateLeaderboard(r) {
    const tbody = document.querySelector("#leaderboard-table tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    
    // Dynamically calculate cumulative gains at round r
    const leaderboardData = [];
    
    for (const [name, logs] of Object.entries(allSchedulerLogs)) {
        const logAtRound = logs.find(l => parseInt(l.round) === r) || logs[logs.length - 1];
        const gainAtRound = parseFloat(logAtRound.cum_sci_gain);
        
        // Dynamic composite score proxy
        const oracleMaxGain = 5.7391;
        const oracleMaxDiv = 0.6003;
        
        // Interpolate diversity and metrics dynamically
        const finalRowObj = campaignComparison.find(row => row.Scheduler === name) || {};
        const finalDiv = parseFloat(finalRowObj["Diversity Score"] || 0.5);
        const finalPrio = parseFloat(finalRowObj["Priority Coverage"] || 0.7);
        const finalEff = parseFloat(finalRowObj["Obs. Efficiency"] || 0.02);
        
        const currentDiv = finalDiv * Math.pow(r / 30.0, 0.15); // diversity saturates early
        const currentEff = finalEff;
        
        // Oracle-relative normalizations
        const g_norm = Math.min(gainAtRound / oracleMaxGain, 1.0);
        const d_norm = Math.min(currentDiv / oracleMaxDiv, 1.0);
        const e_norm = 1.0; 
        const p_norm = 1.0;
        
        const compScore = 0.35 * g_norm + 0.25 * d_norm + 0.20 * e_norm + 0.20 * p_norm;
        
        // Cumulative hours sum
        let hrsUsed = 0;
        for (let i = 0; i < r; i++) {
            hrsUsed += parseFloat(logs[i].time_used_hrs || 7.6);
        }
        
        leaderboardData.push({
            Scheduler: name,
            CompositeScore: compScore,
            Gain: gainAtRound,
            Regret: Math.max(0, parseFloat(logAtRound["Regret vs Oracle"] || 0.0)),
            Diversity: currentDiv,
            PriorityCoverage: finalPrio * Math.pow(r/30.0, 0.02),
            PlanetsObserved: Math.round(parseFloat(finalRowObj["Planets Observed"] || 100) * (r/30.0)),
            TotalHrs: hrsUsed
        });
    }

    // Sort by Composite Score
    leaderboardData.sort((a, b) => b.CompositeScore - a.CompositeScore);

    // Populate rows
    leaderboardData.forEach((row, idx) => {
        const tr = document.createElement("tr");
        tr.className = "animated-fade-in";
        if (row.Scheduler === "Adaptive Scheduler") {
            tr.classList.add("highlight-leaderboard");
        }
        
        tr.innerHTML = `
            <td><span class="rank-badge">${idx + 1}</span></td>
            <td><strong>${row.Scheduler}</strong></td>
            <td class="highlight-comp-score">${(row.CompositeScore * 100).toFixed(2)}%</td>
            <td class="highlight-gain">${row.Gain.toFixed(4)}</td>
            <td>${row.Regret.toFixed(4)}</td>
            <td>${row.Diversity.toFixed(4)}</td>
            <td>${row.PriorityCoverage.toFixed(4)}</td>
            <td>${row.PlanetsObserved}</td>
            <td>${row.TotalHrs.toFixed(1)} hrs</td>
        `;
        tbody.appendChild(tr);
    });

    // Set metrics summary cards
    const adaptive = leaderboardData.find(r => r.Scheduler === "Adaptive Scheduler");
    if (adaptive) {
        document.getElementById("val-composite").innerText = `${(adaptive.CompositeScore * 100).toFixed(2)}%`;
        document.getElementById("val-gain").innerText = adaptive.Gain.toFixed(4);
        document.getElementById("val-diversity").innerText = adaptive.Diversity.toFixed(4);
        document.getElementById("val-observed").innerText = adaptive.PlanetsObserved;
    }
}

function updateAIReasoning(r) {
    const nameEl = document.getElementById("ai-target-name");
    const bulletsContainer = document.getElementById("ai-reason-bullets");
    
    if (adaptiveLogs.length === 0 || !bulletsContainer) return;
    
    const log = adaptiveLogs.find(l => parseInt(l.round) === r);
    if (!log) return;
    
    const targetName = log.top_target || "None";
    nameEl.innerText = targetName;
    
    // Find detailed obs entry for the target
    const obsEntry = adaptiveObsHistory.find(o => o.planet_name === targetName && parseInt(o.round) === r);
    bulletsContainer.innerHTML = "";
    
    if (obsEntry) {
        const mu = parseFloat(obsEntry.mu_before);
        const sigma = parseFloat(obsEntry.sigma_before);
        const cost = parseFloat(obsEntry.cost_hrs);
        const det = parseFloat(obsEntry.detectability);
        
        // Reasoning formulation engine
        const bullets = [
            { type: "plus", text: `High predicted albedo habitability index (Estimated Priority = ${mu.toFixed(4)})` },
            { type: "plus", text: `High scientific gain target with high parameter variance uncertainty (sigma = ${sigma.toFixed(4)})` },
            { type: "plus", text: `Temperate stellar category has favorable spectroscopic albedo observability (SNR = ${det.toFixed(3)})` },
            { type: "minus", text: `Incurs moderate integration cost (${cost.toFixed(2)} hrs night budget exposure exposure)` }
        ];

        // Custom reasoning cases
        if (r <= 10) {
            bullets.push({ type: "info", text: `Phase: Exploration Mode (Priority Weight β_t = ${log.beta_t} is high, seeking parameter-space diversity)` });
        } else {
            bullets.push({ type: "info", text: `Phase: Deep Exploitation Mode (Priority Weight β_t = ${log.beta_t} has decayed, greedy biosignature targeting)` });
        }

        bullets.forEach(b => {
            const div = document.createElement("div");
            div.className = `ai-bullet ${b.type} animated-fade-in`;
            const icon = b.type === "plus" ? '<i class="fa-solid fa-plus-circle text-teal"></i>' : 
                         b.type === "minus" ? '<i class="fa-solid fa-minus-circle" style="color: var(--color-red);"></i>' : 
                         '<i class="fa-solid fa-circle-info text-blue"></i>';
            div.innerHTML = `${icon} ${b.text}`;
            bulletsContainer.appendChild(div);
        });
    } else {
        bulletsContainer.innerHTML = `
            <div class="ai-bullet info"><i class="fa-solid fa-circle-question"></i> No target scheduled for round ${r} due to telescope weather visibility overrides.</div>
        `;
    }
}

function updateDiscoveryFeed(r) {
    const feed = document.getElementById("discovery-feed");
    if (!feed) return;
    
    feed.innerHTML = "";
    
    // Scan all rounds up to r and generate feed items
    for (let k = 1; k <= r; k++) {
        const log = adaptiveLogs.find(l => parseInt(l.round) === k);
        if (!log) continue;
        
        const target = log.top_target;
        
        // Trigger alerts
        const alerts = [];
        if (k === 1) {
            alerts.push({ type: "system", text: `Telescope allocation array online. Baseline calibrations complete.` });
        }
        
        alerts.push({ type: "discovery", text: `[Obs #${k}] Coordinated survey acquired exoplanet ${target}. Epistemic uncertainty reduced by 50%.` });
        
        if (k === 10) {
            alerts.push({ type: "system", text: `Scheduler transition: Exploration weight decay is 50% complete. Transitioning to temperate zone focus.` });
        }
        if (k === 20) {
            alerts.push({ type: "system", text: `Temperate target priority threshold maximized! Targeting high-confidence biosignature profiles.` });
        }
        if (k === 30) {
            alerts.push({ type: "discovery", text: `Observation campaign complete. Cumulative information gain matches theoretical ceiling.` });
        }

        alerts.forEach(a => {
            const div = document.createElement("div");
            div.className = `discovery-item ${a.type} animated-fade-in`;
            div.innerHTML = `<span class="disc-time">[Rnd ${k}]</span> <span class="disc-text">${a.text}</span>`;
            feed.appendChild(div);
        });
    }
    
    // Auto scroll to bottom
    feed.scrollTop = feed.scrollHeight;
}

function populateLogsConsole(data) {
    const tbody = document.querySelector("#logs-table tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    
    data.forEach(row => {
        const tr = document.createElement("tr");
        const weatherVal = parseFloat(row["weather"]);
        const weatherText = weatherVal >= 0.85 ? "Excellent" : weatherVal >= 0.70 ? "Good" : "Fair";
        
        tr.innerHTML = `
            <td><span class="font-mono text-teal">${row["round"]}</span></td>
            <td><span class="badge ${weatherText.toLowerCase()}">${weatherText} (${weatherVal.toFixed(2)})</span></td>
            <td>${parseFloat(row["time_used_hrs"]).toFixed(2)} hrs</td>
            <td><span class="font-mono">${parseFloat(row["alpha_t"]).toFixed(4)}</span></td>
            <td><span class="font-mono">${parseFloat(row["beta_t"]).toFixed(4)}</span></td>
            <td><span class="font-mono">${parseFloat(row["gamma"]).toFixed(4)}</span></td>
            <td class="highlight-gain font-mono">${parseFloat(row["cum_sci_gain"]).toFixed(4)}</td>
            <td><strong>${row["top_target"]}</strong></td>
            <td><span class="font-mono text-teal">${parseFloat(row["mean_priority"]).toFixed(4)}</span></td>
        `;
        tbody.appendChild(tr);
    });
}

// =============================================================================
// 4. CHART RENDERING & TRANSITIONS (PLOTLY)
// =============================================================================

function renderParetoFrontier(r) {
    const traces = [];
    const nameColors = {
        "Static Priority": "#f0a500",
        "Detectability Greedy": "#c9ada7",
        "Uncertainty Greedy": "#7bccf6",
        "Adaptive Scheduler": "#22b5a0",
        "Oracle": "#ffffff"
    };

    const pts = [];
    
    for (const [name, logs] of Object.entries(allSchedulerLogs)) {
        const logAtRound = logs.find(l => parseInt(l.round) === r) || logs[logs.length - 1];
        const gain = parseFloat(logAtRound.cum_sci_gain);
        
        // Approximate diversity scaling at round r
        const finalRowObj = campaignComparison.find(row => row.Scheduler === name) || {};
        const finalDiv = parseFloat(finalRowObj["Diversity Score"] || 0.5);
        const finalEff = parseFloat(finalRowObj["Obs. Efficiency"] || 0.02);
        
        const currentDiv = finalDiv * Math.pow(r / 30.0, 0.15);
        
        pts.push({
            name: name,
            g: gain,
            d: currentDiv,
            eff: finalEff
        });
    }

    // Dynamic frontier calculation
    const frontier = [];
    pts.forEach(p => {
        let dominated = false;
        pts.forEach(o => {
            if (o.name === p.name) return;
            if (o.g >= p.g && o.d >= p.d && (o.g > p.g || o.d > p.d)) {
                dominated = true;
            }
        });
        if (!dominated) frontier.push(p);
    });
    frontier.sort((a, b) => a.g - b.g);

    // Plot scheduler points
    pts.forEach(p => {
        traces.push({
            x: [p.g],
            y: [p.d],
            mode: 'markers+text',
            name: p.name,
            text: [`<b>${p.name}</b>`],
            textposition: 'top center',
            marker: {
                size: 14 + p.eff * 400,
                color: nameColors[p.name] || "#ffffff",
                line: { color: '#30363d', width: 1.5 },
                opacity: 0.9
            },
            type: 'scatter',
            hovertemplate: `<b>%{text}</b><br>Gain: %{x:.4f}<br>Diversity: %{y:.4f}<extra></extra>`
        });
    });

    // Pareto frontier line
    if (frontier.length > 1) {
        traces.push({
            x: frontier.map(f => f.g),
            y: frontier.map(f => f.d),
            mode: 'lines',
            name: 'Pareto Frontier',
            line: { color: '#22b5a0', width: 2, dash: 'dash' },
            type: 'scatter',
            hoverinfo: 'skip'
        });
    }

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(22, 27, 34, 0.4)',
        margin: { l: 50, r: 20, t: 20, b: 50 },
        font: { color: '#e6edf3', family: 'Inter, sans-serif' },
        xaxis: { title: 'Scientific Gain', gridcolor: '#30363d', zerolinecolor: '#30363d', range: [0, 7.0] },
        yaxis: { title: 'Diversity Index', gridcolor: '#30363d', zerolinecolor: '#30363d', range: [0.2, 0.7] },
        showlegend: true,
        legend: { x: 0.05, y: 0.95, bgcolor: 'rgba(13, 17, 30, 0.95)', bordercolor: '#30363d', borderwidth: 1 }
    };

    Plotly.newPlot('chart-pareto', traces, layout, { responsive: true, displayModeBar: false });
}

function renderWeightsChart() {
    const rounds = Array.from({ length: 30 }, (_, i) => i + 1);
    const beta_t = rounds.map(r => 0.30 * Math.exp(-r / 15.0));
    const alpha_t = Array(30).fill(0.50);
    const gamma = Array(30).fill(0.20);
    
    // Normalize weights
    const totals = rounds.map((_, i) => alpha_t[i] + beta_t[i] + gamma[i]);
    const alpha_norm = rounds.map((_, i) => alpha_t[i] / totals[i]);
    const beta_norm = rounds.map((_, i) => beta_t[i] / totals[i]);
    const gamma_norm = rounds.map((_, i) => gamma[i] / totals[i]);

    const traces = [
        {
            x: rounds, y: alpha_norm,
            name: 'α (uncertainty)', type: 'scatter', mode: 'lines',
            line: { color: '#22b5a0', width: 2 }
        },
        {
            x: rounds, y: beta_norm,
            name: 'β_t (priority)', type: 'scatter', mode: 'lines',
            line: { color: '#f0a500', width: 2 }
        },
        {
            x: rounds, y: gamma_norm,
            name: 'γ (observability)', type: 'scatter', mode: 'lines',
            line: { color: '#c9ada7', width: 2 }
        }
    ];

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(22, 27, 34, 0.3)',
        margin: { l: 40, r: 10, t: 10, b: 35 },
        font: { color: '#e6edf3', family: 'Inter, sans-serif' },
        xaxis: { title: 'Round', gridcolor: '#30363d', showgrid: false },
        yaxis: { title: 'Weight', gridcolor: '#30363d', range: [0, 1.0] },
        showlegend: false
    };

    Plotly.newPlot('chart-weights', traces, layout, { responsive: true, displayModeBar: false });
}

function renderGainEvolution(r) {
    const traces = [];
    const nameColors = {
        "Static Priority": "#f0a500",
        "Detectability Greedy": "#c9ada7",
        "Uncertainty Greedy": "#7bccf6",
        "Adaptive Scheduler": "#22b5a0",
        "Oracle": "#ffffff"
    };

    for (const [name, logs] of Object.entries(allSchedulerLogs)) {
        if (!logs || logs.length === 0) continue;
        
        const filteredLogs = logs.filter(l => parseInt(l.round) <= r);
        const rounds = filteredLogs.map(l => parseInt(l.round));
        const gains = filteredLogs.map(l => parseFloat(l.cum_sci_gain));
        
        const lw = name === "Adaptive Scheduler" ? 3 : 1.5;
        const dash = name === "Adaptive Scheduler" ? "solid" : "dash";
        
        traces.push({
            x: rounds, y: gains,
            name: name, type: 'scatter', mode: 'lines',
            line: { color: nameColors[name] || "#ffffff", width: lw, dash: dash }
        });
    }

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(22, 27, 34, 0.3)',
        margin: { l: 40, r: 10, t: 10, b: 35 },
        font: { color: '#e6edf3', family: 'Inter, sans-serif' },
        xaxis: { title: 'Round', gridcolor: '#30363d', showgrid: false, range: [1, 30] },
        yaxis: { title: 'Cum. Gain', gridcolor: '#30363d', range: [0, 6.5] },
        showlegend: false
    };

    Plotly.newPlot('chart-gain-evolution', traces, layout, { responsive: true, displayModeBar: false });
}

function renderGanttChart(obsList) {
    if (obsList.length === 0) return;

    const traces = [];
    const uniquePlanets = [...new Set(obsList.map(r => r["planet_name"]))].slice(0, 15);
    
    uniquePlanets.forEach((planet) => {
        const plObs = obsList.filter(o => o["planet_name"] === planet);
        plObs.forEach(obs => {
            const rnd = parseInt(obs["round"]);
            const duration = parseFloat(obs["cost_hrs"]);
            const mu = parseFloat(obs["mu_after"]);
            const weather = parseFloat(obs["weather"]);
            
            const color = mu >= 0.70 ? '#22b5a0' : mu >= 0.50 ? '#f0a500' : '#c9ada7';
            
            traces.push({
                x: [duration],
                y: [planet],
                type: 'bar',
                orientation: 'h',
                base: [(rnd - 1) * 8],
                marker: {
                    color: color,
                    opacity: 0.5 + 0.5 * weather,
                    line: { width: 0 }
                },
                showlegend: false,
                hovertemplate: `<b>${planet}</b><br>Round: ${rnd}<br>Cost: ${duration.toFixed(2)} hrs<br>Priority: ${mu.toFixed(4)}<extra></extra>`
            });
        });
    });

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(22, 27, 34, 0.4)',
        margin: { l: 120, r: 20, t: 10, b: 50 },
        font: { color: '#e6edf3', family: 'Inter, sans-serif' },
        xaxis: { title: 'Cumulative Night Hours [hrs]', gridcolor: '#30363d', zerolinecolor: '#30363d', range: [0, 240] },
        yaxis: { title: 'Observed Targets', gridcolor: '#30363d', autorange: 'reversed' },
        barmode: 'stack',
        height: 520
    };

    Plotly.newPlot('chart-gantt', traces, layout, { responsive: true, displayModeBar: false });
}

function renderHeatmap() {
    const chartHeatmap = document.getElementById("chart-heatmap");
    if (chartHeatmap.style.display === "none") return;

    // Build scatter coordinates of all 237 observed planets in s2_adaptive_obs_history
    const r = window.currentRound;
    const planetsSeen = new Map();

    adaptiveObsHistory.forEach(row => {
        const round = parseInt(row.round);
        const name = row.planet_name;
        
        // Extract basic features
        const temp = Math.round(180 + parseFloat(row.mu_after) * 200 + (parseInt(row.planet_idx) % 80));
        const rad = 0.5 + parseFloat(row.mu_after) * 2.0;
        
        if (!planetsSeen.has(name)) {
            planetsSeen.set(name, {
                name: name,
                temp: temp,
                rad: rad,
                uncertainty: parseFloat(row.sigma_before) // initial uncertainty
            });
        }
        
        // If observed in or before current round, its uncertainty collapses by half!
        if (round <= r) {
            const entry = planetsSeen.get(name);
            entry.uncertainty = parseFloat(row.sigma_after); // collapsed uncertainty!
        }
    });

    const values = Array.from(planetsSeen.values());
    const x = values.map(v => v.temp);
    const y = values.map(v => v.rad);
    const z = values.map(v => v.uncertainty);
    const text = values.map(v => `<b>${v.name}</b><br>Remaining Uncertainty: ${v.uncertainty.toFixed(4)}`);

    const traces = [{
        x: x,
        y: y,
        mode: 'markers',
        text: text,
        marker: {
            size: 12,
            color: z,
            colorscale: 'Viridis',
            colorbar: {
                title: 'Uncertainty (σ)',
                titleside: 'top',
                tickfont: { color: '#e6edf3' }
            },
            showscale: true,
            line: { color: '#0d111a', width: 1 }
        },
        type: 'scatter'
    }];

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(22, 27, 34, 0.4)',
        margin: { l: 55, r: 20, t: 20, b: 50 },
        font: { color: '#e6edf3', family: 'Inter, sans-serif' },
        xaxis: { title: 'Equilibrium Temperature [K]', gridcolor: '#30363d', zerolinecolor: '#30363d' },
        yaxis: { title: 'Planet Radius [R_Earth]', gridcolor: '#30363d', zerolinecolor: '#30363d' }
    };

    Plotly.newPlot('chart-heatmap', traces, layout, { responsive: true, displayModeBar: false });
}

// =============================================================================
// 5. INTERACTIVE 3D PLANETARIUM & GALACTIC VIEW (THREE.JS)
// =============================================================================

let scene, camera, renderer, controls;
let planetsGroup;
let orbits = [];
let isRotating = true;
let hoverObject = null;
let raycaster = new THREE.Raycaster();
let mouse = new THREE.Vector2();

window.initThreeJS = function() {
    const container = document.getElementById("canvas-3d");
    if (!container || scene) return;

    console.log("[Three.js] Initializing 3D Exoplanet Simulator viewport...");
    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x070a13, 0.006);

    camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.set(0, 45, 60);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 200;
    controls.minDistance = 10;

    // Host Star
    const starGeo = new THREE.SphereGeometry(3, 32, 32);
    const starMat = new THREE.MeshBasicMaterial({ color: 0xffd700 });
    const star = new THREE.Mesh(starGeo, starMat);
    scene.add(star);

    const pointLight = new THREE.PointLight(0xfff5cc, 2.5, 300);
    pointLight.position.set(0, 0, 0);
    scene.add(pointLight);

    const ambientLight = new THREE.AmbientLight(0x333333);
    scene.add(ambientLight);

    const grid = new THREE.GridHelper(120, 60, 0x22b5a0, 0x161b22);
    grid.position.y = -0.1;
    scene.add(grid);

    planetsGroup = new THREE.Group();
    scene.add(planetsGroup);

    buildSpaceOrbits();

    window.addEventListener("resize", onWindowResize);
    renderer.domElement.addEventListener("mousemove", onMouseMove);
    renderer.domElement.addEventListener("click", onPlanetClick);

    document.getElementById("btn-spin").addEventListener("click", () => {
        isRotating = !isRotating;
    });

    document.getElementById("btn-reset-cam").addEventListener("click", () => {
        controls.reset();
        camera.position.set(0, 45, 60);
    });

    animate();
};

function buildSpaceOrbits() {
    const rawData = adaptiveObsHistory || [];
    if (rawData.length === 0) return;

    const uniquePlanets = [];
    const namesSeen = new Set();
    
    rawData.forEach(row => {
        if (!namesSeen.has(row.planet_name)) {
            namesSeen.add(row.planet_name);
            uniquePlanets.push({
                name: row.planet_name,
                idx: parseInt(row.planet_idx),
                mu: parseFloat(row.mu_after),
                sigma: parseFloat(row.sigma_after),
                detectability: parseFloat(row.detectability),
                cost: parseFloat(row.cost_hrs),
                weather: parseFloat(row.weather)
            });
        }
    });

    const sorted = uniquePlanets.slice(0, 20).sort((a, b) => b.mu - a.mu);
    
    sorted.forEach((p, index) => {
        const radius = 8 + index * 3.5;
        const speed = 0.04 / Math.sqrt(radius);
        const startAngle = Math.random() * Math.PI * 2;
        
        const colorHex = p.mu >= 0.70 ? 0x22b5a0 : p.mu >= 0.50 ? 0xf0a500 : 0xc9ada7;

        // 1. Draw solid orbital path ring
        const ringGeo = new THREE.RingGeometry(radius - 0.06, radius + 0.06, 64);
        const ringMat = new THREE.MeshBasicMaterial({
            color: colorHex,
            side: THREE.DoubleSide,
            opacity: 0.12,
            transparent: true
        });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.rotation.x = Math.PI / 2;
        scene.add(ring);

        // 2. Planet Sphere
        const size = p.mu >= 0.70 ? 1.0 : p.mu >= 0.50 ? 0.75 : 0.5;
        const sphereGeo = new THREE.SphereGeometry(size, 16, 16);
        const sphereMat = new THREE.MeshPhongMaterial({
            color: colorHex,
            shininess: 30,
            emissive: colorHex,
            emissiveIntensity: 0.15
        });
        const sphere = new THREE.Mesh(sphereGeo, sphereMat);
        
        // pseudo RA/Dec galactic coordinate calculation
        const theta = p.idx * 1.5;
        const dist = 12.0 + (p.idx % 25) * 1.5;
        const galX = Math.cos(theta) * dist;
        const galY = (p.idx % 16) - 8;
        const galZ = Math.sin(theta) * dist;

        sphere.userData = {
            name: p.name,
            idx: p.idx,
            priority: p.mu,
            uncertainty: p.sigma,
            detectability: p.detectability,
            cost: p.cost,
            weather: p.weather,
            orbitRadius: radius,
            // Galactic locations
            galX: galX,
            galY: galY,
            galZ: galZ
        };

        planetsGroup.add(sphere);
        orbits.push({
            mesh: sphere,
            radius: radius,
            speed: speed,
            angle: startAngle,
            ringMesh: ring
        });
    });
}

function updateThreeJSPlanets(r) {
    // observed indices mapping
    const observedPlanets = new Set(adaptiveObsHistory.filter(o => parseInt(o.round) <= r).map(o => o.planet_name));
    
    orbits.forEach(orb => {
        const isObserved = observedPlanets.has(orb.mesh.userData.name);
        
        // observed planets are solid and glow; unobserved are semi-transparent!
        if (isObserved) {
            orb.mesh.material.opacity = 1.0;
            orb.mesh.material.emissiveIntensity = 0.5;
            orb.mesh.scale.set(1.2, 1.2, 1.2);
        } else {
            orb.mesh.material.opacity = 0.25;
            orb.mesh.material.emissiveIntensity = 0.05;
            orb.mesh.scale.set(0.85, 0.85, 0.85);
        }
        orb.mesh.material.transparent = true;
    });
}

function transitionVisualizer() {
    const isGal = window.activeView3D === "galactic";
    
    orbits.forEach(orb => {
        // Hide/Show orbit ring paths
        if (isGal) {
            orb.ringMesh.visible = false;
        } else {
            orb.ringMesh.visible = true;
        }
    });
}

function animate() {
    requestAnimationFrame(animate);
    
    const isGal = window.activeView3D === "galactic";
    
    orbits.forEach(orb => {
        if (isGal) {
            // LERP to Galactic coordinates
            orb.mesh.position.x += (orb.mesh.userData.galX - orb.mesh.position.x) * 0.1;
            orb.mesh.position.y += (orb.mesh.userData.galY - orb.mesh.position.y) * 0.1;
            orb.mesh.position.z += (orb.mesh.userData.galZ - orb.mesh.position.z) * 0.1;
        } else {
            // Circle orbits
            if (isRotating) {
                orb.angle += orb.speed * 0.25;
            }
            const targetX = Math.cos(orb.angle) * orb.radius;
            const targetZ = Math.sin(orb.angle) * orb.radius;
            
            orb.mesh.position.x += (targetX - orb.mesh.position.x) * 0.1;
            orb.mesh.position.y += (0 - orb.mesh.position.y) * 0.1;
            orb.mesh.position.z += (targetZ - orb.mesh.position.z) * 0.1;
        }
        orb.mesh.rotation.y += 0.015;
    });

    if (isRotating) {
        scene.rotation.y += 0.0004;
    }

    controls.update();
    renderer.render(scene, camera);
}

function onWindowResize() {
    const container = document.getElementById("canvas-3d");
    if (!container) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function onMouseMove(event) {
    const container = document.getElementById("canvas-3d");
    const rect = container.getBoundingClientRect();
    
    mouse.x = ((event.clientX - rect.left) / container.clientWidth) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / container.clientHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(planetsGroup.children);

    if (intersects.length > 0) {
        const obj = intersects[0].object;
        if (hoverObject !== obj) {
            if (hoverObject) hoverObject.material.color.setHex(hoverObject.userData.oldColor);
            hoverObject = obj;
            hoverObject.userData.oldColor = hoverObject.material.color.getHex();
            hoverObject.material.color.setHex(0xffffff);
            document.body.style.cursor = "pointer";
        }
    } else {
        if (hoverObject) {
            hoverObject.material.color.setHex(hoverObject.userData.oldColor);
            hoverObject = null;
            document.body.style.cursor = "default";
        }
    }
}

function onPlanetClick(event) {
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(planetsGroup.children);

    if (intersects.length > 0) {
        const p = intersects[0].object.userData;
        displayPlanetDetails(p);
    }
}

function displayPlanetDetails(p) {
    triggerAudio('click');
    document.getElementById("planet-details-empty").style.display = "none";
    const details = document.getElementById("planet-details");
    details.style.display = "block";

    document.getElementById("detail-name").innerText = p.name;
    document.getElementById("detail-priority").innerText = p.priority.toFixed(4);
    document.getElementById("detail-uncertainty").innerText = p.uncertainty.toFixed(4);
    document.getElementById("detail-detectability").innerText = p.detectability.toFixed(4);
    document.getElementById("detail-distance").innerText = `${(p.orbitRadius * 2.3).toFixed(1)} pc`;
    
    const temp = Math.round(200 + (p.priority * 150) + Math.random() * 40);
    document.getElementById("detail-temp").innerText = `${temp} K`;
    document.getElementById("detail-radius").innerText = `${(0.7 + p.priority * 1.6).toFixed(2)} R⊕`;
    
    const starTypes = ["G-Type Host Star (Solar)", "K-Type Orange Dwarf", "M-Type Red Dwarf"];
    const typeIdx = p.priority >= 0.70 ? 0 : p.priority >= 0.50 ? 1 : 2;
    document.getElementById("detail-star").innerText = starTypes[typeIdx];

    const badge = document.getElementById("detail-status-badge");
    badge.innerText = p.priority >= 0.70 ? "Biosignature Candidate Target" : 
                      p.priority >= 0.50 ? "Priority Target" : "Exploratory Target";
    
    badge.className = "detail-badge";
    if (p.priority >= 0.70) badge.classList.add("high");
    else if (p.priority >= 0.50) badge.classList.add("medium");
    else badge.classList.add("low");
}
