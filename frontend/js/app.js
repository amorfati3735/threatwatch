/**
 * Personal Safety Monitor Dashboard Controller.
 * Manages WebSocket telemetry, dual source switching (WESAD vs Arduino Live),
 * USB serial connection management, audio alarms, and live UI updates.
 */

// Web Audio API Alarm Synthesizer
class SoundEffects {
  constructor() {
    this.audioCtx = null;
    this.isMuted = false;
    this.lastAlarmTime = 0;
  }

  init() {
    if (!this.audioCtx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (AudioContext) this.audioCtx = new AudioContext();
    }
  }

  playBeep(freq = 880, type = "sine", duration = 0.15) {
    if (this.isMuted) return;
    this.init();
    if (!this.audioCtx) return;

    try {
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();

      osc.type = type;
      osc.frequency.setValueAtTime(freq, this.audioCtx.currentTime);

      gain.gain.setValueAtTime(0.2, this.audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.audioCtx.currentTime + duration);

      osc.connect(gain);
      gain.connect(this.audioCtx.destination);

      osc.start();
      osc.stop(this.audioCtx.currentTime + duration);
    } catch (e) {
      console.warn("Audio play error:", e);
    }
  }

  playThreatAlarm() {
    const now = Date.now();
    if (now - this.lastAlarmTime > 2000) {
      this.lastAlarmTime = now;
      this.playBeep(920, "sawtooth", 0.25);
      setTimeout(() => this.playBeep(1240, "sawtooth", 0.35), 250);
    }
  }
}

const sfx = new SoundEffects();

// Charts
let edaChart, respChart, accChart, threatChart;

// App State
let socket = null;
let activeSource = "WESAD_REPLAY";
let arduinoConnected = false;
let currentTheme = "dark";

document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  initTheme();
  connectWebSocket();
  fetchSubjects();
  fetchArduinoPorts();
  setupEventListeners();
});

function initCharts() {
  edaChart = new RealtimeWaveformChart("edaCanvas", 100, "#00f0ff", "EDA / PPG");
  respChart = new RealtimeWaveformChart("respCanvas", 100, "#ccff00", "Resp / Gyro");
  accChart = new RealtimeWaveformChart("accCanvas", 100, "#ffb700", "Acc Mag (g)");
  threatChart = new ThreatHistoryChart("threatCanvas", 120);
}

function initTheme() {
  document.documentElement.setAttribute("data-theme", currentTheme);
  const themeBtn = document.getElementById("themeToggleBtn");
  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      currentTheme = currentTheme === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", currentTheme);
      themeBtn.innerHTML = currentTheme === "dark" ? "☀️ Light" : "🌙 Dark";
    });
  }
}

function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  const statusBadge = document.getElementById("wsStatusBadge");

  socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    if (statusBadge) {
      statusBadge.textContent = "● Live";
      statusBadge.className = "badge-uni badge-neon";
    }
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "TELEMETRY") {
        updateDashboard(data);
      }
    } catch (e) {
      console.error("[WS] Parse error:", e);
    }
  };

  socket.onclose = () => {
    if (statusBadge) {
      statusBadge.textContent = "○ Disconnected";
      statusBadge.className = "badge-uni badge-alert";
    }
    setTimeout(connectWebSocket, 2000);
  };
}

function updateDashboard(data) {
  // 1. Video Feed
  const feed = document.getElementById("webcamFeed");
  if (feed && data.camera_frame) {
    feed.src = data.camera_frame;
  }

  // 2. Header Stats & Active Source
  setText("streamFps", `${data.fps} FPS`);
  setText("activeScenarioDisplay", data.demo_scenario || "Live Free");

  const currentSource = data.active_source_type || "WESAD_REPLAY";
  if (currentSource !== activeSource) {
    activeSource = currentSource;
    updateSourceUI();
  }

  // Arduino connection status badge
  const ardState = data.arduino || {};
  arduinoConnected = ardState.connected || false;
  const ardBadge = document.getElementById("arduinoStatusBadge");
  const btnConnect = document.getElementById("btnArduinoConnect");

  if (ardBadge && btnConnect) {
    if (arduinoConnected) {
      ardBadge.textContent = `Arduino: Connected (${ardState.port || 'USB'})`;
      ardBadge.className = "badge-uni badge-neon";
      btnConnect.textContent = "Disconnect";
      btnConnect.className = "btn-uni btn-dark text-xs py-1 px-2.5 border border-uni-alert text-uni-alert";
    } else {
      ardBadge.textContent = ardState.error ? `Error: ${ardState.error.slice(0, 15)}` : "Hardware: Standby";
      ardBadge.className = "badge-uni text-uni-muted";
      btnConnect.textContent = "Connect";
      btnConnect.className = "btn-uni btn-dark text-xs py-1 px-2.5 border border-uni-border hover:border-uni-neon";
    }
  }

  // 3. Raw Sensor Indicators
  const sp = data.sensor_packet || {};
  setText("valHR", sp.heart_rate ? `${Math.round(sp.heart_rate)} BPM` : "-- BPM");
  setText("valAcc", sp.acc_mag !== undefined ? `${sp.acc_mag.toFixed(2)} g` : "--");

  if (activeSource === "ARDUINO_LIVE") {
    setText("sensorCardTitle", "Wearable Sensors (Arduino Live)");
    setText("groundTruthLabel", "Hardware: MAX30102 + MPU6050");
    setText("lblSlot2", "SpO2 Level");
    setText("valSlot2", sp.spo2 !== undefined && sp.spo2 > 0 ? `${sp.spo2.toFixed(1)}%` : "--");
    setText("lblSlot3", "Gyro Rotation");
    setText("valSlot3", sp.gyro_mag !== undefined ? `${Math.round(sp.gyro_mag)} °/s` : "--");
    setText("lblSlot4", "Finger Sensor");
    setText("valSlot4", sp.finger_detected ? "Contact OK" : "No Finger");

    setText("chartLabel1", "Pulse / HR Tracking");
    setText("chartLabel2", "Gyroscope Energy (°/s)");
    setText("chartLabel3", "Accelerometer Magnitude (g)");

    if (sp.heart_rate) edaChart.push(sp.heart_rate);
    if (sp.gyro_mag !== undefined) respChart.push(sp.gyro_mag);
    if (sp.acc_mag !== undefined) accChart.push(sp.acc_mag);
  } else {
    setText("sensorCardTitle", "Wearable Sensors (WESAD Stream)");
    setText("groundTruthLabel", `Subject: ${data.replay?.subject_id || 'S2'}`);
    setText("lblSlot2", "EDA (Sweat)");
    setText("valSlot2", sp.eda !== undefined ? `${sp.eda.toFixed(2)} µS` : "--");
    setText("lblSlot3", "Respiration");
    setText("valSlot3", sp.respiration !== undefined ? `${sp.respiration.toFixed(2)}` : "--");
    setText("lblSlot4", "Skin Temp");
    setText("valSlot4", sp.temperature !== undefined ? `${sp.temperature.toFixed(1)} °C` : "--");

    setText("chartLabel1", "Electrodermal Activity (EDA)");
    setText("chartLabel2", "Respiration Waveform");
    setText("chartLabel3", "Acceleration Magnitude (g)");

    if (sp.eda !== undefined) edaChart.push(sp.eda);
    if (sp.respiration !== undefined) respChart.push(sp.respiration);
    if (sp.acc_mag !== undefined) accChart.push(sp.acc_mag);
  }

  // 4. Sensor ML Breakdown
  const sm = data.sensor_ml || {};
  const probs = sm.probabilities || { baseline: 0, stress: 0, amusement: 0 };
  setText("sensorDistressScore", `${Math.round((sm.distress_score || 0) * 100)}%`);

  if (activeSource === "ARDUINO_LIVE") {
    setText("modelCardTitle", "Hardware Distress Inference (MAX30102+MPU6050)");
    setText("lblProb1", "Normal Baseline");
    setText("lblProb2", "Tachycardia / Elevated HR");
    setText("lblProb3", "Impact Shock / Struggle Motion");
  } else {
    setText("modelCardTitle", "Sensor Sequence Model (LSTM)");
    setText("lblProb1", "Baseline State (Calm)");
    setText("lblProb2", "Stress / Distress State");
    setText("lblProb3", "Amusement State");
  }

  setText("probBaseline", `${Math.round((probs.baseline || 0) * 100)}%`);
  setText("probStress", `${Math.round((probs.stress || 0) * 100)}%`);
  setText("probAmusement", `${Math.round((probs.amusement || 0) * 100)}%`);
  setBar("barBaseline", (probs.baseline || 0) * 100);
  setBar("barStress", (probs.stress || 0) * 100);
  setBar("barAmusement", (probs.amusement || 0) * 100);
  setText("sensorLatency", `${sm.latency_ms || 0} ms`);

  // 5. Vision ML Breakdown
  const vm = data.vision_ml || {};
  setText("visionDistressScore", `${Math.round((vm.distress_score || 0) * 100)}%`);
  setBar("visionDistressBar", (vm.distress_score || 0) * 100);
  setText("visionPosture", vm.posture || "Unknown");
  setText("visionExpression", vm.facial_expression || "Neutral");
  setText("visionMovement", vm.movement_level || "Normal");

  // 6. Multimodal Threat Score & State Machine
  const fusion = data.fusion || {};
  const threatPct = fusion.threat_score_pct || 0;
  setText("threatScoreText", `${threatPct.toFixed(1)}%`);
  setBar("threatMeterFill", threatPct, fusion.verdict_color);

  threatChart.push(fusion.threat_score || 0);

  // State Banner & Styling
  const smState = data.state_machine || {};
  const currState = smState.current_state || fusion.verdict || "SAFE";
  const stateBadge = document.getElementById("stateBadge");

  if (stateBadge) {
    stateBadge.textContent = currState;
    if (currState.includes("THREAT") || currState.includes("ALERT")) {
      stateBadge.className = "btn-uni btn-alert text-sm px-4 py-2 font-display pulse-threat";
      sfx.playThreatAlarm();
    } else if (currState.includes("CAUTION") || currState.includes("SUSPICIOUS")) {
      stateBadge.className = "btn-uni bg-uni-dark text-uni-cyan border-uni-cyan text-sm px-4 py-2 font-display";
    } else {
      stateBadge.className = "btn-uni btn-neon text-sm px-4 py-2 font-display";
    }
  }

  setText("fusionConsensus", fusion.agreement_status || "Normal / Congruent");
  setText("sensorWeightDisp", `Sensor (${Math.round((fusion.sensor_weight || 0.6) * 100)}%)`);
  setText("visionWeightDisp", `Vision (${Math.round((fusion.vision_weight || 0.4) * 100)}%)`);

  // 7. Replay Telemetry
  const rep = data.replay || {};
  setText("elapsedTime", `${rep.elapsed_seconds || 0}s / ${rep.total_seconds || 0}s`);
  const progressBar = document.getElementById("replayProgressBar");
  if (progressBar && rep.progress_fraction !== undefined) {
    progressBar.style.width = `${rep.progress_fraction * 100}%`;
  }

  // 8. Event Log Updates
  if (data.recent_logs && data.recent_logs.length > 0) {
    renderEventLog(data.recent_logs);
  }
}

function updateSourceUI() {
  const btnWesad = document.getElementById("btnSourceWesad");
  const btnArd = document.getElementById("btnSourceArduino");

  if (btnWesad && btnArd) {
    if (activeSource === "ARDUINO_LIVE") {
      btnArd.className = "btn-uni btn-neon text-xs py-1 px-3";
      btnWesad.className = "btn-uni btn-dark text-xs py-1 px-3";
    } else {
      btnWesad.className = "btn-uni btn-neon text-xs py-1 px-3";
      btnArd.className = "btn-uni btn-dark text-xs py-1 px-3";
    }
  }
}

function renderEventLog(logs) {
  const container = document.getElementById("eventLogContainer");
  if (!container) return;

  const html = logs.map(l => {
    let colorClass = "text-uni-muted";
    if (l.level === "CRITICAL" || l.level === "EMERGENCY") colorClass = "text-uni-alert font-bold";
    else if (l.level === "WARNING") colorClass = "text-uni-cyan";
    else if (l.category === "HARDWARE" || l.category === "SOURCE") colorClass = "text-uni-neon";

    return `
      <div class="flex items-start gap-2 py-1 border-b border-uni-border/40 text-xs font-mono">
        <span class="text-uni-muted shrink-0">[${l.timestamp_str}]</span>
        <span class="badge-uni shrink-0">${l.category}</span>
        <span class="${colorClass}">${l.message}</span>
      </div>
    `;
  }).join("");

  container.innerHTML = html;
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function setBar(id, pct, color = null) {
  const el = document.getElementById(id);
  if (el) {
    el.style.width = `${Math.min(100, Math.max(0, pct))}%`;
    if (color) el.style.backgroundColor = color;
  }
}

async function fetchSubjects() {
  try {
    const res = await fetch("/api/replay/subjects");
    const data = await res.json();
    const select = document.getElementById("subjectSelect");
    if (select && data.subjects) {
      select.innerHTML = data.subjects.map(s => `<option value="${s}">${s}</option>`).join("");
    }
  } catch (e) {
    console.warn("Could not fetch subjects:", e);
  }
}

async function fetchArduinoPorts() {
  try {
    const res = await fetch("/api/arduino/ports");
    const data = await res.json();
    const select = document.getElementById("arduinoPortSelect");
    if (select && data.ports) {
      if (data.ports.length === 0) {
        select.innerHTML = `<option value="COM3">COM3 (Manual)</option>`;
      } else {
        select.innerHTML = data.ports.map(p => `<option value="${p.port}">${p.port} (${p.description})</option>`).join("");
      }
    }
  } catch (e) {
    console.warn("Could not fetch ports:", e);
  }
}

function setupEventListeners() {
  // Source Selection Tabs
  document.getElementById("btnSourceWesad")?.addEventListener("click", () => switchSource("WESAD_REPLAY"));
  document.getElementById("btnSourceArduino")?.addEventListener("click", () => switchSource("ARDUINO_LIVE"));

  // Arduino Connect / Disconnect
  document.getElementById("btnArduinoConnect")?.addEventListener("click", async () => {
    const port = document.getElementById("arduinoPortSelect")?.value || "COM3";
    const action = arduinoConnected ? "disconnect" : "connect";
    try {
      const res = await fetch("/api/arduino/connect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, port }),
      });
      const data = await res.json();
      if (!res.ok) alert(data.detail || "Connection failed");
    } catch (e) {
      alert("Error connecting to Arduino: " + e.message);
    }
  });

  // Replay Controls
  document.getElementById("btnPlay")?.addEventListener("click", () => sendReplayControl({ action: "play" }));
  document.getElementById("btnPause")?.addEventListener("click", () => sendReplayControl({ action: "pause" }));
  document.getElementById("btnReset")?.addEventListener("click", () => sendReplayControl({ action: "reset" }));

  // Preset Segment Jumps
  document.getElementById("jumpBaseline")?.addEventListener("click", () => sendReplayControl({ action: "scenario", scenario: "BASELINE" }));
  document.getElementById("jumpStress")?.addEventListener("click", () => sendReplayControl({ action: "scenario", scenario: "STRESS" }));
  document.getElementById("jumpAmusement")?.addEventListener("click", () => sendReplayControl({ action: "scenario", scenario: "AMUSEMENT" }));

  // Speed buttons
  document.querySelectorAll("[data-speed]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const speed = parseFloat(e.currentTarget.getAttribute("data-speed"));
      sendReplayControl({ action: "speed", speed });
    });
  });

  // Subject Selector
  document.getElementById("subjectSelect")?.addEventListener("change", (e) => {
    sendReplayControl({ action: "subject", subject_id: e.target.value });
  });

  // Timeline scrubber
  const scrubber = document.getElementById("timelineScrubber");
  if (scrubber) {
    scrubber.addEventListener("click", (e) => {
      const rect = scrubber.getBoundingClientRect();
      const fraction = (e.clientX - rect.left) / rect.width;
      sendReplayControl({ action: "seek", seek_fraction: fraction });
    });
  }

  // Demo Scenarios
  document.querySelectorAll("[data-scenario]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const scenario = e.currentTarget.getAttribute("data-scenario");
      triggerScenario(scenario);
    });
  });

  // Mute Alarm
  const muteBtn = document.getElementById("muteBtn");
  if (muteBtn) {
    muteBtn.addEventListener("click", () => {
      sfx.isMuted = !sfx.isMuted;
      muteBtn.textContent = sfx.isMuted ? "🔇 Alarm Off" : "🔊 Alarm On";
      muteBtn.className = sfx.isMuted ? "btn-uni btn-alert text-xs py-1 px-2.5" : "btn-uni btn-dark text-xs py-1 px-2.5";
    });
  }

  // Reset Alarm
  document.getElementById("btnAlertReset")?.addEventListener("click", async () => {
    try {
      await fetch("/api/alert/reset", { method: "POST" });
    } catch (e) {
      console.error("Alert reset error:", e);
    }
  });

  // Fusion Weight Slider
  const slider = document.getElementById("sensorWeightSlider");
  if (slider) {
    slider.addEventListener("input", (e) => {
      const sW = parseFloat(e.target.value);
      const vW = Math.round((1.0 - sW) * 100) / 100;
      setText("sensorWeightLabel", `${Math.round(sW * 100)}%`);
      setText("visionWeightLabel", `${Math.round(vW * 100)}%`);
      sendWeightConfig(sW, vW);
    });
  }
}

async function switchSource(sourceType) {
  try {
    const res = await fetch("/api/source/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_type: sourceType }),
    });
    if (res.ok) {
      activeSource = sourceType;
      updateSourceUI();
    }
  } catch (e) {
    console.error("Failed to switch source:", e);
  }
}

async function sendReplayControl(payload) {
  try {
    await fetch("/api/replay/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (e) {
    console.error("Replay control error:", e);
  }
}

async function triggerScenario(scenario) {
  try {
    await fetch("/api/demo/scenario", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario }),
    });
  } catch (e) {
    console.error("Demo trigger error:", e);
  }
}

async function sendWeightConfig(sW, vW) {
  try {
    await fetch("/api/fusion/weights", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sensor_weight: sW, vision_weight: vW }),
    });
  } catch (e) {
    console.error("Weight config error:", e);
  }
}
