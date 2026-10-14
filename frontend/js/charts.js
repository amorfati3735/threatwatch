/**
 * High-Performance Native Canvas Waveform & Metric Visualizers.
 * Renders real-time continuous sensor traces and threat gauge with UniArchives palette.
 */

class RealtimeWaveformChart {
  constructor(canvasId, maxPoints = 120, lineColor = "#00f0ff", yLabel = "") {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas ? this.canvas.getContext("2d") : null;
    this.maxPoints = maxPoints;
    this.lineColor = lineColor;
    this.yLabel = yLabel;
    this.data = [];
    this.minVal = Infinity;
    this.maxVal = -Infinity;
  }

  push(value) {
    if (value === null || value === undefined || isNaN(value)) return;
    this.data.push(value);
    if (this.data.length > this.maxPoints) {
      this.data.shift();
    }
    this.render();
  }

  render() {
    if (!this.ctx || this.data.length < 2) return;
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    // Clear background
    ctx.clearRect(0, 0, w, h);

    // Dynamic min/max with smoothing
    const currentMin = Math.min(...this.data);
    const currentMax = Math.max(...this.data);
    const padding = Math.max((currentMax - currentMin) * 0.15, 0.05);
    const yMin = currentMin - padding;
    const yMax = currentMax + padding;
    const range = Math.max(yMax - yMin, 0.001);

    // Draw Subtle Grid
    ctx.strokeStyle = "rgba(42, 42, 42, 0.6)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 4]);

    for (let y = 15; y < h; y += 25) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Draw Signal Path
    ctx.beginPath();
    ctx.strokeStyle = this.lineColor;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";

    const stepX = w / (this.maxPoints - 1);
    const startOffset = (this.maxPoints - this.data.length) * stepX;

    for (let i = 0; i < this.data.length; i++) {
      const x = startOffset + i * stepX;
      const y = h - ((this.data[i] - yMin) / range) * (h - 20) - 10;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Fill under curve
    ctx.lineTo(w, h);
    ctx.lineTo(startOffset, h);
    ctx.closePath();
    ctx.fillStyle = this.lineColor === "#ccff00" 
      ? "rgba(204, 255, 0, 0.06)" 
      : (this.lineColor === "#ff003c" ? "rgba(255, 0, 60, 0.08)" : "rgba(0, 240, 255, 0.06)");
    ctx.fill();

    // Draw Latest Point Pulse
    if (this.data.length > 0) {
      const lastVal = this.data[this.data.length - 1];
      const lastX = w;
      const lastY = h - ((lastVal - yMin) / range) * (h - 20) - 10;

      ctx.beginPath();
      ctx.arc(lastX - 2, lastY, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.strokeStyle = this.lineColor;
      ctx.stroke();
    }
  }
}

class ThreatHistoryChart {
  constructor(canvasId, maxPoints = 120) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas ? this.canvas.getContext("2d") : null;
    this.maxPoints = maxPoints;
    this.data = [];
  }

  push(score) {
    if (score === null || isNaN(score)) return;
    this.data.push(score);
    if (this.data.length > this.maxPoints) {
      this.data.shift();
    }
    this.render();
  }

  render() {
    if (!this.ctx || this.data.length < 2) return;
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Draw Safety Threshold Bands
    const ySafe = h - 0.40 * h;
    const yThreat = h - 0.70 * h;

    // Green SAFE Zone
    ctx.fillStyle = "rgba(204, 255, 0, 0.03)";
    ctx.fillRect(0, ySafe, w, h - ySafe);

    // Cyan CAUTION Zone
    ctx.fillStyle = "rgba(0, 240, 255, 0.04)";
    ctx.fillRect(0, yThreat, w, ySafe - yThreat);

    // Red THREAT Zone
    ctx.fillStyle = "rgba(255, 0, 60, 0.07)";
    ctx.fillRect(0, 0, w, yThreat);

    // Threshold lines
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);

    // 0.40 CAUTION line
    ctx.strokeStyle = "rgba(0, 240, 255, 0.5)";
    ctx.beginPath();
    ctx.moveTo(0, ySafe);
    ctx.lineTo(w, ySafe);
    ctx.stroke();

    // 0.70 THREAT line
    ctx.strokeStyle = "rgba(255, 0, 60, 0.6)";
    ctx.beginPath();
    ctx.moveTo(0, yThreat);
    ctx.lineTo(w, yThreat);
    ctx.stroke();

    ctx.setLineDash([]);

    // Draw Threshold Text Labels
    ctx.font = "9px 'JetBrains Mono'";
    ctx.fillStyle = "rgba(255, 0, 60, 0.8)";
    ctx.fillText("0.70 THREAT", 6, yThreat - 4);

    ctx.fillStyle = "rgba(0, 240, 255, 0.7)";
    ctx.fillText("0.40 CAUTION", 6, ySafe - 4);

    ctx.fillStyle = "rgba(204, 255, 0, 0.7)";
    ctx.fillText("0.00 SAFE", 6, h - 4);

    // Draw Fused Threat Path
    ctx.beginPath();
    ctx.lineWidth = 2.5;
    ctx.lineJoin = "round";

    const stepX = w / (this.maxPoints - 1);
    const startOffset = (this.maxPoints - this.data.length) * stepX;

    for (let i = 0; i < this.data.length; i++) {
      const score = Math.max(0, Math.min(1.0, this.data[i]));
      const x = startOffset + i * stepX;
      const y = h - score * (h - 8) - 4;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }

    // Color gradient based on current score
    const currScore = this.data[this.data.length - 1];
    ctx.strokeStyle = currScore < 0.40 ? "#ccff00" : (currScore < 0.70 ? "#00f0ff" : "#ff003c");
    ctx.stroke();
  }
}
