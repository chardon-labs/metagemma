const endpoint = document.querySelector("#endpoint");
const promptInput = document.querySelector("#prompt");
const maxTokens = document.querySelector("#maxTokens");
const temperature = document.querySelector("#temperature");
const topP = document.querySelector("#topP");
const thinking = document.querySelector("#thinking");
const runButton = document.querySelector("#run");
const stopButton = document.querySelector("#stop");
const statusText = document.querySelector("#status");
const tokenCount = document.querySelector("#tokenCount");
const finalConfidence = document.querySelector("#finalConfidence");
const meanConfidence = document.querySelector("#meanConfidence");
const completion = document.querySelector("#completion");
const chart = document.querySelector("#chart");
const ctx = chart.getContext("2d");
const traceCount = 16;
const traceColors = [
  "#1f7a8c",
  "#b7410e",
  "#3d5a80",
  "#6a994e",
  "#8f2d56",
  "#bc6c25",
  "#4361ee",
  "#2a9d8f",
  "#7b2cbf",
  "#d62828",
  "#457b9d",
  "#5f0f40",
  "#588157",
  "#e76f51",
  "#264653",
  "#9a031e",
];

let controller = null;
let traces = [];
let finalResults = [];

function setStatus(text, isError = false) {
  statusText.textContent = text;
  statusText.classList.toggle("error", isError);
}

function formatConfidence(value) {
  return value === null || value === undefined ? "-" : value.toFixed(3);
}

function resetView() {
  traces = Array.from({ length: traceCount }, () => []);
  finalResults = [];
  completion.textContent = "";
  tokenCount.textContent = `0 tokens across ${traceCount} traces`;
  finalConfidence.textContent = "-";
  meanConfidence.textContent = "-";
  drawChart();
}

function meanConfidenceFor(values) {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function allConfidences() {
  return traces.flat();
}

function resizeCanvas() {
  const rect = chart.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  chart.width = Math.max(1, Math.floor(rect.width * scale));
  chart.height = Math.max(1, Math.floor(rect.height * scale));
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  drawChart();
}

function drawChart() {
  const width = chart.clientWidth;
  const height = chart.clientHeight;
  ctx.clearRect(0, 0, width, height);

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const padding = { left: 42, right: 16, top: 18, bottom: 28 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  ctx.strokeStyle = "#d8dee8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i += 1) {
    const y = padding.top + (plotHeight * i) / 4;
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
  }
  ctx.stroke();

  ctx.fillStyle = "#697386";
  ctx.font = "12px system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i += 1) {
    const value = 1 - i / 4;
    const y = padding.top + (plotHeight * i) / 4;
    ctx.fillText(value.toFixed(2), padding.left - 8, y);
  }

  ctx.strokeStyle = "#17202a";
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.stroke();

  const visibleTraces = traces.filter((values) => values.length > 0);
  if (visibleTraces.length === 0) {
    ctx.fillStyle = "#697386";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`Waiting for ${traceCount} streamed traces`, width / 2, height / 2);
    return;
  }

  const yFor = (value) => padding.top + plotHeight * (1 - value);
  const maxLength = Math.max(...visibleTraces.map((values) => values.length));
  const xFor = (index) => {
    if (maxLength === 1) return padding.left;
    return padding.left + (plotWidth * index) / (maxLength - 1);
  };

  traces.forEach((values, traceIndex) => {
    if (values.length === 0) return;
    ctx.strokeStyle = traceColors[traceIndex % traceColors.length];
    ctx.lineWidth = 1.6;
    ctx.globalAlpha = 0.78;
    ctx.beginPath();
    values.forEach((value, tokenIndex) => {
      const x = xFor(tokenIndex);
      const y = yFor(value);
      if (tokenIndex === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  ctx.globalAlpha = 1;
}

function updateStats(finalValue = null, meanValue = null) {
  const values = allConfidences();
  tokenCount.textContent = `${values.length} tokens across ${traceCount} traces`;
  if (values.length > 0) {
    finalConfidence.textContent = formatConfidence(finalValue ?? values[values.length - 1]);
    const mean = meanValue ?? meanConfidenceFor(values);
    meanConfidence.textContent = formatConfidence(mean);
  }
}

function resultMean(result) {
  const summaryMean = result.confidence_summary?.mean;
  if (summaryMean !== null && summaryMean !== undefined) return summaryMean;
  return meanConfidenceFor(result.token_confidences) ?? -Infinity;
}

function renderSelectedOutputs() {
  const completed = finalResults.filter((result) => result !== undefined);
  if (completed.length === 0) return;

  const ranked = [...completed].sort((left, right) => resultMean(right) - resultMean(left));
  const highest = ranked[0];
  const lowest = ranked[ranked.length - 1];
  completion.textContent = [
    `Highest average confidence: trace ${highest.index} (mean ${formatConfidence(resultMean(highest))})`,
    highest.completion || "[empty completion]",
    "",
    `Lowest average confidence: trace ${lowest.index} (mean ${formatConfidence(resultMean(lowest))})`,
    lowest.completion || "[empty completion]",
  ].join("\n");
}

async function runStream() {
  resetView();
  setStatus("Streaming");
  runButton.disabled = true;
  stopButton.disabled = false;
  controller = new AbortController();

  const payload = {
    messages: [{ role: "user", content: promptInput.value }],
    max_new_tokens: Number(maxTokens.value),
    temperature: Number(temperature.value),
    top_p: Number(topP.value),
    enable_thinking: thinking.checked,
    n: traceCount,
  };

  try {
    const response = await fetch(endpoint.value, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!response.ok || !response.body) {
      throw new Error(`HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "token") {
          if (!traces[event.index]) traces[event.index] = [];
          traces[event.index].push(event.confidence);
          updateStats();
          drawChart();
        } else if (event.type === "final") {
          if (!traces[event.index]) traces[event.index] = [];
          finalResults[event.index] = event;
          traces[event.index] = event.token_confidences;
          updateStats();
          renderSelectedOutputs();
          drawChart();
        } else if (event.type === "batch_final") {
          finalResults = event.completions;
          traces = event.completions.map((result) => result.token_confidences);
          updateStats();
          renderSelectedOutputs();
          drawChart();
          setStatus(`Done: ${event.completions.length} traces`);
        }
      }
    }

    if (buffer.trim()) {
      const event = JSON.parse(buffer);
      if (event.type === "batch_final") setStatus(`Done: ${event.completions.length} traces`);
    }
  } catch (error) {
    if (error.name === "AbortError") {
      setStatus("Stopped");
    } else {
      setStatus(error.message, true);
    }
  } finally {
    controller = null;
    runButton.disabled = false;
    stopButton.disabled = true;
  }
}

runButton.addEventListener("click", runStream);
stopButton.addEventListener("click", () => {
  if (controller) controller.abort();
});
window.addEventListener("resize", resizeCanvas);
resizeCanvas();
