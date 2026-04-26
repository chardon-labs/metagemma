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

let controller = null;
let confidences = [];

function setStatus(text, isError = false) {
  statusText.textContent = text;
  statusText.classList.toggle("error", isError);
}

function formatConfidence(value) {
  return value === null || value === undefined ? "-" : value.toFixed(3);
}

function resetView() {
  confidences = [];
  completion.textContent = "";
  tokenCount.textContent = "0 tokens";
  finalConfidence.textContent = "-";
  meanConfidence.textContent = "-";
  drawChart();
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

  if (confidences.length === 0) {
    ctx.fillStyle = "#697386";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("Waiting for streamed tokens", width / 2, height / 2);
    return;
  }

  const xFor = (index) => {
    if (confidences.length === 1) return padding.left;
    return padding.left + (plotWidth * index) / (confidences.length - 1);
  };
  const yFor = (value) => padding.top + plotHeight * (1 - value);

  ctx.strokeStyle = "#1f7a8c";
  ctx.lineWidth = 2;
  ctx.beginPath();
  confidences.forEach((value, index) => {
    const x = xFor(index);
    const y = yFor(value);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "#13505b";
  confidences.forEach((value, index) => {
    const x = xFor(index);
    const y = yFor(value);
    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, Math.PI * 2);
    ctx.fill();
  });
}

function updateStats(finalValue = null, meanValue = null) {
  tokenCount.textContent = `${confidences.length} token${confidences.length === 1 ? "" : "s"}`;
  if (confidences.length > 0) {
    finalConfidence.textContent = formatConfidence(finalValue ?? confidences[confidences.length - 1]);
    const mean = meanValue ?? confidences.reduce((sum, value) => sum + value, 0) / confidences.length;
    meanConfidence.textContent = formatConfidence(mean);
  }
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
          completion.textContent += event.text;
          confidences.push(event.confidence);
          updateStats();
          drawChart();
        } else if (event.type === "final") {
          completion.textContent = event.completion;
          confidences = event.token_confidences;
          updateStats(event.confidence, event.confidence_summary.mean);
          drawChart();
          setStatus(`Done: ${event.finish_reason}`);
        }
      }
    }

    if (buffer.trim()) {
      const event = JSON.parse(buffer);
      if (event.type === "final") setStatus(`Done: ${event.finish_reason}`);
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
