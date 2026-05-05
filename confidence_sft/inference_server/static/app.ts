type NullableNumber = number | null;

type UPlotData = [number[], ...NullableNumber[][]];

type UPlotScaleKey = "x" | "y";

type UPlotSeries = {
  label?: string;
  scale?: UPlotScaleKey;
  stroke?: string;
  width?: number;
  points?: { show: boolean };
};

type UPlotOptions = {
  width: number;
  height: number;
  cursor?: {
    drag?: { x: boolean; y: boolean };
    points?: { show: boolean };
  };
  legend?: { show: boolean };
  scales?: Record<UPlotScaleKey, { time?: boolean; range?: [number, number] }>;
  axes?: Array<{ label?: string; values?: (plot: UPlotInstance, ticks: number[]) => string[] }>;
  series: UPlotSeries[];
};

type UPlotInstance = {
  root: HTMLElement;
  setData: (data: UPlotData, resetScales?: boolean) => void;
  setSize: (size: { width: number; height: number }) => void;
  setSeries: (index: number, options: UPlotSeries, fireHook?: boolean) => void;
  posToVal: (position: number, scale: UPlotScaleKey) => number;
  valToPos: (value: number, scale: UPlotScaleKey) => number;
};

type UPlotConstructor = new (options: UPlotOptions, data: UPlotData, root: HTMLElement) => UPlotInstance;

declare const uPlot: UPlotConstructor;

type StreamTokenEvent = {
  type: "token";
  index: number;
  token_id: number;
  text: string;
  confidence: number;
  position: number;
};

type ConfidenceSummary = {
  final: number | null;
  mean: number | null;
  tail10_mean: number | null;
};

type StreamFinalEvent = {
  type: "final";
  index: number;
  completion: string;
  confidence: number | null;
  token_confidences: number[];
  token_positions: number[];
  token_ids: number[];
  confidence_summary: ConfidenceSummary;
  finish_reason: string;
};

type StreamBatchFinalEvent = {
  type: "batch_final";
  completions: StreamFinalEvent[];
};

type StreamEvent = StreamTokenEvent | StreamFinalEvent | StreamBatchFinalEvent;

type ConfidenceStats = {
  final: number | null;
  mean: number | null;
  min: number | null;
  max: number | null;
};

const endpoint = query<HTMLInputElement>("#endpoint");
const promptInput = query<HTMLTextAreaElement>("#prompt");
const maxTokens = query<HTMLInputElement>("#maxTokens");
const temperature = query<HTMLInputElement>("#temperature");
const topP = query<HTMLInputElement>("#topP");
const thinking = query<HTMLInputElement>("#thinking");
const runButton = query<HTMLButtonElement>("#run");
const stopButton = query<HTMLButtonElement>("#stop");
const statusText = query<HTMLElement>("#status");
const tokenCount = query<HTMLElement>("#tokenCount");
const finalConfidence = query<HTMLElement>("#finalConfidence");
const meanConfidence = query<HTMLElement>("#meanConfidence");
const completion = query<HTMLElement>("#completion");
const traceSelector = query<HTMLElement>("#traceSelector");
const selectedTrace = query<HTMLElement>("#selectedTrace");
const selectedTokens = query<HTMLElement>("#selectedTokens");
const selectedMean = query<HTMLElement>("#selectedMean");
const selectedMin = query<HTMLElement>("#selectedMin");
const selectedMax = query<HTMLElement>("#selectedMax");
const chartRoot = query<HTMLElement>("#chart");
const positionChartRoot = query<HTMLElement>("#positionChart");
const traceCount = 16;
const traceColor = "rgba(31, 122, 140, 0.22)";
const highlightColor = "#0f3f46";
const positionTraceColor = "rgba(181, 95, 31, 0.22)";
const positionHighlightColor = "#8a3f10";
const renderIntervalMs = 100;
const maxPlotPoints = 700;

let controller: AbortController | null = null;
let traces: number[][] = emptyTraceNumbers();
let positionTraces: number[][] = emptyTraceNumbers();
let tokenTexts: string[][] = emptyTraceTexts();
let finalResults: Array<StreamFinalEvent | undefined> = [];
let selectedTraceIndex = 0;
let plot: UPlotInstance | null = null;
let positionPlot: UPlotInstance | null = null;
let renderTimer: number | null = null;
let lastRenderTime = 0;
let renderedTraceIndex = -1;
let renderedTokenCount = 0;

function query<T extends HTMLElement>(selector: string): T {
  const element = document.querySelector(selector);
  if (!(element instanceof HTMLElement)) {
    throw new Error(`Missing element: ${selector}`);
  }
  return element as T;
}

function emptyTraceNumbers(): number[][] {
  return Array.from({ length: traceCount }, () => []);
}

function emptyTraceTexts(): string[][] {
  return Array.from({ length: traceCount }, () => []);
}

function authToken(): string | null {
  const params = new URLSearchParams(window.location.search);
  const queryToken = params.get("token");
  if (queryToken) {
    window.localStorage.setItem("inference_auth_token", queryToken);
    return queryToken;
  }
  return window.localStorage.getItem("inference_auth_token");
}

function setStatus(text: string, isError = false): void {
  statusText.textContent = text;
  statusText.classList.toggle("error", isError);
}

function formatConfidence(value: number | null | undefined): string {
  return value === null || value === undefined ? "-" : value.toFixed(3);
}

function meanConfidenceFor(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function totalTokenCount(): number {
  return traces.reduce((sum, values) => sum + values.length, 0);
}

function confidenceStats(values: number[]): ConfidenceStats {
  if (values.length === 0) {
    return { final: null, mean: null, min: null, max: null };
  }
  return {
    final: values[values.length - 1] ?? null,
    mean: meanConfidenceFor(values),
    min: Math.min(...values),
    max: Math.max(...values),
  };
}

function confidenceColor(value: number): string {
  const clamped = Math.max(0, Math.min(1, value));
  const hue = clamped * 120;
  return `hsla(${hue}, 65%, 48%, 0.28)`;
}

function chartData(sourceTraces: number[][]): UPlotData {
  const maxLength = Math.max(1, ...sourceTraces.map((values) => values.length));
  const pointCount = Math.min(maxLength, maxPlotPoints);
  const stride = Math.max(1, Math.ceil(maxLength / maxPlotPoints));
  const x = Array.from({ length: pointCount }, (_, index) => Math.min(index * stride, maxLength - 1));
  const ySeries = plotTraceOrder().map((traceIndex) =>
    x.map((tokenIndex) => smoothedValueAt(sourceTraces[traceIndex] ?? [], tokenIndex)),
  );
  return [x, ...ySeries];
}

function plotTraceOrder(): number[] {
  const order = Array.from({ length: traceCount }, (_, index) => index).filter((index) => index !== selectedTraceIndex);
  order.push(selectedTraceIndex);
  return order;
}

function smoothedValueAt(values: number[], index: number): NullableNumber {
  const value = values[index];
  if (value === undefined) return null;
  const previous = values[index - 1] ?? value;
  const next = values[index + 1] ?? value;
  return previous * 0.2 + value * 0.6 + next * 0.2;
}

function chartOptions(
  root: HTMLElement,
  sourceTraces: number[][],
  yLabel: string,
  selectedColor: string,
  idleColor: string,
): UPlotOptions {
  const rect = root.getBoundingClientRect();
  return {
    width: Math.max(1, Math.floor(rect.width)),
    height: Math.max(1, Math.floor(rect.height)),
    cursor: {
      drag: { x: false, y: false },
      points: { show: false },
    },
    legend: { show: false },
    scales: {
      x: { time: false },
      y: { time: false, range: [0, 1] },
    },
    axes: [
      {
        values: (_plot, ticks) => ticks.map((tick) => String(Math.round(tick))),
      },
      {
        label: yLabel,
        values: (_plot, ticks) => ticks.map((tick) => tick.toFixed(2)),
      },
    ],
    series: [
      {},
      ...plotTraceOrder().map((traceIndex) => traceSeries(traceIndex, selectedColor, idleColor)),
    ],
  };
}

function traceSeries(index: number, selectedColor: string, idleColor: string): UPlotSeries {
  return {
    label: `Trace ${index}`,
    scale: "y",
    stroke: index === selectedTraceIndex ? selectedColor : idleColor,
    width: index === selectedTraceIndex ? 3 : 1.4,
    points: { show: false },
  };
}

function initPlot(): void {
  plot = new uPlot(chartOptions(chartRoot, traces, "Confidence", highlightColor, traceColor), chartData(traces), chartRoot);
  const overlay = plot.root.querySelector(".u-over");
  if (overlay instanceof HTMLElement) {
    overlay.addEventListener("click", (event) => handleChartClick(event, plot, traces));
  }

  positionPlot = new uPlot(
    chartOptions(positionChartRoot, positionTraces, "Position", positionHighlightColor, positionTraceColor),
    chartData(positionTraces),
    positionChartRoot,
  );
  const positionOverlay = positionPlot.root.querySelector(".u-over");
  if (positionOverlay instanceof HTMLElement) {
    positionOverlay.addEventListener("click", (event) => handleChartClick(event, positionPlot, positionTraces));
  }
}

function updatePlot(): void {
  updateSinglePlot(plot, traces, highlightColor, traceColor);
  updateSinglePlot(positionPlot, positionTraces, positionHighlightColor, positionTraceColor);
}

function updateSinglePlot(
  targetPlot: UPlotInstance | null,
  sourceTraces: number[][],
  selectedColor: string,
  idleColor: string,
): void {
  if (targetPlot === null) return;
  plotTraceOrder().forEach((traceIndex, displayIndex) => {
    targetPlot.setSeries(displayIndex + 1, traceSeries(traceIndex, selectedColor, idleColor), false);
  });
  targetPlot.setData(chartData(sourceTraces), true);
}

function resizePlot(): void {
  resizeSinglePlot(plot, chartRoot);
  resizeSinglePlot(positionPlot, positionChartRoot);
}

function resizeSinglePlot(targetPlot: UPlotInstance | null, root: HTMLElement): void {
  if (targetPlot === null) return;
  const rect = root.getBoundingClientRect();
  targetPlot.setSize({ width: Math.max(1, Math.floor(rect.width)), height: Math.max(1, Math.floor(rect.height)) });
}

function handleChartClick(event: MouseEvent, activePlot: UPlotInstance | null, sourceTraces: number[][]): void {
  if (activePlot === null) return;
  const overlay = activePlot.root.querySelector(".u-over");
  if (!(overlay instanceof HTMLElement)) return;
  const rect = overlay.getBoundingClientRect();
  const tokenIndex = Math.max(0, Math.round(activePlot.posToVal(event.clientX - rect.left, "x")));
  let bestTrace = selectedTraceIndex;
  let bestDistance = Number.POSITIVE_INFINITY;

  sourceTraces.forEach((values, traceIndex) => {
    const value = values[Math.min(tokenIndex, values.length - 1)];
    if (value === undefined) return;
    const y = activePlot.valToPos(value, "y");
    const distance = Math.abs(event.clientY - rect.top - y);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestTrace = traceIndex;
    }
  });
  selectTrace(bestTrace);
}

function renderTraceSelector(): void {
  while (traceSelector.children.length < traceCount) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "trace-button";
    traceSelector.appendChild(button);
  }

  while (traceSelector.children.length > traceCount) {
    traceSelector.lastElementChild?.remove();
  }

  for (let index = 0; index < traceCount; index += 1) {
    const button = traceSelector.children[index];
    if (!(button instanceof HTMLButtonElement)) continue;
    const count = traces[index]?.length ?? 0;
    button.dataset.traceIndex = String(index);
    button.className = `trace-button${index === selectedTraceIndex ? " selected" : ""}`;
    button.ariaPressed = index === selectedTraceIndex ? "true" : "false";
    button.textContent = `${index} · ${count}`;
  }
}

function handleTraceSelectorClick(event: MouseEvent): void {
  const target = event.target;
  if (!(target instanceof Element)) return;
  const button = target.closest(".trace-button");
  if (!(button instanceof HTMLButtonElement) || !traceSelector.contains(button)) return;
  const index = Number(button.dataset.traceIndex);
  if (!Number.isInteger(index) || index < 0 || index >= traceCount) return;
  selectTrace(index);
}

function renderSelectedTrace(): void {
  const values = traces[selectedTraceIndex] ?? [];
  const positions = positionTraces[selectedTraceIndex] ?? [];
  const texts = tokenTexts[selectedTraceIndex] ?? [];
  const stats = confidenceStats(values);
  selectedTrace.textContent = String(selectedTraceIndex);
  selectedTokens.textContent = String(values.length);
  selectedMean.textContent = formatConfidence(stats.mean);
  selectedMin.textContent = formatConfidence(stats.min);
  selectedMax.textContent = formatConfidence(stats.max);
  finalConfidence.textContent = formatConfidence(stats.final);
  meanConfidence.textContent = formatConfidence(stats.mean);

  const needsFullRender = renderedTraceIndex !== selectedTraceIndex || values.length < renderedTokenCount;
  if (needsFullRender) {
    completion.replaceChildren();
    renderedTraceIndex = selectedTraceIndex;
    renderedTokenCount = 0;
  }

  if (values.length === 0) {
    completion.textContent = "No tokens for this trace yet.";
    renderedTokenCount = 0;
    return;
  }

  if (renderedTokenCount === 0) {
    completion.replaceChildren();
  }

  for (let tokenIndex = renderedTokenCount; tokenIndex < values.length; tokenIndex += 1) {
    const confidence = values[tokenIndex];
    if (confidence === undefined) continue;
    const position = positions[tokenIndex];
    const span = document.createElement("span");
    span.className = "token";
    span.style.backgroundColor = confidenceColor(confidence);
    span.title = `token ${tokenIndex} · confidence ${formatConfidence(confidence)} · position ${formatConfidence(position)}`;
    span.textContent = texts[tokenIndex] ?? "";
    completion.appendChild(span);
  }
  renderedTokenCount = values.length;
}

function selectTrace(index: number): void {
  selectedTraceIndex = index;
  renderTraceSelector();
  renderSelectedTrace();
  updatePlot();
}

function updateStats(): void {
  tokenCount.textContent = `${totalTokenCount()} tokens across ${traceCount} traces`;
  renderTraceSelector();
  renderSelectedTrace();
  updatePlot();
}

function scheduleUpdate(immediate = false): void {
  if (immediate) {
    if (renderTimer !== null) {
      window.clearTimeout(renderTimer);
      renderTimer = null;
    }
    lastRenderTime = performance.now();
    updateStats();
    return;
  }

  if (renderTimer !== null) return;
  const elapsed = performance.now() - lastRenderTime;
  const delay = Math.max(0, renderIntervalMs - elapsed);
  renderTimer = window.setTimeout(() => {
    renderTimer = null;
    lastRenderTime = performance.now();
    updateStats();
  }, delay);
}

function resetView(): void {
  traces = emptyTraceNumbers();
  positionTraces = emptyTraceNumbers();
  tokenTexts = emptyTraceTexts();
  finalResults = [];
  selectedTraceIndex = 0;
  renderedTraceIndex = -1;
  renderedTokenCount = 0;
  completion.textContent = "Select a trace to preview token confidences.";
  tokenCount.textContent = `0 tokens across ${traceCount} traces`;
  finalConfidence.textContent = "-";
  meanConfidence.textContent = "-";
  renderTraceSelector();
  renderSelectedTrace();
  updatePlot();
}

function parseStreamEvent(line: string): StreamEvent {
  return JSON.parse(line) as StreamEvent;
}

function applyEvent(event: StreamEvent): void {
  if (event.type === "token") {
    if (!traces[event.index]) traces[event.index] = [];
    if (!positionTraces[event.index]) positionTraces[event.index] = [];
    if (!tokenTexts[event.index]) tokenTexts[event.index] = [];
    traces[event.index].push(event.confidence);
    positionTraces[event.index].push(event.position);
    tokenTexts[event.index].push(event.text);
    scheduleUpdate();
    return;
  }

  if (event.type === "final") {
    if (!traces[event.index]) traces[event.index] = [];
    if (!positionTraces[event.index]) positionTraces[event.index] = [];
    if (!tokenTexts[event.index]) tokenTexts[event.index] = [];
    finalResults[event.index] = event;
    traces[event.index] = event.token_confidences;
    positionTraces[event.index] = event.token_positions;
    if (event.index === selectedTraceIndex) {
      renderedTraceIndex = -1;
    }
    scheduleUpdate();
    return;
  }

  finalResults = event.completions;
  traces = event.completions.map((result) => result.token_confidences);
  positionTraces = event.completions.map((result) => result.token_positions);
  renderedTraceIndex = -1;
  scheduleUpdate(true);
  setStatus(`Done: ${event.completions.length} traces`);
}

async function runStream(): Promise<void> {
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
    const token = authToken();
    const headers: Record<string, string> = { "content-type": "application/json" };
    if (token) headers.authorization = `Bearer ${token}`;

    const response = await fetch(endpoint.value, {
      method: "POST",
      headers,
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
        applyEvent(parseStreamEvent(line));
      }
    }

    if (buffer.trim()) {
      applyEvent(parseStreamEvent(buffer));
    }
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      setStatus("Stopped");
    } else if (error instanceof Error) {
      setStatus(error.message, true);
    } else {
      setStatus("Unknown error", true);
    }
  } finally {
    scheduleUpdate(true);
    controller = null;
    runButton.disabled = false;
    stopButton.disabled = true;
  }
}

runButton.addEventListener("click", () => {
  void runStream();
});
stopButton.addEventListener("click", () => {
  if (controller) controller.abort();
});
traceSelector.addEventListener("click", handleTraceSelectorClick);
window.addEventListener("resize", resizePlot);
initPlot();
resetView();
