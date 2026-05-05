"use strict";
const endpoint = query("#endpoint");
const promptInput = query("#prompt");
const maxTokens = query("#maxTokens");
const temperature = query("#temperature");
const topP = query("#topP");
const thinking = query("#thinking");
const runButton = query("#run");
const stopButton = query("#stop");
const statusText = query("#status");
const tokenCount = query("#tokenCount");
const finalConfidence = query("#finalConfidence");
const meanConfidence = query("#meanConfidence");
const completion = query("#completion");
const traceSelector = query("#traceSelector");
const selectedTrace = query("#selectedTrace");
const selectedTokens = query("#selectedTokens");
const selectedMean = query("#selectedMean");
const selectedMin = query("#selectedMin");
const selectedMax = query("#selectedMax");
const chartRoot = query("#chart");
const positionChartRoot = query("#positionChart");
const traceCount = 16;
const traceColor = "rgba(31, 122, 140, 0.22)";
const highlightColor = "#0f3f46";
const positionTraceColor = "rgba(181, 95, 31, 0.22)";
const positionHighlightColor = "#8a3f10";
const renderIntervalMs = 100;
const maxPlotPoints = 700;
let controller = null;
let traces = emptyTraceNumbers();
let positionTraces = emptyTraceNumbers();
let tokenTexts = emptyTraceTexts();
let finalResults = [];
let selectedTraceIndex = 0;
let plot = null;
let positionPlot = null;
let renderTimer = null;
let lastRenderTime = 0;
let renderedTraceIndex = -1;
let renderedTokenCount = 0;
function query(selector) {
    const element = document.querySelector(selector);
    if (!(element instanceof HTMLElement)) {
        throw new Error(`Missing element: ${selector}`);
    }
    return element;
}
function emptyTraceNumbers() {
    return Array.from({ length: traceCount }, () => []);
}
function emptyTraceTexts() {
    return Array.from({ length: traceCount }, () => []);
}
function authToken() {
    const params = new URLSearchParams(window.location.search);
    const queryToken = params.get("token");
    if (queryToken) {
        window.localStorage.setItem("inference_auth_token", queryToken);
        return queryToken;
    }
    return window.localStorage.getItem("inference_auth_token");
}
function setStatus(text, isError = false) {
    statusText.textContent = text;
    statusText.classList.toggle("error", isError);
}
function formatConfidence(value) {
    return value === null || value === undefined ? "-" : value.toFixed(3);
}
function meanConfidenceFor(values) {
    if (values.length === 0)
        return null;
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}
function totalTokenCount() {
    return traces.reduce((sum, values) => sum + values.length, 0);
}
function confidenceStats(values) {
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
function confidenceColor(value) {
    const clamped = Math.max(0, Math.min(1, value));
    const hue = clamped * 120;
    return `hsla(${hue}, 65%, 48%, 0.28)`;
}
function chartData(sourceTraces) {
    const maxLength = Math.max(1, ...sourceTraces.map((values) => values.length));
    const pointCount = Math.min(maxLength, maxPlotPoints);
    const stride = Math.max(1, Math.ceil(maxLength / maxPlotPoints));
    const x = Array.from({ length: pointCount }, (_, index) => Math.min(index * stride, maxLength - 1));
    const ySeries = plotTraceOrder().map((traceIndex) => x.map((tokenIndex) => smoothedValueAt(sourceTraces[traceIndex] ?? [], tokenIndex)));
    return [x, ...ySeries];
}
function plotTraceOrder() {
    const order = Array.from({ length: traceCount }, (_, index) => index).filter((index) => index !== selectedTraceIndex);
    order.push(selectedTraceIndex);
    return order;
}
function smoothedValueAt(values, index) {
    const value = values[index];
    if (value === undefined)
        return null;
    const previous = values[index - 1] ?? value;
    const next = values[index + 1] ?? value;
    return previous * 0.2 + value * 0.6 + next * 0.2;
}
function chartOptions(root, sourceTraces, yLabel, selectedColor, idleColor) {
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
function traceSeries(index, selectedColor, idleColor) {
    return {
        label: `Trace ${index}`,
        scale: "y",
        stroke: index === selectedTraceIndex ? selectedColor : idleColor,
        width: index === selectedTraceIndex ? 3 : 1.4,
        points: { show: false },
    };
}
function initPlot() {
    plot = new uPlot(chartOptions(chartRoot, traces, "Confidence", highlightColor, traceColor), chartData(traces), chartRoot);
    const overlay = plot.root.querySelector(".u-over");
    if (overlay instanceof HTMLElement) {
        overlay.addEventListener("click", (event) => handleChartClick(event, plot, traces));
    }
    positionPlot = new uPlot(chartOptions(positionChartRoot, positionTraces, "Position", positionHighlightColor, positionTraceColor), chartData(positionTraces), positionChartRoot);
    const positionOverlay = positionPlot.root.querySelector(".u-over");
    if (positionOverlay instanceof HTMLElement) {
        positionOverlay.addEventListener("click", (event) => handleChartClick(event, positionPlot, positionTraces));
    }
}
function updatePlot() {
    updateSinglePlot(plot, traces, highlightColor, traceColor);
    updateSinglePlot(positionPlot, positionTraces, positionHighlightColor, positionTraceColor);
}
function updateSinglePlot(targetPlot, sourceTraces, selectedColor, idleColor) {
    if (targetPlot === null)
        return;
    plotTraceOrder().forEach((traceIndex, displayIndex) => {
        targetPlot.setSeries(displayIndex + 1, traceSeries(traceIndex, selectedColor, idleColor), false);
    });
    targetPlot.setData(chartData(sourceTraces), true);
}
function resizePlot() {
    resizeSinglePlot(plot, chartRoot);
    resizeSinglePlot(positionPlot, positionChartRoot);
}
function resizeSinglePlot(targetPlot, root) {
    if (targetPlot === null)
        return;
    const rect = root.getBoundingClientRect();
    targetPlot.setSize({ width: Math.max(1, Math.floor(rect.width)), height: Math.max(1, Math.floor(rect.height)) });
}
function handleChartClick(event, activePlot, sourceTraces) {
    if (activePlot === null)
        return;
    const overlay = activePlot.root.querySelector(".u-over");
    if (!(overlay instanceof HTMLElement))
        return;
    const rect = overlay.getBoundingClientRect();
    const tokenIndex = Math.max(0, Math.round(activePlot.posToVal(event.clientX - rect.left, "x")));
    let bestTrace = selectedTraceIndex;
    let bestDistance = Number.POSITIVE_INFINITY;
    sourceTraces.forEach((values, traceIndex) => {
        const value = values[Math.min(tokenIndex, values.length - 1)];
        if (value === undefined)
            return;
        const y = activePlot.valToPos(value, "y");
        const distance = Math.abs(event.clientY - rect.top - y);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestTrace = traceIndex;
        }
    });
    selectTrace(bestTrace);
}
function renderTraceSelector() {
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
        if (!(button instanceof HTMLButtonElement))
            continue;
        const count = traces[index]?.length ?? 0;
        button.dataset.traceIndex = String(index);
        button.className = `trace-button${index === selectedTraceIndex ? " selected" : ""}`;
        button.ariaPressed = index === selectedTraceIndex ? "true" : "false";
        button.textContent = `${index} · ${count}`;
    }
}
function handleTraceSelectorClick(event) {
    const target = event.target;
    if (!(target instanceof Element))
        return;
    const button = target.closest(".trace-button");
    if (!(button instanceof HTMLButtonElement) || !traceSelector.contains(button))
        return;
    const index = Number(button.dataset.traceIndex);
    if (!Number.isInteger(index) || index < 0 || index >= traceCount)
        return;
    selectTrace(index);
}
function renderSelectedTrace() {
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
        if (confidence === undefined)
            continue;
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
function selectTrace(index) {
    selectedTraceIndex = index;
    renderTraceSelector();
    renderSelectedTrace();
    updatePlot();
}
function updateStats() {
    tokenCount.textContent = `${totalTokenCount()} tokens across ${traceCount} traces`;
    renderTraceSelector();
    renderSelectedTrace();
    updatePlot();
}
function scheduleUpdate(immediate = false) {
    if (immediate) {
        if (renderTimer !== null) {
            window.clearTimeout(renderTimer);
            renderTimer = null;
        }
        lastRenderTime = performance.now();
        updateStats();
        return;
    }
    if (renderTimer !== null)
        return;
    const elapsed = performance.now() - lastRenderTime;
    const delay = Math.max(0, renderIntervalMs - elapsed);
    renderTimer = window.setTimeout(() => {
        renderTimer = null;
        lastRenderTime = performance.now();
        updateStats();
    }, delay);
}
function resetView() {
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
function parseStreamEvent(line) {
    return JSON.parse(line);
}
function applyEvent(event) {
    if (event.type === "token") {
        if (!traces[event.index])
            traces[event.index] = [];
        if (!positionTraces[event.index])
            positionTraces[event.index] = [];
        if (!tokenTexts[event.index])
            tokenTexts[event.index] = [];
        traces[event.index].push(event.confidence);
        positionTraces[event.index].push(event.position);
        tokenTexts[event.index].push(event.text);
        scheduleUpdate();
        return;
    }
    if (event.type === "final") {
        if (!traces[event.index])
            traces[event.index] = [];
        if (!positionTraces[event.index])
            positionTraces[event.index] = [];
        if (!tokenTexts[event.index])
            tokenTexts[event.index] = [];
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
        const token = authToken();
        const headers = { "content-type": "application/json" };
        if (token)
            headers.authorization = `Bearer ${token}`;
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
            if (done)
                break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";
            for (const line of lines) {
                if (!line.trim())
                    continue;
                applyEvent(parseStreamEvent(line));
            }
        }
        if (buffer.trim()) {
            applyEvent(parseStreamEvent(buffer));
        }
    }
    catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") {
            setStatus("Stopped");
        }
        else if (error instanceof Error) {
            setStatus(error.message, true);
        }
        else {
            setStatus("Unknown error", true);
        }
    }
    finally {
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
    if (controller)
        controller.abort();
});
traceSelector.addEventListener("click", handleTraceSelectorClick);
window.addEventListener("resize", resizePlot);
initPlot();
resetView();
