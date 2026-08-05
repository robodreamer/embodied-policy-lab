const $ = (id) => document.getElementById(id);
let lastState = {};
let editingPrompt = false;
let configured = false;
let pendingPromptSource = "typed";

function format(value, digits = 0) {
  return value !== null && value !== "" && Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
}

function drawSeries(canvas, series, colors, fixedRange = null) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * ratio;
  canvas.height = rect.height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const w = rect.width, h = rect.height, pad = 18;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#202a23"; ctx.lineWidth = 1;
  for (let i = 0; i < 5; i++) {
    const y = pad + (h - 2 * pad) * i / 4;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - pad, y); ctx.stroke();
  }
  const flat = series.flat().filter(Number.isFinite);
  if (!flat.length) return;
  let min = fixedRange ? fixedRange[0] : Math.min(...flat);
  let max = fixedRange ? fixedRange[1] : Math.max(...flat);
  if (max === min) { max += 1; min -= 1; }
  series.forEach((values, index) => {
    if (!values.length) return;
    ctx.strokeStyle = colors[index % colors.length]; ctx.lineWidth = 1.8;
    ctx.beginPath();
    values.forEach((value, point) => {
      const x = pad + (w - 2 * pad) * point / Math.max(1, values.length - 1);
      const y = h - pad - (h - 2 * pad) * (value - min) / (max - min);
      point ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  });
}

function drawAction(chunk) {
  if (!Array.isArray(chunk) || !chunk.length) return drawSeries($("actionChart"), [], []);
  const dimensions = [0,1,2,3,4,5,6].map(d => chunk.map(step => Number(step[d]) || 0));
  drawSeries($("actionChart"), dimensions, ["#b8ff36", "#43d9c5", "#ffba3b", "#9b7cff", "#8467df", "#c795ff", "#ff5f57"]);
}

function drawLatency(values) {
  drawSeries($("latencyChart"), [Array.isArray(values) ? values.map(Number) : []], ["#b8ff36"]);
}

function refreshFrames() {
  const stamp = Date.now();
  $("externalFrame").src = `/frames/external.jpg?t=${stamp}`;
  $("wristFrame").src = `/frames/wrist.jpg?t=${stamp}`;
}

async function postJson(route, payload) {
  const response = await fetch(route, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
  return result;
}

function setControlNote(message, error = false) {
  $("controlNote").textContent = message;
  $("controlNote").style.color = error ? "#ff5f57" : "";
}

function populateTasks(tasks, selected) {
  const select = $("taskSelect");
  if (!Array.isArray(tasks) || select.options.length === tasks.length) {
    select.value = String(selected ?? 0);
    return;
  }
  select.replaceChildren(...tasks.map(task => {
    const option = document.createElement("option");
    option.value = task.id;
    option.textContent = `${task.id}: ${task.prompt}`;
    return option;
  }));
  select.value = String(selected ?? 0);
}

function renderHistory(history) {
  const source = Array.isArray(history) ? history : [];
  const stats = new Map();
  source.filter(item => ["success", "failure"].includes(item.status)).forEach(item => {
    const key = `${item.task_id}\n${item.prompt}`;
    const current = stats.get(key) || {episodes: 0, successes: 0};
    current.episodes += 1; current.successes += Number(item.status === "success"); stats.set(key, current);
  });
  const rows = source.slice().reverse();
  $("historyCard").classList.toggle("hidden", !lastState.interactive);
  $("historyBody").replaceChildren(...rows.map(item => {
    const row = document.createElement("tr");
    const aggregate = stats.get(`${item.task_id}\n${item.prompt}`);
    const rate = aggregate ? `${aggregate.successes}/${aggregate.episodes} · ${Math.round(100 * aggregate.successes / aggregate.episodes)}%` : "excluded";
    [item.attempt, item.task_id, item.prompt, item.status, rate, `${format(item.duration_seconds, 1)} s`].forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value ?? "—";
      if (index === 3) cell.className = `result-${item.status}`;
      row.appendChild(cell);
    });
    return row;
  }));
}

async function configureControls() {
  if (configured) return;
  configured = true;
  const config = await fetch("/api/config", {cache: "no-store"}).then(r => r.json());
  $("generatePrompt").disabled = !config.local_llm_enabled;
  $("generatePrompt").title = config.local_llm_enabled ? `Generate with ${config.local_llm_model}` : "Set LOCAL_LLM_URL and LOCAL_LLM_MODEL to enable";
  $("promptInput").addEventListener("focus", () => { editingPrompt = true; });
  $("promptInput").addEventListener("blur", () => { editingPrompt = false; });
  $("promptInput").addEventListener("input", () => { pendingPromptSource = "typed"; });
  $("applyPrompt").onclick = async () => {
    try { await postJson("/api/control", {action: "set_prompt", prompt: $("promptInput").value, source: pendingPromptSource}); setControlNote("Prompt accepted; the next action chunk will be replanned."); }
    catch (error) { setControlNote(error.message, true); }
  };
  $("resetRun").onclick = async () => {
    try { await postJson("/api/control", {action: "reset"}); setControlNote("Environment reset requested."); }
    catch (error) { setControlNote(error.message, true); }
  };
  $("stopRun").onclick = async () => {
    try { await postJson("/api/control", {action: "stop"}); setControlNote("Stopping after the current simulator tick."); }
    catch (error) { setControlNote(error.message, true); }
  };
  $("taskSelect").onchange = async () => {
    try { await postJson("/api/control", {action: "set_task", task_id: Number($("taskSelect").value)}); setControlNote("Task changed; starting from its canonical initial state."); }
    catch (error) { setControlNote(error.message, true); }
  };
  $("generatePrompt").onclick = async () => {
    const button = $("generatePrompt"); button.disabled = true;
    try {
      const seed = $("promptInput").value.trim() || lastState.canonical_prompt || lastState.prompt;
      const result = await postJson("/api/generate-prompt", {instruction: seed});
      $("promptInput").value = result.prompt;
      pendingPromptSource = "local_llm";
      setControlNote(`Generated locally with ${result.model}; press APPLY + REPLAN to use it.`);
    } catch (error) { setControlNote(error.message, true); }
    finally { button.disabled = !config.local_llm_enabled; }
  };
}

async function updateState() {
  try {
    const state = await fetch("/api/state", {cache: "no-store"}).then(r => r.json());
    lastState = state;
    await configureControls();
    if (state.network_verdict === "loopback_only") {
      $("networkBadge").innerHTML = "<i></i> LOCAL-ONLY VERIFIED";
    } else if (state.network_verdict === "remote_detected") {
      $("networkBadge").innerHTML = "<i></i> REMOTE CONNECTION DETECTED";
      $("networkBadge").style.color = "#ff5f57";
      $("networkBadge").style.borderColor = "#7c2926";
    }
    $("phaseBadge").textContent = String(state.phase || "waiting").toUpperCase();
    $("prompt").textContent = state.prompt || "Waiting for an instruction…";
    $("interactiveControls").classList.toggle("hidden", !state.interactive);
    if (state.interactive) {
      populateTasks(state.available_tasks, state.task_id);
      if (!editingPrompt) $("promptInput").value = state.prompt || "";
      renderHistory(state.attempt_history);
    }
    $("suite").textContent = state.suite || "—";
    $("taskPosition").textContent = state.task_position ? `${state.task_position}/${state.total_tasks}` : "—";
    $("seed").textContent = state.seed ?? "—";
    $("latency").textContent = format(state.inference_latency_ms, 1);
    $("medianLatency").textContent = state.median_inference_latency_ms ? `${format(state.median_inference_latency_ms,1)} ms` : "LIVE";
    $("p95Latency").textContent = state.p95_inference_latency_ms ? `${format(state.p95_inference_latency_ms,1)} ms` : "PENDING";
    $("replan").textContent = state.replan_steps ? `${state.replan_steps} STEPS` : "—";
    $("success").textContent = `${state.successes || 0}/${state.episodes || 0}`;
    $("endpoint").textContent = state.policy_endpoint || "ws://127.0.0.1:8000";
    const progress = Math.max(0, Math.min(1, Number(state.progress) || 0));
    $("progressBar").style.transform = `scaleX(${progress})`;
    $("progressText").textContent = `${Math.round(progress * 100)}%`;
    $("updated").textContent = state.updated_at ? `UPDATED ${new Date(state.updated_at).toLocaleTimeString()}` : "AWAITING TELEMETRY";
    drawAction(state.last_action_chunk || []);
    drawLatency(state.inference_latencies_ms || []);
    refreshFrames();
  } catch (_) { $("phaseBadge").textContent = "RECONNECTING"; }
}

async function updateTelemetry() {
  try {
    const gpu = await fetch("/api/telemetry", {cache: "no-store"}).then(r => r.json());
    const completed = ["stopped", "complete"].includes(lastState.phase);
    $("gpuUtil").textContent = format(completed ? gpu.max_utilization_pct : gpu.utilization_pct);
    $("vram").textContent = format(completed ? gpu.max_memory_mib : gpu.memory_mib);
    $("power").textContent = format(completed ? gpu.max_power_w : gpu.power_w, 1);
  } catch (_) { /* dashboard remains useful without nvidia-smi */ }
}

window.addEventListener("resize", () => { drawAction(lastState.last_action_chunk || []); drawLatency(lastState.inference_latencies_ms || []); });
updateState(); updateTelemetry();
setInterval(updateState, 300);
setInterval(updateTelemetry, 1000);
