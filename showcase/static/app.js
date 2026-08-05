const $ = (id) => document.getElementById(id);
let lastState = {};
let configured = false;
let editingPrompt = false;
let draftDirty = false;
let taskDirty = false;
let pendingPromptSource = "canonical";
let localLlmEnabled = false;
let randomGenerationEnabled = false;
let pendingCommandId = null;

function format(value, digits = 0) {
  return value !== null && value !== "" && Number.isFinite(Number(value))
    ? Number(value).toFixed(digits) : "—";
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
  const samples = Array.isArray(values) ? values.map(Number) : [];
  const warmSamples = samples.length > 1 ? samples.slice(1) : [];
  drawSeries($("latencyChart"), [warmSamples], ["#b8ff36"]);
}

function refreshFrames() {
  const stamp = Date.now();
  $("externalFrame").src = `/frames/external.jpg?t=${stamp}`;
  $("wristFrame").src = `/frames/wrist.jpg?t=${stamp}`;
}

async function postJson(route, payload) {
  const response = await fetch(route, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
  return result;
}

function setControlNote(message, error = false) {
  $("controlNote").textContent = message;
  $("controlNote").classList.toggle("error", error);
}

function selectedTask() {
  const tasks = Array.isArray(lastState.available_tasks) ? lastState.available_tasks : [];
  return tasks.find(task => String(task.id) === $("taskSelect").value);
}

function syncTaskExplanation() {
  const task = selectedTask();
  $("successRule").textContent = task
    ? `the simulator detects completion of: “${task.prompt}”`
    : "the selected simulator goal is completed";
}

function populateTasks(tasks, selected) {
  const select = $("taskSelect");
  if (!Array.isArray(tasks)) return;
  if (select.options.length !== tasks.length) {
    select.replaceChildren(...tasks.map(task => {
      const option = document.createElement("option");
      option.value = task.id;
      option.textContent = `Task ${Number(task.id) + 1} — ${task.prompt}`;
      return option;
    }));
  }
  if (!taskDirty) select.value = String(selected ?? 0);
  syncTaskExplanation();
}

function isRateEligible(item) {
  if (typeof item.rate_eligible === "boolean") return item.rate_eligible;
  return ["success", "failure"].includes(item.status)
    && !item.mixed_prompt
    && item.prompt_source !== "local_llm_exploratory";
}

function renderHistory(history) {
  const source = Array.isArray(history) ? history : [];
  const stats = new Map();
  source.filter(isRateEligible).forEach(item => {
    const key = `${item.task_id}\n${item.prompt}`;
    const current = stats.get(key) || {episodes: 0, successes: 0};
    current.episodes += 1;
    current.successes += Number(item.status === "success");
    stats.set(key, current);
  });
  $("historyCard").classList.toggle("hidden", !lastState.interactive);
  $("historyBody").replaceChildren(...source.slice().reverse().map(item => {
    const row = document.createElement("tr");
    const aggregate = isRateEligible(item) ? stats.get(`${item.task_id}\n${item.prompt}`) : null;
    const rate = aggregate
      ? `${aggregate.successes}/${aggregate.episodes} · ${Math.round(100 * aggregate.successes / aggregate.episodes)}%`
      : (item.prompt_source === "local_llm_exploratory"
        ? "excluded · exploratory"
        : (item.mixed_prompt ? "excluded · mixed prompt" : "excluded · aborted"));
    const values = [
      item.attempt,
      Number(item.task_id) + 1,
      item.mixed_prompt ? `${item.prompt} (changed mid-run)` : item.prompt,
      item.prompt_source || "—",
      item.status,
      rate,
      `${format(item.duration_seconds, 1)} s`,
    ];
    values.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value ?? "—";
      if (index === 4) cell.className = `result-${item.status}`;
      row.appendChild(cell);
    });
    return row;
  }));
}

function humanPhase(state) {
  if (state.phase === "running") return "RUNNING";
  if (state.phase === "awaiting_command") return "READY";
  if (state.phase === "stopped") return "SAVED";
  if (state.phase === "complete") return "COMPLETE";
  return "LOADING";
}

function updateRunStatus(state) {
  const phase = state.phase;
  const scored = Number(state.episodes) || 0;
  const successes = Number(state.successes) || 0;
  const aborted = Number(state.aborted_attempts) || 0;
  const unscored = Number(state.unscored_attempts) || 0;
  if (phase === "running") {
    $("runStatus").textContent = `Running rollout ${state.attempt || 1}`;
    $("runDetail").textContent = `Step ${Number(state.step) || 0} of at most ${state.max_steps || "—"}; it may finish early.`;
    $("frameStatus").innerHTML = "<i></i> LIVE";
    $("progressLabel").textContent = "CURRENT ROLLOUT";
    $("progressText").textContent = `Step ${Number(state.step) || 0} / ${state.max_steps || "—"}`;
  } else if (phase === "stopped" || phase === "complete") {
    $("runStatus").textContent = "Session saved — review mode";
    $("runDetail").textContent = `${successes}/${scored} scored successes; ${unscored} exploratory/mixed and ${aborted} aborted.`;
    $("frameStatus").innerHTML = "<i></i> LAST FRAME";
    $("progressLabel").textContent = "SESSION SUMMARY";
    $("progressText").textContent = `${successes}/${scored} scored · ${unscored} unscored · ${aborted} aborted`;
  } else if (phase === "awaiting_command") {
    $("runStatus").textContent = scored || unscored || aborted ? "Ready for another rollout" : "Ready to run your first rollout";
    $("runDetail").textContent = "Choose a task and instruction below. Nothing runs until you press Start.";
    $("frameStatus").innerHTML = scored || unscored || aborted ? "<i></i> PAUSED" : "<i></i> WAITING";
    $("progressLabel").textContent = "SESSION SUMMARY";
    $("progressText").textContent = `${successes}/${scored} scored · ${unscored} unscored · ${aborted} aborted`;
  } else {
    $("runStatus").textContent = "Loading local simulator and model";
    $("runDetail").textContent = "The first inference includes a one-time JAX compile.";
    $("frameStatus").innerHTML = "<i></i> WAITING";
  }
  $("frameStatus").classList.toggle("paused", phase !== "running");
  $("runDot").className = phase === "running" ? "active" : phase === "stopped" ? "saved" : "";
}

function updateControls(state) {
  const stopped = ["stopped", "complete"].includes(state.phase);
  const running = state.phase === "running";
  const loading = !state.interactive || state.phase === "initializing";
  $("startRun").disabled = stopped || loading;
  $("applyPrompt").disabled = stopped || !running || !draftDirty;
  $("finishRun").disabled = stopped || loading;
  $("generatePrompt").disabled = stopped || !localLlmEnabled;
  $("generatorMode").disabled = stopped || !localLlmEnabled || !randomGenerationEnabled;
  $("taskSelect").disabled = stopped || loading;
  $("promptInput").disabled = stopped || loading;
  const exploratory = pendingPromptSource === "local_llm_exploratory";
  const rolloutType = exploratory ? "EXPLORATORY" : "SCORED";
  $("startRun").textContent = running
    ? `3 · ABORT & START A FRESH ${rolloutType} ROLLOUT`
    : `3 · START A FRESH ${rolloutType} ROLLOUT`;
  $("startHelp").textContent = exploratory
    ? "Resets the scene and runs the generated experiment. It is excluded from success rates because the LIBERO scoring goal is different."
    : "Resets the scene, uses the task and draft above, and counts the result toward the task/prompt success rate.";
  $("applyPrompt").textContent = running ? "APPLY DRAFT TO THIS ROLLOUT" : "AVAILABLE WHILE A ROLLOUT IS RUNNING";
  $("applyHelp").textContent = stopped
    ? "Unavailable in review mode because the simulator has stopped. Launch a new session to continue."
    : (running
      ? "Available now. Keeps the scene and replans; the mixed-prompt attempt is excluded from per-prompt rates."
      : "Disabled until a rollout is running. Starting a rollout already applies the draft above.");
}

async function configureControls() {
  if (configured) return;
  configured = true;
  const config = await fetch("/api/config", {cache: "no-store"}).then(r => r.json());
  localLlmEnabled = Boolean(config.local_llm_enabled);
  randomGenerationEnabled = Array.isArray(config.prompt_generation_modes);
  $("llmStatus").textContent = localLlmEnabled && randomGenerationEnabled
    ? `Local generator ready: ${config.local_llm_model}. Scored variations preserve the goal; exploratory commands are not scored.`
    : (localLlmEnabled
      ? `Local generator ready: ${config.local_llm_model}. Relaunch this session to enable randomized modes.`
      : "Optional local LLM is not configured. Type a prompt directly, or see README setup instructions.");
  $("promptInput").addEventListener("focus", () => { editingPrompt = true; });
  $("promptInput").addEventListener("blur", () => { editingPrompt = false; });
  $("promptInput").addEventListener("input", () => {
    draftDirty = true;
    pendingPromptSource = "typed";
    $("draftSource").textContent = "TYPED DRAFT";
    updateControls(lastState);
  });
  $("taskSelect").addEventListener("change", () => {
    taskDirty = true;
    const task = selectedTask();
    if (task) {
      $("promptInput").value = task.prompt;
      draftDirty = true;
      pendingPromptSource = "canonical";
      $("draftSource").textContent = "CANONICAL DRAFT";
    }
    syncTaskExplanation();
    setControlNote("Task and its canonical instruction are staged. Press Start when ready.");
    updateControls(lastState);
  });
  $("generatorMode").addEventListener("change", () => {
    const exploratory = $("generatorMode").value === "exploratory";
    setControlNote(exploratory
      ? "Exploratory mode invents another plausible action using scene objects. Its rollout will not affect success rates."
      : "Scored-variation mode randomizes the wording while preserving the selected task goal.");
  });
  $("startRun").onclick = async () => {
    try {
      const result = await postJson("/api/control", {
        action: "start_rollout",
        task_id: Number($("taskSelect").value),
        prompt: $("promptInput").value,
        source: pendingPromptSource,
      });
      pendingCommandId = result.command.id;
      taskDirty = false;
      setControlNote("Starting from a fresh simulator state with the task and instruction above.");
    } catch (error) { setControlNote(error.message, true); }
  };
  $("applyPrompt").onclick = async () => {
    if (!window.confirm("Apply this draft during the current rollout? The attempt will be marked mixed-prompt and excluded from per-prompt success rates.")) return;
    try {
      const result = await postJson("/api/control", {
        action: "set_prompt",
        prompt: $("promptInput").value,
        source: pendingPromptSource,
      });
      pendingCommandId = result.command.id;
      setControlNote("Draft sent. π0.5 will replan from the next observation; this attempt is now mixed-prompt.");
    } catch (error) { setControlNote(error.message, true); }
  };
  $("finishRun").onclick = async () => {
    if (!window.confirm("Finish this session and save its report? The dashboard will remain open in review mode.")) return;
    try {
      const result = await postJson("/api/control", {action: "stop"});
      pendingCommandId = result.command.id;
      setControlNote("Finishing the simulator and saving the report. This dashboard will remain open for review.");
    } catch (error) { setControlNote(error.message, true); }
  };
  $("generatePrompt").onclick = async () => {
    const button = $("generatePrompt"); button.disabled = true;
    try {
      const task = selectedTask();
      const mode = randomGenerationEnabled ? $("generatorMode").value : "scored_variation";
      const result = await postJson("/api/generate-prompt", {
        goal: task?.prompt || lastState.canonical_prompt || lastState.prompt,
        instruction: $("promptInput").value.trim(),
        mode,
      });
      $("promptInput").value = result.prompt;
      draftDirty = true;
      pendingPromptSource = result.mode === "exploratory" ? "local_llm_exploratory" : "local_llm";
      $("draftSource").textContent = result.mode === "exploratory" ? "UNSCORED EXPLORATORY DRAFT" : "LOCAL LLM SCORED DRAFT";
      const scoringNote = result.mode === "exploratory"
        ? "This explores another scene action and is excluded from prompt success rates."
        : "This preserves the selected scoring goal.";
      setControlNote(`Generated a new local draft with ${result.model} in ${format(result.duration_ms)} ms. ${scoringNote}`);
    } catch (error) { setControlNote(error.message, true); }
    finally { updateControls(lastState); }
  };
}

async function updateState() {
  try {
    const state = await fetch("/api/state", {cache: "no-store"}).then(r => r.json());
    lastState = state;
    await configureControls();
    if (pendingCommandId && state.command_ack === pendingCommandId) {
      pendingCommandId = null;
      draftDirty = false;
      taskDirty = false;
    }
    if (state.network_verdict === "loopback_only") {
      $("networkBadge").innerHTML = "<i></i> LOCAL-ONLY VERIFIED";
    } else if (state.network_verdict === "remote_detected") {
      $("networkBadge").innerHTML = "<i></i> REMOTE CONNECTION DETECTED";
      $("networkBadge").classList.add("danger-badge");
    }
    $("phaseBadge").textContent = humanPhase(state);
    $("interactiveControls").classList.toggle("hidden", !state.interactive);
    populateTasks(state.available_tasks, state.task_id);
    if (!draftDirty && !editingPrompt) {
      $("promptInput").value = state.prompt || "";
      pendingPromptSource = state.prompt_source || "canonical";
      $("draftSource").textContent = `${pendingPromptSource.toUpperCase()} DRAFT`;
      if (pendingPromptSource === "local_llm_exploratory") {
        $("generatorMode").value = "exploratory";
      }
    }
    updateRunStatus(state);
    updateControls(state);
    $("prompt").textContent = state.model_ack_prompt || state.prompt || "Waiting for an instruction…";
    $("appliedSource").textContent = String(state.model_ack_prompt_source || state.prompt_source || "—").toUpperCase();
    $("appliedState").textContent = state.model_request_status === "waiting_for_response"
      ? `Sending model request #${state.model_request_id}`
      : (state.model_ack_prompt_sha256
        ? `Model response received for request #${state.model_request_id}`
        : "No model request yet");
    $("promptProof").textContent = state.model_ack_prompt_sha256
      ? `ACKNOWLEDGED PROMPT SHA-256 · ${state.model_ack_prompt_sha256}`
      : "No synchronous model request recorded yet";
    $("suite").textContent = state.suite || "—";
    $("taskPosition").textContent = state.task_position ? `Task ${state.task_position} of ${state.total_tasks}` : "—";
    $("seed").textContent = state.seed ?? "—";
    $("latency").textContent = format(state.inference_latency_ms, 1);
    $("medianLatency").textContent = state.median_inference_latency_ms ? `${format(state.median_inference_latency_ms,1)} ms` : "PENDING";
    $("p95Latency").textContent = state.p95_inference_latency_ms ? `${format(state.p95_inference_latency_ms,1)} ms` : "PENDING";
    $("coldLatency").textContent = state.cold_inference_latency_ms ? `${format(state.cold_inference_latency_ms / 1000,1)} s` : "PENDING";
    $("replan").textContent = state.replan_steps ? `${state.replan_steps} STEPS` : "—";
    const episodes = Number(state.episodes) || 0;
    const successes = Number(state.successes) || 0;
    const aborted = Number(state.aborted_attempts) || 0;
    const unscored = Number(state.unscored_attempts) || 0;
    $("success").textContent = episodes ? `${Math.round(100 * successes / episodes)}%` : "—";
    $("successDetail").textContent = `${successes}/${episodes} scored · ${unscored} exploratory/mixed · ${aborted} aborted`;
    $("endpoint").textContent = state.policy_endpoint || "ws://127.0.0.1:8000";
    const runningProgress = state.phase === "running" ? Math.max(0, Math.min(1, Number(state.progress) || 0)) : 0;
    $("progressBar").style.transform = `scaleX(${runningProgress})`;
    $("updated").textContent = state.updated_at ? `UPDATED ${new Date(state.updated_at).toLocaleTimeString()}` : "AWAITING TELEMETRY";
    if (state.control_error) setControlNote(state.control_error, true);
    renderHistory(state.attempt_history);
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
  } catch (_) { /* Dashboard remains useful without nvidia-smi. */ }
}

window.addEventListener("resize", () => {
  drawAction(lastState.last_action_chunk || []);
  drawLatency(lastState.inference_latencies_ms || []);
});
updateState(); updateTelemetry();
setInterval(updateState, 300);
setInterval(updateTelemetry, 1000);
