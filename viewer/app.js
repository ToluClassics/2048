const elements = Object.fromEntries(
  [
    "actions", "action-title", "agent-label", "agent-output", "agent-reasoning", "board", "catalog-episode", "copy-link", "decision-value",
    "download", "end", "environment-label", "episode-id", "error", "manifest-environment", "max-tile",
    "next", "play", "previous", "replay-file", "revision", "score", "score-delta", "seed", "spawn",
    "speed", "start", "terminal", "termination", "timeline", "timeline-event", "turn", "turn-log",
    "turn-total", "validity",
  ].map((id) => [id, document.getElementById(id)])
);

const state = { manifest: null, turns: [], summary: null, index: 0, sourceText: "", timer: null, episodePath: null };

function parseJsonl(text) {
  const records = text.split(/\r?\n/).filter(Boolean).map((line, index) => {
    try { return JSON.parse(line); }
    catch { throw new Error(`Line ${index + 1} is not valid JSON.`); }
  });
  if (records.length < 2 || records[0].record_type !== "manifest") {
    throw new Error("Replay must begin with a manifest.");
  }
  const summary = records.at(-1);
  if (summary.record_type !== "summary") throw new Error("Replay must end with a summary.");
  const turns = records.slice(1, -1);
  if (turns.some((record) => record.record_type !== "turn")) {
    throw new Error("Replay contains an unsupported record type.");
  }
  return { manifest: records[0], turns, summary };
}

function loadReplay(text, requestedTurn = null, episodePath = null) {
  stopPlayback();
  const parsed = parseJsonl(text);
  Object.assign(state, parsed, { sourceText: text, episodePath });
  const queryTurn = requestedTurn ?? Number(new URLSearchParams(location.search).get("turn"));
  state.index = Number.isInteger(queryTurn) ? Math.min(Math.max(queryTurn, 0), state.turns.length) : 0;
  elements.timeline.max = state.turns.length;
  elements["turn-total"].textContent = state.turns.length;
  renderManifest();
  renderLog();
  renderFrame();
  elements.error.textContent = "";
}

function renderManifest() {
  const { manifest, summary } = state;
  elements["agent-label"].textContent = `${manifest.agent.name} · seed ${manifest.seed}`;
  elements["environment-label"].textContent = manifest.environment.id;
  elements["episode-id"].textContent = manifest.episode_id;
  elements["manifest-environment"].textContent = manifest.environment.id;
  elements.seed.textContent = manifest.seed;
  elements.revision.textContent = manifest.source.revision;
  elements.termination.textContent = summary.termination_reason.replaceAll("_", " ");
  elements.actions.textContent = `${summary.valid_actions} valid · ${summary.invalid_actions} invalid`;
}

function renderFrame() {
  const turn = state.index === 0 ? null : state.turns[state.index - 1];
  const board = turn ? turn.board_after : state.manifest.initial_board;
  elements.board.replaceChildren(...board.flat().map(makeTile));
  elements.board.setAttribute("aria-label", `2048 board at turn ${state.index}: ${board.flat().join(", ")}`);
  elements.turn.textContent = state.index;
  elements.timeline.value = state.index;
  elements.score.textContent = turn?.score ?? 0;
  elements["max-tile"].textContent = turn?.max_tile ?? Math.max(...board.flat());

  if (!turn) {
    elements["action-title"].textContent = "Initial state";
    elements.validity.textContent = "Ready";
    elements.validity.className = "status neutral";
    elements["score-delta"].textContent = "—";
    elements.spawn.textContent = "—";
    elements["decision-value"].textContent = "—";
    elements.terminal.textContent = "No";
    elements["agent-output"].textContent = "No model output was recorded for this frame.";
    elements["agent-reasoning"].textContent = "No reasoning trace was recorded for this frame.";
    elements["timeline-event"].textContent = "Initial board";
  } else {
    elements["action-title"].textContent = `Move ${turn.requested_action}`;
    elements.validity.textContent = turn.action_valid ? "Valid" : "Invalid";
    elements.validity.className = `status ${turn.action_valid ? "valid" : "invalid"}`;
    elements["score-delta"].textContent = `+${turn.score_delta}`;
    elements.spawn.textContent = turn.spawn ? `${turn.spawn.value} at r${turn.spawn.row + 1} c${turn.spawn.col + 1}` : "None";
    elements["decision-value"].textContent = turn.agent_event.decision_value ?? "—";
    elements.terminal.textContent = turn.terminal ? "Yes" : "No";
    elements["agent-output"].textContent = turn.agent_event.visible_output || "No model output was recorded for this frame.";
    elements["agent-reasoning"].textContent = turn.agent_event.visible_reasoning || "No reasoning trace was recorded for this frame.";
    elements["timeline-event"].textContent = `${turn.requested_action} · ${turn.action_valid ? "valid" : "invalid"}`;
  }
  [...elements["turn-log"].querySelectorAll("button")].forEach((button, index) => {
    button.classList.toggle("active", index + 1 === state.index);
  });
  const query = new URLSearchParams({ turn: state.index });
  if (state.episodePath) query.set("episode", state.episodePath);
  history.replaceState(null, "", `${location.pathname}?${query}`);
}

function makeTile(value) {
  const tile = document.createElement("div");
  tile.className = "tile";
  tile.dataset.value = value;
  tile.textContent = value || "";
  if (value > 2048) tile.style.background = "#62d8dc";
  return tile;
}

function renderLog() {
  elements["turn-log"].replaceChildren(...state.turns.map((turn, index) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.innerHTML = `<span class="turn-number">${String(index + 1).padStart(2, "0")}</span><strong>${escapeHtml(turn.requested_action)}</strong><span class="turn-score">${turn.score}</span>`;
    button.addEventListener("click", () => setFrame(index + 1));
    item.append(button);
    return item;
  }));
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[character]);
}

function setFrame(index) {
  state.index = Math.min(Math.max(index, 0), state.turns.length);
  renderFrame();
}

function stopPlayback() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  elements.play.textContent = "▶";
  elements.play.setAttribute("aria-label", "Play replay");
}

function togglePlayback() {
  if (state.timer) return stopPlayback();
  if (state.index === state.turns.length) setFrame(0);
  state.timer = setInterval(() => {
    if (state.index >= state.turns.length) return stopPlayback();
    setFrame(state.index + 1);
  }, Number(elements.speed.value));
  elements.play.textContent = "Ⅱ";
  elements.play.setAttribute("aria-label", "Pause replay");
}

elements.start.addEventListener("click", () => setFrame(0));
elements.previous.addEventListener("click", () => setFrame(state.index - 1));
elements.play.addEventListener("click", togglePlayback);
elements.next.addEventListener("click", () => setFrame(state.index + 1));
elements.end.addEventListener("click", () => setFrame(state.turns.length));
elements.timeline.addEventListener("input", (event) => setFrame(Number(event.target.value)));
elements.speed.addEventListener("change", () => { if (state.timer) { stopPlayback(); togglePlayback(); } });
elements["replay-file"].addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) return;
  try { loadReplay(await file.text(), 0); }
  catch (error) { elements.error.textContent = error.message; }
});
elements["catalog-episode"].addEventListener("change", async (event) => {
  if (!event.target.value) return;
  try {
    const response = await fetch(event.target.value);
    if (!response.ok) throw new Error("Published replay was not found.");
    loadReplay(await response.text(), 0, event.target.value);
  } catch (error) { elements.error.textContent = error.message; }
});
elements.download.addEventListener("click", () => {
  const url = URL.createObjectURL(new Blob([state.sourceText], { type: "application/x-ndjson" }));
  const link = Object.assign(document.createElement("a"), { href: url, download: `${state.manifest.episode_id}.jsonl` });
  link.click();
  URL.revokeObjectURL(url);
});
elements["copy-link"].addEventListener("click", async () => {
  await navigator.clipboard.writeText(location.href);
  elements["copy-link"].textContent = "Copied";
  setTimeout(() => { elements["copy-link"].textContent = "Copy turn link"; }, 1200);
});
document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, select")) return;
  if (event.key === "ArrowLeft") setFrame(state.index - 1);
  if (event.key === "ArrowRight") setFrame(state.index + 1);
  if (event.key === " ") { event.preventDefault(); togglePlayback(); }
  if (event.key === "Home") setFrame(0);
  if (event.key === "End") setFrame(state.turns.length);
});

async function initialize() {
  try {
    const catalogResponse = await fetch("catalog.json", { cache: "no-store" });
    if (catalogResponse.ok) {
      const catalog = await catalogResponse.json();
      const options = catalog.entries.flatMap((entry) => entry.episodes.map((episode) => {
        const option = document.createElement("option");
        option.value = episode.replay;
        option.textContent = `${entry.model || entry.agent} · seed ${episode.seed} · ${episode.score}`;
        return option;
      }));
      elements["catalog-episode"].append(...options);
    }
    const requestedEpisode = new URLSearchParams(location.search).get("episode");
    const episodePath = requestedEpisode || "sample_episode.jsonl";
    const response = await fetch(episodePath);
    if (!response.ok) throw new Error("Replay was not found.");
    loadReplay(await response.text(), null, requestedEpisode);
    if (requestedEpisode) elements["catalog-episode"].value = requestedEpisode;
  } catch (error) {
    elements.error.textContent = `${error.message} Open a JSONL replay to continue.`;
  }
}

initialize();
