const elements = Object.fromEntries(
  ["leaderboard-count", "leaderboard-environment", "leaderboard-episodes", "leaderboard-error", "leaderboard-metric", "leaderboard-rows", "leaderboard-turns"]
    .map((id) => [id, document.getElementById(id)])
);

const numberFormatter = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });

function formatNumber(value) {
  return numberFormatter.format(value);
}

function formatPercent(value) {
  return new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 1 }).format(value);
}

function titleCase(value) {
  return String(value).replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function cell(row, value, className = "") {
  const item = document.createElement("td");
  item.className = className;
  if (value instanceof Node) item.append(value);
  else item.textContent = value;
  row.append(item);
  return item;
}

function renderEntry(entry, rank) {
  const row = document.createElement("tr");
  row.className = "leaderboard-entry";
  cell(row, String(rank), "rank-cell");

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "entry-toggle";
  toggle.setAttribute("aria-expanded", "false");
  const identity = document.createElement("div");
  identity.className = "agent-identity";
  const name = document.createElement("strong");
  name.textContent = entry.model || titleCase(entry.agent);
  const type = document.createElement("span");
  type.textContent = entry.model ? `${titleCase(entry.agent)} agent` : "Rule-based baseline";
  const chevron = document.createElement("span");
  chevron.className = "entry-chevron";
  chevron.textContent = "+";
  identity.append(name, type);
  toggle.append(identity, chevron);
  cell(row, toggle);

  cell(row, formatNumber(entry.score_summary.median), "primary-metric");
  cell(row, formatNumber(entry.score_summary.avg));
  cell(row, formatNumber(entry.score_summary.max));
  cell(row, formatNumber(entry.tile_summary.median));
  const isLlm = ["openai", "claude", "vllm"].includes(entry.agent);
  const failure = entry.failure_summary;
  cell(
    row,
    isLlm ? `${formatNumber(failure.failed_turns)} / ${formatNumber(failure.total_turns)} (${formatPercent(failure.rate)})` : "—",
    isLlm ? "failed-turns" : "muted-cell"
  );

  const episodes = document.createElement("div");
  episodes.className = "episode-links";
  entry.episodes.forEach((episode) => {
    const link = document.createElement("a");
    const query = new URLSearchParams({ episode: episode.replay, turn: "0" });
    link.href = `index.html?${query}`;
    link.title = `Open seed ${episode.seed}: score ${formatNumber(episode.score)}, max tile ${formatNumber(episode.max_tile)}`;
    const seed = document.createElement("span");
    seed.textContent = `seed ${episode.seed}`;
    const score = document.createElement("strong");
    score.textContent = formatNumber(episode.score);
    link.append(seed, score);
    episodes.append(link);
  });
  cell(row, episodes);

  const detailsRow = document.createElement("tr");
  detailsRow.className = "entry-details";
  detailsRow.hidden = true;
  const detailsCell = document.createElement("td");
  detailsCell.colSpan = 8;
  const details = document.createElement("div");
  details.className = "entry-details-content";

  const commandBlock = document.createElement("div");
  commandBlock.className = "command-block";
  const commandHeading = document.createElement("p");
  commandHeading.className = "metric-label";
  commandHeading.textContent = "Generation command";
  const command = document.createElement("code");
  const generationCommand = entry.generation_command || "No generation command was recorded for this evaluation.";
  command.textContent = generationCommand;
  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "copy-command";
  copy.textContent = "Copy command";
  copy.addEventListener("click", async () => {
    await navigator.clipboard.writeText(generationCommand);
    copy.textContent = "Copied";
    setTimeout(() => { copy.textContent = "Copy command"; }, 1200);
  });
  commandBlock.append(commandHeading, command, copy);

  const failureBlock = document.createElement("div");
  failureBlock.className = "failure-block";
  const failureHeading = document.createElement("p");
  failureHeading.className = "metric-label";
  failureHeading.textContent = "Failed-turn definition";
  const failureText = document.createElement("p");
  failureText.textContent = isLlm
    ? `${formatNumber(failure.failed_turns)} of ${formatNumber(failure.total_turns)} recorded turns were invalid (${formatPercent(failure.rate)}). This includes unparseable responses and moves that did not change the board.`
    : "This rule-based baseline is not included in the LLM failed-turn comparison.";
  failureBlock.append(failureHeading, failureText);

  details.append(commandBlock, failureBlock);
  detailsCell.append(details);
  detailsRow.append(detailsCell);

  toggle.addEventListener("click", () => {
    const expanded = toggle.getAttribute("aria-expanded") === "true";
    toggle.setAttribute("aria-expanded", String(!expanded));
    chevron.textContent = expanded ? "+" : "−";
    detailsRow.hidden = expanded;
  });
  return [row, detailsRow];
}

async function initialize() {
  try {
    const response = await fetch("catalog.json", { cache: "no-store" });
    if (!response.ok) throw new Error("The published leaderboard catalog was not found.");
    const catalog = await response.json();
    if (catalog.schema_version !== "2048.catalog.v1" || !Array.isArray(catalog.entries)) {
      throw new Error("The published leaderboard catalog is not supported.");
    }

    elements["leaderboard-environment"].textContent = catalog.environment.id;
    elements["leaderboard-episodes"].textContent = `${catalog.seeds.length} fixed seeds`;
    elements["leaderboard-turns"].textContent = formatNumber(catalog.max_turns);
    elements["leaderboard-metric"].textContent = titleCase(catalog.primary_metric);
    elements["leaderboard-count"].textContent = `${catalog.entries.length} ${catalog.entries.length === 1 ? "entry" : "entries"}`;
    elements["leaderboard-rows"].replaceChildren(
      ...catalog.entries.flatMap((entry, index) => renderEntry(entry, index + 1))
    );
  } catch (error) {
    elements["leaderboard-error"].textContent = error.message;
  }
}

initialize();
