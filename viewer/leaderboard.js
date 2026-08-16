const elements = Object.fromEntries(
  ["leaderboard-count", "leaderboard-environment", "leaderboard-episodes", "leaderboard-error", "leaderboard-metric", "leaderboard-rows", "leaderboard-turns"]
    .map((id) => [id, document.getElementById(id)])
);

const numberFormatter = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });

function formatNumber(value) {
  return numberFormatter.format(value);
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
  cell(row, String(rank), "rank-cell");

  const identity = document.createElement("div");
  identity.className = "agent-identity";
  const name = document.createElement("strong");
  name.textContent = entry.model || titleCase(entry.agent);
  const type = document.createElement("span");
  type.textContent = entry.model ? `${titleCase(entry.agent)} agent` : "Rule-based baseline";
  identity.append(name, type);
  cell(row, identity);

  cell(row, formatNumber(entry.score_summary.median), "primary-metric");
  cell(row, formatNumber(entry.score_summary.avg));
  cell(row, formatNumber(entry.score_summary.max));
  cell(row, formatNumber(entry.tile_summary.median));

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
  return row;
}

async function initialize() {
  try {
    const response = await fetch("catalog.json");
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
      ...catalog.entries.map((entry, index) => renderEntry(entry, index + 1))
    );
  } catch (error) {
    elements["leaderboard-error"].textContent = error.message;
  }
}

initialize();
