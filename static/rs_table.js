/* ============ konfiguráció ============ */
const FMP_API_KEY = "RymczcoGjgwOfkxHlCGWEQwknGL1egwi";       // ← ide másold a Starter kulcsot (pl. "demo" is működik 250 hívás/napig)
const PROFILE_URL = (t) =>
  `https://financialmodelingprep.com/api/v3/profile/${t}?apikey=${FMP_API_KEY}`;

/* ======= helyi fallback-szótár ======= */
const fallbackNames = {
  AAPL: "Apple Inc.",
  MSFT: "Microsoft",
  NVDA: "NVIDIA",
  META: "Meta Platforms",
  XOM: "Exxon Mobil",
  // … bővíthető …
};

/* ======= állapot ======= */
let DATA = [];
let NAME_CACHE = {};

/* ======= segédek ======= */
const rsClass = (v) => (v >= 67 ? "rs-high" : v >= 34 ? "rs-mid" : "rs-low");

async function resolveName(ticker) {
  if (NAME_CACHE[ticker]) return NAME_CACHE[ticker];
  if (!FMP_API_KEY) return (NAME_CACHE[ticker] = fallbackNames[ticker] || "");

  try {
    const r = await fetch(PROFILE_URL(ticker));
    const json = await r.json();
    const nm = json?.[0]?.companyName || fallbackNames[ticker] || "";
    return (NAME_CACHE[ticker] = nm);
  } catch {
    return (NAME_CACHE[ticker] = fallbackNames[ticker] || "");
  }
}

/* ======= tábla-render ======= */
async function renderTable() {
  const sectorSel = document.getElementById("sector-filter").value;
  const q = document.getElementById("search-box").value.trim().toLowerCase();
  const tbody = document.getElementById("body_rows");
  tbody.innerHTML = "Loading…";

  let rows = "";
  for (const r of DATA) {
    if (sectorSel !== "ALL" && r.sector !== sectorSel) continue;
    const name = await resolveName(r.ticker);
    if (q && !r.ticker.toLowerCase().includes(q) && !name.toLowerCase().includes(q))
      continue;

    rows += `
      <tr>
        <td>${r.sector}</td>
        <td>
          <span class="ticker">${r.ticker}</span>
          <span class="name">${name}</span>
        </td>
        <td style="text-align:right;" class="${rsClass(r.rs_rating)}">
          ${r.rs_rating}
        </td>
      </tr>`;
  }
  tbody.innerHTML = rows || "<tr><td colspan='3'>No match</td></tr>";
}

/* ======= betöltés ======= */
fetch("../data/rs_snapshot.json")
  .then((r) => r.json())
  .then(async (json) => {
    // ✨ UTC timestamp kiírás (első rekord as_of mezője)
    const ts = json[0]?.as_of;
    if (ts) {
      const d = new Date(ts);
      const fmt =
        d.getUTCFullYear() +
        "-" +
        String(d.getUTCMonth() + 1).padStart(2, "0") +
        "-" +
        String(d.getUTCDate()).padStart(2, "0") +
        " " +
        String(d.getUTCHours()).padStart(2, "0") +
        ":" +
        String(d.getUTCMinutes()).padStart(2, "0") +
        " UTC";
      document.getElementById("last-updated").textContent =
        `Last update: ${fmt}`;
    }

    // adatok rendezése RS szerint (desc)
    DATA = json.sort((a, b) => b.rs_rating - a.rs_rating);

    // szektor-dropdown feltöltése
    const sectors = [...new Set(DATA.map((r) => r.sector))].sort();
    const sel = document.getElementById("sector-filter");
    sel.innerHTML =
      `<option value="ALL">All sectors</option>` +
      sectors.map((s) => `<option value="${s}">${s}</option>`).join("");

    // első render, majd események
    await renderTable();
    sel.addEventListener("change", renderTable);
    document.getElementById("search-box").addEventListener("input", () => {
      clearTimeout(window._deb); // debounce
      window._deb = setTimeout(renderTable, 300);
    });
  });

/* ======= log-panel ======= */
const logBox = document.getElementById("log-box");
function loadLog() {
  fetch("../logs/run.log")
    .then((r) => r.text())
    .then((txt) => (logBox.textContent = txt))
    .catch(() => (logBox.textContent = "log fetch error"));
}
loadLog();
setInterval(loadLog, 10_000);