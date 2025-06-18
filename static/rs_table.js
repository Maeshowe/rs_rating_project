/* rs_table.js
   ------------
   • Rangsort tölt be (rs_snapshot.json) és rajzol.
   • Mini RS-Line sparkline minden sorban (utolsó 60 nap).
   • Szektor-szűrő, keresőmező, élő log-panel.
   • Nagy RS-Line modál logikája a rs_line.js-ben.
*/

/* ---------- Konfiguráció ---------- */
const FMP_API_KEY = "RymczcoGjgwOfkxHlCGWEQwknGL1egwi";
const PROFILE_URL = t =>
  `https://financialmodelingprep.com/api/v3/profile/${t}?apikey=${FMP_API_KEY}`;
const fallbackNames = { AAPL:"Apple Inc.", MSFT:"Microsoft", NVDA:"NVIDIA" };

/* ---------- Állapot ---------- */
let DATA = [];
let NAMES = {};

/* ---------- Segédek ---------- */
const rsClass = v => v>=67 ? "rs-high" : v>=34 ? "rs-mid" : "rs-low";

/* ---------- Cégnév-feloldás ---------- */
async function resolveName(t){
  if (NAMES[t]) return NAMES[t];
  if (!FMP_API_KEY) return NAMES[t] = fallbackNames[t] || "";
  try{
    const j = await (await fetch(PROFILE_URL(t))).json();
    return NAMES[t] = j?.[0]?.companyName || fallbackNames[t] || "";
  }catch{
    return NAMES[t] = fallbackNames[t] || "";
  }
}

/* ---------- Mini sparkline ---------- */
function loadMiniChart(cv){
  const ticker = cv.id.slice(2);          // 'c_AAPL' → 'AAPL'

  /* fix méret, responsive off */
  cv.width  = 110;
  cv.height = 40;

  fetch(`../data/rs_line_${ticker}.json`)
    .then(r => r.json())
    .then(js => {
      if (!js.length) return;

      const dat = js.slice(-60);
      const lbl = dat.map(o => o.date.slice(5,10));
      const val = dat.map(o => o.rs_line);

      /* trend meghatározás */
      const up  = val[val.length-1] >= val[0];
      const col = up ? "#27ae60" : "#d1293d";        // zöld / piros
      const bw  = up ? 0.8 : 0.6;                    // vékonyabb lejtőnél

      new Chart(cv,{
        type:"line",
        data:{labels:lbl,
              datasets:[{data:val,
                         borderWidth:bw,
                         borderColor:col,
                         fill:false,
                         tension:0,
                         pointRadius:0,
                         pointHoverRadius:0,
                         pointHitRadius:0}]},
        options:{
          responsive:false,
          maintainAspectRatio:false,
          plugins:{legend:{display:false},tooltip:{enabled:false}},
          scales:{x:{display:false},y:{display:false}}
        }
      });
    })
    .catch(err => console.warn("miniChart", ticker, err));
}

/* ---------- Táblázat render ---------- */
async function renderTable(){
  const sectorSel = document.getElementById("sector-filter").value;
  const q = document.getElementById("search-box").value.trim().toLowerCase();
  const tbody = document.getElementById("body_rows");
  tbody.innerHTML = "Loading…";

  let rows = "";
  for (const r of DATA){
    if (sectorSel!=="ALL" && r.sector!==sectorSel) continue;
    const name = await resolveName(r.ticker);
    if (q && !r.ticker.toLowerCase().includes(q) && !name.toLowerCase().includes(q))
      continue;

    rows += `<tr>
      <td>${r.sector}</td>
      <td><span class="ticker">${r.ticker}</span><span class="name">${name}</span></td>
      <td style="text-align:right;" class="${rsClass(r.rs_rating)}">${r.rs_rating}</td>
      <td class="rs-cell"><canvas class="rs-mini" id="c_${r.ticker}"></canvas></td>
    </tr>`;
  }
  tbody.innerHTML = rows || "<tr><td colspan='4'>No match</td></tr>";
  tbody.querySelectorAll("canvas").forEach(loadMiniChart);
}

/* ---------- Kezdő betöltés ---------- */
fetch("../data/rs_snapshot.json")
  .then(r => r.json())
  .then(async json => {
    /* Last update felirat */
    const ts = json[0]?.as_of;
    if (ts){
      const d = new Date(ts);
      const fmt = `${d.getUTCFullYear()}-${String(d.getUTCMonth()+1).padStart(2,"0")}-${String(d.getUTCDate()).padStart(2,"0")} ${String(d.getUTCHours()).padStart(2,"0")}:${String(d.getUTCMinutes()).padStart(2,"0")} UTC`;
      document.getElementById("last-updated").textContent = `Last update: ${fmt}`;
    }

    DATA = json.sort((a,b)=>b.rs_rating-a.rs_rating);

    /* Szektor-drop */
    const sectors=[...new Set(DATA.map(r=>r.sector))].sort();
    const sel=document.getElementById("sector-filter");
    sel.innerHTML=`<option value="ALL">All sectors</option>` +
      sectors.map(s=>`<option value="${s}">${s}</option>`).join("");

    await renderTable();
    sel.addEventListener("change", renderTable);
    document.getElementById("search-box")
      .addEventListener("input",()=>{clearTimeout(window._deb);window._deb=setTimeout(renderTable,300);});
  });

/* ---------- Élő log-panel ---------- */
const logBox=document.getElementById("log-box");
function loadLog(){
  fetch("../logs/run.log")
    .then(r=>r.text())
    .then(t=>logBox.textContent=t)
    .catch(()=>logBox.textContent="log fetch error");
}
loadLog(); setInterval(loadLog, 10_000);