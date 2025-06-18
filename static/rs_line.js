/* rs_line.js – teljes RS-Line modál */
(function injectModal(){
  const html = `
  <div id="rs-modal" style="display:none;position:fixed;inset:0;
       background:rgba(0,0,0,.55);backdrop-filter:blur(2px);z-index:9999;
       align-items:center;justify-content:center;">
    <div style="background:#fff;border-radius:10px;padding:24px 28px;
         width:640px;max-width:90%;box-shadow:0 8px 20px rgba(0,0,0,.25);">
      <button id="rs-close" style="float:right;border:none;background:none;
              font-size:1.4rem;cursor:pointer;">×</button>
      <h2 id="rs-title" style="margin:0 0 14px;">RS Line</h2>
      <canvas id="rs-big" height="220"></canvas>
    </div>
  </div>`;
  document.body.insertAdjacentHTML("beforeend", html);
  document.getElementById("rs-close").onclick =
    ()=> (document.getElementById("rs-modal").style.display="none");
})();

async function showBigChart(ticker){
  const url = `../data/rs_line_${ticker}.json`;
  const res = await fetch(url);
  if (!res.ok){ alert(`No RS-Line data for ${ticker}`); return; }
  const js = await res.json();
  if (!js.length){ alert(`Empty RS-Line for ${ticker}`); return; }

  const lbl = js.map(o=>o.date);
  const val = js.map(o=>o.rs_line);
  document.getElementById("rs-title").textContent = `RS Line — ${ticker}`;

  if (window._bigChart) window._bigChart.destroy();
  window._bigChart = new Chart(document.getElementById("rs-big"),{
    type:"line",
    data:{labels:lbl,datasets:[{data:val,borderColor:"#2b6cb0",borderWidth:1.2,fill:false,tension:0}]},
    options:{plugins:{legend:{display:false}},
             scales:{x:{ticks:{maxTicksLimit:10}},y:{title:{display:true,text:"Indexed to 100"}}}}
  });

  document.getElementById("rs-modal").style.display="flex";
}

document.getElementById("rs-table").addEventListener("click",e=>{
  const row=e.target.closest("tr");
  if(!row) return;
  const t=row.querySelector(".ticker")?.textContent;
  if(t) showBigChart(t.trim());
});