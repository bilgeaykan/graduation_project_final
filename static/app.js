async function post(url, body=null){
  const res = await fetch(url, {
    method: "POST",
    headers: body ? {"Content-Type":"application/json"} : {},
    body: body ? JSON.stringify(body) : null
  });
  try { return await res.json(); } catch { return { ok: res.ok }; }
}

async function get(url){
  const res = await fetch(url, { cache:"no-store" });
  return await res.json();
}

function stateBadge(state){
  const s = String(state || "").toUpperCase();
  if(s === "ALERT") return `<span class="badge badge-alert">🔥 ALERT</span>`;
  if(s === "SUSPICIOUS") return `<span class="badge badge-susp">⚠️ SUSPICIOUS</span>`;
  return `<span class="badge badge-ok">✅ NORMAL</span>`;
}

let fId = "";
let fBeh = "";

async function refreshStatusAndTimeline(){
  const statusEl = document.getElementById("netStatus");
  try{
    const data = await get("/api/status");

    statusEl.textContent = "LIVE";
    statusEl.className = "chip chip-live";

    document.getElementById("mFps").innerText = data.fps;
    document.getElementById("mFrame").innerText = data.frame;
    document.getElementById("mCam").innerText = data.camera_enabled ? "ON" : "OFF";
    document.getElementById("mAna").innerText = data.analysis_enabled ? "ON" : "OFF";

    const overlay = document.getElementById("camOverlay");
    overlay.style.display = data.camera_enabled ? "none" : "grid";

    const tbody = document.getElementById("tbody");
    const evs = (data.events || []).slice().reverse();

    const filtered = evs.filter(ev => {
      if(fId && String(ev.person_id) !== String(fId)) return false;
      if(fBeh && String(ev.behavior) !== String(fBeh)) return false;
      return true;
    });

    if(filtered.length === 0){
      tbody.innerHTML = `<tr><td colspan="7" class="muted">No events match filters.</td></tr>`;
      return;
    }

    tbody.innerHTML = filtered.slice(0, 100).map(ev => `
      <tr>
        <td>${ev.time_sec}</td>
        <td>${ev.frame}</td>
        <td><span class="pill">${ev.person_id}</span></td>
        <td>${stateBadge(ev.state)}</td>
        <td><span class="pill pill-soft">${ev.behavior}</span></td>
        <td>${ev.score}</td>
        <td>${ev.speed}</td>
      </tr>
    `).join("");

  }catch(e){
    statusEl.textContent = "DISCONNECTED";
    statusEl.className = "chip chip-dead";
    document.getElementById("tbody").innerHTML =
      `<tr><td colspan="7" class="muted">Connection lost. Retrying...</td></tr>`;
  }
}

async function refreshLogs(){
  // CSV
  const pid = document.getElementById("logPid").value.trim();
  const beh = document.getElementById("logBeh").value.trim();

  const csv = await get(`/api/logs/csv?pid=${encodeURIComponent(pid)}&beh=${encodeURIComponent(beh)}`);
  const head = document.getElementById("csvHead");
  const body = document.getElementById("csvBody");

  if(!csv || !csv.columns || csv.columns.length === 0){
    head.innerHTML = "";
    body.innerHTML = `<tr><td class="muted">No CSV logs yet.</td></tr>`;
  }else{
    head.innerHTML = `<tr>${csv.columns.map(c=>`<th>${c}</th>`).join("")}</tr>`;
    body.innerHTML = csv.rows.map(r => `<tr>${r.map(x=>`<td>${x ?? ""}</td>`).join("")}</tr>`).join("");
  }

  // JSON
  const js = await get("/api/logs/json");
  document.getElementById("jsonBox").textContent = JSON.stringify(js, null, 2);
}

function setupTabs(){
  document.querySelectorAll(".tab").forEach(btn=>{
    btn.addEventListener("click", ()=>{
      document.querySelectorAll(".tab").forEach(b=>b.classList.remove("active"));
      btn.classList.add("active");
      const tab = btn.dataset.tab;
      document.getElementById("tab-csv").classList.toggle("hidden", tab !== "csv");
      document.getElementById("tab-json").classList.toggle("hidden", tab !== "json");
    });
  });
}

window.addEventListener("load", () => {
  setupTabs();

  document.getElementById("btnStart")?.addEventListener("click", async ()=>{
    await post("/api/start_live");
    await refreshStatusAndTimeline();
  });

  document.getElementById("btnStop")?.addEventListener("click", async ()=>{
    await post("/api/stop_live");
    await refreshStatusAndTimeline();
  });

  document.getElementById("btnCamera")?.addEventListener("click", async ()=>{
    if(!confirm("Toggle camera? This may restart the stream.")) return;
    await post("/api/toggle_camera");
    await refreshStatusAndTimeline();
  });

  document.getElementById("btnClear")?.addEventListener("click", async ()=>{
    if(!confirm("Clear timeline?")) return;
    await post("/api/clear_timeline");
    await refreshStatusAndTimeline();
  });

  document.getElementById("btnApplyFilter")?.addEventListener("click", ()=>{
    fId = document.getElementById("filterId").value.trim();
    fBeh = document.getElementById("filterBehavior").value.trim();
    refreshStatusAndTimeline();
  });

  document.getElementById("btnLogFilter")?.addEventListener("click", ()=>{
    refreshLogs();
  });

  document.getElementById("btnSaveCfg")?.addEventListener("click", async ()=>{
    const cfg = {
      mode: document.getElementById("mode").value,
      source_path: document.getElementById("sourcePath").value.trim(),
      conf: Number(document.getElementById("conf").value),
      resize_w: Number(document.getElementById("resizeW").value),
      tracker: document.getElementById("tracker").value,
      use_pose: document.getElementById("usePose").checked,
      speed_th: Number(document.getElementById("speedTh").value),
      dist_th: Number(document.getElementById("distTh").value),
      iou_th: Number(document.getElementById("iouTh").value),
      cooldown: Number(document.getElementById("cooldown").value),
    };

    async function refreshVideoList(){
  const r = await get("/api/videos");
  const sel = document.getElementById("serverVideos");
  if(!sel) return;
  const cur = sel.value;
  sel.innerHTML = `<option value="">Select uploaded video…</option>` +
    (r.videos||[]).map(v=>`<option value="${v}">${v}</option>`).join("");
  if(cur) sel.value = cur;
}

document.getElementById("btnUpload")?.addEventListener("click", async ()=>{
  const input = document.getElementById("videoFile");
  if(!input || !input.files || input.files.length===0){
    alert("Please choose a video file first.");
    return;
  }
  const fd = new FormData();
  fd.append("file", input.files[0]);

  const res = await fetch("/api/upload_video", { method:"POST", body: fd });
  const data = await res.json().catch(()=>({ok:false}));
  if(!data.ok){
    alert("Upload failed: " + (data.error || res.status));
    return;
  }

  await refreshVideoList();

  // Switch to file mode + start analysis automatically (does NOT open webcam)
  await post("/api/mode", { mode:"file", source_path: `data/uploads/${data.filename}` });
  await post("/api/start_live");

  await refreshStatusAndTimeline();
  alert("Uploaded & analyzing: " + data.filename);
});

document.getElementById("btnSelectVideo")?.addEventListener("click", async ()=>{
  const sel = document.getElementById("serverVideos");
  const v = sel?.value;
  if(!v){ alert("Pick a video from the list."); return; }

  await post("/api/mode", { mode:"file", source_path: `data/uploads/${v}` });
  await post("/api/start_live"); // auto start for file

  await refreshStatusAndTimeline();
});

refreshVideoList();

    const r = await post("/api/config", cfg);
    document.getElementById("cfgNote").textContent = r.ok ? "Saved ✅" : "Save failed ❌";
    setTimeout(()=>document.getElementById("cfgNote").textContent="Settings apply to next analysis loop.", 1200);
  });

  document.getElementById("btnApplyMode")?.addEventListener("click", async ()=>{
    await post("/api/mode", {
      mode: document.getElementById("mode").value,
      source_path: document.getElementById("sourcePath").value.trim()
    });
  });

  // first load
  refreshLogs();
  refreshStatusAndTimeline();
  setInterval(refreshStatusAndTimeline, 700);
  setInterval(refreshLogs, 2500);
});
