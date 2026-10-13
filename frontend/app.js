// Base URL of the FastAPI backend. When the page is served by the backend
// itself (http://127.0.0.1:8000/app/) it uses the same origin; otherwise
// (opened as a local file or from another static server) it falls back to
// the default API address. See backend/config.py API_HOST / API_PORT.
const API_BASE = window.location.pathname.startsWith("/app")
  ? window.location.origin
  : "http://127.0.0.1:8000";

const videoSelect = document.getElementById("videoSelect");
const videoPlayer = document.getElementById("videoPlayer");
const emptyVideo = document.getElementById("emptyVideo");
const nowPlaying = document.getElementById("nowPlaying");
const copyLinkBtn = document.getElementById("copyLinkBtn");
const modeButtons = document.querySelectorAll(".mode-btn");
const queryInput = document.getElementById("queryInput");
const searchBtn = document.getElementById("searchBtn");
const askScopeRow = document.getElementById("askScopeRow");
const askThisVideoOnly = document.getElementById("askThisVideoOnly");
const statusMsg = document.getElementById("statusMsg");
const resultsList = document.getElementById("resultsList");
const ingestForm = document.getElementById("ingestForm");
const youtubeUrlInput = document.getElementById("youtubeUrlInput");
const ingestBtn = document.getElementById("ingestBtn");
const jobsList = document.getElementById("jobsList");
const summaryStatus = document.getElementById("summaryStatus");
const summaryOutput = document.getElementById("summaryOutput");
const sectionStart = document.getElementById("sectionStart");
const sectionEnd = document.getElementById("sectionEnd");

const PLACEHOLDERS = {
  text: "e.g. Where is gradient descent explained?",
  visual: "e.g. Where is a neural network diagram shown?",
  ask: "e.g. What is the difference between CNNs and RNNs?",
};

let currentMode = "text"; // "text", "visual" or "ask"
let videosById = {};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatTimestamp(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mmss = `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return h > 0 ? `${h}:${mmss}` : mmss;
}

// "83", "1:23", "01:02:03" -> seconds, or null if invalid
function parseTimestamp(text) {
  const value = text.trim();
  if (!value) return null;
  if (/^\d+(\.\d+)?$/.test(value)) return parseFloat(value);
  const parts = value.split(":");
  if (parts.length < 2 || parts.length > 3 || !parts.every((p) => /^\d+(\.\d+)?$/.test(p))) return null;
  return parts.reduce((acc, p) => acc * 60 + parseFloat(p), 0);
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str == null ? "" : String(str);
  return div.innerHTML;
}

function videoLabel(videoId) {
  const v = videosById[videoId];
  return v ? v.title || v.filename : videoId;
}

// Deep link to a moment: #v=<video_id>&t=<seconds>
function momentHash(videoId, seconds) {
  return `#v=${encodeURIComponent(videoId)}&t=${Math.floor(seconds)}`;
}

function parseMomentHash(hash) {
  const params = new URLSearchParams(hash.replace(/^#/, ""));
  const videoId = params.get("v");
  if (!videoId) return null;
  const t = parseTimestamp(params.get("t") || "0");
  return { videoId, seconds: t == null ? 0 : t };
}

// An <a> that jumps the player to a moment. Plain clicks seek in place;
// middle-click / "open in new tab" / "copy link" get a working deep link.
function momentLink(videoId, seconds, label, className) {
  const a = document.createElement("a");
  a.href = momentHash(videoId, seconds);
  a.className = className;
  a.textContent = label;
  a.addEventListener("click", (e) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    playAt(videoId, seconds);
  });
  return a;
}

async function apiPost(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    throw new Error(detail || `Server returned ${res.status}`);
  }
  return data;
}

// ---------------------------------------------------------------------------
// Videos + player
// ---------------------------------------------------------------------------

async function loadVideos(selectVideoId) {
  try {
    const res = await fetch(`${API_BASE}/videos`);
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const data = await res.json();

    const previous = selectVideoId || videoSelect.value;
    videosById = {};
    videoSelect.innerHTML = "";
    data.videos.forEach((v) => {
      videosById[v.video_id] = v;
      const opt = document.createElement("option");
      opt.value = v.video_id;
      opt.textContent = v.title || v.filename;
      videoSelect.appendChild(opt);
    });

    const hasVideos = data.videos.length > 0;
    videoPlayer.hidden = !hasVideos;
    emptyVideo.hidden = hasVideos;
    copyLinkBtn.hidden = !hasVideos;
    if (!hasVideos) return;
    const target = videosById[previous] ? previous : data.videos[0].video_id;
    if (target !== videoSelect.dataset.active) setActiveVideo(target);
    videoSelect.value = target;
  } catch (err) {
    emptyVideo.textContent = `Could not reach the backend at ${API_BASE}. Is it running?`;
    statusMsg.textContent = `Could not reach the backend at ${API_BASE}. Is it running? (${err.message})`;
  }
}

function setActiveVideo(videoId) {
  videoSelect.value = videoId;
  videoSelect.dataset.active = videoId;
  videoPlayer.src = `${API_BASE}/video/${encodeURIComponent(videoId)}`;
  nowPlaying.textContent = "";
  summaryOutput.innerHTML = "";
  summaryStatus.textContent = "";
}

videoSelect.addEventListener("change", () => {
  setActiveVideo(videoSelect.value);
  history.replaceState(null, "", momentHash(videoSelect.value, 0));
});

function waitForMetadata() {
  if (videoPlayer.readyState >= 1) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const onMeta = () => { cleanup(); resolve(); };
    const onError = () => { cleanup(); reject(new Error("The video could not be loaded.")); };
    const cleanup = () => {
      videoPlayer.removeEventListener("loadedmetadata", onMeta);
      videoPlayer.removeEventListener("error", onError);
    };
    videoPlayer.addEventListener("loadedmetadata", onMeta);
    videoPlayer.addEventListener("error", onError);
  });
}

let seekRequest = 0;

// Opens `videoId` in the player and starts playback at exactly `seconds`.
async function playAt(videoId, seconds) {
  if (!videosById[videoId]) {
    statusMsg.textContent = `Video '${videoId}' is not available.`;
    return;
  }
  const request = ++seekRequest;

  if (videoSelect.dataset.active !== videoId) setActiveVideo(videoId);
  history.replaceState(null, "", momentHash(videoId, seconds));

  try {
    await waitForMetadata();
  } catch (err) {
    statusMsg.textContent = err.message;
    return;
  }
  if (request !== seekRequest) return; // a newer click took over while loading

  const duration = Number.isFinite(videoPlayer.duration) ? videoPlayer.duration : Infinity;
  videoPlayer.currentTime = Math.min(Math.max(0, seconds), Math.max(0, duration - 0.5));
  nowPlaying.textContent = `Playing ${videoLabel(videoId)} from ${formatTimestamp(seconds)}`;
  videoPlayer.play().catch(() => {
    // Autoplay can be blocked (e.g. when opening a shared link); the player
    // is still positioned at the right moment, the user just presses play.
    nowPlaying.textContent = `Ready at ${formatTimestamp(seconds)} — press play`;
  });
  videoPlayer.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

window.addEventListener("hashchange", () => {
  const moment = parseMomentHash(window.location.hash);
  if (moment) playAt(moment.videoId, moment.seconds);
});

copyLinkBtn.addEventListener("click", async () => {
  const videoId = videoSelect.value;
  if (!videoId) return;
  const url = `${window.location.href.split("#")[0]}${momentHash(videoId, videoPlayer.currentTime || 0)}`;
  try {
    await navigator.clipboard.writeText(url);
    copyLinkBtn.textContent = "Link copied";
  } catch {
    window.prompt("Copy this link:", url);
  }
  setTimeout(() => { copyLinkBtn.textContent = "Copy link to this moment"; }, 1500);
});

// ---------------------------------------------------------------------------
// Mode toggle (Text Search / Visual Search / Ask AI)
// ---------------------------------------------------------------------------

modeButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    currentMode = btn.dataset.mode;
    modeButtons.forEach((b) => b.classList.toggle("active", b === btn));
    queryInput.placeholder = PLACEHOLDERS[currentMode];
    searchBtn.textContent = currentMode === "ask" ? "Ask" : "Search";
    askScopeRow.hidden = currentMode !== "ask";
  });
});

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

async function runSearch() {
  const query = queryInput.value.trim();
  if (!query) {
    statusMsg.textContent = "Type a query first.";
    return;
  }

  resultsList.innerHTML = "";
  const mode = currentMode;

  try {
    if (mode === "ask") {
      statusMsg.textContent = "Thinking...";
      const body = { question: query };
      if (askThisVideoOnly.checked && videoSelect.value) body.video_id = videoSelect.value;
      const data = await apiPost("/ask", body);
      renderAnswer(data);
      statusMsg.textContent = data.answerable ? `Answer based on ${data.citations.length} passage(s)` : "";
      return;
    }

    statusMsg.textContent = "Searching...";
    const data = await apiPost(mode === "text" ? "/search/text" : "/search/visual", { query, top_k: 5 });
    renderResults(data.results, mode);
    statusMsg.textContent = `${data.results.length} result(s) for "${data.query}" — click one to play it`;
  } catch (err) {
    statusMsg.textContent = `${mode === "ask" ? "Question" : "Search"} failed: ${err.message}`;
  }
}

searchBtn.addEventListener("click", runSearch);
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") runSearch();
});

// ---------------------------------------------------------------------------
// Render search results — every card is a link to its exact moment
// ---------------------------------------------------------------------------

function renderResults(results, mode) {
  resultsList.innerHTML = "";

  if (!results || results.length === 0) {
    resultsList.innerHTML = `<p class="status">No relevant results found.</p>`;
    return;
  }

  results.forEach((r, i) => {
    const seconds = mode === "text" ? r.start : r.timestamp;
    const card = momentLink(r.video_id, seconds, "", "result-card");

    let bodyHtml = "";
    let range = formatTimestamp(seconds);
    if (mode === "text") {
      range = `${formatTimestamp(r.start)} – ${formatTimestamp(r.end)}`;
      bodyHtml = `<p class="result-text">${escapeHtml(r.text)}</p>`;
    } else {
      const frameFilename = r.frame_path.split(/[\\/]/).pop();
      const frameUrl = `${API_BASE}/frame/${encodeURIComponent(r.video_id)}/${encodeURIComponent(frameFilename)}`;
      bodyHtml = `<img class="result-frame" src="${frameUrl}" alt="frame at ${formatTimestamp(seconds)}">`;
    }

    card.innerHTML = `
      <div class="result-header">
        <span>${i + 1}. ${escapeHtml(videoLabel(r.video_id))}</span>
        <span class="result-score">Score: ${r.score.toFixed(2)}</span>
      </div>
      ${bodyHtml}
      <span class="jump-btn">&#9654; Play from ${formatTimestamp(seconds)}</span>
      <span class="result-range">${range}</span>
    `;
    resultsList.appendChild(card);
  });
}

// Answer text with [n] markers turned into links to the cited moment.
function renderAnswer(data) {
  resultsList.innerHTML = "";
  const byRef = {};
  data.citations.forEach((c) => { byRef[c.ref] = c; });

  const answer = document.createElement("div");
  answer.className = "answer-card";
  const p = document.createElement("p");
  p.className = "answer-text";
  data.answer.split(/(\[\d+\])/).forEach((part) => {
    const m = part.match(/^\[(\d+)\]$/);
    const c = m && byRef[Number(m[1])];
    if (c) {
      p.appendChild(momentLink(c.video_id, c.start, `[${formatTimestamp(c.start)}]`, "time-chip"));
    } else {
      p.appendChild(document.createTextNode(part));
    }
  });
  answer.appendChild(p);
  if (!data.answerable) {
    const note = document.createElement("p");
    note.className = "status";
    note.textContent = "The transcripts don't seem to cover this question.";
    answer.appendChild(note);
  }
  resultsList.appendChild(answer);

  data.citations.forEach((c) => {
    const card = momentLink(c.video_id, c.start, "", "result-card");
    card.innerHTML = `
      <div class="result-header">
        <span>[${c.ref}] ${escapeHtml(videoLabel(c.video_id))}</span>
        <span>${formatTimestamp(c.start)} – ${formatTimestamp(c.end)}</span>
      </div>
      <p class="result-text">${escapeHtml(c.text)}</p>
      <span class="jump-btn">&#9654; Play from ${formatTimestamp(c.start)}</span>
    `;
    resultsList.appendChild(card);
  });
}

// ---------------------------------------------------------------------------
// LLM summaries
// ---------------------------------------------------------------------------

document.querySelectorAll("[data-set-time]").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.getElementById(btn.dataset.setTime).value = formatTimestamp(videoPlayer.currentTime || 0);
  });
});

document.querySelectorAll("[data-summary-mode]").forEach((btn) => {
  btn.addEventListener("click", () => runSummary(btn.dataset.summaryMode));
});

async function runSummary(mode) {
  const videoId = videoSelect.value;
  if (!videoId) {
    summaryStatus.textContent = "Select a video first.";
    return;
  }

  const body = { video_id: videoId, mode };
  if (mode === "section") {
    const start = parseTimestamp(sectionStart.value);
    const end = parseTimestamp(sectionEnd.value);
    if (start == null || end == null || end <= start) {
      summaryStatus.textContent = "Enter a valid section, e.g. from 02:00 to 05:30.";
      return;
    }
    body.start = start;
    body.end = end;
  }

  summaryStatus.textContent = "Generating summary...";
  summaryOutput.innerHTML = "";
  try {
    const data = await apiPost("/summarize", body);
    if (videoSelect.value !== videoId) return; // user switched videos meanwhile
    renderSummary(data);
    const what = { summary: "Video summary", takeaways: "Key takeaways", section: "Section summary" }[mode];
    summaryStatus.textContent = mode === "section"
      ? `${what} (${formatTimestamp(data.start)} – ${formatTimestamp(data.end)})`
      : `${what}${data.cached ? " (cached)" : ""}`;
  } catch (err) {
    summaryStatus.textContent = `Summary failed: ${err.message}`;
  }
}

function renderSummary(data) {
  summaryOutput.innerHTML = "";
  if (data.summary) {
    const p = document.createElement("p");
    p.textContent = data.summary;
    summaryOutput.appendChild(p);
  }
  if (data.points.length) {
    const ul = document.createElement("ul");
    data.points.forEach((pt) => {
      const li = document.createElement("li");
      if (pt.timestamp != null) {
        li.appendChild(momentLink(data.video_id, pt.timestamp, formatTimestamp(pt.timestamp), "time-chip"));
        li.appendChild(document.createTextNode(" "));
      }
      li.appendChild(document.createTextNode(pt.text));
      ul.appendChild(li);
    });
    summaryOutput.appendChild(ul);
  }
}

// ---------------------------------------------------------------------------
// YouTube ingestion
// ---------------------------------------------------------------------------

const jobs = {}; // job_id -> latest job state
const pollers = {};

function renderJobs() {
  const sorted = Object.values(jobs).sort((a, b) => b.created_at - a.created_at);
  jobsList.innerHTML = "";
  sorted.forEach((job) => {
    const row = document.createElement("div");
    row.className = `job-row job-${job.status}`;
    const name = job.title || job.url;
    let detail = escapeHtml(job.message);
    if (job.status === "failed") detail = escapeHtml(job.error || job.message);
    if (job.status === "completed" && job.warnings.length) {
      detail += ` <span class="job-warning">(${escapeHtml(job.warnings.join(" "))})</span>`;
    }
    row.innerHTML = `
      <span class="job-status">${escapeHtml(job.status)}</span>
      <span class="job-name">${escapeHtml(name)}</span>
      <span class="job-detail">${detail}</span>
    `;
    if (job.status === "completed") {
      row.appendChild(momentLink(job.video_id, 0, "Play", "time-chip"));
    }
    jobsList.appendChild(row);
  });
}

function trackJob(job) {
  jobs[job.job_id] = job;
  renderJobs();
  if (job.status === "completed" || job.status === "failed" || pollers[job.job_id]) return;

  pollers[job.job_id] = setInterval(async () => {
    try {
      const res = await fetch(`${API_BASE}/ingest/jobs/${job.job_id}`);
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const updated = await res.json();
      jobs[job.job_id] = updated;
      renderJobs();
      if (updated.status === "completed" || updated.status === "failed") {
        clearInterval(pollers[job.job_id]);
        delete pollers[job.job_id];
        if (updated.status === "completed") await loadVideos(updated.video_id);
      }
    } catch {
      // backend restarted or unreachable: in-memory jobs are gone, stop polling
      clearInterval(pollers[job.job_id]);
      delete pollers[job.job_id];
    }
  }, 2000);
}

ingestForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const url = youtubeUrlInput.value.trim();
  if (!url) return;
  ingestBtn.disabled = true;
  try {
    const job = await apiPost("/ingest/youtube", { url });
    youtubeUrlInput.value = "";
    trackJob(job);
  } catch (err) {
    jobs["error"] = {
      job_id: "error", url, status: "failed", message: "", error: err.message,
      warnings: [], created_at: Date.now() / 1000,
    };
    renderJobs();
  } finally {
    ingestBtn.disabled = false;
  }
});

async function loadJobs() {
  try {
    const res = await fetch(`${API_BASE}/ingest/jobs`);
    if (!res.ok) return;
    const data = await res.json();
    data.jobs.forEach(trackJob);
  } catch {
    // backend unreachable: loadVideos() already reports it
  }
}

// ---------------------------------------------------------------------------
// Init: load videos, then honour a #v=...&t=... deep link if present
// ---------------------------------------------------------------------------

(async function init() {
  const moment = parseMomentHash(window.location.hash);
  await loadVideos(moment ? moment.videoId : undefined);
  if (moment) playAt(moment.videoId, moment.seconds);
  loadJobs();
})();
