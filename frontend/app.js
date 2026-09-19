// Base URL of the FastAPI backend. Change this if you run the API on a
// different host/port (see backend/config.py API_HOST / API_PORT).
const API_BASE = "http://127.0.0.1:8000";

const videoSelect = document.getElementById("videoSelect");
const videoPlayer = document.getElementById("videoPlayer");
const textModeBtn = document.getElementById("textModeBtn");
const visualModeBtn = document.getElementById("visualModeBtn");
const queryInput = document.getElementById("queryInput");
const searchBtn = document.getElementById("searchBtn");
const statusMsg = document.getElementById("statusMsg");
const resultsList = document.getElementById("resultsList");

let currentMode = "text"; // "text" or "visual"

// ---------------------------------------------------------------------------
// Load available videos into the dropdown and video player
// ---------------------------------------------------------------------------

async function loadVideos() {
  try {
    const res = await fetch(`${API_BASE}/videos`);
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const data = await res.json();

    videoSelect.innerHTML = "";
    data.videos.forEach((v) => {
      const opt = document.createElement("option");
      opt.value = v.video_id;
      opt.textContent = v.filename;
      videoSelect.appendChild(opt);
    });

    if (data.videos.length > 0) {
      setActiveVideo(data.videos[0].video_id);
    } else {
      statusMsg.textContent = "No videos found. Run scripts/process_videos.py first.";
    }
  } catch (err) {
    statusMsg.textContent = `Could not reach the backend at ${API_BASE}. Is it running? (${err.message})`;
  }
}

function setActiveVideo(videoId) {
  videoPlayer.src = `${API_BASE}/video/${videoId}`;
}

videoSelect.addEventListener("change", () => setActiveVideo(videoSelect.value));

// ---------------------------------------------------------------------------
// Mode toggle (Text Search vs Visual Search)
// ---------------------------------------------------------------------------

textModeBtn.addEventListener("click", () => {
  currentMode = "text";
  textModeBtn.classList.add("active");
  visualModeBtn.classList.remove("active");
  queryInput.placeholder = "e.g. Where is gradient descent explained?";
});

visualModeBtn.addEventListener("click", () => {
  currentMode = "visual";
  visualModeBtn.classList.add("active");
  textModeBtn.classList.remove("active");
  queryInput.placeholder = "e.g. Where is a neural network diagram shown?";
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

  statusMsg.textContent = "Searching...";
  resultsList.innerHTML = "";

  const endpoint = currentMode === "text" ? "/search/text" : "/search/visual";

  try {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 5 }),
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `Server returned ${res.status}`);
    }

    const data = await res.json();
    renderResults(data.results);
    statusMsg.textContent = `${data.results.length} result(s) for "${data.query}"`;
  } catch (err) {
    statusMsg.textContent = `Search failed: ${err.message}`;
  }
}

searchBtn.addEventListener("click", runSearch);
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") runSearch();
});

// ---------------------------------------------------------------------------
// Render results (different card shape for text vs visual)
// ---------------------------------------------------------------------------

function formatTimestamp(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function renderResults(results) {
  resultsList.innerHTML = "";

  if (!results || results.length === 0) {
    resultsList.innerHTML = `<p class="status">No relevant results found.</p>`;
    return;
  }

  results.forEach((r, i) => {
    const card = document.createElement("div");
    card.className = "result-card";

    const timestampSeconds = currentMode === "text" ? r.start : r.timestamp;
    const timestampDisplay = formatTimestamp(timestampSeconds);

    let bodyHtml = "";
    if (currentMode === "text") {
      bodyHtml = `<p class="result-text">${escapeHtml(r.text)}</p>`;
    } else {
      const frameFilename = r.frame_path.split(/[\\/]/).pop();
      const frameUrl = `${API_BASE}/frame/${r.video_id}/${frameFilename}`;
      bodyHtml = `<img class="result-frame" src="${frameUrl}" alt="frame preview">`;
    }

    card.innerHTML = `
      <div class="result-header">
        <span>Result ${i + 1} — ${escapeHtml(r.video_id)}</span>
        <span class="result-score">Score: ${r.score.toFixed(2)}</span>
      </div>
      ${bodyHtml}
      <button class="jump-btn">Jump to ${timestampDisplay}</button>
    `;

    card.querySelector(".jump-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      jumpToResult(r.video_id, timestampSeconds);
    });

    card.addEventListener("click", () => jumpToResult(r.video_id, timestampSeconds));

    resultsList.appendChild(card);
  });
}

function jumpToResult(videoId, timestampSeconds) {
  if (videoSelect.value !== videoId) {
    videoSelect.value = videoId;
    setActiveVideo(videoId);
    // wait for the new video source to be ready before seeking
    videoPlayer.addEventListener(
      "loadedmetadata",
      () => {
        videoPlayer.currentTime = timestampSeconds;
        videoPlayer.play();
      },
      { once: true }
    );
  } else {
    videoPlayer.currentTime = timestampSeconds;
    videoPlayer.play();
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

loadVideos();
