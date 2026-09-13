/**
 * NSE Stocks AI — Native Mobile Client Controller
 * Pure Vanilla JavaScript: Fast, reactive, and zero-framework overhead.
 */

// Local Storage Keys
const STORAGE_KEY_URL = "nse_backend_url";
const STORAGE_KEY_KEY = "nse_backend_key";

// Default settings
let backendUrl = localStorage.getItem(STORAGE_KEY_URL) || (
  window.location.protocol === "https:"
    ? "https://nse-alpha-backend-sumuk.loca.lt"
    : (window.location.port === "8000" ? window.location.origin : "http://localhost:8000")
);
let apiKey = localStorage.getItem(STORAGE_KEY_KEY) || "nse_secret_alpha_2026";
let activeTab = "tab-markets";
let isOnline = false;

// DOM Elements
const badge = document.getElementById("connectionBadge");
const badgeText = document.getElementById("connectionText");
const urlInput = document.getElementById("backendUrlInput");
const keyInput = document.getElementById("apiKeyInput");
const logEl = document.getElementById("connectionStatusLog");

// Initialize Inputs
urlInput.value = backendUrl;
keyInput.value = apiKey;

// ─── Tab Switching ──────────────────────────────────────────────
function switchTab(tabId) {
  if (!tabId) return;
  activeTab = tabId;
  document.querySelectorAll(".nav-item").forEach(b => {
    b.classList.toggle("active", b.getAttribute("data-tab") === tabId);
  });
  document.querySelectorAll(".tab-view").forEach(v => {
    v.classList.toggle("active", v.id === tabId);
  });
  refreshCurrentTab();
}

document.querySelectorAll(".nav-item").forEach(btn => {
  btn.addEventListener("click", (e) => {
    e.preventDefault();
    const tabId = btn.getAttribute("data-tab");
    switchTab(tabId);
  });
});

// ─── API Helper ─────────────────────────────────────────────────
async function apiFetch(endpoint, options = {}) {
  const url = `${backendUrl.replace(/\/$/, "")}${endpoint}`;
  const headers = {
    "X-API-Key": apiKey,
    "Content-Type": "application/json",
    ...(options.headers || {})
  };

  const res = await fetch(url, { ...options, headers });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  }
  return res.json();
}

// ─── Connection Verification ────────────────────────────────────
async function checkConnection() {
  try {
    badgeText.textContent = "Connecting...";
    const data = await apiFetch("/api/v1/auth/verify", { method: "POST" });
    if (data && data.authenticated) {
      isOnline = true;
      badge.className = "connection-badge connected";
      badgeText.textContent = "LIVE BACKEND";
      if (logEl) logEl.textContent = "Connected successfully to backend engine.";
      return true;
    }
  } catch (err) {
    isOnline = false;
    badge.className = "connection-badge disconnected";
    badgeText.textContent = "OFFLINE";
    if (logEl) logEl.textContent = `Connection error: ${err.message}`;
    return false;
  }
}

// ─── Tab 1: Markets Overview ────────────────────────────────────
async function loadMarkets() {
  if (!isOnline) return;
  try {
    const data = await apiFetch("/api/v1/market/overview");
    if (!data) return;

    // Sentiment Score
    const score = Number(data.sentiment_score || 50).toFixed(1);
    document.getElementById("sentimentScore").textContent = score;
    document.getElementById("sentimentLabel").textContent = data.sentiment || "NEUTRAL";
    document.getElementById("sentimentBar").style.width = `${Math.min(100, Math.max(0, score))}%`;

    // Breadth counts
    document.getElementById("advCount").textContent = data.advance_count || 0;
    document.getElementById("decCount").textContent = data.decline_count || 0;
    document.getElementById("advDecRatio").textContent = (data.adv_dec_ratio || 1.0).toFixed(2);

    // Sector Grid
    const sectorGrid = document.getElementById("sectorGrid");
    const sectors = data.sector_performance || {};
    const keys = Object.keys(sectors);

    if (keys.length === 0) {
      sectorGrid.innerHTML = '<div class="empty-state">No sector data available.</div>';
      return;
    }

    sectorGrid.innerHTML = keys.map(sec => {
      const avg = Number(sectors[sec].avg_return || 0);
      const isPos = avg >= 0;
      const colorClass = isPos ? "positive" : "negative";
      const sign = isPos ? "+" : "";
      return `
        <div class="sector-card">
          <div class="sector-name">${sec}</div>
          <div class="sector-change ${colorClass}">${sign}${avg.toFixed(2)}%</div>
        </div>
      `;
    }).join("");

  } catch (e) {
    console.warn("loadMarkets error:", e);
  }
}

// ─── Tab 2: Screener ────────────────────────────────────────────
async function loadScreener() {
  if (!isOnline) return;
  try {
    const list = await apiFetch("/api/v1/market/screener");
    const container = document.getElementById("screenerList");
    if (!Array.isArray(list) || list.length === 0) {
      container.innerHTML = '<div class="empty-state">No screener data available.</div>';
      return;
    }

    container.innerHTML = list.map(item => {
      const chg = Number(item.change_pct || 0);
      const isPos = chg >= 0;
      const chgClass = isPos ? "positive" : "negative";
      const chgSign = isPos ? "+" : "";
      const act = (item.action || "NEUTRAL").toUpperCase();
      const actClass = act.includes("BUY") ? "buy" : act.includes("SELL") ? "sell" : "neutral";

      return `
        <div class="screener-card">
          <div class="card-top-row">
            <div>
              <span class="stock-symbol">${item.symbol}</span>
              <span style="font-size:0.72rem; color:var(--text-muted); margin-left:6px;">${item.sector || ''}</span>
            </div>
            <div style="text-align:right;">
              <span class="stock-price">₹${Number(item.price || 0).toLocaleString('en-IN', {minimumFractionDigits: 1})}</span>
              <div class="${chgClass}" style="font-size:0.75rem; font-family:'JetBrains Mono';">${chgSign}${chg.toFixed(2)}%</div>
            </div>
          </div>
          <div class="card-mid-row">
            <span class="tech-tag">RSI ${item.rsi_14 || 50}</span>
            <span class="tech-tag">${item.supertrend || 'Neutral'}</span>
            <span class="tech-tag">VWAP ${item.vwap_dist_pct > 0 ? '+' : ''}${item.vwap_dist_pct || 0}%</span>
            <span class="action-pill ${actClass}">${item.action_badge || act}</span>
          </div>
        </div>
      `;
    }).join("");
  } catch (e) {
    console.warn("loadScreener error:", e);
  }
}

// ─── Tab 3: Signals ─────────────────────────────────────────────
async function loadSignals() {
  if (!isOnline) return;
  try {
    const opportunities = await apiFetch("/api/v1/market/opportunities");
    const container = document.getElementById("signalsList");
    if (!Array.isArray(opportunities) || opportunities.length === 0) {
      container.innerHTML = '<div class="empty-state">No active breakout setups currently detected.</div>';
      return;
    }

    container.innerHTML = opportunities.map(opp => {
      const typeLabel = opp.type || opp.setup_type || "BREAKOUT";
      const isBuy = !typeLabel.toUpperCase().includes("SELL");
      const badgeClass = isBuy ? "buy" : "sell";
      const title = opp.title || `${opp.symbol} Algorithmic Signal`;
      const desc = opp.description || opp.reason || "Quantitative setup triggered.";
      const price = Number(opp.price || 0).toFixed(1);
      const target = Number(opp.target || 0).toFixed(1);
      const sl = Number(opp.stop_loss || 0).toFixed(1);

      return `
        <div class="signal-card">
          <div class="card-top-row">
            <div>
              <span class="stock-symbol">${opp.symbol}</span>
              <span style="font-size:0.75rem; color:var(--text-muted); margin-left:6px;">${opp.company || ''}</span>
            </div>
            <span class="action-pill ${badgeClass}">${typeLabel.replace(/_/g, ' ')}</span>
          </div>
          <div style="font-size:0.85rem; color:#fff; font-weight:700; margin:6px 0 2px 0;">${title}</div>
          <div style="font-size:0.76rem; color:var(--text-muted); line-height:1.4;">${desc}</div>
          <div style="display:flex; justify-content:space-between; margin-top:8px; padding-top:6px; border-top:1px solid var(--border-subtle); font-size:0.74rem; font-family:'JetBrains Mono';">
            <div>Entry: <b style="color:#fff;">₹${price}</b></div>
            <div>Target: <b style="color:var(--accent-green);">₹${target}</b></div>
            <div>SL: <b style="color:var(--accent-red);">₹${sl}</b></div>
          </div>
        </div>
      `;
    }).join("");
  } catch (e) {
    console.warn("loadSignals error:", e);
  }
}

// ─── Tab 4: Portfolio ───────────────────────────────────────────
async function loadPortfolio() {
  if (!isOnline) return;
  try {
    const data = await apiFetch("/api/v1/portfolio");
    if (!data) return;

    const total = Number(data.total_capital || 10000);
    const avail = Number(data.available_capital || 10000);
    const risk = total - avail;

    document.getElementById("totalEquity").textContent = `₹${total.toLocaleString('en-IN', {maximumFractionDigits: 0})}`;
    document.getElementById("availCash").textContent = `₹${avail.toLocaleString('en-IN', {maximumFractionDigits: 0})}`;
    document.getElementById("activeRisk").textContent = `₹${risk.toLocaleString('en-IN', {maximumFractionDigits: 0})}`;

    const positions = data.positions || {};
    const posKeys = Object.keys(positions);
    const posContainer = document.getElementById("positionsList");

    if (posKeys.length === 0) {
      posContainer.innerHTML = '<div class="empty-state">No active positions. All capital deployed in cash reserve.</div>';
      return;
    }

    posContainer.innerHTML = posKeys.map(sym => {
      const pos = positions[sym];
      return `
        <div class="pos-card">
          <div class="card-top-row">
            <span class="stock-symbol">${sym}</span>
            <span class="action-pill buy">BUY (${pos.shares || pos.quantity || 1} Qty)</span>
          </div>
          <div class="p-sub-row" style="margin-top:6px;">
            <div>Entry: <b>₹${pos.entry_price || pos.average_price || 0}</b></div>
            <div>Stop Loss: <b style="color:var(--accent-red)">₹${pos.stop_loss || 'Dynamic'}</b></div>
          </div>
        </div>
      `;
    }).join("");
  } catch (e) {
    console.warn("loadPortfolio error:", e);
  }
}

// ─── Run Cycle Button ───────────────────────────────────────────
document.getElementById("btnRunCycle").addEventListener("click", async () => {
  const btn = document.getElementById("btnRunCycle");
  try {
    btn.textContent = "⏳ Executing Scan & Trade Cycle...";
    btn.disabled = true;
    const res = await apiFetch("/api/v1/engine/cycle", { method: "POST" });
    alert(`Cycle completed: Status ${res.status || 'OK'}`);
    loadPortfolio();
  } catch (e) {
    alert(`Cycle execution failed: ${e.message}`);
  } finally {
    btn.textContent = "⚡ Trigger Scan-and-Trade Cycle";
    btn.disabled = false;
  }
});

// ─── Settings Handlers ──────────────────────────────────────────
document.getElementById("btnSaveSettings").addEventListener("click", async () => {
  backendUrl = urlInput.value.trim();
  apiKey = keyInput.value.trim();

  localStorage.setItem(STORAGE_KEY_URL, backendUrl);
  localStorage.setItem(STORAGE_KEY_KEY, apiKey);

  if (logEl) logEl.textContent = "Saved settings. Testing connection...";
  const ok = await checkConnection();
  if (ok) {
    refreshCurrentTab();
  }
});

document.getElementById("btnTestConnection").addEventListener("click", checkConnection);

// ─── Auto Refresh Dispatcher ────────────────────────────────────
function refreshCurrentTab() {
  if (activeTab === "tab-markets") loadMarkets();
  else if (activeTab === "tab-screener") loadScreener();
  else if (activeTab === "tab-signals") loadSignals();
  else if (activeTab === "tab-portfolio") loadPortfolio();
}

// ─── Boot Lifecycle ─────────────────────────────────────────────
(async function init() {
  await checkConnection();
  refreshCurrentTab();

  // 5-second polling loop
  setInterval(() => {
    if (isOnline && activeTab !== "tab-settings") {
      refreshCurrentTab();
    }
  }, 5000);
})();
