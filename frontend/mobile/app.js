/**
 * NSE Alpha — Scalp Compounding Mobile Terminal
 * Pure Vanilla JavaScript: Fast, reactive, zero-framework.
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
let activeTab = "tab-scalp";
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
    "Bypass-Tunnel-Reminder": "true",
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

// ─── Tab: Scalp Compounding Dashboard ───────────────────────────
async function loadScalpDashboard() {
  if (!isOnline) return;
  try {
    const data = await apiFetch("/api/v1/scalp/status");
    if (!data) return;
    renderScalpDashboard(data);
  } catch (e) {
    console.warn("loadScalpDashboard error:", e);
  }
}

function renderScalpDashboard(data) {
  const status = data.status || "idle";
  const pnl = Number(data.total_net_pnl || 0);
  const returnPct = Number(data.overall_net_return_pct || 0);
  const trades = Number(data.total_trades || 0);
  const winRate = data.win_rate || 0;
  const pool = Number(data.current_pool || data.starting_capital || 10000);
  const targetProgress = Number(data.target_progress || 0);
  const chain = data.chain || [];
  const activePos = data.active_position;
  const maxTrades = data.config?.max_trades || 15;

  // Status Badge
  const statusBadge = document.getElementById("scalpStatusBadge");
  statusBadge.textContent = status.toUpperCase();
  statusBadge.className = `scalp-status-badge ${status}`;

  // P&L Display
  const pnlEl = document.getElementById("scalpPnlBig");
  const sign = pnl >= 0 ? "+" : "";
  pnlEl.textContent = `${sign}₹${Math.abs(pnl).toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
  pnlEl.className = `scalp-pnl-big ${pnl > 0 ? 'profit' : pnl < 0 ? 'loss' : 'neutral'}`;

  document.getElementById("scalpPnlSub").textContent =
    `Net P&L • ${returnPct >= 0 ? '+' : ''}${returnPct.toFixed(4)}% return`;

  // Target Progress
  document.getElementById("targetProgressPct").textContent = `${Math.min(100, targetProgress).toFixed(0)}%`;
  document.getElementById("targetProgressBar").style.width = `${Math.min(100, targetProgress)}%`;

  // Stats Grid
  document.getElementById("scalpTrades").textContent = trades;
  document.getElementById("scalpWinRate").textContent = trades > 0 ? `${winRate}%` : "—";
  document.getElementById("scalpPool").textContent = `₹${(pool / 1000).toFixed(1)}K`;

  // Controls
  const btnStart = document.getElementById("btnStartScalp");
  const btnStop = document.getElementById("btnStopScalp");
  if (status === "active" || status === "paused") {
    btnStart.style.display = "none";
    btnStop.style.display = "block";
  } else {
    btnStart.style.display = "block";
    btnStop.style.display = "none";
  }

  // Chain Pipeline Dots
  renderChainPipeline(chain, activePos, maxTrades);

  // Active Position
  renderActivePosition(activePos);

  // Trade Chain List
  renderChainTradeList(chain);
}

function renderChainPipeline(chain, activePos, maxTrades) {
  const container = document.getElementById("chainDots");
  let html = "";
  const totalSlots = Math.max(10, maxTrades);

  for (let i = 0; i < totalSlots; i++) {
    if (i > 0) {
      const connClass = i <= chain.length ? "done" : "";
      html += `<div class="chain-connector ${connClass}"></div>`;
    }

    if (i < chain.length) {
      const trade = chain[i];
      const isWin = Number(trade.net_profit || 0) > 0;
      html += `<div class="chain-dot ${isWin ? 'win' : 'loss'}" title="${trade.symbol}: ${trade.net_return_pct}%">${i + 1}</div>`;
    } else if (i === chain.length && activePos) {
      html += `<div class="chain-dot active">${i + 1}</div>`;
    } else {
      html += `<div class="chain-dot pending">${i + 1}</div>`;
    }
  }

  container.innerHTML = html;
}

function renderActivePosition(pos) {
  const card = document.getElementById("activePositionCard");
  if (!pos) {
    card.style.display = "none";
    return;
  }
  card.style.display = "block";

  document.getElementById("activePosSymbol").textContent = (pos.symbol || "").replace(".NS", "");
  document.getElementById("activePosCompany").textContent = pos.company || "";
  document.getElementById("activePosEntry").textContent = `₹${Number(pos.entry_price).toLocaleString('en-IN', {minimumFractionDigits: 1})}`;
  document.getElementById("activePosTP").textContent = `₹${Number(pos.tp_price).toLocaleString('en-IN', {minimumFractionDigits: 1})}`;
  document.getElementById("activePosSL").textContent = `₹${Number(pos.sl_price).toLocaleString('en-IN', {minimumFractionDigits: 1})}`;

  const netPnl = Number(pos.unrealized_net_pnl || 0);
  const unrealPct = Number(pos.unrealized_pct || 0);
  const pnlEl = document.getElementById("activePosLivePnl");
  pnlEl.textContent = `${netPnl >= 0 ? '+' : ''}₹${Math.abs(netPnl).toFixed(2)} (${unrealPct >= 0 ? '+' : ''}${unrealPct.toFixed(2)}%)`;
  pnlEl.className = `pos-live-pnl ${netPnl >= 0 ? 'positive' : 'negative'}`;

  // Hold time bars
  const barsHeld = Number(pos.bars_held || 0);
  const maxBars = Number(pos.max_bars || 3);
  let barsHtml = "";
  for (let i = 0; i < maxBars; i++) {
    if (i < barsHeld) {
      barsHtml += `<div class="pos-bar filled"></div>`;
    } else if (i === barsHeld) {
      barsHtml += `<div class="pos-bar active-bar"></div>`;
    } else {
      barsHtml += `<div class="pos-bar"></div>`;
    }
  }
  document.getElementById("activePosBars").innerHTML = barsHtml;
}

function renderChainTradeList(chain) {
  const container = document.getElementById("chainTradeList");
  if (!chain || chain.length === 0) {
    container.innerHTML = '<div class="empty-state">No trades yet. Start a session to begin compounding.</div>';
    return;
  }

  // Show most recent first
  const reversed = [...chain].reverse();
  container.innerHTML = reversed.map(trade => {
    const isWin = Number(trade.net_profit || 0) > 0;
    const sym = (trade.symbol || "").replace(".NS", "");
    const netPnl = Number(trade.net_profit || 0);
    const netPct = Number(trade.net_return_pct || 0);
    const reason = (trade.reason || "").replace(/_/g, " ");

    return `
      <div class="chain-trade-item">
        <div class="chain-trade-num ${isWin ? 'win' : 'loss'}">${trade.trade_num || '?'}</div>
        <div class="chain-trade-info">
          <div class="chain-trade-symbol">${sym}</div>
          <div class="chain-trade-detail">${reason} • ${trade.hold_duration || '—'} • Pool: ₹${Number(trade.pool_after || 0).toLocaleString('en-IN')}</div>
        </div>
        <div class="chain-trade-pnl ${isWin ? 'positive' : 'negative'}">
          ${netPnl >= 0 ? '+' : ''}₹${Math.abs(netPnl).toFixed(2)}
          <div style="font-size:0.65rem;color:var(--text-muted);">${netPct >= 0 ? '+' : ''}${netPct.toFixed(3)}%</div>
        </div>
      </div>
    `;
  }).join("");
}

// ─── Scalp Session Controls ─────────────────────────────────────
document.getElementById("btnStartScalp").addEventListener("click", async () => {
  const btn = document.getElementById("btnStartScalp");
  try {
    btn.textContent = "⏳ Starting session...";
    btn.disabled = true;
    await apiFetch("/api/v1/scalp/start", { method: "POST" });
    loadScalpDashboard();
  } catch (e) {
    alert(`Failed to start: ${e.message}`);
  } finally {
    btn.textContent = "🚀 Start Scalp Session";
    btn.disabled = false;
  }
});

document.getElementById("btnStopScalp").addEventListener("click", async () => {
  const btn = document.getElementById("btnStopScalp");
  try {
    btn.textContent = "⏳ Stopping...";
    btn.disabled = true;
    await apiFetch("/api/v1/scalp/stop?reason=manual_stop", { method: "POST" });
    loadScalpDashboard();
  } catch (e) {
    alert(`Failed to stop: ${e.message}`);
  } finally {
    btn.textContent = "⏹ Stop Session";
    btn.disabled = false;
  }
});

// ─── Tab: Markets Overview ──────────────────────────────────────
async function loadMarkets() {
  if (!isOnline) return;
  try {
    const data = await apiFetch("/api/v1/market/overview");
    if (!data) return;

    const score = Number(data.sentiment_score || 50).toFixed(1);
    document.getElementById("sentimentScore").textContent = score;
    document.getElementById("sentimentLabel").textContent = data.sentiment || "NEUTRAL";
    document.getElementById("sentimentBar").style.width = `${Math.min(100, Math.max(0, score))}%`;

    document.getElementById("advCount").textContent = data.advance_count || 0;
    document.getElementById("decCount").textContent = data.decline_count || 0;
    document.getElementById("advDecRatio").textContent = (data.adv_dec_ratio || 1.0).toFixed(2);

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

// ─── Tab: Screener ──────────────────────────────────────────────
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

// ─── Tab: Portfolio ─────────────────────────────────────────────
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
  if (activeTab === "tab-scalp") loadScalpDashboard();
  else if (activeTab === "tab-markets") loadMarkets();
  else if (activeTab === "tab-screener") loadScreener();
  else if (activeTab === "tab-portfolio") loadPortfolio();
}

// ─── Boot Lifecycle ─────────────────────────────────────────────
(async function init() {
  await checkConnection();
  refreshCurrentTab();

  // 3-second polling for scalp dashboard, 5-second for others
  setInterval(() => {
    if (isOnline && activeTab !== "tab-settings") {
      refreshCurrentTab();
    }
  }, activeTab === "tab-scalp" ? 3000 : 5000);
})();
