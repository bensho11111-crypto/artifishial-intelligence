// replay.js — opens its own WebSocket for replay control and UI updates.
// The main Three.js module has already constructed its own WS; monkey-patching
// is unreliable due to module execution order. A second connection is clean and
// lets this module send control commands independently.

let ws;
let paused = false;
let sliderDragging = false;

function formatTime(s) {
  if (!s || isNaN(s)) return '0:00';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60).toString().padStart(2, '0');
  return `${m}:${sec}`;
}

function handleSessionInfo(msg) {
  const dur = msg.replay_duration_s || 0;
  if (msg.mode !== 'LIVE' && dur > 0) {
    document.getElementById('replay-ctrl').style.display = 'flex';
    document.getElementById('replay-time').textContent = '0:00 / ' + formatTime(dur);
  }
  if (msg.has_ground_truth) {
    document.getElementById('gt-panel').style.display = 'block';
  }
}

function handleReplayUpdate(replay) {
  if (!replay) return;
  paused = replay.paused;
  document.getElementById('play-pause-btn').textContent = paused ? '▶' : '⏸';
  document.getElementById('replay-time').textContent =
    formatTime(replay.position_s) + ' / ' + formatTime(replay.duration_s);
  if (!sliderDragging) {
    document.getElementById('replay-slider').value =
      Math.round((replay.fraction || 0) * 1000);
  }
}

function handleGroundTruth(msg) {
  const schools = msg.fish_schools || [];
  const m = msg.detection_metrics || {};
  document.getElementById('gt-count').textContent     = schools.length;
  document.getElementById('gt-detected').textContent  = m.fish_detected_count ?? '—';
  document.getElementById('gt-precision').textContent =
    m.precision !== undefined ? (m.precision * 100).toFixed(0) + '%' : '—';
  document.getElementById('gt-recall').textContent    =
    m.recall !== undefined ? (m.recall * 100).toFixed(0) + '%' : '—';
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws/map`);

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if      (msg.type === 'session_info')          handleSessionInfo(msg);
    else if (msg.type === 'ground_truth_update')   handleGroundTruth(msg);
    else if (msg.type === 'map_update' && msg.replay) handleReplayUpdate(msg.replay);
  };

  ws.onclose = () => setTimeout(connect, 2000);
}

connect();

document.addEventListener('DOMContentLoaded', () => {
  const slider = document.getElementById('replay-slider');
  slider.addEventListener('mousedown',  () => { sliderDragging = true; });
  slider.addEventListener('touchstart', () => { sliderDragging = true; });
  slider.addEventListener('change', () => {
    sliderDragging = false;
    send({ type: 'replay_seek', fraction: parseFloat(slider.value) / 1000 });
  });

  document.getElementById('play-pause-btn').addEventListener('click', () => {
    send({ type: paused ? 'replay_play' : 'replay_pause' });
  });

  const speedInput = document.getElementById('replay-speed-input');
  speedInput.addEventListener('input', () => {
    const mult = Math.pow(2, parseFloat(speedInput.value));
    document.getElementById('replay-speed-val').textContent = mult.toFixed(2) + '×';
    send({ type: 'replay_speed', value: mult });
  });
});
