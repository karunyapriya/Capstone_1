/*
 * AgriSense - UI wiring.
 *
 * Diagnosis and irrigation inference run entirely in this page: no image and no
 * feature vector ever leaves the device. Supabase is contacted only for sign-in
 * and for the text-only scan history, and auth.js is imported dynamically so
 * that a page opened with no network still diagnoses leaves.
 */

import { extractFeaturesDetailed, featureNames, IMG_SIZE } from './features.js';
import { loadModel, predictProba, predictValue, topK, linearContributions } from './inference.js';
import { buildIrrigationFeatures, fuse, pathogenFor, STAGES, SOILS } from './advisory.js';

const $ = id => document.getElementById(id);
const fmt = (v, n = 1) => Number(v).toFixed(n);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const GUEST_KEY = 'agrisense_guest';
const CONFIDENCE_FLOOR = 0.45;

const state = {
  diseaseModel: null,
  irrClf: null,
  irrReg: null,
  lastFeatures: null,
  lastDisease: null,
  lastTop3: null,
  lastFeatureMs: 0,
  lastInferMs: 0,
  lastIrrigation: null,
  names: featureNames(),
  session: null,
  guest: false,
  guestHistory: [],
  saveStatus: null,
};

/* auth.js pulls the Supabase client from a CDN. If that fails - offline, or the
 * CDN blocked - the rest of the page must still work, so it is optional. */
let auth = null;

/* ------------------------------------------------------------------ *
 * Boot                                                                *
 * ------------------------------------------------------------------ */

async function boot() {
  const chips = [];
  try {
    state.irrClf = await loadModel('models/irrigation_classifier.json');
    state.irrReg = await loadModel('models/irrigation_regressor.json');
    const m = state.irrClf.meta || {};
    chips.push(`<span class="chip ok">Irrigation model · acc ${fmt((m.test_accuracy ?? 0) * 100, 1)}%</span>`);
  } catch (e) {
    chips.push('<span class="chip bad">Irrigation model failed to load</span>');
    console.error(e);
  }
  try {
    state.diseaseModel = await loadModel('models/disease_model.json');
    chips.push(`<span class="chip ok">Disease model · ${state.diseaseModel.labels.length} classes</span>`);
    $('predMeta').textContent = `top-3 of ${state.diseaseModel.labels.length} classes`;
  } catch {
    chips.push('<span class="chip bad">Disease model not trained yet</span>');
    $('noModelNote').hidden = false;
  }
  chips.push('<span class="chip">97 features · offline</span>');
  $('statusChips').innerHTML = chips.join('');
  $('gateChips').innerHTML = chips.join('');

  renderAbout();
  updateIrrigation();

  try {
    auth = await import('./auth.js');
  } catch (e) {
    console.warn('Supabase client unavailable — sign-in disabled, guest mode still works.', e);
    $('authForm').hidden = true;
    $('authSwitch').closest('.switchline').hidden = true;
    showAuthNotice('Sign-in is unavailable right now (no network). You can still use the whole app as a guest.');
  }

  if (auth) {
    state.session = await auth.getSession();
    auth.onAuthStateChange(session => {
      state.session = session;
      if (session) { state.guest = false; localStorage.removeItem(GUEST_KEY); }
      applySessionState();
    });
  }
  state.guest = !state.session && localStorage.getItem(GUEST_KEY) === '1';
  applySessionState();
}

/* ------------------------------------------------------------------ *
 * Auth gate                                                           *
 * ------------------------------------------------------------------ */

let authMode = 'signin';

function showAuthError(msg) { const el = $('authError'); el.textContent = msg; el.hidden = !msg; }
function showAuthNotice(msg) { const el = $('authNotice'); el.textContent = msg; el.hidden = !msg; }

function setAuthMode(mode) {
  authMode = mode;
  const signup = mode === 'signup';
  $('authTitle').textContent = signup ? 'Create an account' : 'Sign in';
  $('authSubmit').textContent = signup ? 'Create account' : 'Sign in';
  $('authSwitchLead').textContent = signup ? 'Already have an account?' : 'No account yet?';
  $('authSwitch').textContent = signup ? 'Sign in' : 'Create one';
  $('authPassword').autocomplete = signup ? 'new-password' : 'current-password';
  showAuthError('');
  showAuthNotice('');
}

$('authSwitch').addEventListener('click', e => {
  e.preventDefault();
  setAuthMode(authMode === 'signin' ? 'signup' : 'signin');
});

$('authForm').addEventListener('submit', async e => {
  e.preventDefault();
  if (!auth) return;
  const email = $('authEmail').value.trim();
  const password = $('authPassword').value;
  const btn = $('authSubmit');
  btn.disabled = true;
  showAuthError('');
  showAuthNotice('');
  try {
    if (authMode === 'signup') {
      const r = await auth.signUp(email, password);
      if (!r.ok) { showAuthError(r.error); return; }
      if (r.needsConfirmation) {
        setAuthMode('signin');
        showAuthNotice(`Account created. Confirm ${email} from your inbox, then sign in.`);
        return;
      }
    } else {
      const r = await auth.signInWithPassword(email, password);
      if (!r.ok) { showAuthError(r.error); return; }
    }
    $('authPassword').value = '';
  } finally {
    btn.disabled = false;
  }
});

$('guestBtn').addEventListener('click', () => {
  state.guest = true;
  localStorage.setItem(GUEST_KEY, '1');
  applySessionState();
});

$('signOutBtn').addEventListener('click', async () => {
  if (auth) await auth.signOut();
  state.session = null;
  state.guest = false;
  state.guestHistory = [];
  localStorage.removeItem(GUEST_KEY);
  applySessionState();
});

/** Show either the gate or the app, and keep the header in sync. */
function applySessionState() {
  const inApp = !!state.session || state.guest;
  $('authGate').hidden = inApp;
  $('app').hidden = !inApp;
  if (!inApp) return;

  $('sessionEmail').textContent = state.session ? state.session.user.email : 'Guest — history clears on reload';
  $('signOutBtn').textContent = state.session ? 'Sign out' : 'Sign in';
  renderHistory();
  renderAdvisory();
}

/* ------------------------------------------------------------------ *
 * Tabs                                                                *
 * ------------------------------------------------------------------ */

function showPanel(panelId) {
  document.querySelectorAll('.tab').forEach(x => x.classList.toggle('is-active', x.dataset.panel === panelId));
  document.querySelectorAll('.panel').forEach(x => x.classList.toggle('is-active', x.id === panelId));
  if (panelId === 'panel-advisory') renderAdvisory();
  if (panelId === 'panel-history') renderHistory();
}

document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => showPanel(t.dataset.panel));
});

/* ------------------------------------------------------------------ *
 * Image input                                                         *
 * ------------------------------------------------------------------ */

const dz = $('dropzone');
['dragenter', 'dragover'].forEach(e =>
  dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.add('drag'); }));
['dragleave', 'drop'].forEach(e =>
  dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.remove('drag'); }));
dz.addEventListener('drop', ev => {
  const f = ev.dataTransfer?.files?.[0];
  if (f) handleImage(f);
});
$('fileInput').addEventListener('change', e => e.target.files[0] && handleImage(e.target.files[0]));

$('modelInput').addEventListener('change', async e => {
  const f = e.target.files[0];
  if (!f) return;
  try {
    const m = JSON.parse(await f.text());
    if (m.format !== 'agrisense-model/1') throw new Error('unrecognised model format');
    state.diseaseModel = m;
    $('noModelNote').hidden = true;
    $('predMeta').textContent = `top-3 of ${m.labels.length} classes`;
    document.querySelectorAll('.chip.bad').forEach(c => c.remove());
    $('statusChips').insertAdjacentHTML('beforeend',
      `<span class="chip ok">Disease model · ${m.labels.length} classes (from file)</span>`);
    renderAbout();
    if (state.lastFeatures) runDisease(state.lastFeatures, state.lastFeatureMs);
  } catch (err) {
    alert('Could not read that model file: ' + err.message);
  }
});

async function handleImage(file) {
  const bmp = await createImageBitmap(file);
  const c = document.createElement('canvas');
  c.width = bmp.width; c.height = bmp.height;
  c.getContext('2d', { willReadFrequently: true }).drawImage(bmp, 0, 0);
  const imgData = c.getContext('2d').getImageData(0, 0, bmp.width, bmp.height);

  const t0 = performance.now();
  const { vector, mask, size } = extractFeaturesDetailed(imgData, IMG_SIZE);
  const ms = performance.now() - t0;

  drawPreview(imgData, mask, size);
  state.lastFeatures = vector;
  state.lastFeatureMs = ms;
  state.saveStatus = null;
  renderFeatureTable(vector);
  $('featDetails').hidden = false;
  runDisease(vector, ms);
}

function drawPreview(imgData, mask, size) {
  // source: re-run the same centre-crop + bilinear path via a scratch canvas
  const src = $('canvasSrc').getContext('2d');
  const tmp = document.createElement('canvas');
  tmp.width = imgData.width; tmp.height = imgData.height;
  tmp.getContext('2d').putImageData(imgData, 0, 0);
  const s = Math.min(imgData.width, imgData.height);
  const top = Math.floor((imgData.height - s) / 2);
  const left = Math.floor((imgData.width - s) / 2);
  src.clearRect(0, 0, size, size);
  src.drawImage(tmp, left, top, s, s, 0, 0, size, size);

  const mctx = $('canvasMask').getContext('2d');
  const out = mctx.createImageData(size, size);
  const base = src.getImageData(0, 0, size, size).data;
  for (let i = 0; i < size * size; i++) {
    const on = mask[i];
    out.data[i * 4]     = on ? base[i * 4]     : 22;
    out.data[i * 4 + 1] = on ? base[i * 4 + 1] : 26;
    out.data[i * 4 + 2] = on ? base[i * 4 + 2] : 22;
    out.data[i * 4 + 3] = 255;
  }
  mctx.putImageData(out, 0, 0);
}

/* ------------------------------------------------------------------ *
 * Diagnose panel                                                      *
 * ------------------------------------------------------------------ */

function runDisease(vector, ms) {
  const out = $('diseaseOut');
  out.classList.remove('empty');

  if (!state.diseaseModel) {
    state.lastDisease = null;
    state.lastTop3 = null;
    out.innerHTML = `<div class="note info">Features extracted in ${fmt(ms ?? 0, 0)} ms.
      Segmentation is shown on the left. Train the classifier to get a diagnosis.</div>`;
    renderAdvisory();
    return;
  }

  const t0 = performance.now();
  const top = topK(state.diseaseModel, vector, 3);
  const infMs = performance.now() - t0;
  state.lastDisease = top[0];
  state.lastTop3 = top;
  state.lastInferMs = infMs;

  const p = pathogenFor(top[0].label);
  const uncertain = top[0].prob < CONFIDENCE_FLOOR;
  const healthy = /healthy/i.test(top[0].label);
  const tone = uncertain ? 'uncertain' : (healthy ? 'go' : 'alert');
  const eyebrow = uncertain ? 'UNCERTAIN · TREAT AS PROVISIONAL'
    : healthy ? 'HEALTHY · NO PATHOGEN DETECTED'
    : `${String(p.severity).toUpperCase()} SEVERITY · ${String(p.group).toUpperCase()}`;

  const bars = top.map((t, i) => `
    <div class="bar${i ? ' rest' : ''}">
      <span>${esc(prettyLabel(t.label))}</span><span class="pct">${fmt(t.prob * 100, 1)}%</span>
      <div class="track"><div class="fill" style="width:${(t.prob * 100).toFixed(1)}%"></div></div>
    </div>`).join('');

  const restProb = 1 - top.reduce((a, t) => a + t.prob, 0);
  const restCount = state.diseaseModel.labels.length - top.length;

  let contrib = '';
  const idx = state.diseaseModel.labels.indexOf(top[0].label);
  const lc = linearContributions(state.diseaseModel, vector, idx, state.names);
  if (lc) {
    contrib = `<details><summary>What drove this prediction</summary><div class="feattable">` +
      lc.slice(0, 12).map(c =>
        `<div><span>${esc(c.name)}</span><span>${c.value >= 0 ? '+' : '−'}${fmt(Math.abs(c.value), 3)}</span></div>`
      ).join('') + `</div></details>`;
  }

  out.innerHTML = `
    <div class="hero ${tone}">
      <div class="hero-left">
        <div class="hero-eyebrow">${esc(eyebrow)}</div>
        <div class="hero-title">${esc(prettyLabel(top[0].label))}</div>
        ${p.agent && p.agent !== '-' ? `<div class="hero-sub">${esc(p.agent)}</div>` : ''}
      </div>
      <div class="hero-right">
        <div class="hero-num">${fmt(top[0].prob * 100, 0)}%</div>
        <div class="hero-numlabel">confidence</div>
      </div>
    </div>
    ${uncertain ? '<div class="note warn">Low confidence — re-photograph the leaf against a plain background, filling the frame, in even light. Below the 45% floor no disease modifier is applied to the irrigation advice.</div>' : ''}
    <div class="bars">${bars}</div>
    ${restCount > 0 ? `<p class="muted small">Remaining ${restCount} classes hold ${fmt(Math.max(restProb, 0) * 100, 1)}% between them.</p>` : ''}
    ${contrib}`;

  $('timingOut').textContent =
    `Features ${fmt(ms ?? 0, 0)} ms · inference ${fmt(infMs, 1)} ms · all on this device`;

  renderAdvisory();
  autoSaveScan();
}

function prettyLabel(l) {
  return String(l).replace(/_{2,}/g, ' | ').replace(/_/g, ' ').replace(/\s+/g, ' ').trim();
}

function cropOf(label) {
  return prettyLabel(label).split(/[ |]/)[0] || null;
}

function renderFeatureTable(v) {
  $('featTable').innerHTML = state.names
    .map((n, i) => `<div><span>${esc(n)}</span><span>${fmt(v[i], 4)}</span></div>`).join('');
}

/* ------------------------------------------------------------------ *
 * Irrigation panel                                                    *
 * ------------------------------------------------------------------ */

const RANGE_UNITS = {
  in_vwc: ['o_vwc', v => `${v}% VWC`],
  in_stemp: ['o_stemp', v => `${v} °C`],
  in_atemp: ['o_atemp', v => `${v} °C`],
  in_rh: ['o_rh', v => `${v}%`],
  in_rain: ['o_rain', v => `${v} mm`],
  in_fcast: ['o_fcast', v => `${v} mm`],
  in_et0: ['o_et0', v => `${v} mm/day`],
  in_days: ['o_days', v => `${v} d`],
};

Object.keys(RANGE_UNITS).concat(['in_soil', 'in_stage']).forEach(id => {
  $(id).addEventListener('input', updateIrrigation);
});

document.querySelectorAll('[data-preset]').forEach(b => {
  b.addEventListener('click', () => {
    const p = {
      dry:        { in_vwc: 13, in_rain: 0, in_fcast: 0, in_et0: 6.2, in_days: 9, in_rh: 38, in_atemp: 36 },
      wet:        { in_vwc: 27, in_rain: 22, in_fcast: 14, in_et0: 2.6, in_days: 1, in_rh: 84, in_atemp: 27 },
      borderline: { in_vwc: 20, in_rain: 1, in_fcast: 4, in_et0: 4.8, in_days: 5, in_rh: 55, in_atemp: 32 },
    }[b.dataset.preset];
    Object.entries(p).forEach(([k, v]) => { $(k).value = v; });
    updateIrrigation();
  });
});

function readInputs() {
  Object.entries(RANGE_UNITS).forEach(([id, [outId, f]]) => { $(outId).textContent = f($(id).value); });
  return {
    soil: $('in_soil').value,
    stage: $('in_stage').value,
    soil_vwc_pct: +$('in_vwc').value,
    soil_temp_c: +$('in_stemp').value,
    air_temp_c: +$('in_atemp').value,
    humidity_pct: +$('in_rh').value,
    rain_24h_mm: +$('in_rain').value,
    rain_forecast_mm: +$('in_fcast').value,
    et0_mm: +$('in_et0').value,
    days_since_irrigation: +$('in_days').value,
  };
}

function updateIrrigation() {
  const o = readInputs();
  const { vector, derived } = buildIrrigationFeatures(o);
  const box = $('irrigationOut');

  if (!state.irrClf || !state.irrReg) {
    box.innerHTML = '<div class="note warn">Irrigation models could not be loaded. ' +
      'Serve this folder over HTTP (<code>python -m http.server</code>) rather than opening the file directly.</div>';
    return;
  }

  const p = predictProba(state.irrClf, vector)[1];
  const thr = state.irrClf.meta?.decision_threshold ?? 0.5;
  const irrigate = p >= thr;
  const depth = Math.max(0, predictValue(state.irrReg, vector));

  state.lastIrrigation = { irrigate, probability: p, depth_mm: irrigate ? depth : 0, derived };

  const { taw, raw, trigger, depletion, etc } = derived;
  const pct = v => `${Math.max(0, Math.min(v / Math.max(taw, 1e-6), 1) * 100).toFixed(1)}%`;

  box.innerHTML = `
    <div class="hero ${irrigate ? 'go' : ''}">
      <div class="hero-left">
        <div class="hero-eyebrow">${irrigate ? 'DEPLETION PAST TRIGGER' : 'DEPLETION BELOW TRIGGER'}</div>
        <div class="hero-title">${irrigate ? 'Irrigate today' : 'Hold — no irrigation'}</div>
      </div>
      <div class="hero-right">
        ${irrigate
          ? `<div class="hero-num">${fmt(depth, 1)}<span class="unit"> mm</span></div>
             <div class="hero-numlabel">applied depth · p ${fmt(p, 2)}</div>`
          : `<div class="hero-num sm">p = ${fmt(p, 2)} &lt; ${fmt(thr, 2)}</div>
             <div class="hero-numlabel">below decision threshold</div>`}
      </div>
    </div>

    <div class="gauge">
      <div class="fill" style="width:${pct(depletion)}"></div>
      <div class="mark" style="left:${pct(trigger)}"></div>
      <div class="mark soft" style="left:${pct(raw)}"></div>
    </div>
    <div class="ticks">
      <span class="tick first">0 mm<br>field capacity</span>
      <span class="tick trigger" style="left:${pct(trigger)}">${fmt(trigger, 1)} mm<br>trigger (½ RAW)</span>
      <span class="tick" style="left:${pct(raw)}">${fmt(raw, 1)} mm<br>RAW limit</span>
      <span class="tick last">${fmt(taw, 1)} mm<br>TAW</span>
    </div>
    <p class="muted small">Current root-zone depletion <span class="mono">${fmt(depletion, 1)} mm</span> —
      ${depletion >= trigger
        ? `${fmt(depletion / Math.max(trigger, 1e-6), 1)}× the trigger depth.`
        : `still below the ${fmt(trigger, 1)} mm trigger.`}</p>

    <dl class="kv">
      <dt>Root-zone depletion D<sub>r</sub></dt><dd>${fmt(depletion, 1)} mm</dd>
      <dt>Readily available water RAW</dt><dd>${fmt(raw, 1)} mm</dd>
      <dt>Total available water TAW</dt><dd>${fmt(taw, 1)} mm</dd>
      <dt>Crop water use ET<sub>c</sub></dt><dd>${fmt(etc, 2)} mm/day</dd>
      <dt>Projected end-of-day deficit</dt><dd>${fmt(derived.projected, 1)} mm</dd>
      <dt>Trigger depth</dt><dd>${fmt(trigger, 1)} mm</dd>
      <dt>Model probability</dt><dd>${fmt(p, 3)}</dd>
      <dt>Soil / stage</dt><dd class="plain">${SOILS[o.soil].label} · ${STAGES[o.stage].label}
        <span class="mono small">K<sub>c</sub> ${STAGES[o.stage].kc} · Z<sub>r</sub> ${STAGES[o.stage].zr} m</span></dd>
    </dl>`;

  renderAdvisory();
}

/* ------------------------------------------------------------------ *
 * Fused advisory                                                      *
 * ------------------------------------------------------------------ */

/** One-line summary of the current decision, stored with a saved scan. */
function decisionSummary(f) {
  if (!f.irrigate) return `Hold — p ${fmt(f.probability, 2)} below threshold`;
  const how = f.pathogen ? f.pathogen.irrigation.method : 'grower preference';
  return `Irrigate ${fmt(f.depthMm, 1)} mm · ${how}`;
}

function renderAdvisory() {
  const box = $('advisoryOut');
  if (!box) return;

  if (!state.lastIrrigation) {
    box.innerHTML = '<div class="card"><div class="empty">Set the field inputs first.</div></div>';
    return;
  }

  const f = fuse(state.lastIrrigation, state.lastDisease, CONFIDENCE_FLOOR);
  const p = f.pathogen;
  const factor = state.lastIrrigation.depth_mm > 0 ? f.depthMm / state.lastIrrigation.depth_mm : 1;
  const tone = f.irrigate ? (p && p.severity === 'high' ? 'alert' : 'go') : '';

  const subline = state.lastDisease
    ? `<span>${esc(prettyLabel(state.lastDisease.label))}</span>
       ${p && p.agent !== '-' ? `<span class="agent">${esc(p.agent)}</span>` : ''}
       ${p ? `<span class="tag ${esc(p.severity)}">${esc(String(p.severity).toUpperCase())} SEVERITY</span>` : ''}
       ${!f.confident ? '<span class="tag medium">DIAGNOSIS UNCONFIRMED</span>' : ''}`
    : '<span class="muted">No leaf diagnosed yet — run an image on the Diagnose tab.</span>';

  const why = p && p.group !== 'healthy' && p.group !== 'unknown'
    ? `${p.name} detected: ${p.irrigation.why}`
    : f.confident
      ? 'No disease pressure detected, so no modifier is applied — the irrigation decision stands as the model computed it.'
      : 'No confirmed diagnosis, so no disease modifier is applied. The irrigation decision above is the unmodified model output.';

  const detail = (label, value) => `<div class="dlabel">${label}</div><div class="dvalue">${value}</div>`;

  box.innerHTML = `
    <div class="card">
      <div class="advhero">
        <div>
          <div class="eyebrow">FUSED ADVISORY · DIAGNOSIS-CONDITIONED IRRIGATION</div>
          <div class="headline">${f.irrigate ? `Irrigate ${fmt(f.depthMm, 1)} mm` : 'Hold irrigation'}</div>
          <div class="subline">${subline}</div>
        </div>
        <div class="stattiles">
          <div class="stattile">
            <div class="stattile-label">DEPTH MODIFIER</div>
            <div class="stattile-num">×${fmt(factor, 2)}</div>
            <div class="stattile-sub">${fmt(state.lastIrrigation.depth_mm, 1)} mm base,
              ${Math.abs(factor - 1) < 0.005 ? 'unchanged' : `now ${fmt(f.depthMm, 1)} mm`}</div>
          </div>
          <div class="stattile">
            <div class="stattile-label">CONFIDENCE</div>
            <div class="stattile-num">${state.lastDisease ? fmt(state.lastDisease.prob, 2) : '—'} / ${fmt(f.probability, 2)}</div>
            <div class="stattile-sub">diagnosis / irrigation</div>
          </div>
        </div>
      </div>
    </div>

    <div class="advgrid">
      <div class="card">
        <div class="detailrows">
          ${detail('METHOD', esc(f.method))}
          ${detail('TIMING', esc(f.timing))}
          ${detail('INTERVAL', esc(f.intervalNote))}
          ${p && p.agent !== '-' ? detail('CAUSAL AGENT', `<span class="agent">${esc(p.agent)}</span>`) : ''}
          ${p && p.signs !== '-' ? detail('WHAT TO LOOK FOR', esc(p.signs)) : ''}
          ${p ? detail('CROP ACTION', esc(p.action)) : ''}
        </div>
      </div>

      <div class="sidecol">
        <div class="sidebox">
          <div class="eyebrow">WHY THE ADVICE ${p && p.group !== 'healthy' && p.group !== 'unknown' ? 'CHANGED' : 'STANDS'}</div>
          <p>${esc(why)}</p>
        </div>

        <div class="card">
          <div class="eyebrow mono small muted" style="letter-spacing:.12em">WHAT THE DIAGNOSIS TOUCHES</div>
          <div class="touchrows" style="margin-top:.9rem">
            <div class="touchrow">
              <span class="touchkey">WHETHER</span>
              <span class="muted">Unchanged — the soil water balance alone decides that.</span>
            </div>
            <div class="touchrow${p ? ' changed' : ''}">
              <span class="touchkey">HOW</span>
              <span>${esc(f.method)}</span>
            </div>
            <div class="touchrow${p ? ' changed' : ''}">
              <span class="touchkey">WHEN</span>
              <span>${esc(f.timing)}</span>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="eyebrow mono small muted" style="letter-spacing:.12em">BEFORE YOU ACT</div>
          <p class="muted small">${esc(beforeYouAct(f))}</p>
          <div class="row gap">
            <button class="btn small ghost" id="rephotoBtn">Re-photograph leaf</button>
          </div>
          <p class="muted small" id="saveMsg">${esc(state.saveStatus ?? '')}</p>
        </div>
      </div>
    </div>`;

  $('rephotoBtn').addEventListener('click', () => { showPanel('panel-diagnose'); $('fileInput').click(); });
}

function beforeYouAct(f) {
  const bits = [];
  if (state.lastDisease && state.lastTop3) {
    bits.push(`The diagnosis is ${fmt(state.lastDisease.prob * 100, 0)}% confident`);
    if (state.lastTop3[1]) bits.push(`and the second class is ${prettyLabel(state.lastTop3[1].label)}`);
    bits.push('. ');
  }
  bits.push('This is decision support, not a prescription: confirm with a local extension officer before applying agro-chemicals.');
  return bits.join(' ').replace(/\s+\.\s+/, '. ');
}

/* ------------------------------------------------------------------ *
 * History                                                             *
 * ------------------------------------------------------------------ */

/**
 * Every completed diagnosis is written to history automatically - the scan is
 * the thing worth keeping, and asking the user to press a button for it just
 * loses results. The row records the advisory as it stood at diagnosis time.
 */
async function autoSaveScan() {
  if (!state.lastDisease || !state.lastIrrigation) return;
  const f = fuse(state.lastIrrigation, state.lastDisease, CONFIDENCE_FLOOR);
  state.saveStatus = await saveScan(f);
  renderAdvisory();
  renderHistory();
}

/** @returns {Promise<string>} a human-readable status for the advisory card */
async function saveScan(f) {
  if (!state.lastDisease) return '';
  const row = {
    predicted_class: state.lastDisease.label,
    confidence: state.lastDisease.prob,
    top3: (state.lastTop3 || []).map(t => ({ label: t.label, prob: t.prob })),
    crop: cropOf(state.lastDisease.label),
    irrigate: f.irrigate,
    depth_mm: f.depthMm,
    notes: decisionSummary(f),
  };

  if (state.session) {
    const { error } = await auth.supabase.from('scans').insert({ ...row, user_id: state.session.user.id });
    return error ? `Could not save to history: ${error.message}` : 'Saved to your history.';
  }
  state.guestHistory.unshift({ ...row, id: crypto.randomUUID(), created_at: new Date().toISOString() });
  return 'Saved to this session’s history — sign in to keep it.';
}

async function fetchHistory() {
  if (!state.session) return state.guestHistory;
  const { data, error } = await auth.supabase
    .from('scans')
    .select('id, created_at, predicted_class, confidence, crop, irrigate, depth_mm, notes')
    .order('created_at', { ascending: false })
    .limit(50);
  if (error) { console.error(error); return []; }
  return data;
}

async function deleteScan(id) {
  if (state.session) {
    const { error } = await auth.supabase.from('scans').delete().eq('id', id);
    if (error) console.error(error);
  } else {
    state.guestHistory = state.guestHistory.filter(r => r.id !== id);
  }
  renderHistory();
}

async function clearHistory() {
  if (state.session) {
    const { error } = await auth.supabase.from('scans').delete().eq('user_id', state.session.user.id);
    if (error) console.error(error);
  } else {
    state.guestHistory = [];
  }
  renderHistory();
}

$('clearHistBtn').addEventListener('click', () => {
  if (confirm('Delete every saved scan? This cannot be undone.')) clearHistory();
});

function when(iso) {
  const d = new Date(iso);
  const date = d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' });
  const time = d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
  return `${date} · ${time}`;
}

async function renderHistory() {
  const box = $('historyOut');
  if (!box) return;
  const rows = await fetchHistory();

  $('histSub').textContent = state.session
    ? `${rows.length} scan${rows.length === 1 ? '' : 's'} on your account. Results only — no photograph is ever uploaded.`
    : 'Guest mode: history is kept in memory only and disappears when you reload.';
  $('clearHistBtn').disabled = rows.length === 0;

  if (!rows.length) {
    box.innerHTML = `
      <div class="emptystate">
        <div class="box"></div>
        <div class="title">No scans yet</div>
        <p>Diagnose a leaf and the result — class, confidence and the irrigation decision that
           followed — is kept here${state.session ? ' on your account' : ' for this session'}.</p>
        <button class="btn small" id="goDiagnose">Diagnose a leaf</button>
      </div>`;
    $('goDiagnose').addEventListener('click', () => showPanel('panel-diagnose'));
    return;
  }

  box.innerHTML = `
    <div class="histtable">
      <div class="histhead">
        <div>DATE</div><div>PREDICTED CLASS</div><div style="text-align:right">CONFIDENCE</div>
        <div>IRRIGATION DECISION</div><div></div>
      </div>
      ${rows.map(r => {
        const agent = pathogenFor(r.predicted_class);
        return `
        <div class="histrow">
          <div class="when">${esc(when(r.created_at))}</div>
          <div class="cls">
            <span>${esc(prettyLabel(r.predicted_class))}</span>
            <span class="agent">${esc(agent.agent === '-' ? 'no pathogen detected' : agent.agent)}</span>
          </div>
          <div class="conf">
            <span>${fmt(r.confidence * 100, 1)}%</span>
            <span class="track"><span class="fill" style="width:${fmt(r.confidence * 100, 0)}%"></span></span>
          </div>
          <div class="decision">${esc(r.notes || (r.irrigate ? `Irrigate ${fmt(r.depth_mm, 1)} mm` : 'Hold'))}</div>
          <button class="delbtn" data-id="${esc(r.id)}" title="Delete this scan" aria-label="Delete this scan">×</button>
        </div>`;
      }).join('')}
    </div>
    <p class="timing">${state.session
      ? 'Stored in Supabase under your account, protected by row-level security · results only, never the image'
      : 'Nothing is synced anywhere · sign in to keep history across sessions'}</p>`;

  box.querySelectorAll('.delbtn').forEach(b =>
    b.addEventListener('click', () => deleteScan(b.dataset.id)));
}

/* ------------------------------------------------------------------ *
 * About                                                               *
 * ------------------------------------------------------------------ */

function renderAbout() {
  const rows = [];
  const add = (name, m) => {
    if (!m) { rows.push(`<dt>${name}</dt><dd class="plain">not loaded</dd>`); return; }
    const meta = m.meta || {};
    const bits = [`kind: ${m.kind}`];
    if (m.labels) bits.push(`${m.labels.length} classes`);
    if (meta.test_accuracy != null) bits.push(`accuracy ${fmt(meta.test_accuracy * 100, 1)}%`);
    if (meta.test_f1 != null) bits.push(`F1 ${fmt(meta.test_f1, 3)}`);
    if (meta.test_macro_f1 != null) bits.push(`macro-F1 ${fmt(meta.test_macro_f1, 3)}`);
    if (meta.test_mae_mm != null) bits.push(`MAE ${meta.test_mae_mm} mm`);
    if (meta.test_r2 != null) bits.push(`R² ${meta.test_r2}`);
    rows.push(`<dt>${name}</dt><dd class="plain">${esc(bits.join(' · '))}</dd>`);
  };
  add('Disease classifier', state.diseaseModel);
  add('Irrigation classifier', state.irrClf);
  add('Irrigation depth regressor', state.irrReg);
  $('aboutOut').innerHTML = `<dl class="kv">${rows.join('')}</dl>`;
}

boot();
