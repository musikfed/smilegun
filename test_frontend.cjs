// Logic regressions for the real inline frontend. No browser or dependencies.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const html = fs.readFileSync(path.join(__dirname, 'templates', 'index.html'), 'utf8');
const script = html.match(/<script[^>]*>([\s\S]*?)<\/script>/)[1];
const hooks = `globalThis.game = {
  ready: () => configReady, startGame, stopGame, beginLevel, pollState,
  setSelectedLevel, setAim, updateCrosshairScale, updateHandCursor, tickTargets,
  tickBullets, registerHit, renderLoop,
  get state() { return { gameActive, timeLeft, score, currentLevel, selectedLevel,
    targets, bullets, currentAimPx, crosshairScale, cameraOn }; }
};`;

class Element {
  constructor(tag, attributes = '') {
    this.tag = tag;
    this.attributes = attributes;
    this.id = attributes.match(/\bid="([^"]+)"/)?.[1] || '';
    this.classes = new Set((attributes.match(/\bclass="([^"]+)"/)?.[1] || '').split(/\s+/).filter(Boolean));
    this.classList = {
      contains: name => this.classes.has(name),
      add: (...names) => names.forEach(name => this.classes.add(name)),
      remove: (...names) => names.forEach(name => this.classes.delete(name)),
      toggle: (name, force = !this.classes.has(name)) => {
        if (force) this.classes.add(name); else this.classes.delete(name);
        return force;
      },
    };
    this.dataset = {};
    const level = attributes.match(/\bdata-lvl="([^"]+)"/);
    if (level) this.dataset.lvl = level[1];
    this.disabled = /\sdisabled(?:\s|$)/.test(attributes);
    this.hidden = /\shidden(?:\s|$)/.test(attributes);
    this.style = {};
    this.value = attributes.match(/\bvalue="([^"]+)"/)?.[1] || '';
    this.textContent = '';
    this.children = [];
    this.listeners = new Map();
    this.paused = true;
    this.currentTime = 0;
    this.clientWidth = 1100;
    this.clientHeight = 900;
  }
  addEventListener(name, listener) {
    if (!this.listeners.has(name)) this.listeners.set(name, []);
    this.listeners.get(name).push(listener);
  }
  async dispatch(name, event = {}) {
    for (const listener of this.listeners.get(name) || []) await listener(event);
  }
  click() { return this.disabled ? Promise.resolve() : this.dispatch('click'); }
  appendChild(child) { this.children.push(child); child.parent = this; }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter(child => child !== this);
    this.parent = null;
  }
  closest(selector) {
    return selector.split(',').some(part => this.classes.has(part.trim().slice(1))) ? this : null;
  }
  cloneNode() {
    const clone = new Element(this.tag, this.attributes);
    clone.mediaLog = this.mediaLog;
    return clone;
  }
  play() {
    this.paused = false;
    if (this.mediaLog) this.mediaLog.push({ id: this.id, rate: this.playbackRate || 1 });
    return Promise.resolve();
  }
  pause() { this.paused = true; }
}

function fixture(config = {}) {
  const elements = [...html.matchAll(/<([\w-]+)\b([^>]*?)>/g)].map(match => new Element(match[1], match[2]));
  const byId = new Map(elements.filter(el => el.id).map(el => [el.id, el]));
  const audioPlays = [];
  for (const element of elements.filter(el => el.tag === 'audio')) element.mediaLog = audioPlays;
  const document = new Element('document');
  document.getElementById = id => byId.get(id) || null;
  document.querySelectorAll = selector => elements.filter(el => el.classes.has(selector.slice(1)));
  document.createElement = tag => new Element(tag);
  document.elementFromPoint = () => null;
  const window = new Element('window');
  window.innerWidth = 1440;
  window.innerHeight = 900;
  let now = 0;
  let nextId = 1;
  const timers = new Map();
  const serverConfig = { level_duration: 75, level_difficulty: 4,
    super_blink_gain: 0.30, super_decay_delay: 4, super_decay_rate: 0.10, best_score: 20, ...config };
  let serverState = { camera: true, hand_present: true, aim_x: 0.5, aim_y: 0.8,
    shot_id: 0, super_id: 0, blink_id: 0, super_charge: 0 };
  let cameraStopFails = false;
  const requests = [];
  const createTimer = (callback, delay, interval = 0) => {
    const id = nextId++;
    timers.set(id, { callback, due: now + delay, interval });
    return id;
  };
  const context = vm.createContext({
    document, window, console, performance: { now: () => now },
    setTimeout: (fn, delay = 0) => createTimer(fn, delay),
    clearTimeout: id => timers.delete(id),
    setInterval: (fn, delay) => createTimer(fn, delay, delay),
    clearInterval: id => timers.delete(id),
    // Animation frames are intentionally manual; timer assertions do not run a render loop.
    requestAnimationFrame: () => nextId++,
    alert: message => { throw new Error(message); },
    fetch: async (url, options = {}) => {
      requests.push({ url, options });
      let result;
      if (url === '/config') {
        if (options.method === 'POST') Object.assign(serverConfig, JSON.parse(options.body));
        result = { ...serverConfig };
      } else if (url === '/state') result = { ...serverState };
      else if (url === '/camera/start' || url === '/camera/stop') {
        result = url === '/camera/stop' && cameraStopFails ? { ok: false, error: 'Tracker stop timed out' } : { ok: true };
      } else if (url === '/calibrate' || url === '/calibrate/reset') result = { ok: true };
      else if (url === '/score') {
        serverConfig.best_score = Math.max(serverConfig.best_score, JSON.parse(options.body).score);
        result = { best_score: serverConfig.best_score };
      } else throw new Error(`Unexpected request ${url}`);
      return { ok: result.ok !== false, json: async () => result };
    },
  });
  assert(script.includes('  boot();'), 'Frontend boot hook must exist');
  vm.runInContext(script.replace('  boot();', `${hooks}\n  boot();`), context, { filename: 'templates/index.html' });

  async function flush() { for (let i = 0; i < 8; i++) await Promise.resolve(); }
  async function advance(milliseconds) {
    const end = now + milliseconds;
    while (true) {
      const pending = [...timers].filter(([, timer]) => timer.due <= end).sort((a, b) => a[1].due - b[1].due || a[0] - b[0])[0];
      if (!pending) break;
      const [id, timer] = pending;
      now = Math.max(now, timer.due);
      if (timer.interval) timer.due = now + timer.interval; else timers.delete(id);
      await timer.callback();
      await flush();
    }
    now = end;
    await flush();
  }
  return {
    game: context.game, document, byId, serverConfig, requests, advance, audioPlays,
    setCameraStopFailure: value => { cameraStopFails = value; },
    setState: patch => { serverState = { ...serverState, ...patch }; },
    poll: async patch => {
      serverState = { ...serverState, ...patch };
      await advance(20);
      await context.game.pollState();
    },
    // A suspended tab can advance the monotonic clock without executing callbacks.
    jumpClock: milliseconds => { now += milliseconds; },
  };
}

async function test(name, callback) {
  await callback();
  console.log(`PASS ${name}`);
}

async function main() {
  await test('boot loads persisted duration, difficulty, energy settings, and record quietly', async () => {
    const f = fixture();
    await f.game.ready();
    assert.equal(f.byId.get('menu').classList.contains('hidden'), false);
    assert.equal(Number(f.byId.get('level_duration').value), 75);
    assert.equal(Number(f.byId.get('level_difficulty').value), 4);
    assert.equal(f.game.state.selectedLevel, 3);
    assert.equal(f.byId.get('menu-best').textContent, 20);
    assert(f.byId.get('menu-level-info').textContent.includes('75'));
    assert.equal(Number(f.byId.get('super_blink_gain').value), 0.30);
    assert.equal(Number(f.byId.get('super_decay_delay').value), 4);
    assert.equal(Number(f.byId.get('super_decay_rate').value), 0.10);
    assert.equal(f.audioPlays.length, 0, 'Boot and persisted selection must be silent');
  });

  await test('stop during countdown prevents resurrection and restart creates one active game', async () => {
    const f = fixture();
    await f.game.startGame();
    assert.equal(f.game.state.gameActive, false);
    assert.equal(f.byId.get('btn-game-stop').disabled, false);
    await f.advance(600);
    await f.byId.get('btn-game-stop').click();
    await f.advance(3000);
    assert.equal(f.game.state.gameActive, false);
    assert.equal(f.game.state.targets.length, 0);
    assert.equal(f.byId.get('menu').classList.contains('hidden'), false);
    await f.game.startGame();
    await f.advance(1800);
    assert.equal(f.game.state.gameActive, true);
    assert.equal(f.game.state.timeLeft, 75);
    assert.equal(f.game.state.targets.length, 3);
    assert.equal(f.audioPlays.filter(sound => sound.id === 'sound-countdown').length, 5);
    await f.advance(1000);
    assert.equal(f.game.state.timeLeft, 74);
    await f.document.dispatch('keydown', { key: 'Escape' });
    assert.equal(f.game.state.gameActive, false);
    assert.equal(f.game.state.cameraOn, true, 'Stopping gameplay must leave camera independently controllable');
  });

  await test('difficulty changes target count, size, movement; sliders save for next level', async () => {
    const f = fixture({ level_difficulty: 1 });
    await f.game.startGame();
    await f.advance(1800);
    const easyTarget = f.game.state.targets[0];
    assert.equal(easyTarget.r, 34);
    assert.equal(Number(f.byId.get('hud-left').textContent), 5);
    const easyX = easyTarget.x;
    f.game.tickTargets();
    assert.equal(easyTarget.x, easyX);
    f.byId.get('level_difficulty').value = '5';
    await f.byId.get('level_difficulty').dispatch('input');
    f.byId.get('level_duration').value = '10';
    await f.byId.get('level_duration').dispatch('input');
    await f.advance(80);
    assert.equal(f.serverConfig.level_difficulty, 5);
    assert.equal(f.serverConfig.level_duration, 10);
    assert.equal(f.game.state.currentLevel, 0, 'Changing selection must not mutate the active level');
    assert.equal(f.game.state.timeLeft, 75);
    f.game.stopGame();
    await f.game.startGame();
    await f.advance(1800);
    const hardTarget = f.game.state.targets[0];
    assert.equal(hardTarget.r, 21);
    assert.equal(Number(f.byId.get('hud-left').textContent), 15);
    assert.equal(f.game.state.timeLeft, 10);
    const hardX = hardTarget.x;
    f.game.tickTargets();
    assert(Math.abs(hardTarget.x - hardX) > 1);
  });

  await test('timer uses elapsed deadline and expires without an extra gameplay timer', async () => {
    const f = fixture({ level_duration: 10 });
    await f.game.startGame();
    await f.advance(1800);
    // Simulate delayed interval delivery by making a state poll wait for a clock jump.
    f.jumpClock(5000);
    await f.advance(100);
    assert.equal(f.game.state.timeLeft, 5);
    await f.advance(5000);
    assert.equal(f.game.state.gameActive, false);
    assert.equal(f.game.state.timeLeft, 0);
    assert.equal(f.game.state.targets.length, 0);
    assert.equal(f.byId.get('overlay').classList.contains('hidden'), false);
    assert(f.audioPlays.some(sound => sound.id === 'sound-gameover'));
  });

  await test('real state polling creates bullets and super clears targets with score', async () => {
    const f = fixture();
    await f.game.startGame();
    await f.advance(1800);
    await f.poll({ shot_id: 0 });
    await f.poll({ shot_id: 1 });
    assert.equal(f.game.state.bullets.length, 1);
    assert.equal(Number(f.byId.get('hud-shots').textContent), 1);
    const bullet = f.game.state.bullets[0];
    const target = f.game.state.targets[0];
    bullet.x = target.x;
    bullet.y = target.y + 9;
    f.game.tickBullets();
    assert.equal(f.game.state.score, 10, 'A real bullet collision must add score');
    assert.equal(f.game.state.targets.length, 2);
    await f.poll({ shot_id: 2 });
    f.game.state.bullets[0].y = -30;
    f.game.tickBullets();
    assert(f.audioPlays.some(sound => sound.id === 'sound-miss'));
    await f.poll({ super_id: 1 });
    assert.equal(f.game.state.targets.length, 0);
    assert.equal(f.game.state.score, 50, 'Super destroys two targets and adds its bonus');
    for (const id of ['blaster', 'sound-hit', 'super']) assert(f.audioPlays.some(sound => sound.id === id));
    await f.advance(500);
    assert.equal(f.game.state.targets.length, 3);
    f.game.stopGame();
    await f.advance(1000);
    assert.equal(f.game.state.targets.length, 0, 'Delayed target work cannot survive stop');
    assert.equal(f.game.state.bullets.length, 0);
    assert.equal(f.serverConfig.best_score, 50);
  });

  await test('energy controls persist; blink IDs produce one pitched pulse and decay has no loop', async () => {
    const f = fixture();
    await f.game.startGame();
    await f.advance(1800);
    await f.poll({ blink_id: 0, super_charge: 0 });
    await f.poll({ blink_id: 1, super_charge: 0.3, charging: true });
    assert.equal(f.byId.get('hud-super').textContent, '30%');
    assert.equal(f.audioPlays.filter(sound => sound.id === 'charge').length, 1);
    const firstPitch = f.audioPlays.find(sound => sound.id === 'charge').rate;
    await f.poll({ blink_id: 1, super_charge: 0.3 });
    assert.equal(f.audioPlays.filter(sound => sound.id === 'charge').length, 1, 'Repeated state cannot replay the same blink');
    await f.poll({ super_charge: 0.2, charging: false });
    assert.equal(f.byId.get('hud-super-hint').textContent, '\u0417\u0410\u0420\u042f\u0414 \u0423\u0422\u0415\u041a\u0410\u0415\u0422');
    await f.poll({ blink_id: 2, super_charge: 0.5, charging: true });
    assert.equal(f.audioPlays.filter(sound => sound.id === 'charge').length, 2);
    assert(f.audioPlays.filter(sound => sound.id === 'charge')[1].rate > firstPitch);
    assert(!/\sloop(?:\s|$)/.test(f.byId.get('charge').attributes));
    for (const [id, value] of [['super_blink_gain', '0.5'], ['super_decay_delay', '6'], ['super_decay_rate', '0.02']]) {
      f.byId.get(id).value = value;
      await f.byId.get(id).dispatch('input');
    }
    await f.advance(80);
    assert.equal(f.serverConfig.super_blink_gain, 0.5);
    assert.equal(f.serverConfig.super_decay_delay, 6);
    assert.equal(f.serverConfig.super_decay_rate, 0.02);
  });

  await test('camera stop failure remains visible and success and calibration have separate sounds', async () => {
    const f = fixture();
    await f.game.startGame();
    f.setCameraStopFailure(true);
    await f.byId.get('btn-stop').click();
    assert.equal(f.game.state.cameraOn, true);
    assert.equal(f.byId.get('btn-stop').disabled, false);
    assert.equal(f.byId.get('s-error').textContent, 'Tracker stop timed out');
    assert(!f.audioPlays.some(sound => sound.id === 'sound-camera_off'));
    f.setCameraStopFailure(false);
    await f.byId.get('btn-stop').click();
    assert.equal(f.game.state.cameraOn, false);
    assert(f.audioPlays.some(sound => sound.id === 'sound-camera_off'));
    await f.byId.get('btn-calibrate').click();
    assert(f.audioPlays.some(sound => sound.id === 'sound-calibrate'));
  });

  await test('level completion and final victory use distinct sounds', async () => {
    for (const [difficulty, goal, expectedSound] of [[1, 5, 'sound-level_complete'], [5, 15, 'sound-victory']]) {
      const f = fixture({ level_difficulty: difficulty });
      await f.game.startGame();
      await f.advance(1800);
      for (let i = 0; i < goal; i++) f.game.registerHit({});
      assert.equal(f.game.state.gameActive, false);
      assert(f.audioPlays.some(sound => sound.id === expectedSound));
    }
  });

  await test('60Hz and 120Hz render steps travel equally; tab stalls clamp movement', async () => {
    async function movingFixture() {
      const f = fixture({ level_difficulty: 3 });
      await f.game.startGame();
      await f.advance(1800);
      await f.poll({ shot_id: 0 });
      await f.poll({ shot_id: 1 });
      f.game.renderLoop(2000);
      for (const target of f.game.state.targets) target.y = 100;
      const target = f.game.state.targets[0];
      target.x = 450;
      target.vx = 1;
      const bullet = f.game.state.bullets[0];
      bullet.x = 450;
      bullet.y = 850;
      return { f, target, bullet };
    }
    const hz60 = await movingFixture();
    const hz120 = await movingFixture();
    const frame = 1000 / 60;
    hz60.f.game.renderLoop(2000 + frame);
    hz120.f.game.renderLoop(2000 + frame / 2);
    hz120.f.game.renderLoop(2000 + frame);
    assert(Math.abs(hz60.target.x - hz120.target.x) < 1e-9);
    assert(Math.abs(hz60.bullet.y - hz120.bullet.y) < 1e-9);
    assert(Math.abs(hz60.target.x - 450.6) < 1e-9);
    assert(Math.abs(hz60.bullet.y - 841) < 1e-9);
    const previousX = hz60.target.x;
    const previousY = hz60.bullet.y;
    hz60.f.game.renderLoop(12000 + frame);
    assert(Math.abs(hz60.target.x - previousX - 1.8) < 1e-9);
    assert(Math.abs(hz60.bullet.y - previousY + 27) < 1e-9);
  });

  await test('aim stays inside all edges at enlarged scale and null dwell is safe', async () => {
    const f = fixture();
    await f.game.ready();
    f.jumpClock(Math.PI * 60);
    f.game.updateCrosshairScale({ super_ready: true });
    assert(Math.abs(f.game.state.crosshairScale - 1.8) < 1e-9);
    for (const [x, y] of [[0, 0], [0, 1], [1, 0], [1, 1], [-1, 2]]) {
      f.game.setAim(x, y);
      const { currentAimPx: p, crosshairScale: scale } = f.game.state;
      const radius = 17 * scale;
      assert(p.x - radius >= 0 && p.x + radius <= 900);
      assert(p.y - radius >= 0 && p.y + radius <= 950);
      assert.equal(f.byId.get('crosshair').style.transform, `translate(${p.x}px, ${p.y}px) scale(${scale})`);
    }
    f.game.updateHandCursor({ hand_present: true, hand_x: 0.5, hand_y: 0.5 });
    f.jumpClock(1000);
    assert.doesNotThrow(() => f.game.updateHandCursor({ hand_present: true, hand_x: 0.5, hand_y: 0.5 }));
  });
  console.log('10 frontend logic regressions passed (Node VM; no browser layout validation).');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
