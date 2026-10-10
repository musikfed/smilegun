(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const arena = $("arena"), preview = $("preview"), canvas = $("capture"), ctx = canvas.getContext("2d", {alpha:false});
  const crosshair = $("crosshair"), targetsEl = $("targets"), shotFx = $("shot-fx"), menu = $("menu"), roundOver = $("round-over");
  const cameraSelect = $("camera-select");

  let cfg = {camera_width:640,camera_height:480,analysis_width:416,analysis_fps:15,jpeg_quality:.68,level_duration:75,level_difficulty:1,target_radius:34,best_score:0};
  let stream = null, cameraOn = false, frameBusy = false, lastFrameAt = 0, latestState = null;
  let recognition = null, voiceWanted = false, voiceRunning = false, voiceCooldownUntil = 0, lastVoiceKey = "";
  let gameActive = false, timerId = null, timeLeft = 75, score = 0, best = 0, level = 1;
  let lastHandShotId = 0, lastVoiceShotId = 0, lastManualShotId = 0, lastMouthShotId = 0;
  let lastBlinkId = 0, lastSuperId = 0, lastChargeId = 0;
  let aimX = .5, aimY = .5, currentCharge = 0;

  const sounds = {};
  for (const name of ["blaster","charge","super","hit","miss","ready","start","stop","select","victory","gameover"]) {
    sounds[name] = new Audio(`/static/sounds/${name}.wav`);
    sounds[name].preload = "auto";
  }
  function sound(name, rate=1, volume=.62){
    const src=sounds[name]; if(!src)return;
    const a=src.cloneNode(); a.playbackRate=Math.max(.55,Math.min(2.2,rate)); a.volume=Math.max(0,Math.min(1,volume)); a.play().catch(()=>{});
  }

  function setError(text=""){ $("menu-error").textContent=text; }
  function updateBest(){ $("best").textContent=best; $("menu-best").textContent=best; }
  function postJSON(url, body){ return fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}).then(r=>r.json()); }

  async function loadConfig(){
    try { cfg = {...cfg, ...(await fetch("/api/config",{cache:"no-store"}).then(r=>r.json()))}; } catch(_){}
    best = Number(cfg.best_score)||0; level = Number(cfg.level_difficulty)||1; timeLeft=Number(cfg.level_duration)||75;
    updateBest(); buildLevels(); updateLevelInfo(); updateReticle(0);
  }

  function buildLevels(){
    const box=$("levels"); box.innerHTML="";
    for(let i=1;i<=5;i++){
      const b=document.createElement("button"); b.className="level-btn"+(i===level?" active":""); b.textContent=i;
      b.onclick=()=>{level=i; cfg.level_difficulty=i; buildLevels(); updateLevelInfo(); postJSON("/api/config",{level_difficulty:i}).catch(()=>{}); sound("select");};
      box.appendChild(b);
    }
  }
  function targetCount(){ return 4 + level*2; }
  function updateLevelInfo(){ $("level-info").textContent=`${targetCount()} мишеней · ${level>=3?"часть движется":"мишени стоят"} · ${cfg.level_duration} с`; }

  async function listCameras(){
    try{
      const devices=await navigator.mediaDevices.enumerateDevices();
      const cams=devices.filter(d=>d.kind==="videoinput");
      const current=cameraSelect.value;
      cameraSelect.innerHTML="";
      cams.forEach((d,i)=>{const o=document.createElement("option");o.value=d.deviceId;o.textContent=d.label||`Камера ${i+1}`;cameraSelect.appendChild(o);});
      if(cams.some(c=>c.deviceId===current)) cameraSelect.value=current;
    }catch(_){}
  }

  async function startCamera(deviceId=""){
    setError("");
    if(stream) stopCamera();
    const video={width:{ideal:Number(cfg.camera_width)||640},height:{ideal:Number(cfg.camera_height)||480},frameRate:{ideal:30,max:30}};
    if(deviceId) video.deviceId={exact:deviceId}; else video.facingMode="user";
    try{
      stream=await navigator.mediaDevices.getUserMedia({video,audio:false});
      preview.srcObject=stream; await preview.play(); cameraOn=true;
      $("camera-status").textContent="вкл"; $("camera-btn").textContent="СТОП КАМЕРА";
      await listCameras();
      const track=stream.getVideoTracks()[0]; const settings=track.getSettings();
      if(settings.deviceId) cameraSelect.value=settings.deviceId;
      sound("ready"); return true;
    }catch(err){
      cameraOn=false; $("camera-status").textContent="ошибка";
      const msg = err && err.name === "NotAllowedError" ? "Браузер не получил разрешение на камеру. Нажми значок камеры у адреса и разреши доступ." : `Камера: ${err?.message||err}`;
      setError(msg); return false;
    }
  }
  function stopCamera(){
    if(stream){ for(const t of stream.getTracks()) t.stop(); }
    stream=null; preview.srcObject=null; cameraOn=false; $("camera-status").textContent="выкл"; $("camera-btn").textContent="КАМЕРА"; crosshair.classList.add("off");
  }
  cameraSelect.addEventListener("change",()=>{ if(cameraOn) startCamera(cameraSelect.value); });
  $("camera-btn").onclick=()=> cameraOn ? stopCamera() : startCamera(cameraSelect.value);

  function frameDimensions(){
    const w=Math.max(240,Math.min(640,Number(cfg.analysis_width)||416));
    const aspect=(preview.videoWidth&&preview.videoHeight)?preview.videoHeight/preview.videoWidth:.75;
    return [w,Math.max(180,Math.round(w*aspect))];
  }
  function sendFrame(now){
    if(!cameraOn||frameBusy||preview.readyState<2)return;
    const interval=1000/Math.max(5,Math.min(30,Number(cfg.analysis_fps)||15));
    if(now-lastFrameAt<interval)return;
    lastFrameAt=now; frameBusy=true;
    const [w,h]=frameDimensions(); if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
    ctx.save(); ctx.translate(w,0); ctx.scale(-1,1); ctx.drawImage(preview,0,0,w,h); ctx.restore();
    canvas.toBlob(async blob=>{
      try{
        if(!blob) throw new Error("canvas encode failed");
        const r=await fetch("/api/frame",{method:"POST",headers:{"Content-Type":"image/jpeg"},body:blob,cache:"no-store"});
        if(!r.ok) throw new Error(`HTTP ${r.status}`);
        applyState(await r.json());
      }catch(err){ $("vision-status").textContent="нет связи"; }
      finally{ frameBusy=false; }
    },"image/jpeg",Math.max(.35,Math.min(.95,Number(cfg.jpeg_quality)||.68)));
  }

  function normalizeSpeech(text){ return String(text||"").toLowerCase().replace(/ё/g,"е").replace(/[^a-zа-я]+/g," ").trim(); }
  function editDistance(a,b){
    const m=a.length,n=b.length,dp=Array.from({length:m+1},()=>Array(n+1).fill(0));
    for(let i=0;i<=m;i++)dp[i][0]=i; for(let j=0;j<=n;j++)dp[0][j]=j;
    for(let i=1;i<=m;i++)for(let j=1;j<=n;j++)dp[i][j]=Math.min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+(a[i-1]===b[j-1]?0:1));
    return dp[m][n];
  }
  function speechCommand(text){
    const s=normalizeSpeech(text); if(!s)return null;
    const fireWords=new Set(["пиу","пью","пию","пэу","пеу","пю","пьюу","pew"]);
    const chargeWords=new Set(["пым","пим","пум","пэм","пем","бым","бим","бум","дым","тым","пын","пин","пун","pym"]);
    const tokens=s.split(/\s+/).filter(Boolean);
    for(const t of tokens){
      if(fireWords.has(t))return "fire";
      if(chargeWords.has(t))return "charge";
      if(t.length>=2&&t.length<=4){
        if(editDistance(t,"пиу")<=1)return "fire";
        if(editDistance(t,"пым")<=1)return "charge";
      }
    }
    return null;
  }
  function setupVoice(){
    const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
    if(!SR){ $("voice-status").textContent="нет API"; return false; }
    recognition=new SR(); recognition.lang="ru-RU"; recognition.continuous=true; recognition.interimResults=true; recognition.maxAlternatives=3;
    recognition.onstart=()=>{voiceRunning=true; $("voice-status").textContent="слушаю";};
    recognition.onerror=e=>{ $("voice-status").textContent=e.error||"ошибка"; };
    recognition.onend=()=>{voiceRunning=false;if(voiceWanted)setTimeout(()=>{try{recognition.start();}catch(_){}},160);else $("voice-status").textContent="выкл";};
    recognition.onresult=e=>{
      for(let i=e.resultIndex;i<e.results.length;i++){
        const result=e.results[i]; let chosen=result[0]?.transcript||"", cmd=null;
        for(let j=0;j<result.length;j++){ const candidate=result[j]?.transcript||""; const c=speechCommand(candidate); if(c){chosen=candidate;cmd=c;break;} }
        $("heard").textContent=chosen||"—";
        const now=performance.now(); const key=`${cmd||""}:${normalizeSpeech(chosen)}`;
        if(cmd&&now>=voiceCooldownUntil&&(key!==lastVoiceKey||result.isFinal)){
          voiceCooldownUntil=now+520; lastVoiceKey=key; $("voice-status").textContent=cmd==="charge"?"ПЫМ ✓":"ПИУ ✓"; command(cmd,"voice");
          setTimeout(()=>{if(voiceRunning)$("voice-status").textContent="слушаю";},260);
        }
        if(result.isFinal) setTimeout(()=>{lastVoiceKey="";},180);
      }
    };
    return true;
  }
  function toggleVoice(){
    if(!recognition&&!setupVoice())return;
    voiceWanted=!voiceWanted; $("voice-btn").textContent=voiceWanted?"СТОП ГОЛОС":"ГОЛОС";
    if(voiceWanted&&!voiceRunning){try{recognition.start();}catch(_){}} else if(!voiceWanted&&voiceRunning){try{recognition.stop();}catch(_){}}
  }
  $("voice-btn").onclick=toggleVoice;

  async function command(action,source="key"){
    try{ const s=await postJSON("/api/action",{action,source}); applyState(s); }catch(_){}
  }

  function reticleSizeFor(charge){
    const base=34; const max=Math.max(72,Math.min(156,Math.min(arena.clientWidth||600,arena.clientHeight||500)*.30));
    return Math.round(base+(max-base)*Math.pow(Math.max(0,Math.min(1,charge)),.78));
  }
  function updateReticle(charge){
    currentCharge=Math.max(0,Math.min(1,Number(charge)||0));
    const size=reticleSizeFor(currentCharge), shake=(currentCharge*2.8).toFixed(1);
    crosshair.style.setProperty("--reticle-size",`${size}px`);
    crosshair.style.setProperty("--shake",`${shake}px`);
    crosshair.style.setProperty("--shake-neg",`${-Number(shake)}px`);
    const core=crosshair.querySelector(".reticle-core"); if(core)core.style.animationDuration=`${Math.max(.065,.20-currentCharge*.12)}s`;
  }
  function chargePulse(charge){
    updateReticle(charge); crosshair.classList.remove("charge-pop"); void crosshair.offsetWidth; crosshair.classList.add("charge-pop"); setTimeout(()=>crosshair.classList.remove("charge-pop"),180);
    sound("charge",.72+charge*.95,.48+charge*.25);
  }

  function applyState(s){
    latestState=s; $("vision-status").textContent=`${s.processing_ms||0}ms · ${s.hand_present?"рука":"—"} · ${s.face?"лицо":"—"}`;
    const eye=Number(s.blink_score)||0, th=Number(s.blink_threshold)||0, base=Number(s.eye_baseline)||0;
    $("eye-status").textContent=s.face?`${eye.toFixed(3)} / ${th.toFixed(3)}${base?` · база ${base.toFixed(3)}`:""}`:"—";
    const mouth=Number(s.mouth_score)||0, mouthTh=Number(s.mouth_threshold)||0;
    $("mouth-status").textContent=s.face?`${mouth.toFixed(3)} / ${mouthTh.toFixed(3)}${s.mouth_open?" · ОТКРЫТ":""}`:"—";
    aimX=Number.isFinite(s.aim_x)?s.aim_x:.5; aimY=Number.isFinite(s.aim_y)?s.aim_y:.5;
    crosshair.style.left=`${aimX*100}%`; crosshair.style.top=`${aimY*100}%`;
    crosshair.classList.toggle("off",!s.hand_present);
    const charge=Math.max(0,Math.min(1,Number(s.super_charge)||0)); updateReticle(charge);
    $("power-fill").style.width=`${Math.round(charge*100)}%`;
    $("power-text").textContent=s.super_ready?"100% · УЛЫБНИСЬ ДЛЯ СУПЕРА ИЛИ СТРЕЛЯЙ МАКСИМАЛЬНЫМ КВАДРАТОМ":`${Math.round(charge*100)}% · моргни или скажи «ПЫМ»`;
    if(Number(s.charge_id)>lastChargeId){ chargePulse(charge); lastChargeId=Number(s.charge_id)||lastChargeId; }
    if(Number(s.blink_id)>lastBlinkId) lastBlinkId=Number(s.blink_id)||lastBlinkId;

    // Every trigger has its own monotonic counter: voice, fist and mouth can
    // never gate each other.  This also prevents a frame response from hiding
    // an event produced by another input source.
    let id=Number(s.hand_shot_id)||0;
    if(id>lastHandShotId){for(let i=0;i<Math.min(3,id-lastHandShotId);i++)shoot(Number(s.hand_shot_power)||0,"hand");lastHandShotId=id;}
    id=Number(s.voice_shot_id)||0;
    if(id>lastVoiceShotId){for(let i=0;i<Math.min(3,id-lastVoiceShotId);i++)shoot(Number(s.voice_shot_power)||0,"voice");lastVoiceShotId=id;}
    id=Number(s.manual_shot_id)||0;
    if(id>lastManualShotId){for(let i=0;i<Math.min(3,id-lastManualShotId);i++)shoot(Number(s.manual_shot_power)||0,"manual");lastManualShotId=id;}
    id=Number(s.mouth_shot_id)||0;
    if(id>lastMouthShotId){for(let i=0;i<Math.min(2,id-lastMouthShotId);i++)tripleShoot(Number(s.mouth_shot_power)||0);lastMouthShotId=id;}

    if(Number(s.super_id)>lastSuperId){ superShot(); lastSuperId=Number(s.super_id)||lastSuperId; }
  }

  function spawnTargets(){
    targetsEl.innerHTML=""; const n=targetCount(); const radius=level>=1?34:30;
    const w=Math.max(220,arena.clientWidth),h=Math.max(180,arena.clientHeight),margin=radius+30,topMargin=52;
    for(let i=0;i<n;i++){
      const t=document.createElement("div"); t.className="target"+(level>=3&&i%2?" moving":"");
      t.dataset.value=String(level>=4&&i%3===0?150:100);
      const x=margin+Math.random()*Math.max(1,w-2*margin-50), y=topMargin+margin+Math.random()*Math.max(1,h-topMargin-2*margin);
      t.style.left=`${x}px`;t.style.top=`${y}px`;targetsEl.appendChild(t);
    }
  }
  function pulseReticle(kind){
    const cls=kind==="mouth"?"mouth-fire":(kind==="voice"||kind==="manual")?"voice-fire":"hand-fire";
    crosshair.classList.remove("hand-fire","voice-fire","mouth-fire");
    void crosshair.offsetWidth; crosshair.classList.add(cls);
    setTimeout(()=>crosshair.classList.remove(cls),kind==="mouth"?180:110);
  }
  function makeShotFx(size,kind,x=aimX,y=aimY){
    const fx=document.createElement("div"); fx.className=`shot-burst ${kind}`; fx.style.left=`${Math.max(0,Math.min(1,x))*100}%`;fx.style.top=`${Math.max(0,Math.min(1,y))*100}%`;fx.style.width=`${size}px`;fx.style.height=`${size}px`;shotFx.appendChild(fx);setTimeout(()=>fx.remove(),240);
  }
  function collectHits(boxes){
    const hits=new Set();
    for(const t of targetsEl.querySelectorAll(".target")){
      const r=t.getBoundingClientRect();
      for(const b of boxes){
        if(!(r.right<b.cx-b.half||r.left>b.cx+b.half||r.bottom<b.cy-b.half||r.top>b.cy+b.half)){hits.add(t);break;}
      }
    }
    return [...hits];
  }
  function scoreHits(hits,p){
    if(hits.length){
      for(const hit of hits){score+=Number(hit.dataset.value)||100;hit.classList.add("hit");setTimeout(()=>hit.remove(),130);} $("score").textContent=score; sound("hit",.92+p*.22,.68);
    } else sound("miss");
    if(!targetsEl.querySelector(".target:not(.hit)")) setTimeout(()=>finishRound(true),150);
  }
  function shoot(power=0,kind="manual"){
    if(!gameActive)return;
    const p=Math.max(0,Math.min(1,Number(power)||0)),size=reticleSizeFor(p);
    sound("blaster",kind==="hand"?1.30-p*.35:1.18-p*.42,.62+p*.18); makeShotFx(size,kind); pulseReticle(kind);
    const ar=arena.getBoundingClientRect(),cx=ar.left+aimX*ar.width,cy=ar.top+aimY*ar.height,half=size/2;
    scoreHits(collectHits([{cx,cy,half}]),p);
  }
  function tripleShoot(power=0){
    if(!gameActive)return;
    const p=Math.max(0,Math.min(1,Number(power)||0)),size=reticleSizeFor(p),ar=arena.getBoundingClientRect();
    const spread=Math.max(size*.72,42), spreadNorm=spread/Math.max(1,ar.width);
    const centers=[Math.max(0,aimX-spreadNorm),aimX,Math.min(1,aimX+spreadNorm)];
    sound("blaster",.72-p*.16,.82); setTimeout(()=>sound("blaster",.82-p*.12,.62),45); pulseReticle("mouth");
    const boxes=[];
    for(const x of centers){makeShotFx(size,"mouth",x,aimY);boxes.push({cx:ar.left+x*ar.width,cy:ar.top+aimY*ar.height,half:size/2});}
    scoreHits(collectHits(boxes),p);
  }
  function superShot(){
    if(!gameActive)return; sound("super"); $("flash").classList.remove("on"); void $("flash").offsetWidth; $("flash").classList.add("on");
    const ts=[...targetsEl.querySelectorAll(".target")]; score+=ts.reduce((a,t)=>a+(Number(t.dataset.value)||100),0); $("score").textContent=score; ts.forEach(t=>{t.classList.add("hit");setTimeout(()=>t.remove(),130);}); setTimeout(()=>finishRound(true),180);
  }

  function startRound(){
    gameActive=true; score=0; $("score").textContent="0"; timeLeft=Number(cfg.level_duration)||75; $("time").textContent=timeLeft; menu.classList.add("hidden"); roundOver.classList.add("hidden"); spawnTargets(); sound("start");
    clearInterval(timerId); timerId=setInterval(()=>{timeLeft--;$("time").textContent=timeLeft;if(timeLeft<=0)finishRound(false);},1000);
  }
  function finishRound(cleared){
    if(!gameActive)return; gameActive=false; clearInterval(timerId); timerId=null;
    if(score>best){best=score;updateBest();postJSON("/api/score",{score}).catch(()=>{});} sound(cleared?"victory":"gameover");
    $("round-title").textContent=cleared?"МИШЕНИ УНИЧТОЖЕНЫ":"ВРЕМЯ ВЫШЛО"; $("round-sub").textContent=`СЧЁТ ${score} · РЕКОРД ${best}`; roundOver.classList.remove("hidden");
  }

  async function play(){
    if(!cameraOn && !(await startCamera(cameraSelect.value)))return;
    await postJSON("/api/reset",{}).then(applyState).catch(()=>{});
    lastHandShotId=Number(latestState?.hand_shot_id)||0;lastVoiceShotId=Number(latestState?.voice_shot_id)||0;lastManualShotId=Number(latestState?.manual_shot_id)||0;lastMouthShotId=Number(latestState?.mouth_shot_id)||0;
    lastBlinkId=Number(latestState?.blink_id)||0;lastSuperId=Number(latestState?.super_id)||0;lastChargeId=Number(latestState?.charge_id)||0; startRound();
  }
  $("play-btn").onclick=play; $("again-btn").onclick=()=>{roundOver.classList.add("hidden");play();}; $("menu-btn").onclick=()=>{roundOver.classList.add("hidden");menu.classList.remove("hidden");};
  $("close-btn").onclick=async()=>{stopCamera();voiceWanted=false;if(recognition&&voiceRunning)try{recognition.stop();}catch(_){};try{await fetch("/shutdown",{method:"POST"});}catch(_){};$("menu-error").textContent="Сервер закрывается. Эту вкладку можно закрыть.";};

  document.addEventListener("keydown",e=>{
    if(e.code==="Space"){e.preventDefault();command("fire","keyboard");}
    else if(e.code==="KeyC")command("charge","keyboard");
    else if(e.code==="Escape"){gameActive=false;clearInterval(timerId);menu.classList.remove("hidden");roundOver.classList.add("hidden");}
  });
  window.addEventListener("resize",()=>{if(gameActive)spawnTargets();});
  window.addEventListener("beforeunload",()=>stopCamera());

  function loop(now){sendFrame(now);requestAnimationFrame(loop);} requestAnimationFrame(loop);
  loadConfig(); listCameras();
})();
