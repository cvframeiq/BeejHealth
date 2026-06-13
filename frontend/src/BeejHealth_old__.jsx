import React, { useState, useEffect, useRef, useCallback } from "react";

/* ── API Helper ── */
/* ── App Version — clears stale localStorage on update ── */
const APP_VERSION = '3.5';
(()=>{
  try {
    const stored = localStorage.getItem('bh_app_version');
    if (stored !== APP_VERSION) {
      // Clear old data on version change
      ['bh_token','bh_user','bh_latest_consult','bh_latest_crop','bh_view_consult','bh_sel_expert'].forEach(k=>localStorage.removeItem(k));
      localStorage.setItem('bh_app_version', APP_VERSION);
      console.log('BeejHealth v'+APP_VERSION+' — localStorage cleared');
    }
  } catch(e) {}
})();

const API = {
  BASE: "",   // Vite proxy handles /api → localhost:3000 in dev
  _headers(isJson = false) {
    const token = localStorage.getItem("bh_token");
    const h = {};
    if (isJson) h["Content-Type"] = "application/json";
    if (token)  h["Authorization"] = "Bearer " + token;
    return h;
  },
  async _fetch(method, path, body) {
    const opts = { method, headers: this._headers(!!body) };
    if (body) opts.body = JSON.stringify(body);
    const r    = await fetch(this.BASE + path, opts);
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || "Server error");
    return data;
  },
  get:    (path)        => API._fetch("GET",    path),
  post:   (path, body)  => API._fetch("POST",   path, body),
  patch:  (path, body)  => API._fetch("PATCH",  path, body),
  delete: (path)        => API._fetch("DELETE", path),
};

function saveSession(token, user) {
  localStorage.setItem("bh_token", token);
  localStorage.setItem("bh_user", JSON.stringify(user));
}
function clearSession() {
  localStorage.removeItem("bh_token");
  localStorage.removeItem("bh_user");
}
function loadSession() {
  try {
    const token = localStorage.getItem("bh_token");
    const raw   = localStorage.getItem("bh_user");
    const user  = raw ? JSON.parse(raw) : null;
    // Validate user object has required fields
    if (user && (!user?._id || !user?.name || !user?.type)) {
      // Stale/corrupt user data — clear it
      localStorage.removeItem("bh_token");
      localStorage.removeItem("bh_user");
      return { token: null, user: null };
    }
    return { token, user };
  } catch {
    localStorage.removeItem("bh_token");
    localStorage.removeItem("bh_user");
    return { token: null, user: null };
  }
}

/* ════════════════════════════════════════════════════════════════
   BEEJHEALTH — COMPLETE PRODUCTION APPLICATION
   Farmer Portal + Expert Portal + Full Auth + All Screens
════════════════════════════════════════════════════════════════ */

const APP_CSS = `
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Baloo+2:wght@400;500;600;700;800&display=swap');

*{margin:0;padding:0;box-sizing:border-box;}
:root{
  --g1:#0c3d1e; --g2:#155d30; --g3:#1e7e42; --g4:#2da05a; --g5:#4dbd7a; --g6:#7dd4a0;
  --gp:#eaf7ef; --gpb:#d2eedd; --gb:#f2fbf5;
  --b1:#0d3461; --b2:#1354a0; --b3:#1a6fd4; --b4:#4d9de8; --bp:#e8f2fc; --bpb:#cce0f8;
  --r1:#8b1a1a; --r2:#c62828; --r3:#ef5350; --rp:#fdecea; --rpb:#ffcdd2;
  --a1:#7c4700; --a2:#f0a500; --a3:#ffc940; --ap:#fff8e1;
  --pu:#6a1b9a; --pup:#f3e5f5;
  --t1:#003d35; --t2:#00695c; --t3:#26a69a; --tp:#e0f2f1;
  --tx:#0a1f0e; --tx2:#2d5238; --tx3:#6b8f72; --tx4:#a8c8af;
  --wh:#ffffff; --card:#ffffff; --br:#daeee0; --br2:#b8d9c2;
  --sh:0 2px 16px rgba(12,61,30,.08); --sh2:0 8px 40px rgba(12,61,30,.14); --sh3:0 20px 64px rgba(12,61,30,.20);
  --rad:14px; --rad2:10px; --rad3:20px; --rad4:28px;
}
html{scroll-behavior:smooth;}
body{font-family:'Outfit',sans-serif;background:var(--gb);color:var(--tx);min-height:100vh;overflow-x:hidden;}
button,input,select,textarea{font-family:'Outfit',sans-serif;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--gp);}
::-webkit-scrollbar-thumb{background:var(--g5);border-radius:10px;}
::selection{background:var(--gpb);color:var(--g1);}

/* ── KEYFRAMES ── */
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes slideUp{from{opacity:0;transform:translateY(22px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideDown{from{opacity:0;transform:translateY(-12px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideRight{from{opacity:0;transform:translateX(-18px)}to{opacity:1;transform:translateX(0)}}
@keyframes slideLeft{from{opacity:0;transform:translateX(18px)}to{opacity:1;transform:translateX(0)}}
@keyframes scaleIn{from{opacity:0;transform:scale(.93)}to{opacity:1;transform:scale(1)}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
@keyframes bounce{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
@keyframes shimmer{0%{opacity:.6}50%{opacity:1}100%{opacity:.6}}
@keyframes growW{from{width:0}to{width:100%}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
@keyframes ringPulse{0%{box-shadow:0 0 0 0 rgba(77,189,122,.4)}70%{box-shadow:0 0 0 10px rgba(77,189,122,0)}100%{box-shadow:0 0 0 0 rgba(77,189,122,0)}}

/* ── UTILS ── */
.fade-in{animation:fadeIn .3s ease}
.slide-up{animation:slideUp .35s ease}
.slide-right{animation:slideRight .3s ease}
.scale-in{animation:scaleIn .28s ease}

/* ── NAVBAR ── */
.nav{
  position:fixed;top:0;left:0;right:0;z-index:900;height:62px;
  background:rgba(255,255,255,.97);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
  border-bottom:1.5px solid var(--br);
  display:flex;align-items:center;padding:0 28px;gap:20px;
  box-shadow:0 1px 14px rgba(12,61,30,.06);
}
.nav-logo{display:flex;align-items:center;gap:10px;cursor:pointer;text-decoration:none;flex-shrink:0;}
.nav-logo-mark{
  width:36px;height:36px;border-radius:10px;
  background:linear-gradient(135deg,var(--g3),var(--g5));
  display:flex;align-items:center;justify-content:center;font-size:18px;
  box-shadow:0 2px 10px rgba(30,126,66,.28);
}
.nav-logo-txt{font-family:'Baloo 2',cursive;font-size:21px;font-weight:800;color:var(--g1);letter-spacing:-.3px;}
.nav-links{display:flex;gap:2px;margin-left:28px;}
.nav-a{
  padding:7px 15px;border-radius:8px;font-size:13.5px;font-weight:600;
  color:var(--tx2);background:none;border:none;cursor:pointer;transition:all .18s;position:relative;
}
.nav-a:hover{color:var(--g3);background:var(--gp);}
.nav-a.on{color:var(--g2);}
.nav-a.on::after{content:'';position:absolute;bottom:0;left:12px;right:12px;height:2.5px;background:var(--g4);border-radius:2px 2px 0 0;}
.nav-right{margin-left:auto;display:flex;align-items:center;gap:10px;}
.nav-bell{
  width:36px;height:36px;border-radius:9px;background:var(--gp);
  border:1.5px solid var(--br);display:flex;align-items:center;justify-content:center;
  font-size:16px;cursor:pointer;position:relative;transition:all .18s;
}
.nav-bell:hover{background:var(--gpb);}
.nav-bell-dot{position:absolute;top:7px;right:7px;width:7px;height:7px;background:var(--r2);border-radius:50%;border:1.5px solid white;}
.nav-av{
  width:35px;height:35px;border-radius:50%;
  background:linear-gradient(135deg,var(--g4),var(--g6));
  display:flex;align-items:center;justify-content:center;
  color:white;font-weight:800;font-size:12px;cursor:pointer;
  border:2px solid white;box-shadow:0 2px 10px rgba(30,126,66,.22);transition:all .18s;
}
.nav-av:hover{transform:scale(1.05);}
.nav-av.ex-av{background:linear-gradient(135deg,var(--b3),var(--b4));}
.nav-btn-login{padding:8px 18px;border-radius:8px;font-size:13.5px;font-weight:700;color:var(--g3);background:none;border:2px solid var(--br2);cursor:pointer;transition:all .18s;}
.nav-btn-login:hover{border-color:var(--g3);background:var(--gp);}
.nav-btn-reg{padding:8px 18px;border-radius:8px;font-size:13.5px;font-weight:700;color:white;background:var(--g3);border:none;cursor:pointer;box-shadow:0 2px 12px rgba(30,126,66,.26);transition:all .18s;}
.nav-btn-reg:hover{background:var(--g2);transform:translateY(-1px);}
.dd-menu{
  position:absolute;top:48px;right:0;background:white;border-radius:var(--rad);
  box-shadow:var(--sh3);border:1.5px solid var(--br);min-width:218px;overflow:hidden;
  animation:scaleIn .2s ease;transform-origin:top right;z-index:999;
}
.dd-head{padding:14px 18px;border-bottom:1px solid var(--gp);}
.dd-name{font-size:15px;font-weight:800;color:var(--tx);}
.dd-sub{font-size:12px;color:var(--tx3);margin-top:2px;}
.dd-row{
  padding:11px 18px;font-size:13.5px;font-weight:600;color:var(--tx);
  display:flex;align-items:center;gap:10px;cursor:pointer;transition:background .15s;
}
.dd-row:hover{background:var(--gp);}
.dd-row.red-row{color:var(--r2);}
.dd-div{height:1px;background:var(--gp);}

/* ── LAYOUT ── */
.shell{display:flex;flex-direction:column;min-height:100vh;}
.pg{padding-top:62px;min-height:100vh;}
.pg-chat{padding-top:62px;height:100vh;overflow:hidden;}
.wrap{max-width:1160px;margin:0 auto;padding:40px 28px;}
.wrap-sm{max-width:760px;margin:0 auto;padding:40px 28px;}
.wrap-md{max-width:960px;margin:0 auto;padding:40px 28px;}

/* ── BUTTONS ── */
.btn{display:inline-flex;align-items:center;justify-content:center;gap:7px;font-weight:700;border-radius:10px;border:none;cursor:pointer;transition:all .18s;font-family:'Outfit',sans-serif;}
.btn:disabled{opacity:.5;cursor:not-allowed !important;transform:none !important;}
.btn-g{background:var(--g3);color:white;box-shadow:0 3px 14px rgba(30,126,66,.28);}
.btn-g:hover{background:var(--g2);transform:translateY(-1px);box-shadow:0 5px 20px rgba(30,126,66,.38);}
.btn-b{background:var(--b3);color:white;box-shadow:0 3px 14px rgba(26,111,212,.26);}
.btn-b:hover{background:var(--b2);transform:translateY(-1px);}
.btn-out{background:none;color:var(--g3);border:2px solid var(--br2);}
.btn-out:hover{border-color:var(--g3);background:var(--gp);}
.btn-out-b{background:none;color:var(--b3);border:2px solid var(--bpb);}
.btn-out-b:hover{border-color:var(--b3);background:var(--bp);}
.btn-ghost{background:var(--gp);color:var(--g2);border:none;}
.btn-ghost:hover{background:var(--gpb);}
.btn-red{background:var(--r2);color:white;}
.btn-red:hover{background:var(--r1);}
.btn-sm{padding:7px 16px;font-size:13px;}
.btn-md{padding:10px 22px;font-size:14px;}
.btn-lg{padding:13px 28px;font-size:15px;}
.btn-xl{padding:14px 32px;font-size:16px;}
.btn-full{width:100%;padding:13px;font-size:15px;}

/* ── FORMS ── */
.fgrp{margin-bottom:15px;}
.flbl{display:block;font-size:11px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:.7px;margin-bottom:6px;}
.finp,.fsel,.ftxt{
  width:100%;padding:11px 14px;border-radius:10px;
  border:1.5px solid var(--br);font-size:14px;font-family:'Outfit',sans-serif;
  color:var(--tx);background:var(--gb);outline:none;transition:all .18s;
}
.finp:focus,.fsel:focus,.ftxt:focus{border-color:var(--g4);background:white;box-shadow:0 0 0 3px rgba(77,189,122,.1);}
.finp::placeholder,.ftxt::placeholder{color:var(--tx4);}
.frow{display:grid;grid-template-columns:1fr 1fr;gap:13px;}
.ferr{font-size:12px;color:var(--r2);margin-top:5px;display:flex;align-items:center;gap:4px;}

/* ── CARDS ── */
.card{background:white;border-radius:var(--rad);border:1.5px solid var(--br);box-shadow:var(--sh);}
.card-hov{transition:all .22s;}
.card-hov:hover{transform:translateY(-3px);box-shadow:var(--sh2);border-color:var(--br2);}

/* ── BADGES ── */
.badge{display:inline-flex;align-items:center;gap:5px;padding:4px 10px;border-radius:100px;font-size:11.5px;font-weight:700;}
.bg-g{background:var(--gp);color:var(--g1);}
.bg-b{background:var(--bp);color:var(--b1);}
.bg-r{background:var(--rp);color:var(--r2);}
.bg-a{background:var(--ap);color:var(--a1);}
.bg-pu{background:var(--pup);color:var(--pu);}
.bg-t{background:var(--tp);color:var(--t1);}
.bg-gr{background:#f3f4f6;color:#374151;}

/* ── MODALS ── */
.overlay{position:fixed;inset:0;background:rgba(0,0,0,.48);z-index:1000;display:flex;align-items:center;justify-content:center;padding:16px;animation:fadeIn .2s ease;}
.modal{background:white;border-radius:var(--rad3);box-shadow:var(--sh3);animation:scaleIn .25s ease;position:relative;overflow:hidden;}
.modal-close{position:absolute;top:13px;right:13px;width:29px;height:29px;border-radius:8px;background:rgba(255,255,255,.2);border:none;color:white;font-size:15px;display:flex;align-items:center;justify-content:center;cursor:pointer;z-index:10;}

/* ── TOASTS ── */
.toast-wrap{position:fixed;bottom:26px;right:26px;z-index:9999;display:flex;flex-direction:column;gap:9px;}
.toast{display:flex;align-items:center;gap:11px;padding:13px 20px;background:var(--g1);color:white;border-radius:12px;font-size:13.5px;font-weight:600;box-shadow:0 8px 32px rgba(0,0,0,.22);animation:slideLeft .3s ease;min-width:260px;max-width:340px;}
.toast.err{background:var(--r2);}
.toast.inf{background:var(--b2);}
.toast.warn{background:var(--a1);}

/* ── SPINNER ── */
.spin{width:17px;height:17px;border:2.5px solid rgba(255,255,255,.3);border-top-color:white;border-radius:50%;animation:spin .65s linear infinite;}
.spin-g{border-color:var(--gp);border-top-color:var(--g4);}
.spin-lg{width:44px;height:44px;border-width:4px;}

/* ── PROGRESS ── */
.prog-bar{height:8px;background:var(--gp);border-radius:100px;overflow:hidden;}
.prog-fill{height:100%;border-radius:100px;background:linear-gradient(90deg,var(--g4),var(--g6));transition:width .7s ease;}

/* ── SWITCH ── */
.sw{position:relative;width:43px;height:23px;display:inline-block;flex-shrink:0;}
.sw input{opacity:0;width:0;height:0;}
.sw-sl{position:absolute;inset:0;background:var(--br);border-radius:100px;cursor:pointer;transition:.3s;}
.sw-sl::before{content:'';position:absolute;height:17px;width:17px;left:3px;top:3px;background:white;border-radius:50%;transition:.3s;box-shadow:0 1px 4px rgba(0,0,0,.18);}
.sw input:checked+.sw-sl{background:var(--g4);}
.sw input:checked+.sw-sl::before{transform:translateX(20px);}

/* ── OTP ── */
.otp-row{display:flex;gap:9px;justify-content:center;margin:18px 0;}
.otp-c{width:47px;height:54px;border-radius:12px;border:2px solid var(--br);font-size:21px;font-weight:800;text-align:center;color:var(--g1);background:var(--gb);outline:none;transition:all .18s;}
.otp-c:focus{border-color:var(--g4);background:white;box-shadow:0 0 0 3px rgba(77,189,122,.1);}

/* ── STEP WIZARD ── */
.steps-row{display:flex;align-items:center;gap:0;margin-bottom:26px;}
.step-dot{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;border:2px solid var(--br);background:white;color:var(--tx3);flex-shrink:0;transition:all .3s;}
.step-dot.done{background:var(--g4);border-color:var(--g4);color:white;}
.step-dot.act{background:white;border-color:var(--g4);color:var(--g3);box-shadow:0 0 0 3px rgba(77,189,122,.15);}
.step-ln{flex:1;height:2px;background:var(--br);margin:0 4px;transition:background .3s;}
.step-ln.done{background:var(--g4);}
.step-lbl{font-size:10.5px;font-weight:600;color:var(--tx3);margin-top:4px;}

/* ── TABS ── */
.tabs-row{display:flex;background:var(--gp);border-radius:10px;padding:4px;gap:2px;}
.tab-b{flex:1;padding:9px 14px;border-radius:8px;font-size:13.5px;font-weight:700;border:none;background:none;color:var(--tx2);cursor:pointer;transition:all .18s;}
.tab-b.on{background:white;color:var(--g2);box-shadow:0 1px 8px rgba(12,61,30,.13);}
.tabs-row.ex-tabs .tab-b.on{color:var(--b2);}

/* ══════════════════════════════════════════════════════════
   HOME PAGE
══════════════════════════════════════════════════════════ */
.hero{
  min-height:calc(100vh - 62px);display:flex;align-items:center;
  position:relative;overflow:hidden;
  background:linear-gradient(155deg,#eaf7ef 0%,#f6fef9 40%,#eaf3fa 100%);
}
.hero-in{max-width:1160px;margin:0 auto;padding:60px 28px;display:grid;grid-template-columns:1fr 1fr;gap:60px;align-items:center;width:100%;}
.hero-pill{display:inline-flex;align-items:center;gap:8px;padding:5px 14px;background:white;border:1.5px solid var(--br2);border-radius:100px;font-size:12px;font-weight:700;color:var(--g2);margin-bottom:18px;box-shadow:var(--sh);}
.hero-dot{width:7px;height:7px;background:var(--g5);border-radius:50%;animation:pulse 2s infinite;}
.hero-h1{font-family:'Baloo 2',cursive;font-size:56px;font-weight:900;line-height:1.04;color:var(--g1);margin-bottom:18px;letter-spacing:-1.5px;}
.hero-h1 em{font-style:normal;color:var(--g4);}
.hero-p{font-size:17px;color:var(--tx2);line-height:1.78;margin-bottom:34px;}
.hero-btns{display:flex;gap:14px;flex-wrap:wrap;}
.stats-row{display:flex;gap:36px;margin-top:42px;padding-top:24px;border-top:1.5px solid var(--br);}
.stat-n{font-family:'Baloo 2',cursive;font-size:28px;font-weight:900;color:var(--g1);}
.stat-l{font-size:11.5px;color:var(--tx3);margin-top:1px;}
.hero-card{background:white;border-radius:22px;box-shadow:var(--sh3);border:1.5px solid var(--br);padding:22px;animation:float 5s ease-in-out infinite;}
.hc-lbl{font-size:10.5px;font-weight:700;color:var(--tx3);text-transform:uppercase;letter-spacing:.7px;margin-bottom:13px;}
.dis-card{border-radius:14px;background:linear-gradient(135deg,var(--g2),var(--g1));padding:18px;color:white;margin-bottom:13px;}
.dc-crop{font-size:12.5px;opacity:.75;margin-bottom:5px;}
.dc-name{font-family:'Baloo 2',cursive;font-size:19px;font-weight:800;margin-bottom:2px;}
.dc-sci{font-size:11.5px;opacity:.6;margin-bottom:14px;}
.dc-bar-row{display:flex;justify-content:space-between;font-size:11.5px;font-weight:600;margin-bottom:5px;opacity:.9;}
.dc-bar{height:6px;background:rgba(255,255,255,.2);border-radius:100px;overflow:hidden;margin-bottom:10px;}
.dc-fill{height:100%;background:rgba(255,255,255,.75);border-radius:100px;}
.dc-pill{display:inline-flex;align-items:center;gap:5px;background:rgba(255,255,255,.14);border-radius:8px;padding:5px 11px;font-size:11.5px;font-weight:600;margin-top:10px;}

/* FEATURES */
.feat-sec{background:white;padding:80px 28px;}
.feat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;max-width:1160px;margin:36px auto 0;}
.feat-card{padding:26px;border-radius:var(--rad);border:1.5px solid var(--br);transition:all .22s;cursor:default;}
.feat-card:hover{border-color:var(--br2);transform:translateY(-4px);box-shadow:var(--sh2);}
.feat-icon{font-size:36px;margin-bottom:14px;}
.feat-t{font-size:16px;font-weight:800;color:var(--g1);margin-bottom:7px;}
.feat-p{font-size:13.5px;color:var(--tx2);line-height:1.65;}

/* ══════════════════════════════════════════════════════════
   FARMER ONBOARDING
══════════════════════════════════════════════════════════ */
.ob-wrap{min-height:100vh;background:linear-gradient(160deg,#eaf7ef,#f5fff8);display:flex;align-items:center;justify-content:center;padding:20px;}
.ob-box{width:100%;max-width:520px;background:white;border-radius:var(--rad4);box-shadow:var(--sh3);overflow:hidden;}
.ob-head{background:linear-gradient(135deg,var(--g4),var(--g2));padding:26px 30px;color:white;}
.ob-head.ex{background:linear-gradient(135deg,var(--b3),var(--b1));}
.ob-prog{display:flex;gap:5px;margin-bottom:26px;}
.ob-step{flex:1;height:4px;border-radius:10px;background:rgba(255,255,255,.3);transition:background .3s;}
.ob-step.done{background:rgba(255,255,255,.85);}
.ob-body{padding:24px 30px;}
.ob-sec-t{font-size:18px;font-weight:800;color:var(--g1);margin-bottom:5px;}
.ob-sec-p{font-size:13.5px;color:var(--tx2);margin-bottom:18px;}
.ob-sec-t.ex-t{color:var(--b1);}

/* ══════════════════════════════════════════════════════════
   FARMER DASHBOARD
══════════════════════════════════════════════════════════ */
.greet-card{
  background:linear-gradient(135deg,var(--g3) 0%,var(--g1) 100%);
  border-radius:var(--rad3);padding:26px 30px;color:white;margin-bottom:24px;
  position:relative;overflow:hidden;
}
.gc-ring1{position:absolute;right:-30px;top:-30px;width:180px;height:180px;background:rgba(255,255,255,.06);border-radius:50%;}
.gc-ring2{position:absolute;right:50px;bottom:-40px;width:120px;height:120px;background:rgba(255,255,255,.04);border-radius:50%;}
.gc-top{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:18px;}
.gc-name{font-family:'Baloo 2',cursive;font-size:24px;font-weight:800;margin-bottom:3px;}
.gc-sub{font-size:13.5px;opacity:.82;}
.gc-score-n{font-family:'Baloo 2',cursive;font-size:38px;font-weight:900;line-height:1;}
.gc-score-l{font-size:11px;opacity:.68;margin-top:2px;}
.gc-prog{height:7px;background:rgba(255,255,255,.2);border-radius:100px;overflow:hidden;margin-bottom:6px;}
.gc-prog-f{height:100%;background:rgba(255,255,255,.68);border-radius:100px;}
.gc-btns{display:flex;gap:10px;flex-wrap:wrap;}
.gc-btn{
  padding:9px 20px;border-radius:9px;font-size:13.5px;font-weight:700;cursor:pointer;
  border:2px solid rgba(255,255,255,.3);background:rgba(255,255,255,.12);color:white;
  transition:all .18s;backdrop-filter:blur(6px);
}
.gc-btn:hover{background:rgba(255,255,255,.24);border-color:rgba(255,255,255,.5);}
.gc-btn.prim{background:white;color:var(--g2);border-color:white;}
.gc-btn.prim:hover{background:var(--gp);}

.alert-bar{background:var(--rp);border:1.5px solid var(--rpb);border-radius:var(--rad2);padding:14px 18px;display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;}
.alert-txt{display:flex;align-items:center;gap:9px;font-size:13.5px;font-weight:700;color:var(--r2);}

.crops-scroll{display:flex;gap:13px;overflow-x:auto;padding-bottom:6px;margin-bottom:26px;scrollbar-width:none;}
.crops-scroll::-webkit-scrollbar{display:none;}
.crop-sc-card{flex-shrink:0;width:136px;padding:15px 12px;border-radius:var(--rad2);border:2px solid var(--br);background:white;text-align:center;cursor:pointer;transition:all .18s;}
.crop-sc-card:hover,.crop-sc-card.sel{border-color:var(--g4);background:var(--gp);}
.crop-sc-em{font-size:34px;margin-bottom:7px;}
.crop-sc-nm{font-size:12.5px;font-weight:700;color:var(--tx);margin-bottom:3px;}
.crop-sc-hl{font-size:11px;font-weight:600;}
.crop-sc-st{font-size:11px;color:var(--tx3);}

.dash-2{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-bottom:22px;}
.dash-r{display:flex;flex-direction:column;gap:18px;}

.cons-card{border-radius:var(--rad);border:1.5px solid var(--br);background:white;cursor:pointer;transition:all .22s;overflow:hidden;}
.cons-card:hover{border-color:var(--br2);transform:translateY(-2px);box-shadow:var(--sh2);}
.cons-img{height:72px;display:flex;align-items:center;justify-content:center;font-size:40px;background:linear-gradient(135deg,var(--gp),var(--gpb));}
.cons-body{padding:14px 16px;}
.cons-nm{font-size:14.5px;font-weight:800;color:var(--tx);margin-bottom:3px;}
.cons-issue{font-size:13px;color:var(--tx2);margin-bottom:10px;line-height:1.5;}
.cons-meta{font-size:12px;color:var(--tx3);display:flex;align-items:center;gap:5px;margin-bottom:3px;}
.cons-acts{display:flex;gap:7px;margin-top:12px;}
.ca-rep{flex:1;padding:8px;border-radius:8px;font-size:12.5px;font-weight:700;background:var(--g4);color:white;border:none;cursor:pointer;transition:background .18s;}
.ca-rep:hover{background:var(--g3);}
.ca-chat{flex:1;padding:8px;border-radius:8px;font-size:12.5px;font-weight:700;background:var(--gp);color:var(--g2);border:1.5px solid var(--br2);cursor:pointer;transition:all .18s;}
.ca-chat:hover{background:var(--gpb);}

.weather-card{padding:18px;border-radius:var(--rad);background:linear-gradient(135deg,#e8f4fd,#dbedf8);border:1.5px solid #b8d4ea;}
.wt-main{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;}
.wt-temp{font-family:'Baloo 2',cursive;font-size:38px;font-weight:800;color:var(--g1);}
.wt-risk{padding:5px 13px;border-radius:100px;font-size:12px;font-weight:700;}
.risk-low{background:var(--gp);color:var(--g2);}
.risk-med{background:var(--ap);color:var(--a1);}
.risk-high{background:var(--rp);color:var(--r2);}

.mandi-row{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid var(--gp);}
.mandi-row:last-child{border:none;}
.mandi-crop{font-size:13.5px;font-weight:600;color:var(--tx);}
.mandi-price{font-size:14.5px;font-weight:800;color:var(--g2);}
.mandi-ch{font-size:12px;font-weight:700;}
.ch-up{color:#16a34a;}.ch-dn{color:var(--r2);}.ch-st{color:var(--tx3);}

/* ══════════════════════════════════════════════════════════
   EXPERT DASHBOARD
══════════════════════════════════════════════════════════ */
.ed-head{
  background:linear-gradient(135deg,var(--b3),var(--b1));
  border-radius:var(--rad3);padding:26px 30px;color:white;margin-bottom:24px;position:relative;overflow:hidden;
}
.ed-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin-bottom:24px;}
.ed-stat{padding:18px;border-radius:var(--rad);background:white;border:1.5px solid var(--bpb);text-align:center;}
.ed-stat-n{font-family:'Baloo 2',cursive;font-size:28px;font-weight:900;color:var(--b3);}
.ed-stat-n.red{color:var(--r2);}
.ed-stat-l{font-size:12px;color:var(--tx3);font-weight:600;margin-top:4px;}
.case-card{border-radius:var(--rad);border:1.5px solid var(--br);background:white;padding:16px;margin-bottom:12px;cursor:pointer;transition:all .18s;}
.case-card:hover{border-color:var(--br2);transform:translateX(3px);}
.case-card.urg{border-left:4px solid var(--r2);border-color:var(--rpb);}
.case-card.med{border-left:4px solid var(--a2);}
.case-id{font-size:12px;font-weight:700;color:var(--tx3);}
.case-crop{font-size:14.5px;font-weight:800;color:var(--tx);margin:2px 0;}
.case-issue{font-size:13px;color:var(--tx2);margin-bottom:8px;}
.case-meta-row{display:flex;gap:14px;font-size:12px;color:var(--tx3);}

/* ══════════════════════════════════════════════════════════
   CONSULTATION PAGE
══════════════════════════════════════════════════════════ */
.method-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:26px;}
.method-card{padding:20px 12px;border-radius:var(--rad);border:2px solid var(--br);background:white;text-align:center;cursor:pointer;transition:all .18s;}
.method-card:hover{border-color:var(--br2);transform:translateY(-2px);}
.method-card.sel{border-color:var(--g4);background:var(--gp);}
.method-card.dis{opacity:.5;cursor:not-allowed;}
.method-icon{font-size:30px;margin-bottom:10px;}
.method-t{font-size:13.5px;font-weight:800;color:var(--tx);margin-bottom:3px;}
.method-s{font-size:12px;color:var(--tx3);}
.cs-tag{font-size:10px;font-weight:700;color:var(--a2);margin-top:3px;}

.crops-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:11px;}
.crop-tile{padding:14px 8px;border-radius:var(--rad2);border:2px solid var(--br);background:var(--gb);text-align:center;cursor:pointer;transition:all .18s;}
.crop-tile:hover{border-color:var(--br2);background:var(--gp);}
.crop-tile.sel{border-color:var(--g4);background:var(--gp);}
.ct-em{font-size:28px;margin-bottom:6px;}
.ct-nm{font-size:12px;font-weight:700;color:var(--tx);}

.upload-zone{border:2.5px dashed var(--br2);border-radius:var(--rad);padding:44px;text-align:center;cursor:pointer;transition:all .18s;background:var(--gb);}
.upload-zone:hover{border-color:var(--g4);background:var(--gp);}

.proc-anim{text-align:center;padding:36px;}
.proc-icon{font-size:60px;animation:bounce 1.4s infinite;margin-bottom:18px;}
.proc-steps-list{max-width:300px;margin:20px auto 0;text-align:left;}
.proc-step{display:flex;align-items:center;gap:10px;padding:7px 0;font-size:13.5px;color:var(--tx2);}
.proc-step.done{color:var(--g4);font-weight:600;}
.proc-step.act{color:var(--g1);font-weight:700;}

/* ══════════════════════════════════════════════════════════
   AI REPORT
══════════════════════════════════════════════════════════ */
.rep-head{background:linear-gradient(135deg,var(--g3),var(--g1));padding:26px 28px;color:white;}
.rep-body{padding:26px;}
.rep-sec{margin-bottom:22px;}
.rep-sec-t{font-size:11px;font-weight:700;color:var(--tx3);text-transform:uppercase;letter-spacing:.8px;margin-bottom:11px;}
.sev-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}
.sev-item{padding:13px;border-radius:var(--rad2);text-align:center;}
.sev-n{font-family:'Baloo 2',cursive;font-size:22px;font-weight:800;}
.sev-l{font-size:11px;font-weight:600;margin-top:2px;}
.spread-grid{display:grid;grid-template-columns:1fr 1fr;gap:11px;}
.spread-c{padding:15px;border-radius:var(--rad2);text-align:center;}
.spread-n{font-family:'Baloo 2',cursive;font-size:30px;font-weight:900;}
.spread-l{font-size:12px;font-weight:600;margin-top:3px;}
.treat-ph{border-radius:var(--rad2);padding:13px;margin-bottom:9px;border-left:4px solid var(--g4);}
.treat-ph-lbl{font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.7px;color:var(--g3);margin-bottom:3px;}
.treat-ph-txt{font-size:13.5px;color:var(--tx);}
.med-row{display:flex;justify-content:space-between;align-items:center;padding:11px 13px;border-radius:var(--rad2);border:1.5px solid var(--br);margin-bottom:7px;}
.med-nm{font-size:13.5px;font-weight:700;color:var(--tx);}
.med-ty{font-size:12px;color:var(--tx3);}
.med-pr{font-size:14.5px;font-weight:800;color:var(--g2);}

/* ══════════════════════════════════════════════════════════
   EXPERTS PAGE
══════════════════════════════════════════════════════════ */
.exp-filters{display:flex;gap:10px;flex-wrap:wrap;align-items:center;background:white;padding:16px 20px;border-radius:var(--rad);border:1.5px solid var(--br);margin-bottom:26px;}
.flt-sel{padding:8px 13px;border-radius:8px;border:1.5px solid var(--br);font-size:13px;background:var(--gb);color:var(--tx);outline:none;cursor:pointer;font-family:'Outfit',sans-serif;}
.experts-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;}
.exp-card{padding:22px;border-radius:var(--rad);border:1.5px solid var(--br);background:white;transition:all .22s;box-shadow:var(--sh);}
.exp-card:hover{transform:translateY(-3px);box-shadow:var(--sh2);border-color:var(--br2);}
.exp-av{width:54px;height:54px;border-radius:50%;background:linear-gradient(135deg,var(--g5),var(--g6));display:flex;align-items:center;justify-content:center;font-size:24px;position:relative;flex-shrink:0;}
.exp-av .on-dot{position:absolute;bottom:2px;right:2px;width:12px;height:12px;background:#22c55e;border-radius:50%;border:2px solid white;animation:ringPulse 2s infinite;}
.exp-nm{font-size:15.5px;font-weight:800;color:var(--tx);}
.exp-sp{font-size:13px;color:var(--g4);font-weight:600;}
.exp-rat{display:flex;align-items:center;gap:4px;font-size:13px;font-weight:700;color:#d97706;margin-top:3px;}
.exp-det{font-size:13px;color:var(--tx2);display:flex;align-items:center;gap:6px;margin-bottom:5px;}
.exp-pr{font-family:'Baloo 2',cursive;font-size:19px;font-weight:800;color:var(--g1);margin:12px 0;}
.exp-pr span{font-family:'Outfit';font-size:12.5px;font-weight:500;color:var(--tx3);}

/* ══════════════════════════════════════════════════════════
   CHAT PAGE
══════════════════════════════════════════════════════════ */
.chat-wrap{height:calc(100vh - 62px);display:flex;flex-direction:column;background:var(--gb);}
.chat-hd{padding:14px 22px;background:white;border-bottom:1.5px solid var(--br);display:flex;align-items:center;gap:13px;flex-shrink:0;box-shadow:0 1px 8px rgba(12,61,30,.05);}
.chat-msgs{flex:1;overflow-y:auto;padding:22px;display:flex;flex-direction:column;gap:13px;}
.chat-msg{max-width:68%;}
.chat-msg.mine{align-self:flex-end;}
.chat-msg.theirs{align-self:flex-start;}
.msg-bbl{padding:11px 15px;border-radius:14px;font-size:13.5px;line-height:1.65;white-space:pre-wrap;}
.mine .msg-bbl{background:var(--g4);color:white;border-radius:14px 14px 4px 14px;}
.theirs .msg-bbl{background:white;color:var(--tx);border:1.5px solid var(--br);border-radius:14px 14px 14px 4px;}
.msg-time{font-size:11px;color:var(--tx4);margin-top:4px;}
.mine .msg-time{text-align:right;}
.chat-input-bar{padding:14px 22px;background:white;border-top:1.5px solid var(--br);display:flex;align-items:center;gap:11px;flex-shrink:0;}
.chat-inp{flex:1;padding:10px 16px;border-radius:22px;border:1.5px solid var(--br);font-size:13.5px;outline:none;transition:border .18s;font-family:'Outfit',sans-serif;}
.chat-inp:focus{border-color:var(--g4);}
.chat-send{width:40px;height:40px;border-radius:50%;background:var(--g4);color:white;border:none;font-size:17px;display:flex;align-items:center;justify-content:center;cursor:pointer;transition:background .18s;flex-shrink:0;}
.chat-send:hover{background:var(--g3);}

/* ══════════════════════════════════════════════════════════
   MY FARM
══════════════════════════════════════════════════════════ */
.farm-map{height:220px;border-radius:var(--rad);background:linear-gradient(135deg,#c8e6c9,#a5d6a7);border:1.5px solid var(--br);display:flex;align-items:center;justify-content:center;font-size:60px;position:relative;overflow:hidden;margin-bottom:22px;}
.farm-mark{position:absolute;background:rgba(21,93,48,.85);color:white;padding:5px 11px;border-radius:100px;font-size:11.5px;font-weight:700;}
.farm-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px;}
.fs-item{padding:16px;border-radius:var(--rad);text-align:center;background:white;border:1.5px solid var(--br);}
.fs-n{font-family:'Baloo 2',cursive;font-size:24px;font-weight:800;color:var(--g3);}
.fs-l{font-size:12px;color:var(--tx3);font-weight:600;}
.tl-bars{display:flex;gap:7px;align-items:flex-end;height:75px;margin-top:14px;}
.tl-bw{flex:1;display:flex;flex-direction:column;align-items:center;gap:4px;}
.tl-bar{width:100%;border-radius:4px 4px 0 0;transition:height .5s ease;}
.tl-bl{font-size:10px;color:var(--tx3);font-weight:600;}

/* ══════════════════════════════════════════════════════════
   NOTIFICATIONS
══════════════════════════════════════════════════════════ */
.notif-item{display:flex;gap:13px;padding:15px;border-radius:var(--rad);border:1.5px solid var(--br);background:white;margin-bottom:11px;cursor:pointer;transition:all .18s;}
.notif-item:hover{border-color:var(--br2);transform:translateX(3px);}
.notif-item.unread{border-left:3px solid var(--g4);background:var(--gp);}
.notif-icon{width:40px;height:40px;border-radius:11px;display:flex;align-items:center;justify-content:center;font-size:19px;flex-shrink:0;}
.notif-t{font-size:14px;font-weight:700;color:var(--tx);}
.notif-d{font-size:13px;color:var(--tx2);margin-top:3px;line-height:1.5;}
.notif-time{font-size:11px;color:var(--tx3);margin-top:5px;}

/* ══════════════════════════════════════════════════════════
   BOOKING
══════════════════════════════════════════════════════════ */
.book-types{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:26px;}
.book-type{padding:17px;border-radius:var(--rad);border:2px solid var(--br);background:white;cursor:pointer;transition:all .18s;}
.book-type.sel{border-color:var(--g4);background:var(--gp);}
.book-type:hover{border-color:var(--br2);}

/* ══════════════════════════════════════════════════════════
   PROFILE / SETTINGS
══════════════════════════════════════════════════════════ */
.prof-av-big{width:86px;height:86px;border-radius:50%;background:linear-gradient(135deg,var(--g4),var(--g6));display:flex;align-items:center;justify-content:center;color:white;font-family:'Baloo 2';font-size:30px;font-weight:900;flex-shrink:0;}
.prof-av-big.ex{background:linear-gradient(135deg,var(--b3),var(--b4));}
.info-grid{display:grid;grid-template-columns:1fr 1fr;gap:15px;}
.info-lbl{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--tx3);margin-bottom:5px;}
.info-val{font-size:15px;font-weight:600;color:var(--tx);}
.sett-row{display:flex;justify-content:space-between;align-items:center;padding:13px 0;border-bottom:1px solid var(--gp);}
.sett-row:last-child{border:none;}
.sett-lbl{font-size:14px;font-weight:600;color:var(--tx);}
.sett-sub{font-size:12px;color:var(--tx3);margin-top:2px;}

/* ══════════════════════════════════════════════════════════
   FOOTER
══════════════════════════════════════════════════════════ */
.footer{background:var(--g1);color:white;padding:58px 28px 30px;}
.footer-grid{display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:44px;max-width:1160px;margin:0 auto 44px;}
.footer-bottom{border-top:1px solid rgba(255,255,255,.1);padding-top:22px;text-align:center;font-size:13px;opacity:.5;max-width:1160px;margin:0 auto;}

/* ══════════════════════════════════════════════════════════
   AUTH MODAL
══════════════════════════════════════════════════════════ */
.auth-modal{width:90%;max-width:490px;}
.auth-head{padding:26px 28px;background:linear-gradient(135deg,var(--g4),var(--g1));color:white;}
.auth-head.ex{background:linear-gradient(135deg,var(--b3),var(--b1));}
.auth-body{padding:26px 28px;max-height:78vh;overflow-y:auto;}
.auth-or{text-align:center;color:var(--tx3);font-size:13px;margin:14px 0;position:relative;}
.auth-or::before,.auth-or::after{content:'';position:absolute;top:50%;width:calc(50% - 22px);height:1px;background:var(--br);}
.auth-or::before{left:0;}.auth-or::after{right:0;}
.auth-sw{text-align:center;margin-top:15px;font-size:13px;color:var(--tx2);}
.auth-sw span{color:var(--g4);font-weight:700;cursor:pointer;}
.auth-sw span.ex-link{color:var(--b3);}
.g-btn{width:100%;padding:11px;border-radius:10px;border:1.5px solid var(--br);background:white;color:var(--tx);font-size:13.5px;font-weight:700;display:flex;align-items:center;justify-content:center;gap:10px;cursor:pointer;transition:all .18s;margin-top:8px;font-family:'Outfit',sans-serif;}
.g-btn:hover{border-color:var(--g4);background:var(--gp);}

/* ══════════════════════════════════════════════════════════
   CASE DETAIL (Expert)
══════════════════════════════════════════════════════════ */
.cd-tabs{display:flex;gap:0;border-bottom:2px solid var(--br);margin-bottom:22px;}
.cd-tab{padding:11px 20px;font-size:14px;font-weight:700;color:var(--tx2);border:none;background:none;cursor:pointer;border-bottom:2.5px solid transparent;margin-bottom:-2px;transition:all .18s;}
.cd-tab.on{color:var(--b2);border-bottom-color:var(--b3);}

/* VERIFY STATUS */
.ver-item{display:flex;align-items:center;gap:12px;padding:12px 0;border-bottom:1px solid var(--gp);}
.ver-item:last-child{border:none;}
.ver-ic{width:35px;height:35px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0;}

/* ══════════════════════════════════════════════════════════
   VOICE INPUT
══════════════════════════════════════════════════════════ */
.voice-btn{width:90px;height:90px;border-radius:50%;border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:36px;transition:all .2s;position:relative;}
.voice-btn.idle{background:linear-gradient(135deg,var(--g4),var(--g3));box-shadow:0 4px 20px rgba(30,126,66,.35);}
.voice-btn.listening{background:linear-gradient(135deg,var(--r2),var(--r3));box-shadow:0 0 0 0 rgba(239,83,80,.4);animation:voicePulse 1.2s infinite;}
@keyframes voicePulse{0%{box-shadow:0 0 0 0 rgba(239,83,80,.5)}70%{box-shadow:0 0 0 22px rgba(239,83,80,0)}100%{box-shadow:0 0 0 0 rgba(239,83,80,0)}}
.voice-wave{display:flex;gap:4px;align-items:center;height:40px;}
.vw-bar{width:4px;border-radius:100px;background:var(--g4);animation:waveAnim 1s ease-in-out infinite;}
.vw-bar:nth-child(2){animation-delay:.1s}.vw-bar:nth-child(3){animation-delay:.2s}.vw-bar:nth-child(4){animation-delay:.3s}.vw-bar:nth-child(5){animation-delay:.4s}
@keyframes waveAnim{0%,100%{height:8px}50%{height:32px}}

/* ══════════════════════════════════════════════════════════
   SATELLITE MONITOR
══════════════════════════════════════════════════════════ */
.sat-map{height:280px;border-radius:var(--rad);background:linear-gradient(145deg,#1a3a1a,#0d2b0d);border:1.5px solid #2d5a2d;position:relative;overflow:hidden;margin-bottom:18px;cursor:pointer;}
.sat-grid{position:absolute;inset:0;background-image:linear-gradient(rgba(77,189,122,.08) 1px,transparent 1px),linear-gradient(90deg,rgba(77,189,122,.08) 1px,transparent 1px);background-size:40px 40px;}
.sat-field{position:absolute;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:white;cursor:pointer;transition:all .2s;border:2px solid transparent;}
.sat-field:hover{transform:scale(1.05);border-color:white;}
.sat-overlay{position:absolute;top:12px;left:12px;background:rgba(0,0,0,.7);border-radius:8px;padding:8px 12px;color:white;font-size:12px;font-weight:600;backdrop-filter:blur(8px);}
.sat-legend{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:16px;}
.sat-leg-item{display:flex;align-items:center;gap:6px;font-size:12px;font-weight:600;color:var(--tx2);}
.sat-leg-dot{width:12px;height:12px;border-radius:3px;}
.ndvi-bar{height:12px;border-radius:100px;background:linear-gradient(90deg,#ef5350,#ffc940,#4dbd7a,#1e7e42);margin:8px 0;position:relative;}
.ndvi-marker{position:absolute;top:-4px;width:4px;height:20px;background:white;border-radius:2px;box-shadow:0 0 6px rgba(0,0,0,.4);}

/* ══════════════════════════════════════════════════════════
   PREDICTIVE FORECAST
══════════════════════════════════════════════════════════ */
.forecast-card{border-radius:var(--rad);border:1.5px solid var(--br);background:white;padding:20px;margin-bottom:14px;transition:all .2s;}
.forecast-card:hover{border-color:var(--br2);box-shadow:var(--sh);}
.risk-meter{height:10px;border-radius:100px;background:var(--gp);overflow:hidden;margin:10px 0;}
.risk-fill{height:100%;border-radius:100px;transition:width .8s ease;}
.risk-fill.low{background:linear-gradient(90deg,var(--g5),var(--g4));}
.risk-fill.med{background:linear-gradient(90deg,var(--a3),var(--a2));}
.risk-fill.high{background:linear-gradient(90deg,var(--r3),var(--r2));}
.forecast-timeline{display:flex;gap:0;overflow-x:auto;padding-bottom:6px;scrollbar-width:none;}
.forecast-timeline::-webkit-scrollbar{display:none;}
.ftl-day{flex-shrink:0;width:72px;padding:12px 8px;text-align:center;border-right:1px solid var(--gp);position:relative;}
.ftl-day:last-child{border:none;}
.ftl-risk-dot{width:10px;height:10px;border-radius:50%;margin:6px auto;}

/* ══════════════════════════════════════════════════════════
   IoT SOIL SENSORS
══════════════════════════════════════════════════════════ */
.sensor-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-bottom:22px;}
.sensor-card{padding:18px;border-radius:var(--rad);border:1.5px solid var(--br);background:white;position:relative;overflow:hidden;}
.sensor-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.sensor-card.ok::before{background:var(--g4);}
.sensor-card.warn::before{background:var(--a2);}
.sensor-card.bad::before{background:var(--r2);}
.sensor-val{font-family:'Baloo 2',cursive;font-size:32px;font-weight:900;line-height:1;}
.sensor-unit{font-size:13px;color:var(--tx3);font-weight:600;}
.sensor-lbl{font-size:12px;font-weight:700;color:var(--tx2);margin-bottom:8px;text-transform:uppercase;letter-spacing:.5px;}
.sensor-gauge{height:6px;background:var(--gp);border-radius:100px;overflow:hidden;margin-top:10px;}
.sensor-gauge-fill{height:100%;border-radius:100px;transition:width .8s ease;}
.sensor-hist{display:flex;gap:3px;align-items:flex-end;height:32px;margin-top:8px;}
.sh-bar{flex:1;border-radius:2px 2px 0 0;min-height:4px;}

/* ══════════════════════════════════════════════════════════
   B2B DATA INTELLIGENCE
══════════════════════════════════════════════════════════ */
.b2b-stat{padding:20px;border-radius:var(--rad);border:1.5px solid var(--br);background:white;text-align:center;}
.b2b-n{font-family:'Baloo 2',cursive;font-size:28px;font-weight:900;color:var(--b3);}
.b2b-l{font-size:12px;color:var(--tx3);font-weight:600;margin-top:4px;}
.heatmap-grid{display:grid;grid-template-columns:repeat(7,1fr);gap:3px;margin:14px 0;}
.hm-cell{height:22px;border-radius:4px;cursor:pointer;transition:transform .15s;}
.hm-cell:hover{transform:scale(1.2);}
.disease-bar-row{display:flex;align-items:center;gap:10px;margin-bottom:9px;}
.disease-bar-track{flex:1;height:8px;background:var(--gp);border-radius:100px;overflow:hidden;}
.disease-bar-fill{height:100%;border-radius:100px;background:linear-gradient(90deg,var(--b4),var(--b3));}

/* ══════════════════════════════════════════════════════════
   INPUT MARKETPLACE
══════════════════════════════════════════════════════════ */
.mkt-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;}
.mkt-card{border-radius:var(--rad);border:1.5px solid var(--br);background:white;overflow:hidden;transition:all .22s;cursor:pointer;}
.mkt-card:hover{transform:translateY(-3px);box-shadow:var(--sh2);border-color:var(--br2);}
.mkt-img{height:100px;display:flex;align-items:center;justify-content:center;font-size:46px;background:linear-gradient(135deg,var(--gp),var(--gpb));}
.mkt-body{padding:13px;}
.mkt-nm{font-size:13.5px;font-weight:800;color:var(--tx);margin-bottom:3px;}
.mkt-type{font-size:11.5px;color:var(--tx3);margin-bottom:7px;}
.mkt-pr{font-family:'Baloo 2',cursive;font-size:18px;font-weight:900;color:var(--g2);}
.mkt-ai{font-size:11px;color:var(--g4);font-weight:700;background:var(--gp);padding:2px 8px;border-radius:100px;display:inline-block;margin-bottom:8px;}
.cart-badge{position:fixed;bottom:26px;left:26px;background:var(--g3);color:white;border-radius:14px;padding:13px 22px;font-size:14px;font-weight:700;cursor:pointer;box-shadow:var(--sh3);animation:slideUp .3s ease;z-index:888;display:flex;align-items:center;gap:10px;}

/* ══════════════════════════════════════════════════════════
   INSURANCE CLAIM
══════════════════════════════════════════════════════════ */
.ins-step{display:flex;gap:14px;padding:16px;border-radius:var(--rad);border:1.5px solid var(--br);background:white;margin-bottom:12px;transition:all .2s;}
.ins-step.active{border-color:var(--g4);background:var(--gp);}
.ins-step.done{border-color:var(--g5);opacity:.8;}
.ins-step-num{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:900;flex-shrink:0;}
.ins-step-num.done{background:var(--g4);color:white;}
.ins-step-num.active{background:var(--g3);color:white;}
.ins-step-num.wait{background:var(--gp);color:var(--tx3);}
.claim-status{padding:20px;border-radius:var(--rad3);text-align:center;margin-bottom:22px;}

/* ══════════════════════════════════════════════════════════
   GOVT DISEASE MAP
══════════════════════════════════════════════════════════ */
.gov-map{height:300px;border-radius:var(--rad);background:linear-gradient(160deg,#e8f4fd,#dbedf8);border:1.5px solid var(--bpb);position:relative;overflow:hidden;margin-bottom:18px;}
.map-district{position:absolute;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;color:white;cursor:pointer;transition:all .2s;}
.map-district:hover{transform:scale(1.12);z-index:10;}
.map-tooltip{position:absolute;background:rgba(0,0,0,.82);color:white;padding:7px 12px;border-radius:8px;font-size:12px;font-weight:600;pointer-events:none;white-space:nowrap;z-index:20;backdrop-filter:blur(6px);}
.outbreak-row{display:flex;gap:12px;padding:13px;border-radius:var(--rad2);border:1.5px solid var(--br);background:white;margin-bottom:9px;cursor:pointer;transition:all .18s;}
.outbreak-row:hover{border-color:var(--br2);transform:translateX(3px);}
.outbreak-sev{width:40px;height:40px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:17px;flex-shrink:0;}

/* ══ ROBOT CSS ══ */
@keyframes roboBlink{0%,100%{opacity:1}50%{opacity:.3}}
@keyframes roboScan{0%{transform:translateY(-100%)}100%{transform:translateY(400%)}}
@keyframes roboPing{0%{transform:scale(1);opacity:1}100%{transform:scale(2.5);opacity:0}}
@keyframes roboGlow{0%,100%{box-shadow:0 0 8px #00d4ff}50%{box-shadow:0 0 24px #00d4ff,0 0 48px rgba(0,212,255,.3)}}
.rob-shell{background:#0a0f1e;min-height:calc(100vh - 62px);color:white;}
.rob-wrap{max-width:1200px;margin:0 auto;padding:32px 24px;}
.rob-wrap-sm{max-width:860px;margin:0 auto;padding:32px 24px;}
.rob-card{background:rgba(255,255,255,.04);border:1px solid rgba(0,212,255,.2);border-radius:16px;}
.rob-card-glow{border-color:#00d4ff;box-shadow:0 0 20px rgba(0,212,255,.15);}
.rob-badge{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:100px;font-size:11.5px;font-weight:700;}
.rob-badge.online{background:rgba(0,255,157,.12);color:#00ff9d;border:1px solid rgba(0,255,157,.3);}
.rob-badge.offline{background:rgba(255,255,255,.06);color:rgba(255,255,255,.4);border:1px solid rgba(255,255,255,.1);}
.rob-badge.busy{background:rgba(255,215,0,.12);color:#ffd700;border:1px solid rgba(255,215,0,.3);}
.rob-badge.error{background:rgba(255,68,68,.15);color:#ff4444;border:1px solid rgba(255,68,68,.3);}
.rob-dot{width:7px;height:7px;border-radius:50%;display:inline-block;}
.rob-dot.online{background:#00ff9d;animation:roboBlink 1.8s infinite;}
.rob-dot.offline{background:rgba(255,255,255,.3);}
.rob-dot.busy{background:#ffd700;animation:roboBlink 1s infinite;}
.rob-dot.error{background:#ff4444;animation:roboBlink .6s infinite;}
.rob-stat{padding:18px 20px;border-radius:14px;border:1px solid rgba(0,212,255,.15);background:rgba(0,212,255,.04);transition:all .2s;}
.rob-stat:hover{border-color:#00d4ff;background:rgba(0,212,255,.08);}
.rob-stat-n{font-family:'Baloo 2',cursive;font-size:30px;font-weight:900;line-height:1;}
.rob-stat-l{font-size:11.5px;font-weight:600;margin-top:5px;opacity:.65;}
.rob-prog{height:6px;background:rgba(255,255,255,.08);border-radius:100px;overflow:hidden;}
.rob-prog-fill{height:100%;border-radius:100px;transition:width .8s ease;}
.rob-prog-fill.cyan{background:linear-gradient(90deg,#0088aa,#00d4ff);}
.rob-prog-fill.green{background:linear-gradient(90deg,#00aa66,#00ff9d);}
.rob-prog-fill.red{background:linear-gradient(90deg,#aa0000,#ff4444);}
.rob-prog-fill.yellow{background:linear-gradient(90deg,#aa8800,#ffd700);}
.rob-btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;padding:10px 20px;border-radius:10px;font-size:13.5px;font-weight:700;cursor:pointer;transition:all .18s;font-family:'Outfit',sans-serif;border:none;}
.rob-btn.primary{background:#00d4ff;color:#0a0f1e;box-shadow:0 0 18px rgba(0,212,255,.3);}
.rob-btn.primary:hover{background:#33ddff;}
.rob-btn.danger{background:#ff4444;color:white;}
.rob-btn.ghost{background:rgba(0,212,255,.1);color:#00d4ff;border:1px solid rgba(0,212,255,.3);}
.rob-btn.ghost:hover{background:rgba(0,212,255,.2);}
.rob-btn.green{background:#00ff9d;color:#0a0f1e;}
.rob-btn:disabled{opacity:.4;cursor:not-allowed;}
.robot-row{display:flex;align-items:center;gap:14px;padding:16px 18px;border-radius:12px;border:1px solid rgba(255,255,255,.07);background:rgba(255,255,255,.03);cursor:pointer;transition:all .18s;margin-bottom:9px;}
.robot-row:hover{border-color:rgba(0,212,255,.3);background:rgba(0,212,255,.05);}
.robot-row.sel{border-color:#00d4ff;background:rgba(0,212,255,.08);}
.robot-av{width:46px;height:46px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0;}
.cam-feed{border-radius:14px;overflow:hidden;position:relative;background:#0d1b3e;border:1px solid rgba(0,212,255,.2);height:220px;}
.cam-scanline{position:absolute;left:0;right:0;height:3px;background:linear-gradient(90deg,transparent,#00d4ff,transparent);opacity:.5;animation:roboScan 2.5s linear infinite;z-index:3;}
.cam-overlay-tl,.cam-overlay-tr,.cam-overlay-bl,.cam-overlay-br{position:absolute;width:20px;height:20px;border-color:#00d4ff;border-style:solid;z-index:4;}
.cam-overlay-tl{top:10px;left:10px;border-width:2px 0 0 2px;}
.cam-overlay-tr{top:10px;right:10px;border-width:2px 2px 0 0;}
.cam-overlay-bl{bottom:10px;left:10px;border-width:0 0 2px 2px;}
.cam-overlay-br{bottom:10px;right:10px;border-width:0 2px 2px 0;}
.cam-rec{position:absolute;top:12px;left:50%;transform:translateX(-50%);display:flex;align-items:center;gap:6px;background:rgba(0,0,0,.65);padding:4px 10px;border-radius:100px;font-size:11px;font-weight:700;color:#ff4444;z-index:5;}
.cam-rec-dot{width:7px;height:7px;border-radius:50%;background:#ff4444;animation:roboBlink .8s infinite;}
.rob-field-map{height:280px;background:#0d1b3e;border-radius:14px;border:1px solid rgba(0,212,255,.2);position:relative;overflow:hidden;}
.rob-grid-bg{position:absolute;inset:0;background-image:linear-gradient(rgba(0,212,255,.05) 1px,transparent 1px),linear-gradient(90deg,rgba(0,212,255,.05) 1px,transparent 1px);background-size:36px 36px;}
.rob-robot-icon{width:28px;height:28px;border-radius:8px;background:#00d4ff;display:flex;align-items:center;justify-content:center;font-size:14px;animation:roboGlow 2s infinite;position:absolute;z-index:5;}
.rob-ping{position:absolute;width:28px;height:28px;border-radius:50%;border:2px solid #00d4ff;animation:roboPing 1.5s infinite;z-index:4;}
.joystick-wrap{width:160px;height:160px;border-radius:50%;background:rgba(0,212,255,.06);border:2px solid rgba(0,212,255,.22);position:relative;display:flex;align-items:center;justify-content:center;margin:0 auto;}
.joystick-knob{width:56px;height:56px;border-radius:50%;background:linear-gradient(135deg,#00d4ff,#0088aa);box-shadow:0 0 20px rgba(0,212,255,.4);display:flex;align-items:center;justify-content:center;font-size:22px;cursor:grab;}
.joy-dir-btn{width:44px;height:44px;border-radius:10px;background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.3);color:#00d4ff;font-size:18px;display:flex;align-items:center;justify-content:center;cursor:pointer;transition:all .15s;user-select:none;}
.joy-dir-btn:active{background:rgba(0,212,255,.25);box-shadow:0 0 12px rgba(0,212,255,.3);}
.spray-zone{border-radius:12px;border:1px solid rgba(0,212,255,.15);background:rgba(255,255,255,.03);padding:16px;margin-bottom:10px;cursor:pointer;transition:all .18s;}
.spray-zone:hover{border-color:rgba(0,212,255,.3);}
.spray-zone.active{border-color:#00d4ff;background:rgba(0,212,255,.08);}
.rob-chart-bar{display:flex;align-items:flex-end;gap:6px;height:80px;}
.rcb-col{flex:1;display:flex;flex-direction:column;align-items:center;gap:3px;}
.rcb-bar{width:100%;border-radius:4px 4px 0 0;transition:height .7s ease;min-height:4px;}
.rcb-lbl{font-size:9px;color:rgba(255,255,255,.4);font-weight:600;}
/* RESPONSIVE */
@media(max-width:768px){
  .hero-in{grid-template-columns:1fr;}
  .hero-card{display:none;}
  .hero-h1{font-size:36px;}
  .feat-grid,.dash-2,.experts-grid,.ed-stats,.farm-stats,.info-grid,.frow,.crops-grid,.sensor-grid,.mkt-grid{grid-template-columns:1fr;}
  .method-grid{grid-template-columns:repeat(2,1fr);}
  .crops-grid{grid-template-columns:repeat(3,1fr);}
  .nav-links{display:none;}
  .book-types{grid-template-columns:1fr;}
}

/* ── Q PROGRESS DOTS ── */
.qf-prog{display:flex;align-items:center;gap:0;margin-bottom:24px;}
.qf-dot{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;border:2px solid var(--br);background:white;color:var(--tx4);flex-shrink:0;transition:all .3s;z-index:1;}
.qf-dot.done{background:var(--g4);border-color:var(--g4);color:white;}
.qf-dot.active{background:white;border-color:var(--g4);color:var(--g2);box-shadow:0 0 0 4px rgba(77,189,122,.15);}
.qf-line{flex:1;height:2.5px;background:var(--br);margin:0 -1px;transition:background .3s;}
.qf-line.done{background:var(--g4);}

/* ── Q CARD ── */
.qf-card{background:white;border:1.5px solid var(--br);border-radius:var(--rad3);padding:24px;margin-bottom:18px;animation:slideUp .3s ease both;box-shadow:var(--sh);}
.qf-num{font-size:11px;font-weight:700;color:var(--g4);letter-spacing:.6px;text-transform:uppercase;margin-bottom:8px;}
.qf-q{font-family:'Baloo 2',cursive;font-size:20px;font-weight:800;color:var(--g1);line-height:1.25;margin-bottom:5px;}
.qf-hint{font-size:13px;color:var(--tx3);margin-bottom:18px;}

/* ── OPTIONS ── */
.qf-opts{display:grid;grid-template-columns:1fr 1fr;gap:9px;}
.qf-opt{padding:12px 14px;border-radius:10px;border:1.5px solid var(--br);background:white;text-align:left;font-size:13px;font-weight:600;color:var(--tx);display:flex;align-items:center;gap:9px;cursor:pointer;transition:all .18s;}
.qf-opt:hover{border-color:var(--gpb);background:var(--gp);transform:translateX(3px);}
.qf-opt.sel{border-color:var(--g4);background:var(--gp);color:var(--g1);}
.qf-opt.sel::after{content:'✓';margin-left:auto;color:var(--g2);font-weight:900;}
.qf-opt-icon{font-size:18px;flex-shrink:0;}
.qf-opt-desc{font-size:11px;color:var(--tx3);margin-top:1px;font-weight:400;}

/* ── ANSWERED Q ── */
.qf-answered{background:var(--gp);border:1.5px solid var(--gpb);border-radius:10px;padding:11px 15px;margin-bottom:9px;display:flex;align-items:center;gap:11px;animation:fadeIn .25s ease;}
.qf-ans-check{width:22px;height:22px;background:var(--g4);border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-size:11px;flex-shrink:0;}
.qf-ans-q{font-size:11px;color:var(--tx3);}
.qf-ans-a{font-size:13px;font-weight:700;color:var(--g1);}

/* ── REPORT STYLES ── */
.rep-sheet{background:white;border-radius:var(--rad3);overflow:hidden;box-shadow:var(--sh3);}

.rep-header{background:linear-gradient(135deg,#0d3a21 0%,#1a6b3c 60%,#2a8a52 100%);padding:26px 30px 22px;position:relative;overflow:hidden;}
.rep-header::before{content:'';position:absolute;top:-50px;right:-50px;width:220px;height:220px;background:rgba(255,255,255,.04);border-radius:50%;}
.rep-header::after{content:'';position:absolute;bottom:-60px;left:160px;width:160px;height:160px;background:rgba(255,255,255,.03);border-radius:50%;}
.rh-top{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:18px;position:relative;z-index:1;}
.rh-logo{display:flex;align-items:center;gap:9px;}
.rh-logo-img{height:28px;object-fit:contain;filter:brightness(0) invert(1);opacity:.9;}
.rh-logo-sep{width:1px;height:20px;background:rgba(255,255,255,.25);margin:0 3px;}
.rh-platform{font-size:11px;color:rgba(255,255,255,.55);letter-spacing:.5px;text-transform:uppercase;}
.rh-meta{text-align:right;}
.rh-badge{display:inline-flex;align-items:center;gap:5px;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.2);border-radius:100px;padding:3px 10px;font-size:11px;font-weight:600;color:rgba(255,255,255,.85);margin-bottom:4px;}
.rh-badge-dot{width:6px;height:6px;background:#4ade80;border-radius:50%;animation:pulse 2s infinite;}
.rh-id{font-size:10px;color:rgba(255,255,255,.4);}
.rh-main{position:relative;z-index:1;}
.rh-crop-line{font-size:11px;color:rgba(255,255,255,.5);margin-bottom:5px;letter-spacing:.4px;}
.rh-disease{font-family:'Baloo 2',cursive;font-size:28px;font-weight:900;color:#fff;line-height:1.1;margin-bottom:4px;}
.rh-sci{font-size:12px;color:rgba(255,255,255,.5);font-style:italic;margin-bottom:16px;}
.rh-scores{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;}
.rhs{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);border-radius:9px;padding:10px 12px;}
.rhs-val{font-family:'Baloo 2',cursive;font-size:22px;font-weight:900;color:#fff;line-height:1;}
.rhs-lbl{font-size:9px;color:rgba(255,255,255,.45);margin-top:3px;letter-spacing:.4px;text-transform:uppercase;}
.rhs-bar{height:3px;background:rgba(255,255,255,.15);border-radius:2px;margin-top:7px;overflow:hidden;}
.rhs-fill{height:100%;background:rgba(255,255,255,.55);border-radius:2px;}

.rep-body{padding:22px 26px 28px;}
.r-sec{margin-bottom:18px;}
.r-sec-hd{display:flex;align-items:center;gap:7px;margin-bottom:10px;}
.r-sec-num{width:19px;height:19px;border-radius:5px;background:var(--g3);color:white;font-size:10px;font-weight:800;display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.r-sec-title{font-size:13px;font-weight:800;color:var(--tx);}
.r-sec-tag{margin-left:auto;font-size:9px;padding:2px 8px;border-radius:100px;font-weight:700;}
.r-tag-p{background:#eff6ff;color:#1e40af;}
.r-tag-q{background:var(--ap);color:var(--a1);}
.r-tag-b{background:var(--gp);color:var(--g2);}
.r-tag-ai{background:#fdf2f8;color:#86198f;}

.r-card{background:var(--gb);border:1px solid var(--br);border-radius:10px;padding:14px 16px;}
.r-card.gl{border-left:3px solid var(--g4);}
.r-card.al{border-left:3px solid var(--a3);}
.r-card.rl{border-left:3px solid var(--r3);}

.r-fr{display:flex;align-items:baseline;gap:8px;padding:5.5px 0;border-bottom:1px solid var(--gb);}
.r-fr:last-child{border:none;}
.r-fl{font-size:11px;color:var(--tx3);min-width:118px;flex-shrink:0;}
.r-fv{font-size:13px;color:var(--tx);flex:1;}
.r-fv.b{font-weight:700;}
.r-fv.g{color:var(--g2);font-weight:700;}
.r-fv.r{color:var(--r2);font-weight:700;}

.r-pill{display:inline-flex;align-items:center;gap:3px;font-size:9px;font-weight:700;padding:1px 6px;border-radius:100px;margin-left:5px;vertical-align:middle;}
.r-pill-p{background:#eff6ff;color:#1e40af;}
.r-pill-q{background:var(--ap);color:var(--a1);}
.r-pill-b{background:var(--gp);color:var(--g2);}

.r-conf-row{display:flex;justify-content:space-between;font-size:11px;color:var(--tx3);margin:8px 0 4px;}
.r-conf-bar{height:7px;background:#e8f0ea;border-radius:100px;overflow:hidden;}
.r-conf-fill{height:100%;border-radius:100px;background:linear-gradient(90deg,var(--g4),var(--g5));}

.r-two{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.r-spread{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px;}
.r-sc{padding:11px 13px;border-radius:8px;text-align:center;}
.r-sc.un{background:var(--rp);border:1px solid var(--rpb);}
.r-sc.tr{background:var(--gp);border:1px solid var(--gpb);}
.r-sc-num{font-family:'Baloo 2',cursive;font-size:26px;font-weight:900;line-height:1;}
.r-sc.un .r-sc-num{color:var(--r2);}
.r-sc.tr .r-sc-num{color:var(--g2);}
.r-sc-lbl{font-size:10px;font-weight:700;margin-top:2px;}
.r-sc.un .r-sc-lbl{color:var(--r2);}
.r-sc.tr .r-sc-lbl{color:var(--g2);}

.r-phases{display:flex;flex-direction:column;gap:7px;}
.r-ph{padding:10px 13px;border-radius:8px;display:flex;gap:9px;align-items:flex-start;}
.r-ph.t{background:#fef2f2;border:1px solid #fecaca;}
.r-ph.w{background:var(--ap);border:1px solid #fde68a;}
.r-ph.s{background:var(--gp);border:1px solid var(--gpb);}
.r-ph-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;margin-top:4px;}
.r-ph.t .r-ph-dot{background:var(--r3);}
.r-ph.w .r-ph-dot{background:var(--a3);}
.r-ph.s .r-ph-dot{background:var(--g4);}
.r-ph-tag{font-size:9px;font-weight:800;letter-spacing:.4px;text-transform:uppercase;margin-bottom:2px;}
.r-ph.t .r-ph-tag{color:var(--r2);}
.r-ph.w .r-ph-tag{color:var(--a2);}
.r-ph.s .r-ph-tag{color:var(--g2);}
.r-ph-txt{font-size:12.5px;color:var(--tx);line-height:1.5;}

.r-med-list{display:flex;flex-direction:column;gap:7px;}
.r-med{display:flex;align-items:center;gap:9px;padding:9px 12px;border-radius:8px;border:1px solid var(--br);}
.r-med.top{border-color:var(--gpb);background:var(--gp);}
.r-med-rank{width:18px;height:18px;border-radius:50%;background:var(--g4);color:white;font-size:9px;font-weight:800;display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.r-med.top .r-med-rank{background:var(--g2);}
.r-med-nm{font-size:12px;font-weight:700;color:var(--tx);flex:1;}
.r-med-ty{font-size:10px;color:var(--tx3);}
.r-med-pr{font-family:'Baloo 2',cursive;font-size:16px;font-weight:900;color:var(--g2);}
.r-rec-badge{font-size:8px;font-weight:800;background:var(--g2);color:white;padding:1px 6px;border-radius:100px;margin-left:5px;}

.r-alert{background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:11px 13px;}
.r-alert-title{font-size:10px;font-weight:800;color:#c2410c;text-transform:uppercase;letter-spacing:.4px;margin-bottom:7px;}

.rep-footer{border-top:1px solid var(--br);padding:14px 26px;display:flex;align-items:center;justify-content:space-between;background:white;}
.rf-logo{display:flex;align-items:center;gap:7px;}
.rf-logo-img{height:20px;opacity:.6;}
.rf-txt{font-size:10px;color:var(--tx4);}
.rf-disc{font-size:10px;color:var(--tx4);text-align:right;max-width:280px;line-height:1.5;}

@media(max-width:600px){
  .rh-scores{grid-template-columns:1fr 1fr;}
  .r-two{grid-template-columns:1fr;}
  .qf-opts{grid-template-columns:1fr;}
  .rep-footer{flex-direction:column;gap:8px;text-align:center;}
  .rf-disc{text-align:center;}
}`;

/* ════════════════════════════════════════════════════════════════
   DATA
════════════════════════════════════════════════════════════════ */
const SOILS = [
  'Clay (Chipchipa mitti)','Loamy (Domat mitti)','Black Cotton (Kali mitti)',
  'Red & Yellow (Lal-Peeli mitti)','Laterite (Lat mitti)',
  'Arid/Desert (Reti wali mitti)','Saline & Alkaline (Namak wali mitti)',
  'Peaty & Marshy (Daldali mitti)','Forest & Mountain (Jungle/Pahadi mitti)',
  'Alluvial (Jalodum mitti)',
];

const INDIA_STATES = [
  'Andhra Pradesh','Arunachal Pradesh','Assam','Bihar','Chhattisgarh',
  'Goa','Gujarat','Haryana','Himachal Pradesh','Jharkhand','Karnataka',
  'Kerala','Madhya Pradesh','Maharashtra','Manipur','Meghalaya','Mizoram',
  'Nagaland','Odisha','Punjab','Rajasthan','Sikkim','Tamil Nadu','Telangana',
  'Tripura','Uttar Pradesh','Uttarakhand','West Bengal',
  'Andaman & Nicobar Islands','Chandigarh','Dadra & Nagar Haveli and Daman & Diu',
  'Delhi','Jammu & Kashmir','Ladakh','Lakshadweep','Puducherry',
];

/* ── Complete State-wise District Mapping ── */
const INDIA_DISTRICTS = {
  'Maharashtra':['Ahmednagar','Akola','Amravati','Aurangabad','Beed','Bhandara','Buldhana','Chandrapur','Dhule','Gadchiroli','Gondia','Hingoli','Jalgaon','Jalna','Kolhapur','Latur','Mumbai City','Mumbai Suburban','Nagpur','Nanded','Nandurbar','Nashik','Osmanabad','Palghar','Parbhani','Pune','Raigad','Ratnagiri','Sangli','Satara','Sindhudurg','Solapur','Thane','Wardha','Washim','Yavatmal'],
  'Uttar Pradesh':['Agra','Aligarh','Ambedkar Nagar','Amethi','Amroha','Auraiya','Ayodhya','Azamgarh','Baghpat','Bahraich','Ballia','Balrampur','Banda','Barabanki','Bareilly','Basti','Bhadohi','Bijnor','Budaun','Bulandshahr','Chandauli','Chitrakoot','Deoria','Etah','Etawah','Farrukhabad','Fatehpur','Firozabad','Gautam Buddha Nagar','Ghaziabad','Ghazipur','Gonda','Gorakhpur','Hamirpur','Hapur','Hardoi','Hathras','Jalaun','Jaunpur','Jhansi','Kannauj','Kanpur Dehat','Kanpur Nagar','Kasganj','Kaushambi','Kushinagar','Lakhimpur Kheri','Lalitpur','Lucknow','Maharajganj','Mahoba','Mainpuri','Mathura','Mau','Meerut','Mirzapur','Moradabad','Muzaffarnagar','Pilibhit','Pratapgarh','Prayagraj','Raebareli','Rampur','Saharanpur','Sambhal','Sant Kabir Nagar','Shahjahanpur','Shamli','Shravasti','Siddharthnagar','Sitapur','Sonbhadra','Sultanpur','Unnao','Varanasi'],
  'Punjab':['Amritsar','Barnala','Bathinda','Faridkot','Fatehgarh Sahib','Fazilka','Ferozepur','Gurdaspur','Hoshiarpur','Jalandhar','Kapurthala','Ludhiana','Mansa','Moga','Mohali','Muktsar','Nawanshahr','Pathankot','Patiala','Rupnagar','Sangrur','Tarn Taran'],
  'Haryana':['Ambala','Bhiwani','Charkhi Dadri','Faridabad','Fatehabad','Gurugram','Hisar','Jhajjar','Jind','Kaithal','Karnal','Kurukshetra','Mahendragarh','Nuh','Palwal','Panchkula','Panipat','Rewari','Rohtak','Sirsa','Sonipat','Yamunanagar'],
  'Rajasthan':['Ajmer','Alwar','Banswara','Baran','Barmer','Bharatpur','Bhilwara','Bikaner','Bundi','Chittorgarh','Churu','Dausa','Dholpur','Dungarpur','Hanumangarh','Jaipur','Jaisalmer','Jalore','Jhalawar','Jhunjhunu','Jodhpur','Karauli','Kota','Nagaur','Pali','Pratapgarh','Rajsamand','Sawai Madhopur','Sikar','Sirohi','Sri Ganganagar','Tonk','Udaipur'],
  'Gujarat':['Ahmedabad','Amreli','Anand','Aravalli','Banaskantha','Bharuch','Bhavnagar','Botad','Chhota Udaipur','Dahod','Dang','Devbhoomi Dwarka','Gandhinagar','Gir Somnath','Jamnagar','Junagadh','Kheda','Kutch','Mahisagar','Mehsana','Morbi','Narmada','Navsari','Panchmahal','Patan','Porbandar','Rajkot','Sabarkantha','Surat','Surendranagar','Tapi','Vadodara','Valsad'],
  'Madhya Pradesh':['Agar Malwa','Alirajpur','Anuppur','Ashoknagar','Balaghat','Barwani','Betul','Bhind','Bhopal','Burhanpur','Chhatarpur','Chhindwara','Damoh','Datia','Dewas','Dhar','Dindori','Guna','Gwalior','Harda','Hoshangabad','Indore','Jabalpur','Jhabua','Katni','Khandwa','Khargone','Mandla','Mandsaur','Morena','Narsinghpur','Neemuch','Niwari','Panna','Raisen','Rajgarh','Ratlam','Rewa','Sagar','Satna','Sehore','Seoni','Shahdol','Shajapur','Sheopur','Shivpuri','Sidhi','Singrauli','Tikamgarh','Ujjain','Umaria','Vidisha'],
  'Bihar':['Araria','Arwal','Aurangabad','Banka','Begusarai','Bhagalpur','Bhojpur','Buxar','Darbhanga','East Champaran','Gaya','Gopalganj','Jamui','Jehanabad','Kaimur','Katihar','Khagaria','Kishanganj','Lakhisarai','Madhepura','Madhubani','Munger','Muzaffarpur','Nalanda','Nawada','Patna','Purnia','Rohtas','Saharsa','Samastipur','Saran','Sheikhpura','Sheohar','Sitamarhi','Siwan','Supaul','Vaishali','West Champaran'],
  'Karnataka':['Bagalkot','Ballari','Belagavi','Bengaluru Rural','Bengaluru Urban','Bidar','Chamarajanagar','Chikkaballapur','Chikkamagaluru','Chitradurga','Dakshina Kannada','Davanagere','Dharwad','Gadag','Hassan','Haveri','Kalaburagi','Kodagu','Kolar','Koppal','Mandya','Mysuru','Raichur','Ramanagara','Shivamogga','Tumakuru','Udupi','Uttara Kannada','Vijayapura','Yadgir'],
  'Tamil Nadu':['Ariyalur','Chengalpattu','Chennai','Coimbatore','Cuddalore','Dharmapuri','Dindigul','Erode','Kallakurichi','Kancheepuram','Kanyakumari','Karur','Krishnagiri','Madurai','Mayiladuthurai','Nagapattinam','Namakkal','Nilgiris','Perambalur','Pudukkottai','Ramanathapuram','Ranipet','Salem','Sivaganga','Tenkasi','Thanjavur','Theni','Thoothukudi','Tiruchirappalli','Tirunelveli','Tirupathur','Tiruppur','Tiruvallur','Tiruvannamalai','Tiruvarur','Vellore','Viluppuram','Virudhunagar'],
  'Andhra Pradesh':['Alluri Sitharama Raju','Anakapalli','Ananthapuramu','Bapatla','Chittoor','Dr. B.R. Ambedkar Konaseema','East Godavari','Eluru','Guntur','Kakinada','Krishna','Kurnool','Nandyal','NTR','Palnadu','Parvathipuram Manyam','Prakasam','Sri Potti Sriramulu Nellore','Sri Sathya Sai','Srikakulam','Tirupati','Visakhapatnam','Vizianagaram','West Godavari','YSR Kadapa'],
  'Telangana':['Adilabad','Bhadradri Kothagudem','Hanumakonda','Hyderabad','Jagtial','Jangaon','Jayashankar Bhupalpally','Jogulamba Gadwal','Kamareddy','Karimnagar','Khammam','Komaram Bheem','Mahabubabad','Mahabubnagar','Mancherial','Medak','Medchal-Malkajgiri','Mulugu','Nagarkurnool','Nalgonda','Narayanpet','Nirmal','Nizamabad','Peddapalli','Rajanna Sircilla','Rangareddy','Sangareddy','Siddipet','Suryapet','Vikarabad','Wanaparthy','Warangal','Yadadri Bhuvanagiri'],
  'West Bengal':['Alipurduar','Bankura','Birbhum','Cooch Behar','Dakshin Dinajpur','Darjeeling','Hooghly','Howrah','Jalpaiguri','Jhargram','Kalimpong','Kolkata','Malda','Murshidabad','Nadia','North 24 Parganas','Paschim Bardhaman','Paschim Medinipur','Purba Bardhaman','Purba Medinipur','Purulia','South 24 Parganas','Uttar Dinajpur'],
  'Odisha':['Angul','Balangir','Balasore','Bargarh','Boudh','Cuttack','Deogarh','Dhenkanal','Gajapati','Ganjam','Jagatsinghpur','Jajpur','Jharsuguda','Kalahandi','Kandhamal','Kendrapara','Kendujhar','Khordha','Koraput','Malkangiri','Mayurbhanj','Nabarangpur','Nayagarh','Nuapada','Puri','Rayagada','Sambalpur','Subarnapur','Sundargarh'],
  'Assam':['Bajali','Baksa','Barpeta','Biswanath','Bongaigaon','Cachar','Charaideo','Chirang','Darrang','Dhemaji','Dhubri','Dibrugarh','Dima Hasao','Goalpara','Golaghat','Hailakandi','Hojai','Jorhat','Kamrup','Kamrup Metropolitan','Karbi Anglong','Karimganj','Kokrajhar','Lakhimpur','Majuli','Morigaon','Nagaon','Nalbari','Sivasagar','Sonitpur','South Salmara-Mankachar','Tamulpur','Tinsukia','Udalguri','West Karbi Anglong'],
  'Himachal Pradesh':['Bilaspur','Chamba','Hamirpur','Kangra','Kinnaur','Kullu','Lahaul & Spiti','Mandi','Shimla','Sirmaur','Solan','Una'],
  'Uttarakhand':['Almora','Bageshwar','Chamoli','Champawat','Dehradun','Haridwar','Nainital','Pauri Garhwal','Pithoragarh','Rudraprayag','Tehri Garhwal','Udham Singh Nagar','Uttarkashi'],
  'Jharkhand':['Bokaro','Chatra','Deoghar','Dhanbad','Dumka','East Singhbhum','Garhwa','Giridih','Godda','Gumla','Hazaribagh','Jamtara','Khunti','Koderma','Latehar','Lohardaga','Pakur','Palamu','Ramgarh','Ranchi','Sahebganj','Seraikela Kharsawan','Simdega','West Singhbhum'],
  'Chhattisgarh':['Balod','Baloda Bazar','Balrampur','Bastar','Bemetara','Bijapur','Bilaspur','Dantewada','Dhamtari','Durg','Gariyaband','Gaurela-Pendra-Marwahi','Janjgir-Champa','Jashpur','Kabirdham','Kanker','Kondagaon','Korba','Koriya','Mahasamund','Mungeli','Narayanpur','Raigarh','Raipur','Rajnandgaon','Sakti','Sukma','Surajpur','Surguja'],
  'Kerala':['Alappuzha','Ernakulam','Idukki','Kannur','Kasaragod','Kollam','Kottayam','Kozhikode','Malappuram','Palakkad','Pathanamthitta','Thiruvananthapuram','Thrissur','Wayanad'],
  'Goa':['North Goa','South Goa'],
  'Manipur':['Bishnupur','Chandel','Churachandpur','Imphal East','Imphal West','Jiribam','Kakching','Kamjong','Kangpokpi','Noney','Pherzawl','Senapati','Tamenglong','Tengnoupal','Thoubal','Ukhrul'],
  'Meghalaya':['East Garo Hills','East Jaintia Hills','East Khasi Hills','Eastern West Khasi Hills','North Garo Hills','Ri Bhoi','South Garo Hills','South West Garo Hills','South West Khasi Hills','West Garo Hills','West Jaintia Hills','West Khasi Hills'],
  'Nagaland':['Chumoukedima','Dimapur','Kiphire','Kohima','Longleng','Mokokchung','Mon','Niuland','Noklak','Peren','Phek','Tuensang','Wokha','Zunheboto'],
  'Arunachal Pradesh':['Anjaw','Changlang','Dibang Valley','East Kameng','East Siang','Kamle','Kra Daadi','Kurung Kumey','Lepa Rada','Lohit','Longding','Lower Dibang Valley','Lower Siang','Lower Subansiri','Namsai','Pakke Kessang','Papum Pare','Shi Yomi','Siang','Tawang','Tirap','Upper Siang','Upper Subansiri','West Kameng','West Siang'],
  'Mizoram':['Aizawl','Champhai','Hnahthial','Khawzawl','Kolasib','Lawngtlai','Lunglei','Mamit','Saiha','Saitual','Serchhip'],
  'Tripura':['Dhalai','Gomati','Khowai','North Tripura','Sepahijala','South Tripura','Unakoti','West Tripura'],
  'Sikkim':['East Sikkim','North Sikkim','Pakyong','Soreng','South Sikkim','West Sikkim'],
  'Delhi':['Central Delhi','East Delhi','New Delhi','North Delhi','North East Delhi','North West Delhi','Shahdara','South Delhi','South East Delhi','South West Delhi','West Delhi'],
  'Jammu & Kashmir':['Anantnag','Bandipora','Baramulla','Budgam','Doda','Ganderbal','Jammu','Kathua','Kishtwar','Kulgam','Kupwara','Poonch','Pulwama','Rajouri','Ramban','Reasi','Samba','Shopian','Srinagar','Udhampur'],
  'Ladakh':['Kargil','Leh'],
  'Puducherry':['Karaikal','Mahe','Puducherry','Yanam'],
  'Chandigarh':['Chandigarh'],
  'Andaman & Nicobar Islands':['Nicobar','North & Middle Andaman','South Andaman'],
  'Dadra & Nagar Haveli and Daman & Diu':['Dadra & Nagar Haveli','Daman','Diu'],
  'Lakshadweep':['Lakshadweep'],
};

/* ── Maharashtra District → Taluka Mapping ── */
const MAHARASHTRA_TALUKAS = {
  'Pune':['Ambegaon','Baramati','Bhor','Daund','Haveli','Indapur','Junnar','Khed','Maval','Mulshi','Purandar','Shirur','Velhe'],
  'Nashik':['Baglan','Chandwad','Deola','Dindori','Igatpuri','Kalwan','Malegaon','Nandgaon','Nifad','Peint','Sinnar','Surgana','Trimbakeshwar','Yeola'],
  'Aurangabad':['Aurangabad','Fulambri','Gangapur','Kannad','Khuldabad','Paithan','Sillod','Soegaon','Vaijapur'],
  'Nagpur':['Bhiwapur','Hingna','Kamptee','Katol','Kuhi','Mauda','Nagpur Rural','Nagpur Urban','Narkhed','Parseoni','Ramtek','Savner','Umred'],
  'Kolhapur':['Ajra','Bhudargad','Chandgad','Gadhinglaj','Hatkanangle','Kagal','Karvir','Panhala','Radhanagari','Shahuwadi','Shirol'],
  'Solapur':['Akkalkot','Barshi','Karmala','Madha','Malshiras','Mangalvedha','Mohol','North Solapur','Pandharpur','Sangola','South Solapur'],
  'Satara':['Jaoli','Karad','Khandala','Khatav','Koregaon','Mahabaleshwar','Man','Patan','Phaltan','Satara','Wai'],
  'Sangli':['Atpadi','Jat','Kadegaon','Kavthe Mahankal','Khanapur','Miraj','Palus','Shirala','Tasgaon','Walwa'],
  'Ahmednagar':['Akole','Jamkhed','Karjat','Kopargaon','Nagar','Nevasa','Parner','Pathardi','Rahata','Rahuri','Sangamner','Shevgaon','Shrirampur','Shrigonda'],
  'Jalgaon':['Amalner','Bhadgaon','Bhusawal','Bodwad','Chalisgaon','Chopda','Dharangaon','Erandol','Jalgaon','Jamner','Muktainagar','Pachora','Parola','Raver','Yawal'],
  'Amravati':['Achalpur','Amravati','Anjangaon Surji','Chandur Bazar','Chandur Railway','Chikhaldara','Daryapur','Dhamangaon Railway','Morshi','Nandgaon Khandeshwar','Teosa','Warud'],
  'Yavatmal':['Arni','Babulgaon','Darwha','Digras','Ghatanji','Kalamb','Kelapur','Mahagaon','Maregaon','Ner','Pusad','Ralegaon','Umarkhed','Wani','Yavatmal','Zari Jamani'],
  'Latur':['Ahmadpur','Ausa','Chakur','Deoni','Jalkot','Latur','Nilanga','Renapur','Shirur Anantpal','Udgir'],
  'Nanded':['Ardhapur','Biloli','Deglur','Dharmabad','Hadgaon','Himayatnagar','Kandhar','Kinwat','Loha','Mudkhed','Mukhed','Naigaon','Nanded','Umri'],
  'Akola':['Akola','Akot','Balapur','Barshitakli','Murtizapur','Patur','Telhara'],
  'Buldhana':['Buldana','Chikhli','Deulgaon Raja','Jalgaon Jamod','Khamgaon','Lonar','Malkapur','Mehkar','Motala','Nandura','Sangrampur','Shegaon','Sindkhed Raja'],
  'Wardha':['Arvi','Ashti','Deoli','Hinganghat','Karanja','Samudrapur','Seloo','Wardha'],
  'Bhandara':['Bhandara','Lakhani','Lakhandur','Mohadi','Pauni','Sakoli','Tumsar'],
  'Gondia':['Amgaon','Arjuni Morgaon','Deori','Gondiya','Goregaon','Sadak Arjuni','Salekasa','Tirora'],
  'Gadchiroli':['Aheri','Armori','Bhamragad','Chamorshi','Dhanora','Etapalli','Gadchiroli','Kurkheda','Mulchera','Sironcha'],
  'Chandrapur':['Bhadravati','Brahmapuri','Chandrapur','Chimur','Gondpipri','Jivati','Korpana','Mul','Nagbhid','Pombhurna','Rajura','Sawoli','Sindewahi','Warora'],
  'Dhule':['Dhule','Sakri','Shindkheda','Shirpur'],
  'Nandurbar':['Akkalkuwa','Akrani','Nandurbar','Nawapur','Shahada','Taloda'],
  'Raigad':['Alibag','Karjat','Khalapur','Mahad','Mangaon','Mhasla','Murud','Panvel','Pen','Poladpur','Roha','Shrivardhan','Sudhagad','Tala','Uran'],
  'Ratnagiri':['Chiplun','Dapoli','Guhagar','Khed','Lanja','Mandangad','Rajapur','Ratnagiri','Sangameshwar'],
  'Sindhudurg':['Devgad','Dodamarg','Kankavli','Kudal','Malvan','Sawantwadi','Vaibhavwadi','Vengurla'],
  'Palghar':['Dahanu','Jawhar','Mokhada','Palghar','Talasari','Vada','Vasai','Vikramgad','Wada'],
  'Thane':['Ambarnath','Bhiwandi','Kalyan','Murbad','Shahapur','Thane','Ulhasnagar'],
  'Osmanabad':['Bhum','Kalamb','Lohara','Osmanabad','Paranda','Tuljapur','Umarga','Washi'],
  'Parbhani':['Gangakhed','Jintur','Manwath','Parbhani','Pathri','Purna','Sailu','Selu','Sonpeth'],
  'Hingoli':['Aundha Nagnath','Basmath','Hingoli','Kalamnuri','Sengaon'],
  'Jalna':['Ambad','Badnapur','Bhokardan','Ghansawangi','Jalna','Jafrabad','Mantha','Partur'],
  'Beed':['Ambajogai','Ashti','Beed','Dharur','Georai','Kaij','Manjlegaon','Parli','Patoda','Shirur Kasar','Wadwani'],
  'Washim':['Karanja','Malegaon','Mangrulpir','Manora','Risod','Washim'],
  'Mumbai City':['Borivali','Kandivali','Dahisar','Malad','Goregaon','Jogeshwari','Andheri','Vile Parle','Kurla','Ghatkopar'],
  'Mumbai Suburban':['Panvel','Uran','Pen','Alibag','Khalapur','Karjat'],
};

/* ── Helper Functions (MUST be before any JSX that uses them) ── */
function getDistricts(state) {
  if (!state) return [];
  return INDIA_DISTRICTS[state] || [];
}
function getStateTalukas(state, district) {
  if (!state || !district) return [];
  if (state === 'Maharashtra' && MAHARASHTRA_TALUKAS[district]) {
    return MAHARASHTRA_TALUKAS[district];
  }
  return [district + ' - Taluka 1', district + ' - Taluka 2', district + ' - Taluka 3'];
}

/* ── Legacy aliases ── */
const DISTRICTS = INDIA_DISTRICTS['Maharashtra'];
const TALUKAS   = MAHARASHTRA_TALUKAS['Pune'];

const CROPS = [
  {id:'tomato',name:'Tomato',emoji:'🍅',health:75,stage:'Flowering'},
  {id:'wheat',name:'Wheat',emoji:'🌾',health:88,stage:'Tillering'},
  {id:'potato',name:'Potato',emoji:'🥔',health:92,stage:'Vegetative'},
  {id:'cotton',name:'Cotton',emoji:'🌸',health:68,stage:'Boll Formation'},
  {id:'corn',name:'Corn',emoji:'🌽',health:86,stage:'Tasseling'},
  {id:'apple',name:'Apple',emoji:'🍎',health:95,stage:'Fruiting'},
  {id:'grape',name:'Grape',emoji:'🍇',health:91,stage:'Veraison'},
  {id:'orange',name:'Orange',emoji:'🍊',health:84,stage:'Fruiting'},
  {id:'pepper',name:'Pepper',emoji:'🫑',health:89,stage:'Flowering'},
  {id:'soybean',name:'Soybean',emoji:'🫘',health:90,stage:'R3 Pod'},
  {id:'strawberry',name:'Strawberry',emoji:'🍓',health:73,stage:'Fruiting'},
  {id:'peach',name:'Peach',emoji:'🍑',health:77,stage:'Stone Hardening'},
  {id:'cherry',name:'Cherry',emoji:'🍒',health:79,stage:'Fruiting'},
  {id:'blueberry',name:'Blueberry',emoji:'🫐',health:82,stage:'Ripening'},
  {id:'squash',name:'Squash',emoji:'🎃',health:85,stage:'Fruit Set'},
  {id:'raspberry',name:'Raspberry',emoji:'🍓',health:81,stage:'Ripening'},
  {id:'coconut', name:'Coconut',  emoji:'🥥',health:72,stage:'Fruiting'},
];

const EXPERTS = [
  {id:1,name:'Dr. Rajesh Kumar',spec:'Plant Pathologist',exp:'15+ yrs',langs:['Hindi','English','Punjabi'],price:800,rating:4.9,reviews:243,online:true,emoji:'👨‍🔬',crops:'Tomato, Wheat, Cotton',cases:1240,response:'45 min',success:96},
  {id:2,name:'Dr. Priya Sharma',spec:'Horticulture Expert',exp:'8+ yrs',langs:['English','Hindi','Tamil'],price:600,rating:4.8,reviews:198,online:true,emoji:'👩‍🔬',crops:'Fruits, Vegetables',cases:890,response:'38 min',success:94},
  {id:3,name:'Prof. Amit Verma',spec:'Soil Scientist',exp:'20+ yrs',langs:['English','Hindi'],price:1000,rating:5.0,reviews:312,online:false,emoji:'👨‍🏫',crops:'All Crops',cases:2100,response:'60 min',success:98},
  {id:4,name:'Dr. Manoj Desai',spec:'Crop Scientist',exp:'10+ yrs',langs:['Hindi','Marathi'],price:500,rating:4.7,reviews:128,online:true,emoji:'🧑‍🔬',crops:'Cereals, Pulses',cases:670,response:'52 min',success:93},
  {id:5,name:'Dr. Ravi Singh',spec:'Plant Pathologist',exp:'15+ yrs',langs:['Hindi','Marathi'],price:800,rating:4.9,reviews:243,online:true,emoji:'👨‍🔬',crops:'Cotton, Soybean',cases:980,response:'41 min',success:95},
  {id:6,name:'Dr. Kavita Patel',spec:'Horticulture Expert',exp:'8+ yrs',langs:['Hindi','Gujarati'],price:600,rating:4.8,reviews:198,online:false,emoji:'👩‍🏫',crops:'Spices, Vegetables',cases:750,response:'47 min',success:94},
];

const CONSULTATIONS = [
  {id:1,crop:'Tomato Plant',emoji:'🍅',issue:'Early blight on lower leaves — spreading rapidly',date:'Oct 24, 2026 • 11:30 AM',expert:'Dr. Rajesh Kumar',status:'expert',statusLabel:'Expert Assigned',sev:2,conf:94},
  {id:2,crop:'Wheat Crop',emoji:'🌾',issue:'Nitrogen deficiency — yellowing of leaves',date:'Oct 22, 2026 • 4:05 PM',expert:'Dr. Priya Sharma',status:'ai',statusLabel:'AI Report Ready',sev:1,conf:89},
  {id:3,crop:'Cotton Plant',emoji:'🌸',issue:'Leaf wilting and stem discoloration',date:'Oct 20, 2026 • 2:40 PM',expert:'Dr. Manoj Desai',status:'pending',statusLabel:'Report Pending',sev:3,conf:82},
  {id:4,crop:'Potato Crop',emoji:'🥔',issue:'Late Blight risk — brown spots visible',date:'Oct 18, 2026 • 9:15 AM',expert:'Dr. Ravi Singh',status:'completed',statusLabel:'Completed',sev:1,conf:91},
];

const NOTIFICATIONS = [
  {id:1,type:'alert',icon:'🔴',col:'#fee2e2',title:'Disease Outbreak Alert!',desc:'Late Blight detected — 8 cases within 5km of your farm. Take preventive action now.',time:'2 hours ago',unread:true},
  {id:2,type:'chat',icon:'💬',col:'#e3f2fd',title:'Dr. Rajesh Kumar replied',desc:'Maine aapki tomato ki photo dekhi. Treatment plan bhej raha hoon.',time:'3 hours ago',unread:true},
  {id:3,type:'weather',icon:'🌧️',col:'#fff8e1',title:'Weather Warning',desc:'Baarish ki sambhavna agle 2 din — spray band rakhein.',time:'5 hours ago',unread:true},
  {id:4,type:'reminder',icon:'💊',col:'#eaf7ef',title:'Treatment Reminder',desc:'Aaj Mancozeb spray ka time hai — Tomato field.',time:'Today 9 AM',unread:false},
  {id:5,type:'market',icon:'📈',col:'#f3e5f5',title:'Market Price Alert',desc:'Tomato prices ↑ 18% this week — ₹1,500/quintal. Sell karne ka sahi waqt!',time:'Yesterday',unread:false},
  {id:6,type:'report',icon:'✅',col:'#eaf7ef',title:'AI Report Ready',desc:'Wheat crop analysis complete — Nitrogen deficiency confirmed.',time:'Oct 22',unread:false},
];

const MESSAGES_DATA = [
  {id:1,from:'expert',text:'Namaskar! Maine aapki tomato ki photo carefully dekhi. AI report sahi hai — Early Blight confirmed. Kab se symptoms dekh rahe hain?',time:'2:15 PM'},
  {id:2,from:'farmer',text:'3-4 din pehle se neeche ke patte kaale ho rahe hain. Kal aur zyada badh gaya.',time:'2:18 PM'},
  {id:3,from:'expert',text:'Samajh gaya. Pichle hafte barish aayi thi? Aur koi spray kiya tha recently?',time:'2:20 PM'},
  {id:4,from:'farmer',text:'Haan, 5 din pehle barish thi. Spray nahi kiya pichle 2 hafte se.',time:'2:22 PM'},
  {id:5,from:'expert',text:'Theek hai. High humidity + no fungicide spray — isliye infection faila.\n\nTreatment plan:\n1. Aaj: Affected patte turant hatao\n2. Kal: Mancozeb 75% WP (2.5g/L) spray\n3. 7 din baad: Follow-up photo bhejo\n\nGhabhraiye mat — yeh treatable hai! 💚',time:'2:25 PM'},
];


const DISEASE_DB = {
  tomato:{disease:'Early Blight',sci:'Alternaria solani',hindi:'Aagat Jhulsa Rog',conf:94,sev:2,sevLabel:'Stage 2/5 — Moderate',aff:23,utr:60,tr:15,cause:'Fungal infection — Alternaria solani',humidity:'High humidity + baarish pichle hafte',risk:'3rd consecutive season — same variety',
    phases:['Affected patte hatao. Spray band karo. Waterlogging check karo.','Mancozeb 75% WP — 2.5g/L spray, 2x in 7 days. Poori fasal cover karein.','Crop rotation karein. Resistant variety use karein. Soil health maintain karein.'],
    meds:[{nm:'Mancozeb 75% WP',ty:'Chemical Fungicide',pr:'₹280/kg',top:true},{nm:'Copper Oxychloride 50%',ty:'Preventive Fungicide',pr:'₹320/kg'},{nm:'Neem Oil 1500 PPM',ty:'Organic Option',pr:'₹180/L'}]},
  wheat:{disease:'Leaf Rust',sci:'Puccinia triticina',hindi:'Patti Ka Tamba Rog',conf:89,sev:1,sevLabel:'Stage 1/5 — Early',aff:15,utr:45,tr:10,cause:'Fungal spores — wind se spread hote hain',humidity:'Moderate',risk:'Low — early stage',
    phases:['Infected tillers identify karo. Healthy seed ke paas mat rakho.','Propiconazole 25% EC — 1mL/L spray. Doosre hafte dobara karein.','Resistant variety plan karo. Seed treatment next season.'],
    meds:[{nm:'Propiconazole 25% EC',ty:'Systemic Fungicide',pr:'₹380/L',top:true},{nm:'Mancozeb 75% WP',ty:'Contact Fungicide',pr:'₹280/kg'},{nm:'Tebuconazole 25% EC',ty:'Systemic Fungicide',pr:'₹420/L'}]},
  potato:{disease:'Late Blight',sci:'Phytophthora infestans',hindi:'Picheti Jhulsa',conf:91,sev:3,sevLabel:'Stage 3/5 — High',aff:40,utr:75,tr:30,cause:'Phytophthora infestans — cool + wet mausam',humidity:'Very High',risk:'High — epidemic possible',
    phases:['URGENT: Affected plants immediately hatao. Baaki crop isolate karo.','Metalaxyl + Mancozeb spray — har 5 din pe. Drainage improve karo.','Resistant variety lagao. Crop rotation must. Seed treatment karo.'],
    meds:[{nm:'Metalaxyl 8% + Mancozeb 64%',ty:'Systemic + Contact Fungicide',pr:'₹420/kg',top:true},{nm:'Cymoxanil + Mancozeb',ty:'Combination Fungicide',pr:'₹380/kg'},{nm:'Copper Sulphate',ty:'Traditional Fungicide',pr:'₹120/kg'}]},
  cotton:{disease:'Bacterial Blight',sci:'Xanthomonas malvacearum',hindi:'Jaivanu Jhulsa',conf:86,sev:3,sevLabel:'Stage 3/5 — Moderate-High',aff:35,utr:70,tr:25,cause:'Bacterial — seed aur wind se phailta hai',humidity:'High',risk:'Moderate — containable',
    phases:['Infected branches prune karo. Spray nozzles clean karo.','Streptomycin 90% + Tetracycline spray — wettable powder form.','Crop spacing badhaao. Rain-resistant variety use karein.'],
    meds:[{nm:'Streptomycin Sulfate 90%',ty:'Antibiotic Bactericide',pr:'₹480/100g',top:true},{nm:'Copper Oxychloride 50%',ty:'Bactericide + Fungicide',pr:'₹320/kg'},{nm:'Kasugamycin 3% SL',ty:'Antibiotic',pr:'₹350/250mL'}]},
  coconut:{disease:'Gray Leaf Spot',sci:'Pestalotiopsis palmarum',hindi:'Naariyal Patta Dhabb Rog',conf:91,sev:2,sevLabel:'Stage 2/5 — Moderate',aff:28,utr:65,tr:18,cause:'Pestalotiopsis palmarum fungus — high humidity mein badhta hai',humidity:'High',risk:'Moderate — manageable',
    phases:['Sankat wale patte kaat ke jala dein. Mancozeb 75% WP spray aaj hi shuru karein.','3 din baad dobara spray. Zinc Sulphate foliar spray karein. Nayi pattiyaan monitor karein.','Monthly preventive spray. Soil testing — potassium + zinc check. Bagaan mein drainage improve karo.'],
    meds:[{nm:'Mancozeb 75% WP',ty:'Contact Fungicide',pr:'₹180-320/250g',top:true},{nm:'Copper Oxychloride 50%',ty:'Contact Fungicide',pr:'₹200-380/500g'},{nm:'Propiconazole 25% EC',ty:'Systemic Fungicide',pr:'₹350-550/250ml'},{nm:'Zinc Sulphate',ty:'Micronutrient',pr:'₹80-150/500g'}]},
  coconut_stem_bleeding:{disease:'Stem Bleeding',sci:'Thielaviopsis paradoxa',hindi:'Naariyal Tanaa Khoon Rog',conf:93,sev:4,sevLabel:'Stage 4/5 — Serious',aff:45,utr:80,tr:35,cause:'Thielaviopsis paradoxa — wound se enter karta hai',humidity:'High',risk:'High — spread possible',
    phases:['TURANT: Kaale affected bark ko saaf kato. Bordeaux Paste wound pe lagao. Paani tanay se door rakho.','Har roz wound check karo. Tridemorph injection plan karo. Drainage fix karo.','Annual trunk inspection karo. Harvesting tools sterilize karo. Prophylactic treatment schedule.'],
    meds:[{nm:'Bordeaux Paste (4:4:50)',ty:'Protective Fungicide Paste',pr:'₹50-100 DIY',top:true},{nm:'Tridemorph 80% EC',ty:'Systemic Injection',pr:'₹400-700/500ml'},{nm:'Hexaconazole 5% SC',ty:'Systemic Fungicide',pr:'₹300-500/500ml'},{nm:'Copper Sulphate',ty:'Bordeaux ingredient',pr:'₹80-150/500g'}]},
  coconut_bud_rot:{disease:'Bud Rot',sci:'Phytophthora palmivora',hindi:'Naariyal Ankur Sadna',conf:90,sev:5,sevLabel:'Stage 5/5 — Critical',aff:70,utr:95,tr:60,cause:'Phytophthora palmivora — most destructive coconut disease',humidity:'Very High',risk:'Critical — ped mar sakta hai',
    phases:['EMERGENCY: Infected soft bud tissue spoon se hata do. Bud cavity mein Ridomil paste bharo. Expert turant bulao.','Har 3 din check karo. Aas-paas ke ped mein preventive spray. Arecanut se doori rakho.','Resistant variety future mein lagao. Monsoon se pehle preventive spray. Drainage + airflow improve karo.'],
    meds:[{nm:'Ridomil MZ 72 WP (Metalaxyl)',ty:'Systemic Phytophthora killer',pr:'₹350-600/500g',top:true},{nm:'Bordeaux Mixture 1%',ty:'Protective Copper Fungicide',pr:'₹50-80 DIY'},{nm:'Potassium Phosphonate',ty:'Excellent for Phytophthora',pr:'₹300-500/L'},{nm:'Cymoxanil + Mancozeb',ty:'Combination Fungicide',pr:'₹250-450/250g'}]},
  coconut_gray_leaf_spot:{disease:'Gray Leaf Spot',sci:'Pestalotiopsis palmarum',hindi:'Naariyal Bhoora Patta Rog',conf:91,sev:2,sevLabel:'Stage 2/5 — Moderate',aff:28,utr:65,tr:18,cause:'Pestalotiopsis palmarum — baarish + humidity mein zyada',humidity:'High',risk:'Moderate',
    phases:['Infected patte kaat ke jalao. Mancozeb 75% WP 2.5g/L spray shuru karo. Drainage check karo.','3 din baad dobara spray. Zinc Sulphate foliar spray karein. Crown area pe dhyan do.','Monthly preventive spray banao. Soil testing — K + Zn check. Resistant variety ki jankari lo.'],
    meds:[{nm:'Mancozeb 75% WP',ty:'Contact Fungicide',pr:'₹180-320/250g',top:true},{nm:'Copper Oxychloride 50%',ty:'Protective Fungicide',pr:'₹200-380/500g'},{nm:'Propiconazole 25% EC',ty:'Systemic Fungicide',pr:'₹350-550/250ml'},{nm:'Zinc Sulphate',ty:'Micronutrient',pr:'₹80-150/500g'}]},
  coconut_leaf_rot:{disease:'Leaf Rot',sci:'Colletotrichum gloeosporioides',hindi:'Naariyal Patta Sadna',conf:88,sev:3,sevLabel:'Stage 3/5 — Moderate-High',aff:38,utr:72,tr:28,cause:'Colletotrichum + Boron deficiency combination',humidity:'High',risk:'Moderate-High',
    phases:['Affected fronds kaat ke bagaan se bahar le jao. Carbendazim + Mancozeb combo spray aaj karo.','Borax foliar spray shuru karo. Crown area pe focus. Pest (mites) bhi check karo.','Complete NPK + micronutrient schedule banao. Bagaan mein air circulation maintain karo. Har season boron dein.'],
    meds:[{nm:'Carbendazim 50% WP',ty:'Systemic Fungicide',pr:'₹150-280/250g',top:true},{nm:'Mancozeb 75% WP',ty:'Contact Fungicide',pr:'₹180-320/250g'},{nm:'Borax (Boron Source)',ty:'Micronutrient',pr:'₹60-120/500g'},{nm:'Copper Hydroxide',ty:'Protective Fungicide',pr:'₹220-400/500g'}]},
  default:{disease:'Powdery Mildew',sci:'Erysiphe spp.',hindi:'Safed Chhati Rog',conf:88,sev:2,sevLabel:'Stage 2/5 — Moderate',aff:18,utr:50,tr:12,cause:'Fungal spores — dry + warm conditions',humidity:'Low-Moderate',risk:'Moderate — control possible with treatment',
    phases:['White powdery coating wale leaves hatao.','Wettable Sulfur 80% WP — 3g/L spray. Har 7 din pe.','Airflow improve karo. Resistant variety plan karo.'],
    meds:[{nm:'Wettable Sulfur 80% WP',ty:'Contact Fungicide',pr:'₹120/kg',top:true},{nm:'Hexaconazole 5% SC',ty:'Systemic Fungicide',pr:'₹340/L'},{nm:'Neem Oil + Garlic Extract',ty:'Organic Combo',pr:'₹200/L'}]}
};

/* ════════════════════════════════════════════════════════════════
   QUESTION FLOW STYLES
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   CONSULT PAGE  — Crop → Upload → Question Flow → Processing
════════════════════════════════════════════════════════════════ */
/* ── Error Boundary ─────────────────────────────────────────── */
class ErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { hasError: false, error: null }; }
  static getDerivedStateFromError(error) { return { hasError: true, error }; }
  componentDidCatch(error, info) { console.error('BeejHealth Crash:', error, info); }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{minHeight:'100vh',display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',fontFamily:'sans-serif',padding:24,background:'#f0fdf4'}}>
          <div style={{fontSize:56,marginBottom:16}}>🌱</div>
          <div style={{fontSize:22,fontWeight:800,color:'#166534',marginBottom:8}}>Kuch gadbad ho gayi</div>
          <div style={{fontSize:14,color:'#4b5563',marginBottom:24,textAlign:'center',maxWidth:400}}>
            App mein ek error aaya. Refresh karne se theek ho sakta hai.
            <br/><br/>
            <small style={{color:'#9ca3af'}}>Error: {this.state.error?.message}</small>
          </div>
          <button
            onClick={()=>{
              localStorage.clear();
              window.location.reload();
            }}
            style={{padding:'12px 28px',background:'#16a34a',color:'white',border:'none',borderRadius:10,fontSize:15,fontWeight:700,cursor:'pointer',marginBottom:10}}>
            🔄 Reset Karke Dobara Try Karo
          </button>
          <button
            onClick={()=>window.location.reload()}
            style={{padding:'10px 24px',background:'none',color:'#16a34a',border:'2px solid #16a34a',borderRadius:10,fontSize:14,fontWeight:600,cursor:'pointer'}}>
            Sirf Refresh Karo
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

function useToasts() {
  const [toasts,setToasts] = useState([]);
  const add = useCallback((msg,type='ok')=>{
    const id = Date.now();
    setToasts(p=>[...p,{id,msg,type}]);
    setTimeout(()=>setToasts(p=>p.filter(t=>t.id!==id)),3800);
  },[]);
  return {toasts,add};
}

/* ════════════════════════════════════════════════════════════════
   AUTH MODAL
════════════════════════════════════════════════════════════════ */
function AuthModal({mode,setMode,onClose,onDone,initType='farmer'}) {
  const [utype,setUtype] = useState(initType);
  const [lmethod,setLmethod] = useState('otp');
  const [otpStage,setOtpStage] = useState(false);
  const [regStep,setRegStep] = useState(1);
  const [busy,setBusy] = useState(false);
  const [otp,setOtp] = useState(['','','','','','']);
  const [f,setF] = useState({name:'',mobile:'',email:'',password:'',confirm:'',state:'Maharashtra',district:'',taluka:'',village:'',soil:'',spec:'',fee:'',university:'',langs:''});
  const [errs,setErrs] = useState({});
  const isEx = utype==='expert';
  const thm = isEx ? 'var(--b3)' : 'var(--g4)';

  const upd=(k,v)=>setF(p=>({...p,[k]:v}));
  const setE=(k,v)=>setErrs(p=>({...p,[k]:v}));
  const clr=()=>setErrs({});

  const handleOtp=(i,v)=>{
    if(!/^\d*$/.test(v)) return;
    const n=[...otp]; n[i]=v; setOtp(n);
    if(v && i<5) document.getElementById(`oc${i+1}`)?.focus();
    if(!v && i>0) document.getElementById(`oc${i-1}`)?.focus();
  };

  const vLogin=()=>{ const e={}; if(!f.mobile||f.mobile.length<10) e.mobile='Valid 10-digit number enter karein'; if(lmethod==='password'&&!f.password) e.password='Password required'; setErrs(e); return !Object.keys(e).length; };
  const vR1=()=>{ const e={}; if(!f.name||f.name.length<3) e.name='Naam 3+ characters ka hona chahiye'; if(!f.mobile||f.mobile.length<10) e.mobile='Valid mobile number'; if(f.email&&!/\S+@\S+\.\S+/.test(f.email)) e.email='Valid email'; setErrs(e); return !Object.keys(e).length; };
  const vR2=()=>{ const e={}; if(!f.district) e.district='District zaroor select karein'; setErrs(e); return !Object.keys(e).length; };
  const vR3=()=>{ const e={}; if(!f.password||f.password.length<8) e.password='Password 8+ characters'; if(f.password!==f.confirm) e.confirm='Passwords match nahi kar rahe'; setErrs(e); return !Object.keys(e).length; };

  const doLogin=async()=>{
    if(lmethod==='otp'&&!otpStage){
      if(!vLogin()) return;
      setBusy(true);
      try { await API.post('/api/auth/send-otp',{mobile:f.mobile}); } catch(e){ setE('mobile',e.message); setBusy(false); return; }
      setBusy(false); setOtpStage(true); return;
    }
    if(lmethod==='otp'&&otp.join('').length<6){ setE('otp','6-digit OTP enter karein'); return; }
    if(lmethod==='password'&&!vLogin()) return;
    setBusy(true);
    try {
      const res = await API.post('/api/auth/login',{mobile:f.mobile,password:f.password,otp:otp.join(''),method:lmethod,type:utype});
      saveSession(res.token, res.user);
      onDone(res.user);
    } catch(e){ setE('mobile',e.message); }
    setBusy(false);
  };

  const doRegNext=async()=>{
    if(regStep===1&&!vR1()) return;
    if(regStep===2&&!vR2()) return;
    if(regStep===3){
      if(!vR3()) return;
      setBusy(true);
      try {
        const res = await API.post('/api/auth/register',{
          name:f.name,mobile:f.mobile,email:f.email,password:f.password,
          type:utype,
          state:f.state||'Maharashtra',
          district:f.district,taluka:f.taluka,village:f.village,soil:f.soil,
          farmSize:Number(f.farmSize)||0,irrigation:f.irrigation||'',
          spec:f.spec,fee:Number(f.fee)||0,university:f.university,
          langs:f.langs||'Hindi',expYrs:Number(f.expYrs)||0,
          crops:['tomato','wheat']
        });
        saveSession(res.token, res.user);
        onDone(res.user);
      } catch(e){ setE('password',e.message); }
      setBusy(false);
      return;
    }
    setRegStep(p=>p+1);
  };

  return (
    <div className="overlay" onClick={onClose}>
      <div className="modal auth-modal" onClick={e=>e.stopPropagation()}>
        <div className={`auth-head${isEx?' ex':''}`}>
          <button className="modal-close" onClick={onClose}>✕</button>
          <div style={{fontSize:12,opacity:.78,marginBottom:4}}>🌱 BeejHealth — {mode==='login'?'Wapas Aayein':'Account Banayein'}</div>
          <div style={{fontFamily:"'Baloo 2',cursive",fontSize:22,fontWeight:900}}>{mode==='login'?(isEx?'Expert Login':'Farmer Login'):(isEx?'Expert Register':'Farmer Register')}</div>
        </div>
        <div className="auth-body">
          {/* Type switcher */}
          <div className={`tabs-row${isEx?' ex-tabs':''}`} style={{marginBottom:20}}>
            <button className={`tab-b${!isEx?' on':''}`} onClick={()=>{setUtype('farmer');clr();setOtpStage(false);setRegStep(1);}}>🌾 Farmer</button>
            <button className={`tab-b${isEx?' on':''}`} onClick={()=>{setUtype('expert');clr();setOtpStage(false);setRegStep(1);}}>👨‍⚕️ Expert</button>
          </div>

          {mode==='login' ? (
            <>
              {!otpStage ? (
                <>
                  <div style={{display:'flex',gap:8,marginBottom:18}}>
                    {['otp','password'].map(m=>(
                      <button key={m} onClick={()=>setLmethod(m)} style={{flex:1,padding:'9px',borderRadius:9,fontSize:13,fontWeight:700,border:`2px solid ${lmethod===m?thm:'var(--br)'}`,background:lmethod===m?(isEx?'var(--bp)':'var(--gp)'):'none',color:lmethod===m?thm:'var(--tx2)',cursor:'pointer',fontFamily:"'Outfit',sans-serif",transition:'all .18s'}}>
                        {m==='otp'?'📱 OTP Login':'🔑 Password'}
                      </button>
                    ))}
                  </div>
                  <div className="fgrp">
                    <label className="flbl">Mobile Number</label>
                    <input className="finp" placeholder="10-digit mobile number" value={f.mobile} onChange={e=>upd('mobile',e.target.value.replace(/\D/,'').slice(0,10))} maxLength={10}/>
                    {errs.mobile&&<div className="ferr">⚠️ {errs.mobile}</div>}
                  </div>
                  {lmethod==='password'&&(
                    <div className="fgrp">
                      <label className="flbl">Password</label>
                      <input className="finp" type="password" placeholder="Aapka password" value={f.password} onChange={e=>upd('password',e.target.value)}/>
                      {errs.password&&<div className="ferr">⚠️ {errs.password}</div>}
                    </div>
                  )}
                  <button className="btn btn-g btn-full" style={{background:thm,boxShadow:`0 3px 14px ${isEx?'rgba(26,111,212,.26)':'rgba(30,126,66,.28)'}`}} onClick={doLogin}>
                    {busy?<><div className="spin"/>Processing...</>:lmethod==='otp'?'OTP Bhejo →':'Login Karo →'}
                  </button>
                </>
              ):(
                <>
                  <div style={{textAlign:'center',fontSize:13.5,color:'var(--tx2)',marginBottom:4}}>
                    6-digit OTP bheja gaya: <strong>+91 {f.mobile}</strong>
                  </div>
                  <div style={{textAlign:'center',fontSize:12,color:'var(--tx3)',marginBottom:4}}>(Demo: koi bhi 6 digits chalenge)</div>
                  <div className="otp-row">
                    {otp.map((v,i)=>(
                      <input key={i} id={`oc${i}`} className="otp-c" maxLength={1} value={v} onChange={e=>handleOtp(i,e.target.value)}/>
                    ))}
                  </div>
                  {errs.otp&&<div className="ferr" style={{justifyContent:'center',marginBottom:8}}>⚠️ {errs.otp}</div>}
                  <div style={{textAlign:'center',fontSize:13,color:'var(--tx2)',margin:'4px 0 14px'}}>
                    Code nahi mila? <span style={{color:thm,fontWeight:700,cursor:'pointer'}}>Resend</span>
                  </div>
                  <button className="btn btn-g btn-full" style={{background:thm}} onClick={doLogin}>
                    {busy?<><div className="spin"/>Login Ho Raha Hai...</>:'✅ Login Karo'}
                  </button>
                  <button style={{width:'100%',marginTop:9,padding:'10px',border:'none',background:'none',color:'var(--tx3)',fontSize:13,cursor:'pointer'}} onClick={()=>setOtpStage(false)}>← Wapas Jao</button>
                </>
              )}
              <div className="auth-or">OR</div>
              <button className="g-btn"><span style={{fontWeight:900,fontSize:16,color:'#4285f4'}}>G</span> Continue with Google</button>
              <div className="auth-sw">Account nahi hai? <span className={isEx?'ex-link':''} onClick={()=>{setMode('register');clr();setRegStep(1);}}>Register Karein</span></div>
            </>
          ):(
            <>
              {/* Step indicator */}
              <div className="steps-row" style={{marginBottom:22}}>
                {['Personal','Location','Password'].map((s,i)=>(
                  <div key={i} style={{display:'flex',alignItems:'center',flex:1}}>
                    <div style={{display:'flex',flexDirection:'column',alignItems:'center'}}>
                      <div className={`step-dot${regStep>i+1?' done':regStep===i+1?' act':''}`} style={regStep===i+1?{borderColor:thm,color:isEx?'var(--b3)':'var(--g3)'}:{}}>
                        {regStep>i+1?'✓':i+1}
                      </div>
                      <div className="step-lbl">{s}</div>
                    </div>
                    {i<2&&<div className={`step-ln${regStep>i+1?' done':''}`} style={regStep>i+1?{background:thm}:{}}/>}
                  </div>
                ))}
              </div>

              {regStep===1&&(
                <div className="fade-in">
                  <div className="fgrp">
                    <label className="flbl">Poora Naam *</label>
                    <input className="finp" placeholder="Aapka poora naam" value={f.name} onChange={e=>upd('name',e.target.value)}/>
                    {errs.name&&<div className="ferr">⚠️ {errs.name}</div>}
                  </div>
                  <div className="fgrp">
                    <label className="flbl">Mobile Number *</label>
                    <input className="finp" placeholder="10-digit mobile" value={f.mobile} onChange={e=>upd('mobile',e.target.value.replace(/\D/,'').slice(0,10))} maxLength={10}/>
                    {errs.mobile&&<div className="ferr">⚠️ {errs.mobile}</div>}
                  </div>
                  <div className="fgrp">
                    <label className="flbl">Email (Optional)</label>
                    <input className="finp" type="email" placeholder="Email address" value={f.email} onChange={e=>upd('email',e.target.value)}/>
                    {errs.email&&<div className="ferr">⚠️ {errs.email}</div>}
                  </div>
                  {isEx&&(
                    <>
                      <div className="frow">
                        <div className="fgrp">
                          <label className="flbl">Specialization *</label>
                          <select className="fsel" value={f.spec} onChange={e=>upd('spec',e.target.value)}>
                            <option value="">Select</option>
                            <option>Plant Pathologist</option>
                            <option>Horticulture Expert</option>
                            <option>Soil Scientist</option>
                            <option>Crop Scientist</option>
                            <option>Agri Economist</option>
                          </select>
                        </div>
                        <div className="fgrp">
                          <label className="flbl">Fee (₹/session)</label>
                          <input className="finp" type="number" placeholder="e.g. 800" value={f.fee} onChange={e=>upd('fee',e.target.value)}/>
                        </div>
                      </div>
                      <div className="fgrp">
                        <label className="flbl">University / Institution</label>
                        <input className="finp" placeholder="e.g. IARI New Delhi" value={f.university} onChange={e=>upd('university',e.target.value)}/>
                      </div>
                    </>
                  )}
                </div>
              )}

              {regStep===2&&(
                <div className="fade-in">
                  {!isEx ? (
                    <>
                      {/* STATE */}
                      <div className="fgrp">
                        <label className="flbl">State *</label>
                        <select className="fsel" value={f.state} onChange={e=>{upd('state',e.target.value);upd('district','');upd('taluka','');upd('village','');}}>
                          <option value="">-- State Select Karein --</option>
                          {INDIA_STATES.map(s=><option key={s} value={s}>{s}</option>)}
                        </select>
                      </div>
                      {/* DISTRICT + TALUKA */}
                      <div className="frow">
                        <div className="fgrp">
                          <label className="flbl">District *</label>
                          <select className="fsel" value={f.district} onChange={e=>{upd('district',e.target.value);upd('taluka','');upd('village','');}} disabled={!f.state}>
                            <option value="">-- District --</option>
                            {getDistricts(f.state).map(d=><option key={d} value={d}>{d}</option>)}
                          </select>
                          {errs.district&&<div className="ferr">⚠️ {errs.district}</div>}
                        </div>
                        <div className="fgrp">
                          <label className="flbl">Taluka / Block</label>
                          <select className="fsel" value={f.taluka} onChange={e=>{upd('taluka',e.target.value);upd('village','');}} disabled={!f.district}>
                            <option value="">-- Taluka --</option>
                            {getStateTalukas(f.state,f.district).map(t=><option key={t} value={t}>{t}</option>)}
                          </select>
                        </div>
                      </div>
                      {/* VILLAGE + SOIL */}
                      <div className="frow">
                        <div className="fgrp">
                          <label className="flbl">Village / Gaon</label>
                          <input className="finp" placeholder="Aapke gaon ka naam" value={f.village} onChange={e=>upd('village',e.target.value)}/>
                        </div>
                        <div className="fgrp">
                          <label className="flbl">Soil Type</label>
                          <select className="fsel" value={f.soil} onChange={e=>upd('soil',e.target.value)}>
                            <option value="">-- Soil Type --</option>
                            {SOILS.map(s=><option key={s} value={s}>{s}</option>)}
                          </select>
                        </div>
                      </div>
                      {/* FARM SIZE */}
                      <div className="frow">
                        <div className="fgrp">
                          <label className="flbl">Farm Size (Acres)</label>
                          <input className="finp" type="number" placeholder="e.g. 5" value={f.farmSize||''} onChange={e=>upd('farmSize',e.target.value)}/>
                        </div>
                        <div className="fgrp">
                          <label className="flbl">Irrigation Source</label>
                          <select className="fsel" value={f.irrigation||''} onChange={e=>upd('irrigation',e.target.value)}>
                            <option value="">Select</option>
                            {['Borewell / Tube well','Canal / Nahr','Rainwater / Baarish','Drip Irrigation','River / Nadi','Tank / Taalaab','None / Barani'].map(o=><option key={o}>{o}</option>)}
                          </select>
                        </div>
                      </div>
                    </>
                  ):(
                    <>
                      {/* Expert State + District */}
                      <div className="frow">
                        <div className="fgrp">
                          <label className="flbl">State *</label>
                          <select className="fsel" value={f.state} onChange={e=>{upd('state',e.target.value);upd('district','');}}>
                            <option value="">-- State --</option>
                            {INDIA_STATES.map(s=><option key={s} value={s}>{s}</option>)}
                          </select>
                        </div>
                        <div className="fgrp">
                          <label className="flbl">District *</label>
                          <select className="fsel" value={f.district} onChange={e=>upd('district',e.target.value)} disabled={!f.state}>
                            <option value="">-- District --</option>
                            {getDistricts(f.state).map(d=><option key={d} value={d}>{d}</option>)}
                          </select>
                          {errs.district&&<div className="ferr">⚠️ {errs.district}</div>}
                        </div>
                      </div>
                      <div className="fgrp">
                        <label className="flbl">Languages Spoken</label>
                        <input className="finp" placeholder="e.g. Hindi, English, Marathi, Punjabi" value={f.langs} onChange={e=>upd('langs',e.target.value)}/>
                      </div>
                      <div className="frow">
                        <div className="fgrp">
                          <label className="flbl">Years of Experience</label>
                          <input className="finp" type="number" placeholder="e.g. 10" value={f.expYrs||''} onChange={e=>upd('expYrs',e.target.value)}/>
                        </div>
                        <div className="fgrp">
                          <label className="flbl">Crop Specializations</label>
                          <input className="finp" placeholder="e.g. Tomato, Wheat, Cotton" value={f.crops||''} onChange={e=>upd('crops',e.target.value)}/>
                        </div>
                      </div>
                      <div style={{padding:13,background:'var(--bp)',borderRadius:10,fontSize:13,color:'var(--b1)',fontWeight:600}}>
                        ℹ️ Documents verify honge approval ke baad (2–3 din)
                      </div>
                    </>
                  )}
                </div>
              )}

              {regStep===3&&(
                <div className="fade-in">
                  <div style={{textAlign:'center',padding:14,background:'var(--gp)',borderRadius:10,marginBottom:18,fontSize:13.5,color:'var(--g2)',fontWeight:600}}>
                    🎉 Almost done! Sirf password set karein.
                  </div>
                  <div className="fgrp">
                    <label className="flbl">Password Banayein *</label>
                    <input className="finp" type="password" placeholder="Min 8 characters" value={f.password} onChange={e=>upd('password',e.target.value)}/>
                    {errs.password&&<div className="ferr">⚠️ {errs.password}</div>}
                    <div style={{fontSize:11,color:'var(--tx3)',marginTop:4}}>Letters, numbers & symbols mix karein</div>
                  </div>
                  <div className="fgrp">
                    <label className="flbl">Confirm Password *</label>
                    <input className="finp" type="password" placeholder="Dobara likhein" value={f.confirm} onChange={e=>upd('confirm',e.target.value)}/>
                    {errs.confirm&&<div className="ferr">⚠️ {errs.confirm}</div>}
                  </div>
                  <div style={{fontSize:12,color:'var(--tx3)',marginBottom:14,lineHeight:1.6}}>
                    Account banake aap BeejHealth Terms of Service aur Privacy Policy se agree karte hain.
                  </div>
                </div>
              )}

              <div style={{display:'flex',gap:9,marginTop:6}}>
                {regStep>1&&<button className="btn btn-out btn-md" style={{flex:1}} onClick={()=>setRegStep(p=>p-1)}>← Wapas</button>}
                <button className="btn btn-g btn-md" style={{flex:2,background:thm}} onClick={doRegNext}>
                  {busy?<><div className="spin"/>Saving...</>:regStep<3?'Aage Badho →':'✅ Account Banao'}
                </button>
              </div>
              <div className="auth-or">OR</div>
              <button className="g-btn"><span style={{fontWeight:900,fontSize:16,color:'#4285f4'}}>G</span> Continue with Google</button>
              <div className="auth-sw">Account hai? <span className={isEx?'ex-link':''} onClick={()=>{setMode('login');clr();setRegStep(1);}}>Login Karein</span></div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   FARMER ONBOARDING
════════════════════════════════════════════════════════════════ */
function FarmerOnboarding({user,onDone,setUser}) {
  const [step,setStep]=useState(1);
  const [sel,setSel]=useState({district:user?.district||'Pune',crops:['tomato','wheat'],notifs:{disease:true,weather:true,market:true,expert:true}});
  const [saving,setSaving]=useState(false);
  const steps=4;

  const finishOnboarding=async()=>{
    setSaving(true);
    try{
      const res=await API.patch('/api/auth/profile',{
        district: sel.district,
        crops:    sel.crops,
        langs:    'Hindi',
      });
      if(res.user){
        saveSession(localStorage.getItem('bh_token'),res.user);
        if(setUser) setUser(res.user);
      }
    }catch(e){ console.warn('Onboarding save:',e.message); }
    setSaving(false);
    onDone();
  };
  return (
    <div className="ob-wrap">
      <div className="ob-box">
        <div className="ob-head">
          <div style={{fontSize:12,opacity:.76,marginBottom:5}}>🌱 BeejHealth Setup — {step}/{steps}</div>
          <div style={{fontFamily:"'Baloo 2',cursive",fontSize:24,fontWeight:900}}>Swagat Hai, {user?.name?.split(' ')?.[0]||'Kisan'}! 🎉</div>
          <div style={{fontSize:13.5,opacity:.8,marginTop:3}}>Sirf 2 minute — platform taiyar karte hain</div>
        </div>
        <div style={{padding:'0 30px 4px'}}>
          <div className="ob-prog" style={{margin:'0',paddingTop:18,paddingBottom:4}}>
            {Array.from({length:steps},(_,i)=><div key={i} className={`ob-step${i<step?' done':''}`}/>)}
          </div>
        </div>
        <div className="ob-body">
          {step===1&&<div className="slide-up">
            <div className="ob-sec-t">📍 Aapka Location</div>
            <div className="ob-sec-p">Sahi experts aur disease alerts ke liye</div>
            <div className="frow">
              <div className="fgrp"><label className="flbl">District</label>
                <select className="fsel" value={sel.district} onChange={e=>setSel(p=>({...p,district:e.target.value}))}>
                  {DISTRICTS.map(d=><option key={d}>{d}</option>)}
                </select>
              </div>
              <div className="fgrp"><label className="flbl">Taluka</label>
                <select className="fsel">{TALUKAS.map(t=><option key={t}>{t}</option>)}</select>
              </div>
            </div>
            <button style={{width:'100%',padding:'12px',background:'var(--gp)',border:'2px dashed var(--br2)',borderRadius:10,fontSize:13.5,fontWeight:600,color:'var(--g2)',cursor:'pointer',marginBottom:6}}>
              📍 GPS Se Location Use Karo
            </button>
          </div>}
          {step===2&&<div className="slide-up">
            <div className="ob-sec-t">🌾 Aapki Fasalein</div>
            <div className="ob-sec-p">Kaunsi crops ughate hain? (Multiple select)</div>
            <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:9}}>
              {CROPS.slice(0,8).map(c=>(
                <div key={c.id} onClick={()=>setSel(p=>({...p,crops:p.crops.includes(c.id)?p.crops.filter(x=>x!==c.id):[...p.crops,c.id]}))}
                  style={{padding:'11px 6px',borderRadius:10,border:`2px solid ${sel.crops.includes(c.id)?'var(--g4)':'var(--br)'}`,background:sel.crops.includes(c.id)?'var(--gp)':'white',textAlign:'center',cursor:'pointer',transition:'all .18s'}}>
                  <div style={{fontSize:26,marginBottom:5}}>{c.emoji}</div>
                  <div style={{fontSize:11,fontWeight:700,color:'var(--tx)'}}>{c.name.split(' ')[0]}</div>
                </div>
              ))}
            </div>
          </div>}
          {step===3&&<div className="slide-up">
            <div className="ob-sec-t">🔔 Notification Preferences</div>
            <div className="ob-sec-p">Kaunsi alerts chahiye?</div>
            {[{k:'disease',i:'🦠',l:'Disease Alerts',s:'Nearby outbreak mein',},{k:'weather',i:'🌦️',l:'Weather Warnings',s:'Spray timing ke liye'},{k:'market',i:'📈',l:'Market Price Alerts',s:'Jab crop ka bhav badhe'},{k:'expert',i:'💬',l:'Expert Replies',s:'Consultation updates'}].map(n=>(
              <div key={n.k} style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'12px 0',borderBottom:'1px solid var(--gp)'}}>
                <div style={{display:'flex',gap:11,alignItems:'center'}}>
                  <span style={{fontSize:20}}>{n.i}</span>
                  <div>
                    <div style={{fontSize:14,fontWeight:700,color:'var(--tx)'}}>{n.l}</div>
                    <div style={{fontSize:12,color:'var(--tx3)'}}>{n.s}</div>
                  </div>
                </div>
                <label className="sw">
                  <input type="checkbox" checked={sel.notifs[n.k]} onChange={e=>setSel(p=>({...p,notifs:{...p.notifs,[n.k]:e.target.checked}}))}/>
                  <span className="sw-sl"/>
                </label>
              </div>
            ))}
          </div>}
          {step===4&&<div className="slide-up" style={{textAlign:'center'}}>
            <div style={{fontSize:68,animation:'bounce 1.2s infinite',marginBottom:14}}>🎉</div>
            <div style={{fontFamily:"'Baloo 2',cursive",fontSize:24,fontWeight:900,color:'var(--g1)',marginBottom:7}}>Sab Ready Hai!</div>
            <div style={{fontSize:14,color:'var(--tx2)',lineHeight:1.75,marginBottom:20}}>Aapka BeejHealth account fully setup ho gaya.</div>
            <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:10}}>
              {[['🔬','AI Diagnosis','Photo se instant disease'],['👨‍⚕️','200+ Experts','Certified specialists'],['📊','Farm Analytics','Complete tracking'],['🌦️','Smart Alerts','Weather + disease']].map(([ic,t,d])=>(
                <div key={t} style={{padding:13,background:'var(--gp)',borderRadius:10,textAlign:'left'}}>
                  <div style={{fontSize:22,marginBottom:5}}>{ic}</div>
                  <div style={{fontSize:13,fontWeight:700,color:'var(--g1)'}}>{t}</div>
                  <div style={{fontSize:11,color:'var(--tx3)',marginTop:2}}>{d}</div>
                </div>
              ))}
            </div>
          </div>}
          <div style={{display:'flex',gap:9,marginTop:22}}>
            {step>1&&step<4&&<button className="btn btn-out btn-md" style={{flex:1}} onClick={()=>setStep(p=>p-1)}>← Wapas</button>}
            <button className="btn btn-g btn-md" style={{flex:2}} onClick={()=>step<4?setStep(p=>p+1):finishOnboarding()} disabled={saving}>
              {step===4?'🚀 Dashboard Par Jao':'Aage Badho →'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   EXPERT ONBOARDING
════════════════════════════════════════════════════════════════ */
function ExpertOnboarding({user,onDone,setUser}) {
  const finishExpertOnboarding=async(data)=>{
    try{
      const res=await API.patch('/api/auth/profile',{
        spec:       data.spec||'Agricultural Expert',
        fee:        Number(data.fee)||500,
        university: data.university||'',
        langs:      data.langs||'Hindi',
        available:  true,
      });
      if(res.user){
        saveSession(localStorage.getItem('bh_token'),res.user);
        if(setUser) setUser(res.user);
      }
    }catch(e){ console.warn('Expert onboarding save:',e.message); }
    onDone();
  };
  const [uploaded,setUploaded]=useState({id:false,degree:false,exp:false});
  return (
    <div className="ob-wrap">
      <div className="ob-box">
        <div className="ob-head ex">
          <div style={{fontSize:12,opacity:.76,marginBottom:5}}>👨‍⚕️ Expert Verification</div>
          <div style={{fontFamily:"'Baloo 2',cursive",fontSize:22,fontWeight:900}}>Namaskar, {user?.name||'Expert'}! 👋</div>
          <div style={{fontSize:13,opacity:.8,marginTop:3}}>Verification complete karo — 2–3 din mein approval</div>
        </div>
        <div className="ob-body">
          <div style={{background:'white',border:'1.5px solid var(--bpb)',borderRadius:'var(--rad)',padding:20}}>
            <div style={{fontSize:15,fontWeight:800,color:'var(--b1)',marginBottom:14}}>🔐 Verification Checklist</div>
            {[{k:'mobile',l:'Mobile Verified',s:'OTP se complete',done:true},{k:'id',l:'Government ID',s:'Aadhaar / PAN upload',done:uploaded.id},{k:'degree',l:'Degree Certificate',s:'University certificate',done:uploaded.degree},{k:'exp',l:'Experience Letter',s:'Employer / registration',done:uploaded.exp}].map(item=>(
              <div key={item.k} className="ver-item">
                <div className="ver-ic" style={{background:item.done?'var(--gp)':'var(--bp)'}}>{item.done?'✅':'📄'}</div>
                <div style={{flex:1}}>
                  <div style={{fontSize:14,fontWeight:700,color:'var(--tx)'}}>{item.l}</div>
                  <div style={{fontSize:12,color:'var(--tx3)'}}>{item.s}</div>
                </div>
                {!item.done&&<button className="btn btn-b btn-sm" onClick={()=>setUploaded(p=>({...p,[item.k]:true}))}>📤 Upload</button>}
              </div>
            ))}
          </div>
          <div style={{padding:13,background:'var(--bp)',borderRadius:10,margin:'16px 0',fontSize:13,color:'var(--b1)',lineHeight:1.65}}>
            <strong>ℹ️ Process:</strong> Documents review 2–3 business days. Tab tak platform explore kar sakte hain.
          </div>
          <div style={{display:'flex',gap:9}}>
            <button className="btn btn-out-b btn-md" style={{flex:1}} onClick={()=>finishExpertOnboarding({})}>Baad Mein</button>
            <button className="btn btn-b btn-md" style={{flex:2}} onClick={()=>finishExpertOnboarding({})}>📋 Dashboard Dekho →</button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   HOME PAGE
════════════════════════════════════════════════════════════════ */
function HomePage({nav,setAuth,setAuthMode,user}) {
  const [contactForm,setContactForm]=useState({name:'',mobile:'',email:'',company:'',subject:'',message:'',type:'farmer'});
  const [contactBusy,setContactBusy]=useState(false);
  const [contactDone,setContactDone]=useState(false);
  const [faqOpen,setFaqOpen]=useState(null);

  const submitContact=async()=>{
    if(!contactForm.name.trim()||!contactForm.message.trim()){
      alert('Naam aur message zaroor bharo'); return;
    }
    setContactBusy(true);
    try{
      await API.post('/api/consultations',{
        cropId:'contact', cropName:'Website Enquiry', cropEmoji:'📩',
        method:'manual', disease:'Contact Form: '+contactForm.subject,
        confidence:100, severity:1, answers:{...contactForm}
      });
      setContactDone(true);
    }catch(e){ alert('Send fail hua, dobara try karein'); }
    setContactBusy(false);
  };

  const FAQS=[
    {q:'BeejHealth kaise kaam karta hai?',a:'Aap apni fasal ki photo upload karte hain, AI (EfficientNetV2 model) se disease detect hoti hai 94%+ accuracy ke saath, phir certified expert se consultation lete hain.'},
    {q:'Kya ye free hai?',a:'Basic AI scan bilkul free hai. Expert consultation ₹300-1000 per session hai. Insurance claims aur B2B pricing alag hai.'},
    {q:'Konsi crops support karta hai?',a:'Abhi 17 crops: Tomato, Wheat, Potato, Cotton, Corn, Apple, Grape, Orange, Pepper, Soybean, Strawberry, Peach, Cherry, Blueberry, Squash, Raspberry, aur Coconut.'},
    {q:'Expert real doctors hain kya?',a:'Haan — sab certified agricultural scientists hain. Plant Pathologists, Soil Scientists, Horticulture Experts — IARI, PAU, BHU jaise institutions se trained.'},
    {q:'Data private rahega?',a:'Haan. Aapki farm data, photos aur consultations fully encrypted hain. Kisi third party ko share nahi karte.'},
    {q:'Coconut ke liye special kya hai?',a:'Coconut ke liye EfficientNetV2-S Transfer Learning model se 8 disease classes detect hoti hain — Gray Leaf Spot, Leaf Rot, Bud Rot, Stem Bleeding. 30 adaptive questions sirf coconut ke liye.'},
    {q:'Robot farming kab aayegi?',a:'BeejHealth Robot 2025-26 mein launch hoga — autonomous field scanning, targeted spray, real-time crop monitoring.'},
    {q:'Kya offline bhi kaam karta hai?',a:'Basic features offline mein bhi kaam karte hain. Expert consultation ke liye internet chahiye. PWA support coming soon.'},
  ];

  const TESTIMONIALS=[
    {name:'Ramesh Patil',loc:'Pune, Maharashtra',crop:'🍅 Tomato',text:'BeejHealth ne meri poori tomato fasal bachi. AI ne exactly bata diya — Early Blight hai aur Mancozeb spray karo. 7 din mein result dikha.',rating:5,saved:'₹45,000'},
    {name:'Gurpreet Singh',loc:'Ludhiana, Punjab',crop:'🌾 Wheat',text:'Expert Dr. Rajesh Kumar ne video call pe ek ghante mein poori problem solve kar di. Itni sasti expert consultation pehle kabhi nahi mili.',rating:5,saved:'₹28,000'},
    {name:'Kavita Devi',loc:'Solapur, Maharashtra',crop:'🌸 Cotton',text:'Cotton mein Bacterial Blight aa gayi thi, photo upload ki, 30 second mein report aa gayi. Kamaal ki service!',rating:5,saved:'₹62,000'},
    {name:'Mohammad Rizwan',loc:'Nanded, Maharashtra',crop:'🥥 Coconut',text:'Nariyal ke ped mein Bud Rot ho rahi thi. BeejHealth ke coconut specialist questions se exact diagnosis hua. Shukriya!',rating:5,saved:'₹38,000'},
    {name:'Priya Shinde',loc:'Nashik, Maharashtra',crop:'🍇 Grape',text:'Grape export quality maintain karna tha. AI scan + expert se spray schedule banaya. Is season pehle se zyada production mili.',rating:5,saved:'₹1,20,000'},
    {name:'Suresh Kumar',loc:'Nagpur, Maharashtra',crop:'🍊 Orange',text:'Satellite map se apna bagaan dekha, soil sensor data samjha. Technology ne farming ko alag hi level pe le gaya.',rating:5,saved:'₹55,000'},
  ];

  const TEAM=[
    {name:'Dr. Anjali Sharma',role:'Founder & CEO',bg:'var(--g4)',init:'AS',desc:'Ex-IARI scientist, 15 years agricultural AI research'},
    {name:'Rahul Mehta',role:'CTO',bg:'var(--b3)',init:'RM',desc:'IIT Bombay, Ex-Google, ML & computer vision expert'},
    {name:'Dr. Priya Nair',role:'Chief Agro Officer',bg:'var(--a2)',init:'PN',desc:'Kerala Agri Univ, plant pathology specialist'},
    {name:'Vikram Singh',role:'Head of Operations',bg:'var(--g3)',init:'VS',desc:'Ex-Mahindra Agri, farmer network across 12 states'},
  ];

  return (
    <>
      {/* ══ HERO — Light green, original style ══════════════ */}
      <section className="hero">
        <div style={{position:'absolute',width:600,height:600,background:'radial-gradient(circle,rgba(77,189,122,.1),transparent)',top:-100,right:-80,borderRadius:'50%',pointerEvents:'none'}}/>
        <div style={{position:'absolute',width:350,height:350,background:'radial-gradient(circle,rgba(26,111,212,.07),transparent)',bottom:-60,left:-40,borderRadius:'50%',pointerEvents:'none'}}/>
        <div className="hero-in">
          <div style={{animation:'slideUp .5s ease'}}>
            <div className="hero-pill" style={{marginBottom:18}}>
              <div className="hero-dot"/>🏆 India's #1 AI Farming Platform — 50,000+ Farmers Trust Us
            </div>
            <h1 className="hero-h1">Apni Fasal Ko<br/><em>Smart Banao</em> 🌱</h1>
            <p className="hero-p">
              BeejHealth — AI se crop disease instant detect karo, certified experts se real-time consult karo,
              weather alerts pao, aur apni farm ko digitally manage karo.{' '}
              <strong style={{color:'var(--g3)'}}>Free mein shuru karo.</strong>
            </p>
            <div className="hero-btns" style={{marginBottom:28}}>
              <button className="btn btn-g btn-xl" onClick={()=>nav('consultation')}>🔬 Free Scan Karo</button>
              <button className="btn btn-out btn-xl" onClick={()=>nav('experts')}>👨‍⚕️ Expert Dhundho</button>
            </div>
            {/* Trust badges */}
            <div style={{display:'flex',gap:14,flexWrap:'wrap',marginBottom:28}}>
              {['🔒 100% Secure','✅ ICAR Certified','🌐 12 Languages','📱 Works Offline'].map(b=>(
                <div key={b} style={{fontSize:12,fontWeight:700,color:'var(--tx3)',display:'flex',alignItems:'center',gap:4,background:'white',padding:'5px 12px',borderRadius:100,border:'1px solid var(--br)',boxShadow:'0 1px 4px rgba(0,0,0,.04)'}}>{b}</div>
              ))}
            </div>
            <div className="stats-row">
              {[['50K+','Farmers'],['200+','Experts'],['94%','Accuracy'],['₹12Cr+','Saved']].map(([n,l])=>(
                <div key={l}><div className="stat-n">{n}</div><div className="stat-l">{l}</div></div>
              ))}
            </div>
          </div>

          {/* Hero Demo Card */}
          <div className="hero-card" style={{animation:'slideUp .55s .1s ease both'}}>
            <div className="hc-lbl">🤖 Live AI Analysis Demo</div>
            <div className="dis-card">
              <div className="dc-crop">🥥 Coconut • Palakkad, Kerala</div>
              <div className="dc-name">Bud Rot Detected</div>
              <div className="dc-sci">Phytophthora palmivora • Stage 4/5</div>
              <div className="dc-bar-row"><span>AI Confidence</span><span style={{color:'var(--r2)',fontWeight:800}}>90.2%</span></div>
              <div className="dc-bar"><div className="dc-fill" style={{width:'90%',background:'var(--r3)'}}/></div>
              <div className="dc-bar-row" style={{marginTop:6}}><span>Treatment Urgency</span><span style={{color:'var(--a1)',fontWeight:800}}>HIGH</span></div>
              <div className="dc-bar"><div className="dc-fill" style={{width:'80%',background:'var(--a3)'}}/></div>
              <div style={{display:'flex',gap:6,marginTop:10,flexWrap:'wrap'}}>
                <span style={{padding:'4px 10px',background:'var(--rp)',borderRadius:100,fontSize:11,fontWeight:700,color:'var(--r2)'}}>🚨 Critical</span>
                <span style={{padding:'4px 10px',background:'var(--bp)',borderRadius:100,fontSize:11,fontWeight:700,color:'var(--b2)'}}>EfficientNetV2-S</span>
                <span style={{padding:'4px 10px',background:'var(--gp)',borderRadius:100,fontSize:11,fontWeight:700,color:'var(--g3)'}}>Expert Assigned</span>
              </div>
              <div className="dc-pill" style={{marginTop:10}}>💊 Ridomil MZ 72 WP — Apply immediately</div>
            </div>
            <div style={{display:'flex',gap:8,marginTop:12}}>
              <button className="btn btn-g btn-sm" style={{flex:1}} onClick={()=>nav('consultation')}>🔬 Try Free Scan</button>
              <button className="btn btn-out btn-sm" style={{flex:1}} onClick={()=>nav('experts')}>👨‍⚕️ Book Expert</button>
            </div>
            <div style={{marginTop:10,padding:'8px 12px',background:'var(--gp)',borderRadius:8,fontSize:12,color:'var(--tx3)',display:'flex',alignItems:'center',gap:8}}>
              <div style={{width:6,height:6,borderRadius:'50%',background:'var(--g4)',flexShrink:0}}/>
              <span>23 farmers currently online • 4 experts available</span>
            </div>
          </div>
        </div>
      </section>

      {/* ══ ACHIEVEMENT STATS ══════════════════════════════════ */}
      <section style={{padding:'32px 28px',background:'white',borderBottom:'1px solid var(--br)'}}>
        <div style={{maxWidth:1160,margin:'0 auto'}}>
          <div style={{display:'grid',gridTemplateColumns:'repeat(8,1fr)',gap:8}}>
            {[['👨‍🌾','50K+','Active Farmers'],['👨‍⚕️','200+','Certified Experts'],['🎯','94.3%','AI Accuracy'],['🦠','58+','Disease Classes'],['🌾','17','Crops Supported'],['💰','₹12Cr+','Farmer Savings'],['⭐','4.9★','App Rating'],['🕐','24/7','Expert Support']].map(([ic,n,l])=>(
              <div key={l} style={{textAlign:'center',padding:'12px 4px'}}>
                <div style={{fontSize:20,marginBottom:4}}>{ic}</div>
                <div style={{fontFamily:"'Baloo 2',cursive",fontSize:20,fontWeight:900,color:'var(--g1)'}}>{n}</div>
                <div style={{fontSize:10,color:'var(--tx3)',fontWeight:600,marginTop:2,lineHeight:1.3}}>{l}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ══ HOW IT WORKS ══════════════════════════════════════ */}
      <section style={{padding:'72px 28px',background:'var(--gb)'}}>
        <div style={{maxWidth:1160,margin:'0 auto'}}>
          <div style={{textAlign:'center',marginBottom:48}}>
            <div style={{fontFamily:"'Baloo 2',cursive",fontSize:34,fontWeight:900,color:'var(--g1)',marginBottom:8}}>⚡ Kaise Kaam Karta Hai?</div>
            <div style={{fontSize:15,color:'var(--tx2)'}}>3 easy steps — 30 seconds mein AI diagnosis</div>
          </div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:28,position:'relative'}}>
            <div style={{position:'absolute',top:52,left:'16%',right:'16%',height:2,background:'linear-gradient(90deg,var(--g4),var(--b3))',zIndex:0,borderRadius:2}}/>
            {[
              {n:'01',i:'📸',t:'Photo Lo ya Upload Karo',d:'Apni fasal ki koi bhi achi photo lo. Gallery se upload karo ya seedha camera se khicho.',c:'var(--g4)',bg:'var(--gp)'},
              {n:'02',i:'🤖',t:'AI Instant Diagnosis',d:'EfficientNetV2 model 30 seconds mein disease pehchanta hai. 58+ diseases, 94.3% accuracy.',c:'var(--b3)',bg:'var(--bp)'},
              {n:'03',i:'👨‍⚕️',t:'Expert Report & Solution',d:'Certified expert aapko AI report ke saath contact karta hai. Treatment plan, medicine list — sab detail mein.',c:'var(--a2)',bg:'var(--ap)'},
            ].map((s,i)=>(
              <div key={s.n} className="card" style={{padding:28,position:'relative',zIndex:1}}>
                <div style={{width:52,height:52,borderRadius:14,background:s.bg,border:`2px solid ${s.c}44`,display:'flex',alignItems:'center',justifyContent:'center',fontSize:24,marginBottom:16}}>{s.i}</div>
                <div style={{position:'absolute',top:18,right:20,fontFamily:"'Baloo 2',cursive",fontSize:44,fontWeight:900,color:'var(--br)',lineHeight:1}}>{s.n}</div>
                <div style={{fontSize:16,fontWeight:800,color:'var(--tx)',marginBottom:8}}>{s.t}</div>
                <div style={{fontSize:13.5,color:'var(--tx2)',lineHeight:1.7}}>{s.d}</div>
              </div>
            ))}
          </div>
          <div style={{textAlign:'center',marginTop:32}}>
            <button className="btn btn-g btn-xl" onClick={()=>nav('consultation')}>🚀 Abhi Shuru Karo — Free</button>
          </div>
        </div>
      </section>

      {/* ══ FEATURES GRID ══════════════════════════════════════ */}
      <section style={{padding:'72px 28px',background:'white'}}>
        <div style={{maxWidth:1160,margin:'0 auto'}}>
          <div style={{textAlign:'center',marginBottom:48}}>
            <div style={{fontFamily:"'Baloo 2',cursive",fontSize:34,fontWeight:900,color:'var(--g1)',marginBottom:8}}>🚀 Complete Farming Platform</div>
            <div style={{fontSize:15,color:'var(--tx2)'}}>Ek app mein sab kuch — AI, Experts, Weather, Market, Robots</div>
          </div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:20}}>
            {[
              {i:'🤖',t:'AI Disease Detection',d:'CNN + Transfer Learning. 30 seconds mein 58+ diseases detect. 94.3% accuracy on Indian crops.',tag:'Free',bg:'var(--gp)',tc:'var(--g3)'},
              {i:'👨‍⚕️',t:'Certified Expert Network',d:'200+ verified Plant Pathologists, Soil Scientists, Horticulture Experts. Video call, voice call, written report.',tag:'₹300+',bg:'var(--bp)',tc:'var(--b2)'},
              {i:'🥥',t:'Coconut Specialist AI',d:'Special EfficientNetV2-S model. 8 disease classes: Bud Rot, Stem Bleeding, Gray Leaf Spot, Leaf Rot.',tag:'New',bg:'var(--ap)',tc:'var(--a1)'},
              {i:'🌦️',t:'Weather Intelligence',d:'OpenWeatherMap + IMD data. Disease outbreak prediction. Spray timing alerts. District level forecast.',tag:'Live',bg:'var(--tp)',tc:'var(--t2)'},
              {i:'📊',t:'Farm Analytics',d:'Digital farm twin. Crop health history, disease trends, yield tracking, cost analysis. Satellite NDVI map.',tag:'Pro',bg:'var(--pup)',tc:'var(--pu)'},
              {i:'🛒',t:'AgriMart + Price Tracker',d:'AI-recommended medicines, seeds, fertilizers. Real-time mandi prices for 12 crops. Village delivery.',tag:'Live',bg:'var(--gp)',tc:'var(--g3)'},
              {i:'🏦',t:'Insurance & Govt Schemes',d:'PMFBY, WBCIS, Coconut Palm Insurance — ek click mein apply. Subsidy calculator.',tag:'Free',bg:'var(--bp)',tc:'var(--b2)'},
              {i:'🎤',t:'Voice Input (Hindi)',d:'Hindi mein bolo — AI samjhega. Crop naam, disease symptoms voice se describe karo.',tag:'Beta',bg:'var(--ap)',tc:'var(--a1)'},
              {i:'🤖',t:'Robot Fleet (2025+)',d:'BeejHealth autonomous robot — field scan, targeted spray, real-time monitoring. Early access open.',tag:'Soon',bg:'var(--tp)',tc:'var(--t2)'},
            ].map(f=>(
              <div key={f.t} className="card card-hov" style={{padding:22,background:f.bg,border:'none',position:'relative'}}>
                <div style={{position:'absolute',top:14,right:14,fontSize:10,fontWeight:800,padding:'3px 9px',borderRadius:100,background:'rgba(255,255,255,.8)',color:f.tc,border:`1px solid ${f.tc}33`}}>{f.tag}</div>
                <div style={{fontSize:30,marginBottom:12}}>{f.i}</div>
                <div style={{fontSize:15,fontWeight:800,color:'var(--tx)',marginBottom:7}}>{f.t}</div>
                <div style={{fontSize:13,color:'var(--tx2)',lineHeight:1.65}}>{f.d}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ══ TESTIMONIALS ══════════════════════════════════════ */}
      <section style={{padding:'72px 28px',background:'var(--gb)'}}>
        <div style={{maxWidth:1160,margin:'0 auto'}}>
          <div style={{textAlign:'center',marginBottom:44}}>
            <div style={{fontFamily:"'Baloo 2',cursive",fontSize:34,fontWeight:900,color:'var(--g1)',marginBottom:8}}>❤️ Farmers Ki Kahaniyan</div>
            <div style={{fontSize:15,color:'var(--tx2)'}}>50,000+ farmers ne apni fasal bachaayi — unhi ki zubani</div>
          </div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:20}}>
            {TESTIMONIALS.map((t,i)=>(
              <div key={i} className="card" style={{padding:22,borderTop:`3px solid var(--g4)`}}>
                <div style={{display:'flex',gap:2,marginBottom:10}}>
                  {[...Array(t.rating)].map((_,j)=><span key={j} style={{color:'var(--a2)',fontSize:14}}>★</span>)}
                </div>
                <div style={{fontSize:13.5,color:'var(--tx)',lineHeight:1.72,marginBottom:16,fontStyle:'italic'}}>"{t.text}"</div>
                <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
                  <div>
                    <div style={{fontWeight:800,fontSize:14,color:'var(--tx)'}}>{t.name}</div>
                    <div style={{fontSize:12,color:'var(--tx3)'}}>{t.crop} • {t.loc}</div>
                  </div>
                  <div style={{textAlign:'right'}}>
                    <div style={{fontSize:10,color:'var(--tx3)',marginBottom:2}}>Saved</div>
                    <div style={{fontFamily:"'Baloo 2',cursive",fontSize:18,fontWeight:900,color:'var(--g4)'}}>{t.saved}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ══ CROPS COVERAGE ══════════════════════════════════════ */}
      <section style={{padding:'56px 28px',background:'white'}}>
        <div style={{maxWidth:1160,margin:'0 auto'}}>
          <div style={{textAlign:'center',marginBottom:32}}>
            <div style={{fontFamily:"'Baloo 2',cursive",fontSize:30,fontWeight:900,color:'var(--g1)',marginBottom:6}}>🌾 17 Crops — Growing List</div>
            <div style={{fontSize:14,color:'var(--tx2)'}}>Har crop ke liye specialized disease database aur treatment protocols</div>
          </div>
          <div style={{display:'flex',flexWrap:'wrap',gap:10,justifyContent:'center'}}>
            {[['🍅','Tomato','58 diseases'],['🌾','Wheat','24 diseases'],['🥔','Potato','31 diseases'],['🌸','Cotton','28 diseases'],['🌽','Corn','22 diseases'],['🍎','Apple','35 diseases'],['🍇','Grape','29 diseases'],['🍊','Orange','26 diseases'],['🫑','Pepper','33 diseases'],['🫘','Soybean','18 diseases'],['🍓','Strawberry','21 diseases'],['🍑','Peach','27 diseases'],['🍒','Cherry','24 diseases'],['🫐','Blueberry','19 diseases'],['🎃','Squash','22 diseases'],['🍓','Raspberry','18 diseases'],['🥥','Coconut','8 AI classes','NEW']].map(([em,nm,cnt,tag])=>(
              <div key={nm} className="card card-hov" style={{padding:'10px 16px',display:'flex',alignItems:'center',gap:10,cursor:'pointer',minWidth:140,border:tag?'1.5px solid var(--g4)':'1.5px solid var(--br)'}} onClick={()=>nav('consultation')}>
                <span style={{fontSize:24}}>{em}</span>
                <div>
                  <div style={{fontSize:13,fontWeight:800,color:'var(--tx)',display:'flex',alignItems:'center',gap:5}}>
                    {nm}{tag&&<span style={{fontSize:9,background:'var(--g4)',color:'white',padding:'1px 6px',borderRadius:100,fontWeight:700}}>{tag}</span>}
                  </div>
                  <div style={{fontSize:11,color:'var(--tx3)'}}>{cnt}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ══ TEAM SECTION ══════════════════════════════════════ */}
      <section style={{padding:'72px 28px',background:'var(--gb)'}}>
        <div style={{maxWidth:1160,margin:'0 auto'}}>
          <div style={{textAlign:'center',marginBottom:44}}>
            <div style={{fontFamily:"'Baloo 2',cursive",fontSize:34,fontWeight:900,color:'var(--g1)',marginBottom:8}}>👥 Hamaari Team</div>
            <div style={{fontSize:15,color:'var(--tx2)'}}>Scientists, engineers aur farmers — milke farming ka future bana rahe hain</div>
          </div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:22}}>
            {TEAM.map(m=>(
              <div key={m.name} className="card" style={{padding:24,textAlign:'center'}}>
                <div style={{width:68,height:68,borderRadius:'50%',background:m.bg,display:'flex',alignItems:'center',justifyContent:'center',color:'white',fontSize:22,fontWeight:900,margin:'0 auto 14px'}}>{m.init}</div>
                <div style={{fontWeight:800,fontSize:15,color:'var(--tx)',marginBottom:3}}>{m.name}</div>
                <div style={{fontSize:12,color:'var(--g4)',fontWeight:700,marginBottom:8}}>{m.role}</div>
                <div style={{fontSize:12,color:'var(--tx2)',lineHeight:1.6}}>{m.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ══ FAQ SECTION ══════════════════════════════════════ */}
      <section style={{padding:'72px 28px',background:'white'}}>
        <div style={{maxWidth:760,margin:'0 auto'}}>
          <div style={{textAlign:'center',marginBottom:44}}>
            <div style={{fontFamily:"'Baloo 2',cursive",fontSize:34,fontWeight:900,color:'var(--g1)',marginBottom:8}}>❓ Aksar Pooche Jaane Wale Sawaal</div>
            <div style={{fontSize:15,color:'var(--tx2)'}}>Koi sawaal hai? Hum yahan hain</div>
          </div>
          <div style={{display:'flex',flexDirection:'column',gap:10}}>
            {FAQS.map((f,i)=>(
              <div key={i} className="card" style={{overflow:'hidden',cursor:'pointer'}} onClick={()=>setFaqOpen(faqOpen===i?null:i)}>
                <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'15px 20px'}}>
                  <div style={{fontWeight:700,fontSize:14.5,color:'var(--tx)',paddingRight:16}}>{f.q}</div>
                  <div style={{fontSize:20,color:'var(--g4)',flexShrink:0,transform:faqOpen===i?'rotate(45deg)':'none',transition:'transform .2s',lineHeight:1}}>+</div>
                </div>
                {faqOpen===i&&(
                  <div style={{padding:'0 20px 16px',fontSize:13.5,color:'var(--tx2)',lineHeight:1.72,borderTop:'1px solid var(--br)',paddingTop:12}}>
                    {f.a}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ══ CONTACT / ENQUIRY FORM ══════════════════════════ */}
      <section style={{padding:'72px 28px',background:'var(--gb)'}}>
        <div style={{maxWidth:1100,margin:'0 auto',display:'grid',gridTemplateColumns:'1fr 1fr',gap:48,alignItems:'start'}}>
          <div>
            <div style={{fontFamily:"'Baloo 2',cursive",fontSize:34,fontWeight:900,color:'var(--g1)',marginBottom:10}}>📩 Humse Baat Karo</div>
            <div style={{fontSize:15,color:'var(--tx2)',lineHeight:1.75,marginBottom:28}}>
              Farmer ho, expert banna chahte ho, B2B partnership chahiye, ya koi bhi sawaal hai — hum yahan hain. 24 ghante mein reply guaranteed.
            </div>
            <div style={{display:'flex',flexDirection:'column',gap:18}}>
              {[
                {i:'📧',t:'Email',v:'support@beejhealth.com'},
                {i:'📱',t:'WhatsApp / Phone',v:'+91 98765 43210'},
                {i:'📍',t:'Office',v:'Bandra Kurla Complex, Mumbai, Maharashtra 400051'},
                {i:'🕐',t:'Support Hours',v:'Mon-Sat: 8 AM – 8 PM IST'},
              ].map(c=>(
                <div key={c.t} style={{display:'flex',gap:14,alignItems:'flex-start'}}>
                  <div style={{width:40,height:40,borderRadius:10,background:'var(--gp)',border:'1px solid var(--br)',display:'flex',alignItems:'center',justifyContent:'center',fontSize:18,flexShrink:0}}>{c.i}</div>
                  <div>
                    <div style={{fontSize:11,fontWeight:700,color:'var(--tx3)',textTransform:'uppercase',letterSpacing:.4,marginBottom:2}}>{c.t}</div>
                    <div style={{fontSize:14,fontWeight:600,color:'var(--tx)'}}>{c.v}</div>
                  </div>
                </div>
              ))}
            </div>
            <div style={{marginTop:24}}>
              <div style={{fontSize:12,fontWeight:700,color:'var(--tx3)',marginBottom:10,textTransform:'uppercase',letterSpacing:.4}}>Social Media</div>
              <div style={{display:'flex',gap:9}}>
                {['📷','🐦','💼','📘','▶️'].map((ic,j)=>(
                  <div key={j} style={{width:38,height:38,borderRadius:9,background:'white',border:'1px solid var(--br)',display:'flex',alignItems:'center',justifyContent:'center',cursor:'pointer',fontSize:16}}>{ic}</div>
                ))}
              </div>
            </div>
          </div>

          <div className="card" style={{padding:28}}>
            {contactDone?(
              <div style={{textAlign:'center',padding:'36px 16px'}}>
                <div style={{fontSize:52,marginBottom:14}}>🎉</div>
                <div style={{fontFamily:"'Baloo 2',cursive",fontSize:22,fontWeight:900,color:'var(--g1)',marginBottom:8}}>Message Bhej Diya!</div>
                <div style={{fontSize:14,color:'var(--tx2)',lineHeight:1.7}}>24 ghante mein humari team aapse contact karegi.</div>
                <button className="btn btn-g btn-md" style={{marginTop:18}} onClick={()=>setContactDone(false)}>Dobara Bhejo</button>
              </div>
            ):(
              <>
                <div style={{fontFamily:"'Baloo 2',cursive",fontSize:20,fontWeight:900,color:'var(--g1)',marginBottom:18}}>✉️ Enquiry Form</div>
                <div className="fgrp">
                  <label className="flbl">Aap Kaun Hain? *</label>
                  <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:8}}>
                    {[['farmer','👨‍🌾 Farmer'],['expert','👨‍⚕️ Expert Banna'],['business','💼 Business']].map(([v,l])=>(
                      <button key={v} onClick={()=>setContactForm(p=>({...p,type:v}))}
                        style={{padding:'9px 6px',borderRadius:9,fontSize:12,fontWeight:700,border:`2px solid ${contactForm.type===v?'var(--g4)':'var(--br)'}`,background:contactForm.type===v?'var(--gp)':'white',color:contactForm.type===v?'var(--g3)':'var(--tx2)',cursor:'pointer'}}>{l}</button>
                    ))}
                  </div>
                </div>
                <div className="frow">
                  <div className="fgrp">
                    <label className="flbl">Naam *</label>
                    <input className="finp" placeholder="Aapka poora naam" value={contactForm.name} onChange={e=>setContactForm(p=>({...p,name:e.target.value}))}/>
                  </div>
                  <div className="fgrp">
                    <label className="flbl">Mobile *</label>
                    <input className="finp" placeholder="10-digit" value={contactForm.mobile} maxLength={10} onChange={e=>setContactForm(p=>({...p,mobile:e.target.value.replace(/[^0-9]/g,'')}))}/>
                  </div>
                </div>
                <div className="frow">
                  <div className="fgrp">
                    <label className="flbl">Email</label>
                    <input className="finp" type="email" placeholder="email@example.com" value={contactForm.email} onChange={e=>setContactForm(p=>({...p,email:e.target.value}))}/>
                  </div>
                  <div className="fgrp">
                    <label className="flbl">Company / Farm</label>
                    <input className="finp" placeholder="Optional" value={contactForm.company} onChange={e=>setContactForm(p=>({...p,company:e.target.value}))}/>
                  </div>
                </div>
                <div className="fgrp">
                  <label className="flbl">Subject</label>
                  <select className="fsel" value={contactForm.subject} onChange={e=>setContactForm(p=>({...p,subject:e.target.value}))}>
                    <option value="">-- Topic Select Karein --</option>
                    <option>Crop Disease Help</option>
                    <option>Expert Registration</option>
                    <option>B2B / Bulk Partnership</option>
                    <option>Investment / Funding</option>
                    <option>Technical Support</option>
                    <option>Media / Press</option>
                    <option>Other</option>
                  </select>
                </div>
                <div className="fgrp">
                  <label className="flbl">Message *</label>
                  <textarea className="ftxt" rows={4} placeholder="Aapka sawaal ya message..." value={contactForm.message} onChange={e=>setContactForm(p=>({...p,message:e.target.value}))}/>
                </div>
                <button className="btn btn-g btn-full" onClick={submitContact} disabled={contactBusy||!contactForm.name||!contactForm.message}>
                  {contactBusy?<><div className="spin"/>Sending...</>:'📩 Message Bhejo →'}
                </button>
                <div style={{fontSize:11,color:'var(--tx4)',textAlign:'center',marginTop:8}}>🔒 Aapki jankari 100% safe hai. Koi spam nahi.</div>
              </>
            )}
          </div>
        </div>
      </section>

      {/* ══ CTA BANNER — green theme (no dark) ══════════════ */}
      <section style={{padding:'60px 28px',background:'var(--g1)',textAlign:'center'}}>
        <div style={{maxWidth:680,margin:'0 auto'}}>
          <div style={{fontFamily:"'Baloo 2',cursive",fontSize:34,fontWeight:900,color:'white',marginBottom:10}}>🌱 Aaj Se Hi Shuru Karo</div>
          <div style={{fontSize:15,color:'rgba(255,255,255,.82)',marginBottom:26,lineHeight:1.75}}>
            50,000+ farmers already BeejHealth use kar rahe hain. Free scan karo, expert se milo, apni fasal ko bachao.
          </div>
          <div style={{display:'flex',gap:12,justifyContent:'center',flexWrap:'wrap'}}>
            <button className="btn btn-xl" style={{padding:'13px 34px',background:'var(--g5)',color:'white',border:'none',fontSize:15,borderRadius:12}} onClick={()=>nav('consultation')}>
              🔬 Free Scan Karo
            </button>
            <button className="btn btn-xl" style={{padding:'13px 34px',background:'transparent',color:'white',border:'1.5px solid rgba(255,255,255,.4)',fontSize:15,borderRadius:12}} onClick={()=>{setAuthMode('register');setAuth(true);}}>
              📝 Register Karo
            </button>
          </div>
          <div style={{marginTop:16,fontSize:12,color:'rgba(255,255,255,.5)'}}>No credit card required • Free forever for basic features</div>
        </div>
      </section>

      {/* ══ FOOTER ══════════════════════════════════════════ */}
      <footer className="footer">
        <div style={{maxWidth:1160,margin:'0 auto'}}>
          <div style={{display:'grid',gridTemplateColumns:'2fr 1fr 1fr 1fr 1fr',gap:36,marginBottom:36}}>
            <div>
              <div style={{fontFamily:"'Baloo 2',cursive",fontSize:24,fontWeight:900,marginBottom:10}}>🌱 BeejHealth</div>
              <div style={{fontSize:13.5,opacity:.72,lineHeight:1.75,marginBottom:18,maxWidth:260}}>India ka #1 AI farming platform. Crop disease detection, expert consultation, weather intelligence — sab ek jagah.</div>
              <div style={{marginBottom:14}}>
                <div style={{fontSize:10,opacity:.45,textTransform:'uppercase',letterSpacing:.6,marginBottom:7}}>Certified By</div>
                <div style={{display:'flex',gap:8,flexWrap:'wrap'}}>
                  {['ICAR','IARI','NABARD','Startup India'].map(c=><span key={c} style={{fontSize:10,padding:'2px 9px',border:'1px solid rgba(255,255,255,.2)',borderRadius:100,opacity:.65}}>{c}</span>)}
                </div>
              </div>
              <div style={{display:'flex',gap:9}}>
                {['📷','🐦','💼','📘','▶️'].map((s,i)=><div key={i} style={{width:34,height:34,borderRadius:8,background:'rgba(255,255,255,.1)',display:'flex',alignItems:'center',justifyContent:'center',cursor:'pointer',fontSize:14}}>{s}</div>)}
              </div>
            </div>
            {[
              ['Platform',['AI Scan (Free)','Expert Consult','Crop Forecast','AgriMart','Insurance','Robot Fleet']],
              ['Company',['About Us','Our Team','Careers','Press & Media','Blog','Investors']],
              ['Support',['Help Center','Video Tutorials','Terms of Service','Privacy Policy','Refund Policy','API Docs']],
              ['Contact',['support@beejhealth.com','+91 98765 43210','WhatsApp Support','Mumbai, Maharashtra','Mon-Sat 8AM-8PM','Emergency: 24/7']],
            ].map(([title,items])=>(
              <div key={title}>
                <div style={{fontSize:10,fontWeight:800,textTransform:'uppercase',letterSpacing:.8,opacity:.45,marginBottom:12}}>{title}</div>
                {items.map(item=>(
                  <div key={item} style={{fontSize:13,opacity:.68,cursor:'pointer',marginBottom:8}}
                    onMouseEnter={e=>e.currentTarget.style.opacity='1'} onMouseLeave={e=>e.currentTarget.style.opacity='.68'}>
                    {item}
                  </div>
                ))}
              </div>
            ))}
          </div>
          <div style={{borderTop:'1px solid rgba(255,255,255,.1)',paddingTop:18,display:'flex',justifyContent:'space-between',alignItems:'center',flexWrap:'wrap',gap:10}}>
            <div style={{fontSize:12,opacity:.5}}>© 2026 BeejHealth Technologies Pvt. Ltd. All rights reserved.</div>
            <div style={{fontSize:12,opacity:.5}}>Made with 💚 for 140 Million Indian Farmers</div>
          </div>
        </div>
      </footer>
    </>
  );
}
function ConsultPage({user,nav,toast,selCrop,setSelCrop,qAnswers,setQAnswers}) {
  const [step,setStep]=useState(1); // 1=crop 2=upload 3=questions 4=processing
  const [photos,setPhotos]=useState([]);            // [{preview, photoId, uploading, error, sizeKB}]
  const [uploadErrors,setUploadErrors]=useState([]);
  const uploadDone = photos.length > 0 && !photos.some(p=>p.uploading);
  const [preAiResult,setPreAiResult]=useState(null);  // Quick AI scan after photo upload
  const [aiScanning,setAiScanning]=useState(false);   // AI scan in progress
  // Camera state
  const [showCamera,setShowCamera]=useState(false);
  const [cameraError,setCameraError]=useState('');
  const videoRef=useRef(null);
  const canvasRef=useRef(null);
  const streamRef=useRef(null);
  const [q1Answer,setQ1Answer]=useState(null);
  const [activeQs,setActiveQs]=useState([]);
  const isCoconut = selCrop?.id === 'coconut';
  const currentQ1 = isCoconut ? COCONUT_Q1_DATA : Q1_DATA;
  const [qIndex,setQIndex]=useState(0);
  const [currAns,setCurrAns]=useState(null);
  const [answeredList,setAnsweredList]=useState([]);
  const [procStep,setProcStep]=useState(0);

  const PROC_STEPS=['Image quality check...','Disease patterns scan...','Severity calculate ho rahi hai...','Treatment plan generate ho raha hai...'];

  const startProcessing=async()=>{
    setStep(4);setProcStep(0);
    for(let i=0;i<4;i++){await new Promise(r=>setTimeout(r,680));setProcStep(i+1);}
    if(user){
      try{
        const isCoconutCrop = selCrop?.id==='coconut';
        const photoBase64 = photos[0]?.preview||null;
        const photoId0 = photos[0]?.photoId||null;

        if(isCoconutCrop && (photoBase64||photoId0)){
          // ── COCONUT: Use real AI model ────────────────────────────
          const res=await API.post('/api/coconut/analyze',{
            photoBase64:     photoBase64,
            photoId:         photoId0,
            questionAnswers: qAnswers,
            preAiDisease:    preAiResult?.disease||null,  // pass pre-scan result
          });
          if(res?.consultation_id){
            localStorage.setItem('bh_latest_consult', res.consultation_id);
            localStorage.setItem('bh_latest_crop', JSON.stringify({id:'coconut',name:'Naariyal',emoji:'🥥'}));
            // Store AI result for immediate display
            localStorage.setItem('bh_ai_result', JSON.stringify(res.ai_result));
          }
        } else {
          // ── OTHER CROPS: Existing flow ────────────────────────────
          const d=DISEASE_DB[selCrop?.id]||DISEASE_DB.default;
          const res=await API.post('/api/consultations',{
            cropId:      selCrop?.id     ||'tomato',
            cropName:    selCrop?.name   ||'Crop',
            cropEmoji:   selCrop?.emoji  ||'🌱',
            method:      photos.length>0?'photo':'manual',
            photoUploaded: photos.length>0,
            photoId:     photoId0,
            photoUrl:    photos[0]?.url||null,
            photoIds:    photos.filter(p=>p.photoId).map(p=>p.photoId),
            photoUrls:   photos.filter(p=>p.photoId).map(p=>p.url),
            photoCount:  photos.length,
            answers:     qAnswers,
            disease:     d.disease,
            confidence:  d.conf,
            severity:    d.sev,
          });
          if(res?.consultation?._id){
            localStorage.setItem('bh_latest_consult', res.consultation._id);
            localStorage.setItem('bh_latest_crop', JSON.stringify({id:selCrop?.id,name:selCrop?.name,emoji:selCrop?.emoji}));
          }
        }
      }catch(e){
        if(e.message?.includes('AI_SERVER_DOWN')||e.message?.includes('AI server band')||e.message?.includes('ai_down')){
          toast('⚠️ AI server band hai — Photo save ho gayi, baad mein analyze hogi','warn');
        } else {
          console.warn('Consultation save:',e.message);
        }
      }
    }
    nav('ai-report');
  };

  const handleSelectCrop=(c)=>{setSelCrop(c);};
  const handleNextCrop=()=>{
    if(!selCrop){toast('Pehle crop select karein','warn');return;}
    setStep(2);
  };
  const handleFileChange=async(fileList)=>{
    if(!fileList||fileList.length===0) return;
    const files=Array.from(fileList);
    const remaining=5-photos.length;
    if(remaining<=0){toast('Maximum 5 photos allowed','warn');return;}
    const toProcess=files.slice(0,remaining);
    if(files.length>remaining) toast(`Sirf ${remaining} aur photo add ho sakti hain (max 5)`,'warn');
    const baseIdx=photos.length;
    const errors=[];
    for(const file of toProcess){
      if(file.size>10*1024*1024){errors.push(`${file.name}: 10MB se zyada`);continue;}
      if(!file.type.startsWith('image/')){errors.push(`${file.name}: Image nahi hai`);continue;}
      const idx=baseIdx+toProcess.indexOf(file);
      // Add with uploading state
      const tempId='temp_'+Date.now()+'_'+idx;
      const reader=new FileReader();
      reader.onload=async(ev)=>{
        const base64=ev.target.result;
        setPhotos(prev=>[...prev,{tempId,preview:base64,uploading:true,error:null,sizeKB:Math.round(file.size/1024)}]);
        try{
          const res=await API.post('/api/photos',{
            base64, type:file.type, index:idx, totalExpected:toProcess.length,
            consultationId:localStorage.getItem('bh_latest_consult')||null,
          });
          setPhotos(prev=>prev.map(p=>p.tempId===tempId?{...p,uploading:false,photoId:res.photoId,url:res.url,sizeKB:res.sizeKB||p.sizeKB}:p));
        }catch(err){
          setPhotos(prev=>prev.map(p=>p.tempId===tempId?{...p,uploading:false,error:err.message}:p));
          toast(`Photo upload fail: ${err.message}`,'err');
        }
      };
      reader.readAsDataURL(file);
    }
    if(errors.length>0) setUploadErrors(errors);
  };

  const removePhoto=(index)=>{
    const ph=photos[index];
    if(ph?.photoId){
      API.delete('/api/photos/'+ph.photoId).catch(()=>{});
    }
    setPhotos(prev=>prev.filter((_,i)=>i!==index));
  };

  const handleNextUpload=async()=>{
    if(photos.length===0){toast('Kam se kam 1 photo upload karein','warn');return;}
    if(photos.some(p=>p.uploading)){toast('Photos upload ho rahi hain, thoda wait karein...','warn');return;}

    // For coconut: Quick AI scan to set disease-specific questions
    if(selCrop?.id==='coconut' && photos[0]?.preview){
      setAiScanning(true);
      try{
        const quickRes = await fetch('/api/coconut/ai-status');
        const status = await quickRes.json();
        if(status.ai_online){
          const aiRes = await API.post('/api/coconut/quick-scan',{photoBase64: photos[0]?.preview});
          if(aiRes?.disease){
            setPreAiResult(aiRes);
            // Set disease-specific questions
            const dqs = COCONUT_DISEASE_QS[aiRes.disease] || COCONUT_DISEASE_QS['Healthy'];
            setActiveQs(dqs.slice(0,4));
            toast(`🥥 AI ne detect kiya: ${aiRes.disease} (${aiRes.confidence}%)`, 'inf');
          }
        }
      }catch(e){console.warn('Quick scan:',e.message);}
      finally{setAiScanning(false);}
    }

    setStep(3);setQIndex(0);setCurrAns(null);setAnsweredList([]);setQAnswers({});setQ1Answer(null);
    if(!preAiResult || selCrop?.id!=='coconut'){
      setActiveQs(isCoconut?(BRANCH_QS['coconut_spots']||[]):(BRANCH_QS['spots']||[]));
    }
  };

  // Camera handlers using getUserMedia
  const openCamera=async()=>{
    if(photos.length>=5){toast('Maximum 5 photos allowed','warn');return;}
    setCameraError('');
    setShowCamera(true);
    try{
      const stream=await navigator.mediaDevices.getUserMedia({
        video:{facingMode:'environment',width:{ideal:1920},height:{ideal:1080}},
        audio:false
      });
      streamRef.current=stream;
      if(videoRef.current){
        videoRef.current.srcObject=stream;
        videoRef.current.play();
      }
    }catch(err){
      console.warn('Camera error:',err);
      if(err.name==='NotAllowedError') setCameraError('Camera permission denied. Browser settings mein allow karo.');
      else if(err.name==='NotFoundError') setCameraError('Camera nahi mila. Gallery use karein.');
      else setCameraError('Camera nahi khula: '+err.message);
    }
  };

  const stopCamera=()=>{
    if(streamRef.current){
      streamRef.current.getTracks().forEach(t=>t.stop());
      streamRef.current=null;
    }
    if(videoRef.current) videoRef.current.srcObject=null;
    setShowCamera(false);
    setCameraError('');
  };

  const capturePhoto=()=>{
    const video=videoRef.current;
    const canvas=canvasRef.current;
    if(!video||!canvas) return;
    canvas.width=video.videoWidth||1280;
    canvas.height=video.videoHeight||720;
    canvas.getContext('2d').drawImage(video,0,0);
    const base64=canvas.toDataURL('image/jpeg',0.85);
    // Process as uploaded file
    const sizeKB=Math.round((base64.length*3/4)/1024);
    const tempId='cam_'+Date.now();
    setPhotos(prev=>[...prev,{tempId,preview:base64,uploading:true,error:null,sizeKB}]);
    stopCamera();
    // Upload to backend
    API.post('/api/photos',{
      base64,type:'image/jpeg',
      index:photos.length,totalExpected:1,
      consultationId:localStorage.getItem('bh_latest_consult')||null,
    }).then(res=>{
      setPhotos(prev=>prev.map(p=>p.tempId===tempId?{...p,uploading:false,photoId:res.photoId,url:res.url,sizeKB:res.sizeKB||sizeKB}:p));
      toast('📸 Photo capture ho gayi! ✅');
    }).catch(err=>{
      setPhotos(prev=>prev.map(p=>p.tempId===tempId?{...p,uploading:false,error:err.message}:p));
      toast('Photo upload fail: '+err.message,'err');
    });
  };

  // For coconut with pre-AI result: skip Q1, show 5 disease-specific questions directly
  const coconutSkipQ1 = isCoconut && preAiResult && activeQs.length > 0;
  const currentQ = coconutSkipQ1 
    ? (activeQs[qIndex] || activeQs[0] || activeQs[activeQs.length-1])
    : (qIndex===0 ? currentQ1 : (activeQs[qIndex-1] || currentQ1));

  const handleSelectOpt=(opt)=>{
    setCurrAns(opt);
    if(qIndex===0){
      setQ1Answer(opt.id);
      setActiveQs(isCoconut?(BRANCH_QS[opt.id]||BRANCH_QS.coconut_none):(BRANCH_QS[opt.id]||BRANCH_QS.none));
    }
  };

  const handleNextQ=()=>{
    if(!currAns||!currentQ) return;
    const newAnswers={...qAnswers,[currentQ.id]:{id:currAns.id,label:currAns.label}};
    setQAnswers(newAnswers);
    setAnsweredList(p=>[...p,{q:currentQ.text||currentQ.id,a:currAns.label,icon:currAns.icon}]);
    const nextIdx=qIndex+1;
    // Coconut with AI: 5 disease-specific questions total
    const totalQs = coconutSkipQ1 ? Math.min(5, activeQs.length) : 5;
    if(nextIdx>=totalQs){startProcessing();return;}
    setQIndex(nextIdx);setCurrAns(null);
  };

  const totalQ=5;
  const progress=step===1?10:step===2?35:step===3?(55+(qIndex/5)*30):90;

  return (
    <div className="wrap-md">

      {/* Page Title */}
      <div style={{marginBottom:24}}>
        <div style={{fontFamily:"'Baloo 2',cursive",fontSize:26,fontWeight:900,color:'var(--g1)'}}>
          🔬 Crop Consultation
        </div>
        <div style={{fontSize:14,color:'var(--tx3)',marginTop:4}}>
          {step===1&&'Step 1/3 — Apni fasal choose karein'}
          {step===2&&'Step 2/3 — Affected part ki photo lo'}
          {step===3&&(isCoconut&&preAiResult
            ? `🥥 Q${qIndex+1}/5 — ${preAiResult.disease} ke baare mein`
            : `Step 3/3 — Sawaal ${Math.min(qIndex+1,5)} of ${totalQ}`
          )}
          {step===4&&'AI analysis ho rahi hai...'}
        </div>
      </div>

      {/* Progress Bar */}
      <div className="prog-bar" style={{marginBottom:24}}>
        <div className="prog-fill" style={{width:`${progress}%`,transition:'width .5s ease'}}/>
      </div>

      {/* ── STEP 1: CROP SELECT ── */}
      {step===1&&(
        <div className="slide-up">
          <div className="card" style={{padding:26,marginBottom:18}}>
            <div style={{fontSize:15,fontWeight:800,color:'var(--g1)',marginBottom:4}}>🌱 Apni Fasal Select Karein</div>
            <div style={{fontSize:13,color:'var(--tx3)',marginBottom:18}}>Kaunsi crop mein problem hai?</div>
            <div className="crops-grid">
              {CROPS.map(c=>(
                <div key={c.id} className={`crop-tile${selCrop?.id===c.id?' sel':''}`} onClick={()=>handleSelectCrop(c)}>
                  <div className="ct-em">{c.emoji}</div>
                  <div className="ct-nm">{c.name}</div>
                </div>
              ))}
            </div>
          </div>
          {selCrop&&(
            <div style={{padding:'12px 16px',background:'var(--gp)',border:'1.5px solid var(--gpb)',borderRadius:'var(--rad)',marginBottom:16,display:'flex',alignItems:'center',gap:12}} className="fade-in">
              <span style={{fontSize:28}}>{selCrop.emoji}</span>
              <div><div style={{fontSize:14,fontWeight:800,color:'var(--g1)'}}>{selCrop.name} ✅</div>
              <div style={{fontSize:12,color:'var(--tx3)'}}>Health: {selCrop.health}% · {selCrop.stage}</div></div>
              <button style={{marginLeft:'auto'}} className="btn btn-ghost btn-sm" onClick={()=>setSelCrop(null)}>✕</button>
            </div>
          )}
          <button className="btn btn-g" style={{width:'100%',padding:'13px',fontSize:15,borderRadius:12}} disabled={!selCrop} onClick={handleNextCrop}>
            Aage Badho — Photo Upload →
          </button>
        </div>
      )}

      {/* ── STEP 2: UPLOAD ── */}
      {step===2&&(
        <div className="slide-up">
          <button className="btn btn-ghost btn-sm" style={{marginBottom:18}} onClick={()=>setStep(1)}>← Crop Change Karo</button>

          {/* Crop badge */}
          <div style={{display:'flex',alignItems:'center',gap:10,marginBottom:20,padding:'11px 15px',background:'var(--gp)',border:'1.5px solid var(--gpb)',borderRadius:'var(--rad2)'}}>
            <span style={{fontSize:24}}>{selCrop?.emoji}</span>
            <span style={{fontSize:14,fontWeight:700,color:'var(--g1)'}}>{selCrop?.name} — Photo Upload</span>
            <span style={{marginLeft:'auto',fontSize:12,color:'var(--tx3)',fontWeight:600}}>{photos.length}/5 photos</span>
          </div>

          {/* Gallery file input */}
          <input id="galleryInput" type="file" accept="image/*" multiple style={{display:'none'}}
            onChange={e=>handleFileChange(e.target.files)}/>

          {/* Camera Modal */}
          {showCamera&&(
            <div style={{position:'fixed',inset:0,background:'#000',zIndex:9999,display:'flex',flexDirection:'column'}}>
              <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'12px 16px',background:'rgba(0,0,0,.8)'}}>
                <span style={{color:'white',fontWeight:700,fontSize:16}}>📷 Camera</span>
                <button style={{background:'rgba(255,255,255,.2)',border:'none',color:'white',padding:'6px 14px',borderRadius:8,cursor:'pointer',fontWeight:700}} onClick={stopCamera}>✕ Band Karo</button>
              </div>
              <video ref={videoRef} autoPlay playsInline muted style={{flex:1,objectFit:'cover',width:'100%'}}/>
              <canvas ref={canvasRef} style={{display:'none'}}/>
              <div style={{padding:20,background:'rgba(0,0,0,.8)',display:'flex',gap:12,justifyContent:'center',alignItems:'center'}}>
                {cameraError?(
                  <div style={{color:'#ff6b6b',fontSize:13,textAlign:'center'}}>⚠️ {cameraError}<br/><small>Gallery use karein</small></div>
                ):(
                  <button style={{width:68,height:68,borderRadius:'50%',background:'white',border:'4px solid rgba(255,255,255,.4)',cursor:'pointer',fontSize:28,display:'flex',alignItems:'center',justifyContent:'center',boxShadow:'0 0 0 3px rgba(255,255,255,.3)'}}
                    onClick={capturePhoto}>📸</button>
                )}
              </div>
            </div>
          )}

          {/* Upload buttons */}
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:12,marginBottom:20}}>
            <button className="btn btn-g" style={{padding:'16px 10px',fontSize:14,borderRadius:12,display:'flex',flexDirection:'column',alignItems:'center',gap:6}}
              onClick={openCamera}>
              <span style={{fontSize:28}}>📷</span>
              <span>Camera Se Lo</span>
              <span style={{fontSize:11,opacity:.7,fontWeight:400}}>Seedha photo khicho</span>
            </button>
            <button className="btn btn-out" style={{padding:'16px 10px',fontSize:14,borderRadius:12,display:'flex',flexDirection:'column',alignItems:'center',gap:6}}
              onClick={()=>document.getElementById('galleryInput').click()}>
              <span style={{fontSize:28}}>🖼️</span>
              <span>Gallery Se Upload</span>
              <span style={{fontSize:11,opacity:.7,fontWeight:400}}>1–5 photos select karo</span>
            </button>
          </div>

          {/* Tips */}
          <div style={{display:'flex',gap:8,flexWrap:'wrap',marginBottom:16}}>
            {['💡 Natural light mein lo','🔍 Close-up zaroor lo','📐 Alag angles se lo','🌿 Healthy + sick dono','📍 1 se 5 photos allowed'].map(t=>(
              <div key={t} style={{background:'white',border:'1px solid var(--br)',borderRadius:100,padding:'4px 11px',fontSize:11.5,color:'var(--tx3)'}}>{t}</div>
            ))}
          </div>

          {/* Upload progress / errors */}
          {uploadErrors.length>0&&(
            <div style={{background:'#fff0f0',border:'1.5px solid var(--r2)',borderRadius:10,padding:'10px 14px',marginBottom:14}}>
              {uploadErrors.map((err,i)=>(
                <div key={i} style={{fontSize:12.5,color:'var(--r2)',fontWeight:600}}>⚠️ {err}</div>
              ))}
            </div>
          )}

          {/* Photo grid — show uploaded photos */}
          {photos.length>0&&(
            <div style={{marginBottom:18}}>
              <div style={{fontSize:13,fontWeight:700,color:'var(--g1)',marginBottom:10}}>
                📸 Uploaded Photos ({photos.length}/5)
                {photos.some(p=>p.uploading)&&<span style={{marginLeft:8,fontSize:11,color:'var(--a2)',fontWeight:600}}>⏳ uploading...</span>}
                {photos.every(p=>!p.uploading&&p.photoId)&&<span style={{marginLeft:8,fontSize:11,color:'var(--g4)',fontWeight:600}}>✅ sab ready</span>}
              </div>
              <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:10}}>
                {photos.map((ph,i)=>(
                  <div key={i} style={{position:'relative',borderRadius:10,overflow:'hidden',border:`2px solid ${ph.uploading?'var(--a2)':ph.error?'var(--r2)':'var(--g4)'}`,aspectRatio:'1',background:'var(--gp)'}}>
                    <img src={ph.preview} alt={`photo ${i+1}`} style={{width:'100%',height:'100%',objectFit:'cover',display:'block'}}/>
                    {/* Status overlay */}
                    {ph.uploading&&(
                      <div style={{position:'absolute',inset:0,background:'rgba(0,0,0,.45)',display:'flex',alignItems:'center',justifyContent:'center'}}>
                        <div className="spin" style={{width:20,height:20,borderColor:'white',borderTopColor:'transparent'}}/>
                      </div>
                    )}
                    {ph.error&&(
                      <div style={{position:'absolute',inset:0,background:'rgba(255,0,0,.25)',display:'flex',alignItems:'center',justifyContent:'center',fontSize:20}}>⚠️</div>
                    )}
                    {!ph.uploading&&!ph.error&&(
                      <div style={{position:'absolute',top:4,left:4,background:'rgba(30,126,66,.85)',borderRadius:100,padding:'2px 7px',fontSize:10,color:'white',fontWeight:700}}>✓ {i+1}</div>
                    )}
                    {/* Delete button */}
                    <button style={{position:'absolute',top:4,right:4,width:22,height:22,borderRadius:'50%',background:'rgba(0,0,0,.6)',border:'none',color:'white',fontSize:12,cursor:'pointer',display:'flex',alignItems:'center',justifyContent:'center'}}
                      onClick={()=>removePhoto(i)}>✕</button>
                    {/* Size badge */}
                    {ph.sizeKB&&<div style={{position:'absolute',bottom:4,right:4,background:'rgba(0,0,0,.55)',borderRadius:6,padding:'1px 5px',fontSize:9,color:'white'}}>{ph.sizeKB}KB</div>}
                  </div>
                ))}
                {/* Add more button */}
                {photos.length<5&&(
                  <div style={{borderRadius:10,border:'2px dashed var(--br2)',aspectRatio:'1',display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',gap:6,cursor:'pointer',background:'var(--gp)'}}
                    onClick={()=>document.getElementById('galleryInput').click()}>
                    <span style={{fontSize:24}}>➕</span>
                    <span style={{fontSize:11,fontWeight:700,color:'var(--g3)'}}>Add More</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Empty state — no photos yet */}
          {photos.length===0&&(
            <div style={{textAlign:'center',padding:'28px 20px',background:'var(--gp)',borderRadius:'var(--rad)',border:'2px dashed var(--br2)',marginBottom:16}}>
              <div style={{fontSize:48,marginBottom:8}}>🌿</div>
              <div style={{fontSize:14,fontWeight:700,color:'var(--tx)',marginBottom:4}}>Koi photo nahi upload hui</div>
              <div style={{fontSize:12.5,color:'var(--tx3)'}}>Upar Camera ya Gallery button use karein</div>
            </div>
          )}

          <button className="btn btn-g" style={{width:'100%',padding:'13px',fontSize:15,borderRadius:12}}
            disabled={photos.length===0||photos.some(p=>p.uploading)||aiScanning}
            onClick={handleNextUpload}>
            {photos.some(p=>p.uploading)
              ? <><div className="spin" style={{width:16,height:16,display:'inline-block',marginRight:8}}/> Upload ho raha hai...</>
              : aiScanning
              ? <><div className="spin" style={{width:16,height:16,display:'inline-block',marginRight:8}}/> 🥥 AI Analysis ho rahi hai...</>
              : selCrop?.id==='coconut'
              ? `🥥 ${photos.length} Photo${photos.length>1?'s':''} Ready — Coconut Diagnosis Shuru Karo →`
              : `🤖 ${photos.length} Photo${photos.length>1?'s':''} Ready — Diagnosis Shuru Karo →`
            }
          </button>
        </div>
      )}

            {/* ── STEP 3: QUESTION FLOW ── */}
      {step===3&&currentQ&&(
        <div>
          <button className="btn btn-ghost btn-sm" style={{marginBottom:18}} onClick={()=>{setStep(2);}}>← Photo Change Karo</button>

          {/* Q Progress Dots */}
          <div className="qf-prog">
            {Array.from({length:totalQ},(_,i)=>[
              i>0&&<div key={'l'+i} className={`qf-line${i<=qIndex?' done':''}`}/>,
              <div key={'d'+i} className={`qf-dot${i<qIndex?' done':i===qIndex?' active':''}`}>
                {i<qIndex?'✓':i+1}
              </div>
            ]).flat().filter(Boolean)}
          </div>

          {/* Answered Questions (collapsed) */}
          {answeredList.map((a,i)=>(
            <div key={i} className="qf-answered">
              <div className="qf-ans-check">✓</div>
              <div>
                <div className="qf-ans-q">{a.q}</div>
                <div className="qf-ans-a">{a.icon} {a.a}</div>
              </div>
            </div>
          ))}

          {/* Current Question Card */}
          <div className="qf-card" key={currentQ.id}>
            <div className="qf-num">Sawaal {qIndex+1} of {totalQ}</div>
            <div className="qf-q">{currentQ.text}</div>
            {currentQ.hint&&<div className="qf-hint">{currentQ.hint}</div>}
            <div className="qf-opts">
              {currentQ.options.map(o=>(
                <button key={o.id} className={`qf-opt${currAns?.id===o.id?' sel':''}`} onClick={()=>handleSelectOpt(o)}>
                  <span className="qf-opt-icon">{o.icon}</span>
                  <span>{o.label}</span>
                </button>
              ))}
            </div>
          </div>

          <button className="btn btn-g" style={{width:'100%',padding:'13px',fontSize:15,borderRadius:12}} disabled={!currAns} onClick={handleNextQ}>
            {qIndex<4?'Agle Sawaal →':'✅ Diagnosis Shuru Karo →'}
          </button>
        </div>
      )}

      {/* ── STEP 4: PROCESSING ── */}
      {step===4&&(
        <div className="card" style={{padding:40,textAlign:'center'}}>
          <div className="proc-icon">🤖</div>
          <div style={{fontFamily:"'Baloo 2',cursive",fontSize:22,fontWeight:900,color:'var(--g1)',marginBottom:7}}>AI Analyze Kar Raha Hai...</div>
          <div style={{fontSize:14,color:'var(--tx3)',marginBottom:22}}>Aapki photo aur {answeredList.length} answers process ho rahe hain</div>
          <div style={{maxWidth:300,margin:'0 auto'}}>
            <div className="prog-bar" style={{marginBottom:18}}><div className="prog-fill" style={{width:`${(procStep/4)*100}%`}}/></div>
            <div className="proc-steps-list">
              {PROC_STEPS.map((s,i)=>(
                <div key={i} className={`proc-step${i<procStep?' done':i===procStep?' act':''}`}>
                  <span style={{fontSize:15}}>{i<procStep?'✅':i===procStep?'🔄':'⏳'}</span>
                  <span>{s}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   AI REPORT PAGE  — Full Branded Report with Logo
════════════════════════════════════════════════════════════════ */
function AIReportPage({selCrop,nav,toast,qAnswers,viewConsultId}) {
  const crop=selCrop||CROPS[0];
  const [dbConsult,setDbConsult]=useState(null);
  const [loadingConsult,setLoadingConsult]=useState(false);
  const [aiResult,setAiResult]=useState(()=>{
    try{ return JSON.parse(localStorage.getItem('bh_ai_result')||'null'); }catch{return null;}
  });

  useEffect(()=>{
    // Reset purana data taaki pichla report flash na kare
    setDbConsult(null);
    setAiResult(()=>{ try{ return JSON.parse(localStorage.getItem('bh_ai_result')||'null'); }catch{return null;} });
    const consultId = viewConsultId || localStorage.getItem('bh_view_consult') || localStorage.getItem('bh_latest_consult');
    if(!consultId) return;
    setLoadingConsult(true);
    API.get('/api/consultations/'+consultId)
      .then(d=>{ if(d.consultation) setDbConsult(d.consultation); })
      .catch(()=>{})
      .finally(()=>setLoadingConsult(false));
  },[viewConsultId]); // viewConsultId change hone par naya data fetch hoga

  // Use coconut AI result first, then DB, then fallback
  const isCoconutReport = (dbConsult?.cropId==='coconut') || (crop?.id==='coconut') || !!aiResult;
  const effectiveCropId = dbConsult?.cropId || crop.id;
  const effectiveAnswers = dbConsult?.answers || qAnswers || {};
  const effectiveDisease = aiResult?.disease || dbConsult?.disease;
  const effectiveConf    = aiResult?.confidence || dbConsult?.confidence;
  const effectiveSev     = aiResult?.severity || dbConsult?.severity;
  const effectiveHindi   = aiResult?.disease_hindi || dbConsult?.disease_hindi || '';
  const effectiveTreatments = aiResult?.treatments || dbConsult?.ai_treatments || [];
  const effectiveTop3    = aiResult?.top3 || dbConsult?.ai_top3 || [];
  const effectiveUrgency = aiResult?.urgency || dbConsult?.urgency || 'medium';

  const d = DISEASE_DB[effectiveCropId] || DISEASE_DB.default;
  const reportD = {...d, disease: effectiveDisease||d.disease, conf: effectiveConf||d.conf, sev: effectiveSev||d.sev, sevLabel: effectiveSev?`Stage ${effectiveSev}/5`:d.sevLabel};
  const answers=qAnswers||{};

  const today=new Date().toLocaleDateString('en-IN',{day:'2-digit',month:'short',year:'numeric'});
  const reportId='BH-'+today.replace(/ /g,'')+'-'+Math.floor(Math.random()*90000+10000);

  /* Q-derived values */
  const q2=answers.q2;const q3=answers.q3;const q9=answers.q9;const q20=answers.q20;
  const triggerText=q9?(q9.id==='heavy'||q9.id==='light'?'Baarish + humidity 68–75%':'Dry conditions — monitoring needed'):(d.humidity);
  const sprayText=q20?(q20.id==='never'||q20.id==='old'?'Spray nahi kiya tha lamba time':'Recent spray tha — phir bhi infection'):d.humidity;

  return (
    <div className="wrap-sm">
      <button className="btn btn-ghost btn-sm" style={{marginBottom:18}} onClick={()=>nav('consultation')}>← Naya Diagnosis</button>

      <div className="rep-sheet">
        {/* ── HEADER ── */}
        <div className="rep-header">
          <div className="rh-top">
            <div className="rh-logo">
              <img className="rh-logo-img" src={`data:image/png;base64,${LOGO_FULL_B64}`} alt="FrameIQ"/>
              <div className="rh-logo-sep"/>
              <div className="rh-platform">BeejHealth AI Report</div>
            </div>
            <div className="rh-meta">
              <div className="rh-badge"><span className="rh-badge-dot"/>&nbsp;AI Analysis Complete</div>
              <div className="rh-id">Report #{reportId}</div>
            </div>
          </div>
          <div className="rh-main">
            <div className="rh-crop-line">{crop.emoji} {crop.name.toUpperCase()} · {today}</div>
            <div className="rh-disease">{reportD.disease} Detected</div>
            <div className="rh-sci">{reportD.sci} · {reportD.hindi}</div>
            <div className="rh-scores">
              <div className="rhs"><div className="rhs-val">{reportD.conf}%</div><div className="rhs-lbl">AI Confidence</div><div className="rhs-bar"><div className="rhs-fill" style={{width:`${reportD.conf}%`}}/></div></div>
              <div className="rhs"><div className="rhs-val">{reportD.sev}/5</div><div className="rhs-lbl">Severity</div><div className="rhs-bar"><div className="rhs-fill" style={{width:`${reportD.sev*20}%`}}/></div></div>
              <div className="rhs"><div className="rhs-val">{reportD.aff}%</div><div className="rhs-lbl">Area Affected</div><div className="rhs-bar"><div className="rhs-fill" style={{width:`${reportD.aff}%`}}/></div></div>
              <div className="rhs"><div className="rhs-val">{reportD.sev>=3?'High':d.sev===2?'Med':'Low'}</div><div className="rhs-lbl">Risk Level</div><div className="rhs-bar"><div className="rhs-fill" style={{width:d.sev>=3?'80%':d.sev===2?'50%':'25%'}}/></div></div>
            </div>
          </div>
        </div>

        {/* ── BODY ── */}
        <div className="rep-body">

          {/* S1: Disease ID */}
          <div className="r-sec">
            <div className="r-sec-hd"><div className="r-sec-num">1</div><div className="r-sec-title">Disease Identification</div><span className="r-sec-tag r-tag-b">Photo + Q1</span></div>
            {(dbConsult?.photoUrls?.length>0||dbConsult?.photoUrl)&&(
              <div style={{marginBottom:14}}>
                <div style={{fontSize:12,fontWeight:700,color:'var(--tx3)',marginBottom:8,textTransform:'uppercase',letterSpacing:'.5px'}}>
                  📸 Uploaded Photos ({dbConsult.photoCount||dbConsult.photoUrls?.length||1})
                </div>
                <div style={{display:'grid',gridTemplateColumns:`repeat(${Math.min((dbConsult.photoUrls||[dbConsult.photoUrl]).length,3)},1fr)`,gap:8}}>
                  {(dbConsult.photoUrls||[dbConsult.photoUrl]).filter(Boolean).map((url,i)=>(
                    <div key={i} style={{borderRadius:8,overflow:'hidden',border:'1.5px solid var(--br)',aspectRatio:'1'}}>
                      <img src={url} alt={`Photo ${i+1}`} style={{width:'100%',height:'100%',objectFit:'cover',display:'block'}}/>
                    </div>
                  ))}
                </div>
                <div style={{marginTop:6,padding:'5px 10px',background:'var(--gp)',borderRadius:7,fontSize:11.5,color:'var(--tx3)',fontWeight:600}}>
                  📸 AI Analysis in {dbConsult.photoCount||1} photo{(dbConsult.photoCount||1)>1?'s':''} se ki gayi
                </div>
              </div>
            )}
            <div className="r-card gl">
              <div className="r-fr"><div className="r-fl">Disease</div><div className="r-fv b">{reportD.disease} <span className="r-pill r-pill-p">Photo se</span></div></div>
              <div className="r-fr"><div className="r-fl">Scientific name</div><div className="r-fv"><em>{reportD.sci}</em></div></div>
              <div className="r-fr"><div className="r-fl">Hindi naam</div><div className="r-fv">{reportD.hindi}</div></div>
              <div className="r-fr"><div className="r-fl">Severity stage</div><div className="r-fv b">{reportD.sevLabel} <span className="r-pill r-pill-b">Dono se</span></div></div>
              <div className="r-conf-row"><span>AI Confidence</span><span style={{color:'var(--g4)',fontWeight:700}}>{reportD.conf}%</span></div>
              <div className="r-conf-bar"><div className="r-conf-fill" style={{width:`${reportD.conf}%`}}/></div>
            </div>
          </div>

          {/* COCONUT AI: Real Model Results */}
          {isCoconutReport&&effectiveTop3&&effectiveTop3.length>0&&(
            <div className="r-sec" style={{marginBottom:14}}>
              <div className="r-sec-hd">
                <div className="r-sec-num" style={{background:'#1565C0'}}>AI</div>
                <div className="r-sec-title">EfficientNetV2-S Model Results</div>
                <span className="r-sec-tag" style={{background:'#E3F0FB',color:'#1565C0'}}>Real Model</span>
              </div>
              <div style={{padding:'14px 16px',background:'#E3F0FB',borderRadius:10,marginBottom:10,border:'1.5px solid #1565C0'}}>
                <div style={{fontSize:12,fontWeight:700,color:'#1565C0',marginBottom:8,textTransform:'uppercase',letterSpacing:'.5px'}}>
                  🤖 Top-3 Predictions
                </div>
                {effectiveTop3.map((p,i)=>(
                  <div key={i} style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'7px 0',borderBottom:i<2?'1px solid rgba(21,101,192,.15)':'none'}}>
                    <div style={{display:'flex',gap:9,alignItems:'center'}}>
                      <div style={{width:22,height:22,borderRadius:'50%',background:i===0?'#1565C0':'rgba(21,101,192,.2)',display:'flex',alignItems:'center',justifyContent:'center',fontSize:11,fontWeight:800,color:i===0?'white':'#1565C0'}}>#{p.rank}</div>
                      <span style={{fontSize:13,fontWeight:i===0?800:600,color:i===0?'#1565C0':'var(--tx2)'}}>{p.disease}</span>
                    </div>
                    <div style={{display:'flex',gap:8,alignItems:'center'}}>
                      <span style={{fontSize:13,fontWeight:800,color:i===0?'#1565C0':'var(--tx3)'}}>{p.confidence}%</span>
                      <div style={{width:60,height:6,background:'rgba(21,101,192,.15)',borderRadius:3}}>
                        <div style={{width:`${p.confidence}%`,height:'100%',background:i===0?'#1565C0':'rgba(21,101,192,.4)',borderRadius:3}}/>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              {effectiveHindi&&(
                <div style={{fontSize:12.5,color:'var(--tx2)',padding:'8px 12px',background:'var(--gp)',borderRadius:8}}>
                  🌐 Hindi: <strong>{effectiveHindi}</strong>
                  {effectiveUrgency==='critical'&&<span style={{marginLeft:10,background:'#FEE2E2',color:'#DC2626',padding:'2px 8px',borderRadius:100,fontSize:11,fontWeight:700}}>⚠️ Critical</span>}
                  {effectiveUrgency==='high'&&<span style={{marginLeft:10,background:'#FEF3C7',color:'#D97706',padding:'2px 8px',borderRadius:100,fontSize:11,fontWeight:700}}>⚡ High</span>}
                </div>
              )}
            </div>
          )}

          {/* COCONUT AI: Treatment Plan from Model */}
          {isCoconutReport&&effectiveTreatments&&effectiveTreatments.length>0&&(
            <div className="r-sec" style={{marginBottom:14}}>
              <div className="r-sec-hd">
                <div className="r-sec-num" style={{background:'#16a34a'}}>Rx</div>
                <div className="r-sec-title">AI Treatment Plan</div>
                <span className="r-sec-tag r-tag-b">Model Based</span>
              </div>
              <div style={{background:'var(--gp)',borderRadius:10,padding:14}}>
                {effectiveTreatments.map((t,i)=>(
                  <div key={i} style={{display:'flex',gap:10,padding:'8px 0',borderBottom:i<effectiveTreatments.length-1?'1px solid var(--br)':'none',alignItems:'flex-start'}}>
                    <div style={{width:22,height:22,borderRadius:'50%',background:'var(--g4)',display:'flex',alignItems:'center',justifyContent:'center',fontSize:11,fontWeight:800,color:'white',flexShrink:0,marginTop:1}}>{i+1}</div>
                    <span style={{fontSize:13,color:'var(--tx)',lineHeight:1.55}}>{t}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* S2 + S3 */}
          <div className="r-two">
            <div className="r-sec">
              <div className="r-sec-hd"><div className="r-sec-num">2</div><div className="r-sec-title">7-Day Spread</div><span className="r-sec-tag r-tag-b">Photo + Q2,Q3</span></div>
              <div className="r-card">
                <div className="r-fr"><div className="r-fl">Affected now</div><div className="r-fv b">{reportD.aff}% leaves</div></div>
                {q2&&<div className="r-fr"><div className="r-fl">Daag kahan</div><div className="r-fv">{q2.label} <span className="r-pill r-pill-q">Q2</span></div></div>}
                {q3&&<div className="r-fr" style={{borderBottom:'none'}}><div className="r-fl">Daag rang</div><div className="r-fv">{q3.label} <span className="r-pill r-pill-q">Q3</span></div></div>}
                {!q2&&!q3&&<div className="r-fr" style={{borderBottom:'none'}}><div className="r-fl">Status</div><div className="r-fv">Photo analysis complete</div></div>}
                <div className="r-spread">
                  <div className="r-sc un"><div className="r-sc-num">{reportD.utr}%</div><div className="r-sc-lbl">Untreated</div><div style={{fontSize:10,opacity:.65}}>7 din baad</div></div>
                  <div className="r-sc tr"><div className="r-sc-num">{reportD.tr}%</div><div className="r-sc-lbl">Treated</div><div style={{fontSize:10,opacity:.65}}>5 din mein</div></div>
                </div>
              </div>
            </div>
            <div className="r-sec">
              <div className="r-sec-hd"><div className="r-sec-num">3</div><div className="r-sec-title">Root Cause</div><span className="r-sec-tag r-tag-q">Q9, Q20</span></div>
              <div className="r-card">
                <div className="r-fr"><div className="r-fl">Primary cause</div><div className="r-fv b">{reportD.cause}</div></div>
                <div className="r-fr"><div className="r-fl">Trigger {q9&&<span className="r-pill r-pill-q">Q9</span>}</div><div className="r-fv">{triggerText}</div></div>
                <div className="r-fr"><div className="r-fl">Contributing {q20&&<span className="r-pill r-pill-q">Q20</span>}</div><div className="r-fv">{sprayText}</div></div>
                <div className="r-fr" style={{borderBottom:'none'}}><div className="r-fl">Risk</div><div className="r-fv r">{reportD.risk}</div></div>
              </div>
            </div>
          </div>

          {/* S4: Treatment */}
          <div className="r-sec">
            <div className="r-sec-hd"><div className="r-sec-num">4</div><div className="r-sec-title">Treatment Plan — 3 Phases</div><span className="r-sec-tag r-tag-ai">AI Generated</span></div>
            <div className="r-phases">
              {[['t','Aaj Karo','🔴'],['w','Is Hafte','🟡'],['s','Is Season','🟢']].map(([cls,lbl,ic],i)=>(
                <div key={cls} className={`r-ph ${cls}`}>
                  <div className="r-ph-dot"/>
                  <div><div className="r-ph-tag">{ic} {lbl}</div><div className="r-ph-txt">{reportD.phases[i]}</div></div>
                </div>
              ))}
            </div>
          </div>

          {/* S5 + S6+S7 */}
          <div className="r-two">
            <div className="r-sec">
              <div className="r-sec-hd"><div className="r-sec-num">5</div><div className="r-sec-title">Medicines</div><span className="r-sec-tag r-tag-ai">AI Matched</span></div>
              <div className="r-med-list">
                {reportD.meds.map((m,i)=>(
                  <div key={m.nm} className={`r-med${m.top?' top':''}`}>
                    <div className="r-med-rank" style={!m.top?{background:'var(--tx4)'}:{}}>{i+1}</div>
                    <div style={{flex:1}}>
                      <div className="r-med-nm">{m.nm}{m.top&&<span className="r-rec-badge">Best</span>}</div>
                      <div className="r-med-ty">{m.ty}</div>
                    </div>
                    <div className="r-med-pr">{m.pr}</div>
                  </div>
                ))}
              </div>
              <div style={{marginTop:9,padding:'8px 11px',background:'var(--gp)',borderRadius:8,fontSize:12,color:'var(--g2)',fontWeight:600}}>
                📍 Ram Agri Store — 2.3 km away
              </div>
            </div>
            <div style={{display:'flex',flexDirection:'column',gap:12}}>
              <div className="r-sec">
                <div className="r-sec-hd"><div className="r-sec-num">6</div><div className="r-sec-title">Weather Risk</div><span className="r-sec-tag r-tag-q">Q9</span></div>
                <div className="r-alert">
                  <div className="r-alert-title">⚠️ High Risk — Act Today</div>
                  <div className="r-fr"><div className="r-fl">Humidity</div><div className="r-fv r">68–75% Dangerous</div></div>
                  <div className="r-fr"><div className="r-fl">Spray window</div><div className="r-fv">Aaj 4–6 PM</div></div>
                  <div className="r-fr" style={{borderBottom:'none'}}><div className="r-fl">Nearby cases</div><div className="r-fv r">8 within 5km</div></div>
                </div>
              </div>
              <div className="r-sec">
                <div className="r-sec-hd"><div className="r-sec-num">7</div><div className="r-sec-title">Expert Advice</div><span className="r-sec-tag r-tag-b">Stage based</span></div>
                <div className="r-card">
                  <div className="r-fr"><div className="r-fl">Consult</div><div className="r-fv g">Recommended</div></div>
                  <div className="r-fr"><div className="r-fl">Expert</div><div className="r-fv">Dr. Rajesh Kumar</div></div>
                  <div className="r-fr" style={{borderBottom:'none'}}><div className="r-fl">Follow-up</div><div className="r-fv">7 din baad photo</div></div>
                </div>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:9,marginTop:20}}>
            <button className="btn btn-g btn-md" onClick={()=>nav('experts')}>👨‍⚕️ Expert Se Confirm</button>
            <button className="btn btn-out btn-md" onClick={()=>toast('PDF download ho rahi hai...','inf')}>📄 PDF Download</button>
            <button className="btn btn-ghost btn-md" onClick={()=>toast('Treatment reminder set ✅')}>🔔 Reminder Set Karo</button>
            <button className="btn btn-ghost btn-md" onClick={()=>nav('marketplace')}>🛒 Medicine Order Karo</button>
          </div>
        </div>

        {/* ── FOOTER ── */}
        <div className="rep-footer">
          <div className="rf-logo">
            <img className="rf-logo-img" src={`data:image/png;base64,${LOGO_FULL_B64}`} alt="FrameIQ"/>
            <div className="rf-txt">BeejHealth · Powered by FrameIQ AI</div>
          </div>
          <div className="rf-disc">AI-generated report for informational use only. Consult a certified agricultural expert before major treatment decisions.</div>
        </div>
      </div>
    </div>
  );
}


/* ════════════════════════════════════════════════════════════════
   MY CONSULTATIONS
════════════════════════════════════════════════════════════════ */
function MyConsultPage({user,nav,toast}) {
  const [filter,setFilter]=useState('all');
  const [apiConsults,setApiConsults]=useState([]);
  const [loading,setLoading]=useState(false);
  const [ratingConsult,setRatingConsult]=useState(null);
  const [myRating,setMyRating]=useState(0);
  const [ratingSubmitted,setRatingSubmitted]=useState(false);

  // Status helpers
  const stColor=(s)=>s==='completed'?'bg-g':s==='expert'?'bg-b':s==='pending'?'bg-a':'bg-p';
  const stIcon=(s)=>s==='completed'?'✅':s==='expert'?'👨‍⚕️':s==='pending'?'⏳':'🔵';

  const submitRating=async()=>{
    if(!myRating||!ratingConsult?.expertId) return;
    try{
      await API.post('/api/experts/'+ratingConsult.expertId+'/rate',{rating:myRating,consultationId:ratingConsult._id});
      setRatingSubmitted(true);
      toast('Rating submit ho gayi! ⭐ Shukriya.');
      setTimeout(()=>{setRatingConsult(null);setMyRating(0);setRatingSubmitted(false);},2000);
    }catch(e){ toast('Rating fail hua','err'); }
  };

  useEffect(()=>{
    if(!user) return;
    setLoading(true);
    API.get('/api/consultations')
      .then(d=>{
        if(d.consultations) setApiConsults(d.consultations.map(c=>({
          _id:c._id, id:c._id,
          crop:c.cropName, emoji:c.cropEmoji||'🌱',
          issue:`${c.disease||'Analysis'} detected (${c.confidence||0}% confidence)`,
          date:new Date(c.createdAt).toLocaleDateString('en-IN',{day:'2-digit',month:'short',year:'numeric'}),
          expert:c.expertName||'Auto-assign',
          expertId:c.expertId||null,
          expertName:c.expertName||'Expert',
          status:c.status==='completed'?'completed':c.status==='expert_assigned'?'expert':c.status==='pending'?'pending':'ai',
          statusLabel:c.status==='completed'?'Completed':c.status==='expert_assigned'?'Expert Assigned':c.status==='pending'?'Pending':'AI Report Ready',
          sev:c.severity||1, conf:c.confidence||0,
          rated:c.rated||false,
        })));
        setLoading(false);
      })
      .catch(()=>setLoading(false));
  },[user]);

  if(!user) return (
    <div className="wrap" style={{textAlign:'center',padding:'80px 28px'}}>
      <div style={{fontSize:60,marginBottom:16}}>🔒</div>
      <div style={{fontSize:22,fontWeight:800,color:'var(--g1)',marginBottom:8}}>Login Karein</div>
      <div style={{fontSize:15,color:'var(--tx2)',marginBottom:24}}>Apni consultations dekhne ke liye login zaroor hai</div>
      <button className="btn btn-g btn-lg" onClick={()=>nav('home')}>Login Karo</button>
    </div>
  );

  const displayConsults=apiConsults.length>0?apiConsults:[];
  const filtered=filter==='all'?displayConsults:displayConsults.filter(c=>c.status===filter);

  return (
    <div className="wrap">
      {/* Rating Modal */}
      {ratingConsult&&!ratingSubmitted&&(
        <div className="overlay" onClick={()=>setRatingConsult(null)}>
          <div className="modal" style={{maxWidth:380,padding:28}} onClick={e=>e.stopPropagation()}>
            <div style={{fontFamily:"'Baloo 2',cursive",fontSize:20,fontWeight:900,color:'var(--g1)',marginBottom:8}}>⭐ Expert Rate Karein</div>
            <div style={{fontSize:13,color:'var(--tx3)',marginBottom:18}}>Dr. {ratingConsult?.expertName||'Expert'} ke liye aapka feedback</div>
            <div style={{display:'flex',gap:12,justifyContent:'center',marginBottom:20}}>
              {[1,2,3,4,5].map(s=>(
                <button key={s} onClick={()=>setMyRating(s)}
                  style={{fontSize:28,background:'none',border:'none',cursor:'pointer',
                          opacity:s<=myRating?1:0.3,transform:s<=myRating?'scale(1.2)':'scale(1)',transition:'all .15s'}}>⭐</button>
              ))}
            </div>
            <button className="btn btn-g btn-full" onClick={submitRating} disabled={!myRating}>
              {myRating?`${myRating} Star Dein`:'Star select karein'}
            </button>
          </div>
        </div>
      )}
      {ratingConsult&&ratingSubmitted&&(
        <div className="overlay">
          <div className="modal" style={{maxWidth:300,padding:28,textAlign:'center'}}>
            <div style={{fontSize:50,marginBottom:8}}>🎉</div>
            <div style={{fontFamily:"'Baloo 2',cursive",fontSize:18,fontWeight:900,color:'var(--g1)'}}>Shukriya!</div>
            <div style={{fontSize:13,color:'var(--tx3)',marginTop:4}}>Aapka feedback expert ke liye valuable hai.</div>
          </div>
        </div>
      )}

      {loading&&<div style={{textAlign:'center',padding:40,fontSize:14,color:'var(--tx3)'}}>⏳ Loading consultations...</div>}
      <div style={{marginBottom:26}}>
        <div style={{fontFamily:"'Baloo 2',cursive",fontSize:28,fontWeight:900,color:'var(--g1)'}}>📋 My Consultations</div>
        <div style={{fontSize:15,color:'var(--tx2)',marginTop:5}}>Apni plant diagnosis history</div>
      </div>

      {/* Filter tabs */}
      <div style={{display:'flex',gap:8,marginBottom:22,flexWrap:'wrap'}}>
        {['all','ai','expert','pending','completed'].map(f=>(
          <button key={f} onClick={()=>setFilter(f)} style={{padding:'7px 17px',borderRadius:100,fontSize:13,fontWeight:700,border:`2px solid ${filter===f?'var(--g4)':'var(--br)'}`,background:filter===f?'var(--gp)':'white',color:filter===f?'var(--g3)':'var(--tx2)',cursor:'pointer',transition:'all .18s'}}>
            {f==='all'?'All':f==='ai'?'🔵 AI Ready':f==='expert'?'🟢 Expert Assigned':f==='pending'?'🟠 Pending':'⚫ Completed'}
          </button>
        ))}
      </div>

      {/* Empty state */}
      {!loading&&filtered.length===0&&(
        <div style={{textAlign:'center',padding:'60px 20px',color:'var(--tx4)'}}>
          <div style={{fontSize:48,marginBottom:12}}>🌿</div>
          <div style={{fontSize:16,fontWeight:700,color:'var(--tx2)',marginBottom:8}}>Koi consultation nahi mili</div>
          <div style={{fontSize:13,color:'var(--tx3)',marginBottom:20}}>Pehli crop consultation shuru karein</div>
          <button className="btn btn-g btn-md" onClick={()=>nav('consultation')}>🔬 Consultation Start Karo</button>
        </div>
      )}

      {/* Consultation grid */}
      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(310px,1fr))',gap:18}}>
        {filtered.map(c=>(
          <div key={c.id} className="card card-hov" style={{overflow:'hidden',cursor:'pointer'}} onClick={()=>{
            localStorage.setItem('bh_view_consult', c.id);
            nav('ai-report');
          }}>
            <div style={{height:75,display:'flex',alignItems:'center',justifyContent:'center',fontSize:40,background:'linear-gradient(135deg,var(--gp),var(--gpb))'}}>{c.emoji}</div>
            <div className="cons-body">
              <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',marginBottom:5}}>
                <div className="cons-nm">{c.crop}</div>
                <span className={`badge ${stColor(c.status)}`}>{stIcon(c.status)} {c.statusLabel}</span>
              </div>
              <div className="cons-issue">{c.issue}</div>
              <div className="cons-meta">📅 {c.date}</div>
              <div className="cons-meta">👨‍⚕️ {c.expert}</div>
              <div style={{display:'flex',gap:6,margin:'7px 0'}}>
                <span className="badge bg-b">AI: {c.conf}%</span>
                <span className={`badge ${c.sev>=3?'bg-r':c.sev===2?'bg-a':'bg-g'}`}>Stage {c.sev}/5</span>
              </div>
              <div className="cons-acts">
                <button className="ca-rep" onClick={e=>{e.stopPropagation();localStorage.setItem('bh_view_consult',c.id);nav('ai-report');}}>📄 Report</button>
                <button className="ca-chat" onClick={e=>{e.stopPropagation();localStorage.setItem('bh_latest_consult',c.id);nav('chat');}}>💬 Chat →</button>
                {c.status==='completed'&&c.expertId&&(
                  <button className="ca-rep" style={{background:'var(--ap)',color:'var(--a1)'}} onClick={e=>{e.stopPropagation();setRatingConsult(c);}}>⭐ Rate</button>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ExpertDash({user,nav,toast}) {
  const [tab,setTab]=useState('urgent');
  const [avail,setAvail]=useState(user?.available!==false);
  const [cases,setCases]=useState(CONSULTATIONS);
  useEffect(()=>{
    if(!user) return;
    API.get('/api/consultations')
      .then(d=>{ if(d.consultations&&d.consultations.length>0) setCases(d.consultations.map(c=>({
        id:c._id, crop:c.cropName, emoji:c.cropEmoji||'🌱',
        issue:`${c.disease} — ${c.confidence}% confidence`,
        date:new Date(c.createdAt).toLocaleDateString('en-IN'),
        expert:user?.name||'Expert',
        status:c.status,
        statusLabel:c.status==='completed'?'Completed':c.status==='expert_assigned'?'Expert Assigned':c.status==='pending'?'Pending':'AI Report Ready',
        sev:c.severity||1, conf:c.confidence||0,
      }))); }).catch(()=>{});
  },[user]);

  // Poll for new cases every 15 seconds — real-time notification
  const [newCaseCount,setNewCaseCount]=useState(0);
  useEffect(()=>{
    if(!user||user?.type!=='expert') return;
    let prevCount=0;
    const poll=setInterval(()=>{
      API.get('/api/consultations')
        .then(d=>{
          const n=(d.consultations||[]).length;
          if(prevCount>0&&n>prevCount) setNewCaseCount(n-prevCount);
          prevCount=n;
        }).catch(()=>{});
    },15000);
    return ()=>clearInterval(poll);
  },[user]);

  return (
    <div className="wrap">
      {newCaseCount>0&&(
        <div style={{background:'var(--b3)',color:'white',padding:'10px 18px',borderRadius:10,marginBottom:16,display:'flex',justifyContent:'space-between',alignItems:'center',fontWeight:700,fontSize:13.5}}>
          🔔 {newCaseCount} naya case aaya! <button style={{background:'white',color:'var(--b1)',border:'none',borderRadius:7,padding:'4px 12px',fontWeight:800,cursor:'pointer'}} onClick={()=>setNewCaseCount(0)}>Dekho →</button>
        </div>
      )}
      {/* ── EXPERT HEADER CARD ── */}
      <div className="ed-head">
        <div style={{position:'absolute',right:-30,top:-30,width:180,height:180,background:'rgba(255,255,255,.05)',borderRadius:'50%'}}/>
        <div style={{position:'absolute',right:80,bottom:-50,width:120,height:120,background:'rgba(255,255,255,.03)',borderRadius:'50%'}}/>
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',position:'relative'}}>
          <div>
            <div style={{fontSize:12,opacity:.76,marginBottom:4,fontWeight:600}}>👨‍⚕️ Expert Dashboard</div>
            <div style={{fontFamily:"'Baloo 2',cursive",fontSize:24,fontWeight:900,color:'white'}}>Namaskar, Dr. {user?.name?.split(' ')?.slice(-1)?.[0]||'Expert'}! 👋</div>
            <div style={{fontSize:13.5,opacity:.8,color:'white',marginTop:3}}>{user?.spec||'Plant Pathologist'}</div>
            <div style={{marginTop:10,display:'flex',alignItems:'center',gap:10}}>
              <div style={{display:'flex',alignItems:'center',gap:8,padding:'6px 12px',background:'rgba(255,255,255,.12)',borderRadius:100,backdropFilter:'blur(6px)'}}>
                <div style={{width:8,height:8,borderRadius:'50%',background:avail?'#4ade80':'#f87171',animation:avail?'ringPulse 2s infinite':'none'}}/>
                <span style={{fontSize:12.5,fontWeight:700,color:'white'}}>{avail?'Available Now':'Busy / Offline'}</span>
              </div>
              <button onClick={async()=>{const nv=!avail;try{await API.patch('/api/experts/availability',{available:nv});setAvail(nv);toast(nv?'🟢 Online — cases milenge':'🔴 Offline ho gaye');}catch(e){toast(e.message,'err');}}} className="btn btn-sm" style={{background:'rgba(255,255,255,.18)',color:'white',border:'1.5px solid rgba(255,255,255,.3)',borderRadius:8,padding:'5px 12px',fontSize:12,fontWeight:700,cursor:'pointer'}}>
                {avail ? '🔴 Offline Ho Jao' : '🟢 Online Ho Jao'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ── MAIN 2-COL ── */}
      <div className="dash-2">
        {/* LEFT: Case Queue */}
        <div>
          <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:13}}>
            <div style={{fontSize:16,fontWeight:800,color:'var(--b1)'}}>📋 Case Queue</div>
            <button className="btn btn-ghost btn-sm" style={{color:'var(--b3)'}} onClick={()=>nav('case-detail')}>Sab Cases →</button>
          </div>
          <div style={{display:'flex',gap:5,marginBottom:14}}>
            {[{k:'urgent',l:'🚨 Urgent',n:3},{k:'pending',l:'📋 Pending',n:5},{k:'active',l:'🔄 Active',n:2}].map(t=>(
              <button key={t.k} onClick={()=>setTab(t.k)} style={{flex:1,padding:'8px 6px',borderRadius:8,fontSize:12,fontWeight:700,border:`2px solid ${tab===t.k?'var(--b3)':'var(--br)'}`,background:tab===t.k?'var(--bp)':'none',color:tab===t.k?'var(--b3)':'var(--tx2)',cursor:'pointer',fontFamily:"'Outfit',sans-serif",transition:'all .18s'}}>
                {t.l}({t.n})
              </button>
            ))}
          </div>
          {cases.map((c,i)=>(
            <div key={c.id} className={`case-card${i===0?' urg':i===2?' med':''}`} onClick={()=>nav('case-detail')}>
              <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',marginBottom:7}}>
                <div>
                <div className="case-id">#{String(c.id||'').slice(-6).toUpperCase()} • {c.date.split('•')[0].trim()}</div>
                  <div className="case-crop">{c.emoji} {c.crop}</div>
                </div>
                <div style={{display:'flex',flexDirection:'column',gap:4,alignItems:'flex-end'}}>
                  <span className={`badge ${c.sev>=3?'bg-r':c.sev===2?'bg-a':'bg-g'}`}>Stage {c.sev}/5</span>
                  <span className="badge bg-b">AI {c.conf}%</span>
                </div>
              </div>
              <div className="case-issue">{c.issue}</div>
              <div className="case-meta-row">
                <span>👤 Farmer #{1000+c.id}</span><span>•</span><span>📍 Pune, MH</span><span>•</span><span>🤖 AI {c.conf}% confident</span>
              </div>
              <div style={{display:'flex',gap:7,marginTop:11}}>
                <button className="btn btn-b btn-sm" style={{flex:2}} onClick={e=>{e.stopPropagation();nav('case-detail');}}>🔍 Review Case →</button>
                <button className="btn btn-ghost btn-sm" style={{flex:1,color:'var(--b2)'}} onClick={e=>{e.stopPropagation();nav('chat');}}>💬 Chat</button>
              </div>
            </div>
          ))}
        </div>

        {/* RIGHT: Performance + Availability + Earnings */}
        <div style={{display:'flex',flexDirection:'column',gap:18}}>
          {/* Performance */}
          <div className="card" style={{padding:20}}>
            <div style={{fontSize:15,fontWeight:800,color:'var(--b1)',marginBottom:14}}>📊 Is Hafte Ka Performance</div>
            {[['Cases Solved','24','✅'],['Avg Response','38 min','⚡'],['Rating','4.9 ⭐','🏆'],['Accuracy','96%','🎯'],['Repeat Clients','68%','🔄']].map(([l,v,i])=>(
              <div key={l} style={{display:'flex',justifyContent:'space-between',padding:'8px 0',borderBottom:'1px solid var(--gp)'}}>
                <span style={{fontSize:13,color:'var(--tx2)',fontWeight:600}}>{i} {l}</span>
                <span style={{fontSize:14,fontWeight:800,color:'var(--b3)'}}>{v}</span>
              </div>
            ))}
          </div>

          {/* Availability */}
          <div className="card" style={{padding:20}}>
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:14}}>
              <div style={{fontSize:15,fontWeight:800,color:'var(--b1)'}}>📅 Aaj Ki Availability</div>
              <button className="btn btn-b btn-sm" onClick={()=>toast('Schedule editor — coming soon','inf')}>✏️ Edit</button>
            </div>
            {[{t:'9 AM–11 AM',s:'🟢 Available',c:'var(--g4)'},{t:'11 AM–2 PM',s:'🔴 Busy',c:'var(--r2)'},{t:'2 PM–5 PM',s:'🟢 Available',c:'var(--g4)'},{t:'5 PM–6 PM',s:'🟡 Limited',c:'var(--a2)'}].map(({t,s,c})=>(
              <div key={t} style={{display:'flex',justifyContent:'space-between',padding:'8px 0',borderBottom:'1px solid var(--gp)',fontSize:13,fontWeight:600}}>
                <span style={{color:'var(--tx2)'}}>{t}</span>
                <span style={{color:c}}>{s}</span>
              </div>
            ))}
          </div>

          {/* Earnings Mini */}
          <div className="card" style={{padding:20,background:'linear-gradient(135deg,#eef5ff,#dbeafe)',border:'1.5px solid var(--bpb)'}}>
            <div style={{fontSize:15,fontWeight:800,color:'var(--b1)',marginBottom:13}}>💰 Earnings Summary</div>
            {[['Aaj','₹4,800','6 cases'],['Is Mahine','₹38,400','48 cases'],['Net Payout','₹28,800','after deductions']].map(([l,v,s])=>(
              <div key={l} style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'8px 0',borderBottom:'1px solid rgba(147,197,253,.4)'}}>
                <div>
                  <div style={{fontSize:13,fontWeight:700,color:'var(--b1)'}}>{l}</div>
                  <div style={{fontSize:11,color:'var(--tx3)'}}>{s}</div>
                </div>
                <div style={{fontFamily:"'Baloo 2',cursive",fontSize:18,fontWeight:900,color:'var(--b3)'}}>{v}</div>
              </div>
            ))}
            <button className="btn btn-b btn-sm" style={{width:'100%',marginTop:12}} onClick={()=>toast('Withdrawal feature — coming soon','inf')}>
              💸 Bank Mein Withdraw Karo
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   CONSULTATION PAGE
════════════════════════════════════════════════════════════════ */
const LOGO_FULL_B64 = "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";


/* ════════════════════════════════════════════════════════════════
   QUESTION FLOW DATA
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   COCONUT AI — Disease-specific questions (30 total, 5 shown)
   After AI detects disease, show relevant questions for that disease
════════════════════════════════════════════════════════════════ */

// Maps AI disease name → question branch
const COCONUT_DISEASE_BRANCH = {
  'Gray Leaf Spot':         'coconut_spots',
  'Gray Leaf Spot_multiple':'coconut_spots',
  'Leaf rot':               'coconut_spots',
  'Leaf rot_multiple':      'coconut_spots',
  'bud rot':                'coconut_wilt',
  'stem bleeding':          'coconut_wilt',
  'stem bleeding_multiple': 'coconut_wilt',
  'Healthy':                'coconut_none',
};

// Per-disease targeted questions — confirm & detail the AI detection
const COCONUT_DISEASE_QS = {
  'Gray Leaf Spot': [
    {id:'cgls1', text:'Daag ka rang kaisa hai?', hint:'Gray Leaf Spot confirm karne ke liye',
      options:[{id:'gray',label:'Bhoora/Gray rang',icon:'⬜'},{id:'brown',label:'Dark Brown',icon:'🟤'},
               {id:'yellow',label:'Peele edges ke saath',icon:'🟡'},{id:'black',label:'Kala daag',icon:'⚫'}]},
    {id:'cgls2', text:'Daag patte ke kahan hain?', hint:'Location batao',
      options:[{id:'tip',label:'Patte ki nauk pe',icon:'📍'},{id:'middle',label:'Beech mein',icon:'🎯'},
               {id:'all',label:'Poore patte pe',icon:'🍃'},{id:'base',label:'Neeche base pe',icon:'⬇️'}]},
    {id:'cgls3', text:'Kitne patte prabhavit hain?', hint:'Infection ka phailaav',
      options:[{id:'one',label:'1-2 patte',icon:'1️⃣'},{id:'few',label:'3-5 patte',icon:'🔢'},
               {id:'many',label:'Adhe se zyada',icon:'📊'},{id:'all',label:'Saare patte',icon:'⚠️'}]},
    {id:'cgls4', text:'Daag ke aas-paas peela ring hai?', hint:'Halo pattern check',
      options:[{id:'yes',label:'Haan, peela ring hai',icon:'🟡'},{id:'no',label:'Nahi, sirf daag',icon:'❌'},
               {id:'wet',label:'Geela aur dark border',icon:'💧'},{id:'dry',label:'Sukha aur crispy',icon:'🍂'}]},
    {id:'cgls5', text:'Pichle 2 hafte mein zyada nami ya baarish?', hint:'Fungal trigger',
      options:[{id:'heavy',label:'Haan, bahut baarish',icon:'🌧️'},{id:'light',label:'Thodi baarish',icon:'🌦️'},
               {id:'no',label:'Bilkul sukha',icon:'☀️'},{id:'fog',label:'Kohra/Fog tha',icon:'🌫️'}]},
  ],
  'Gray Leaf Spot_multiple': [
    {id:'cglsm1', text:'Kitne patte ek saath prabhavit hain?', hint:'Multiple infection scale',
      options:[{id:'few',label:'3-5 patte',icon:'🔢'},{id:'many',label:'10+ patte',icon:'📊'},
               {id:'half',label:'Adhi fasal',icon:'⚠️'},{id:'all',label:'Poori fasal',icon:'💀'}]},
    {id:'cglsm2', text:'Daag aapas mein jud rahe hain?', hint:'Merging spots — serious sign',
      options:[{id:'yes_merge',label:'Haan, daag jud gaye',icon:'🔴'},{id:'spreading',label:'Fail rahe hain',icon:'⚠️'},
               {id:'separate',label:'Alag alag hain abhi',icon:'🟡'},{id:'no',label:'Nahi',icon:'✅'}]},
    {id:'cglsm3', text:'Infected patte gir bhi rahe hain?', hint:'Defoliation check',
      options:[{id:'yes',label:'Haan, bahut gir rahe',icon:'🍂'},{id:'some',label:'Kuch gir rahe',icon:'🌿'},
               {id:'no',label:'Nahi gir rahe',icon:'✅'},{id:'dry',label:'Sukh ke gir rahe',icon:'💀'}]},
    {id:'cglsm4', text:'Pehle koi treatment kiya tha?', hint:'Previous spray history',
      options:[{id:'fungicide',label:'Fungicide spray kiya',icon:'💊'},{id:'neem',label:'Neem spray kiya',icon:'🌿'},
               {id:'nothing',label:'Kuch nahi kiya',icon:'❌'},{id:'failed',label:'Kiya par kaam nahi aaya',icon:'⚠️'}]},
    {id:'cglsm5', text:'Aas-paas ke ped bhi prabhavit hain?', hint:'Spread risk',
      options:[{id:'yes_many',label:'Haan, bahut pedo mein',icon:'⚠️'},{id:'yes_few',label:'1-2 pedo mein',icon:'🔢'},
               {id:'no',label:'Sirf is ped mein',icon:'✅'},{id:'unknown',label:'Check nahi kiya',icon:'❓'}]},
  ],
  'Leaf rot': [
    {id:'clr1', text:'Patte kahan se galana shuru hua?', hint:'Leaf rot starting point',
      options:[{id:'tip',label:'Nauk se shuru',icon:'📍'},{id:'edge',label:'Kinare se',icon:'↔️'},
               {id:'middle',label:'Beech se',icon:'🎯'},{id:'base',label:'Base se',icon:'⬇️'}]},
    {id:'clr2', text:'Gale hue hisse ka rang?', hint:'Rot color identification',
      options:[{id:'brown',label:'Brown/Bhoora',icon:'🟤'},{id:'black',label:'Kala',icon:'⚫'},
               {id:'gray',label:'Bhoora-Gray',icon:'⬜'},{id:'yellow_brown',label:'Peela-Bhoora',icon:'🟡'}]},
    {id:'clr3', text:'Galane ki gati kaisi hai?', hint:'Progression speed',
      options:[{id:'fast',label:'Tezi se fail raha',icon:'🏃'},{id:'slow',label:'Dheere dheere',icon:'🐌'},
               {id:'stable',label:'Ruka hua lagta hai',icon:'⏸️'},{id:'unknown',label:'Pata nahi',icon:'❓'}]},
    {id:'clr4', text:'Koi buri smell aa rahi hai?', hint:'Bacterial vs fungal',
      options:[{id:'yes_strong',label:'Haan, bahut buri smell',icon:'👃'},{id:'yes_mild',label:'Halki smell',icon:'😐'},
               {id:'no',label:'Koi smell nahi',icon:'✅'},{id:'earthy',label:'Mitti jaisi smell',icon:'🌍'}]},
    {id:'clr5', text:'Patte mein paani ka bhoora daag hai?', hint:'Water soaking check',
      options:[{id:'yes',label:'Haan, wet daag hain',icon:'💧'},{id:'dry_rot',label:'Sukha rot hai',icon:'🍂'},
               {id:'both',label:'Dono hain',icon:'🤔'},{id:'no',label:'Normal lag raha',icon:'✅'}]},
  ],
  'Leaf rot_multiple': [
    {id:'clrm1', text:'Kitni pattiyan ek saath gal rahi hain?', hint:'Scale of infection',
      options:[{id:'few',label:'3-5 patte',icon:'🔢'},{id:'many',label:'10+ patte',icon:'📊'},
               {id:'half',label:'Adha ped',icon:'⚠️'},{id:'all',label:'Poora ped',icon:'💀'}]},
    {id:'clrm2', text:'Kali (bud) bhi prabhavit hai?', hint:'Critical check — bud rot spreading?',
      options:[{id:'yes_bud',label:'Haan, kali bhi gal rahi',icon:'💀'},{id:'no_bud',label:'Nahi, kali theek hai',icon:'✅'},
               {id:'unsure',label:'Pata nahi',icon:'❓'},{id:'discolored',label:'Kali ka rang badla',icon:'🟫'}]},
    {id:'clrm3', text:'Pani bharav (waterlogging) ki problem hai?', hint:'Root cause',
      options:[{id:'yes',label:'Haan, paani ruka rehta',icon:'🌊'},{id:'sometimes',label:'Kabhi kabhi',icon:'🤔'},
               {id:'no',label:'Nahi, drain ho jaata',icon:'✅'},{id:'recent',label:'Haal hi mein baarish',icon:'🌧️'}]},
    {id:'clrm4', text:'Trunk pe koi daag ya liquid?', hint:'Check for stem bleeding too',
      options:[{id:'yes_liquid',label:'Haan, liquid nikal raha',icon:'🩸'},{id:'yes_spot',label:'Haan, daag hain trunk pe',icon:'🟫'},
               {id:'no',label:'Trunk theek lagta',icon:'✅'},{id:'bark',label:'Chhal uth rahi hai',icon:'🌴'}]},
    {id:'clrm5', text:'Yeh problem kab se hai?', hint:'Duration',
      options:[{id:'days',label:'2-3 din se',icon:'📅'},{id:'week',label:'1 hafte se',icon:'🗓️'},
               {id:'month',label:'1+ mahine se',icon:'📆'},{id:'old',label:'3+ mahine se',icon:'⏳'}]},
  ],
  'bud rot': [
    {id:'cbr1', text:'Naariyal ki kali (growing tip) ka kya haal hai?', hint:'Bud rot central symptom',
      options:[{id:'brown_dead',label:'Kali bhoori/kali ho gayi',icon:'💀'},{id:'wilting',label:'Murjha rahi hai',icon:'🥀'},
               {id:'soft',label:'Soft aur geeli lag rahi',icon:'💧'},{id:'smell',label:'Buri smell aa rahi',icon:'👃'}]},
    {id:'cbr2', text:'Kali khinchne pe kya hota hai?', hint:'Pull test for bud rot',
      options:[{id:'comes_out',label:'Aasaani se nikal aati',icon:'⚠️'},{id:'stuck',label:'Atki hui hai',icon:'✅'},
               {id:'broken',label:'Toot jaati hai',icon:'💔'},{id:'not_tried',label:'Check nahi kiya',icon:'❓'}]},
    {id:'cbr3', text:'Inner leaves (andar ke patte) ka rang?', hint:'Early bud rot sign',
      options:[{id:'yellow',label:'Peele pad rahe',icon:'🟡'},{id:'brown',label:'Bhoore ho gaye',icon:'🟤'},
               {id:'normal',label:'Normal hare',icon:'✅'},{id:'rotting',label:'Galne lage',icon:'💀'}]},
    {id:'cbr4', text:'Pichle mahine mein zyada baarish ya nami?', hint:'Phytophthora trigger',
      options:[{id:'heavy',label:'Bahut zyada baarish',icon:'🌧️'},{id:'humid',label:'Nami bahut thi',icon:'💧'},
               {id:'normal',label:'Normal mausam',icon:'🌤️'},{id:'dry',label:'Sukha mausam',icon:'☀️'}]},
    {id:'cbr5', text:'Aas-paas ke naariyal pedo mein bhi?', hint:'Spread check — urgent',
      options:[{id:'yes_many',label:'Haan, kai pedo mein',icon:'🚨'},{id:'yes_one',label:'1-2 pedo mein',icon:'⚠️'},
               {id:'no',label:'Sirf is ped mein',icon:'✅'},{id:'check',label:'Abhi check karunga',icon:'🔍'}]},
  ],
  'stem bleeding': [
    {id:'csb1', text:'Trunk se kaunsa liquid nikal raha hai?', hint:'Stem bleeding identification',
      options:[{id:'dark_red',label:'Dark red/maroon liquid',icon:'🩸'},{id:'brown',label:'Brown ranga liquid',icon:'🟤'},
               {id:'black',label:'Kala chipchipa',icon:'⚫'},{id:'clear',label:'Transparent/Paani',icon:'💧'}]},
    {id:'csb2', text:'Trunk pe daag ya kharaabi kahan hai?', hint:'Location of bleeding',
      options:[{id:'base',label:'Neeche (jaad ke paas)',icon:'⬇️'},{id:'middle',label:'Beech mein',icon:'🎯'},
               {id:'top',label:'Upar (crown ke paas)',icon:'⬆️'},{id:'all',label:'Poore trunk pe',icon:'🌴'}]},
    {id:'csb3', text:'Trunk pe daag ka size kaisa hai?', hint:'Size of affected area',
      options:[{id:'small',label:'Chota (10cm se kam)',icon:'🔵'},{id:'medium',label:'Medium (10-30cm)',icon:'🟡'},
               {id:'large',label:'Bada (30cm+)',icon:'🔴'},{id:'ring',label:'Poore trunk ka chakkar',icon:'⭕'}]},
    {id:'csb4', text:'Affected jagah ki chhal (bark) kaisi lag rahi?', hint:'Bark condition',
      options:[{id:'soft',label:'Soft aur sunken',icon:'💧'},{id:'cracked',label:'Phaati hui',icon:'💔'},
               {id:'peeling',label:'Uth rahi hai',icon:'🌿'},{id:'normal',label:'Theek lagti',icon:'✅'}]},
    {id:'csb5', text:'Ped ka production prabhavit hua hai?', hint:'Yield impact',
      options:[{id:'less',label:'Haan, naariyal kam aaye',icon:'📉'},{id:'none',label:'Bilkul nahi aaye',icon:'❌'},
               {id:'normal',label:'Normal production hai',icon:'✅'},{id:'young',label:'Abhi young ped hai',icon:'🌱'}]},
  ],
  'stem bleeding_multiple': [
    {id:'csbm1', text:'Trunk pe kitni jagah se bleeding ho rahi?', hint:'Multiple sites = serious',
      options:[{id:'two',label:'2 jagah',icon:'2️⃣'},{id:'three',label:'3-4 jagah',icon:'🔢'},
               {id:'many',label:'5+ jagah',icon:'⚠️'},{id:'ring',label:'Poore trunk pe',icon:'🚨'}]},
    {id:'csbm2', text:'Aas-paas ke pedo mein bhi yahi problem?', hint:'Farm-wide assessment',
      options:[{id:'yes_many',label:'Haan, kai pedo mein',icon:'🚨'},{id:'yes_few',label:'1-2 pedo mein',icon:'⚠️'},
               {id:'no',label:'Sirf is ped mein',icon:'✅'},{id:'new',label:'Naye pedo mein bhi',icon:'⚡'}]},
    {id:'csbm3', text:'Kya pehle bhi yeh ped beemar tha?', hint:'Chronic vs new infection',
      options:[{id:'yes_old',label:'Haan, pehle bhi tha',icon:'📆'},{id:'first',label:'Pehli baar dekha',icon:'📍'},
               {id:'partial',label:'Pehle thoda tha',icon:'🤔'},{id:'unknown',label:'Pata nahi',icon:'❓'}]},
    {id:'csbm4', text:'Koi treatment pehle try kiya?', hint:'Treatment history',
      options:[{id:'yes_success',label:'Haan, kuch faayda hua',icon:'✅'},{id:'yes_fail',label:'Kiya par kaam nahi aaya',icon:'❌'},
               {id:'nothing',label:'Kuch nahi kiya abhi tak',icon:'🚨'},{id:'natural',label:'Natural/organic try kiya',icon:'🌿'}]},
    {id:'csbm5', text:'Ped kitne saalon ka hai?', hint:'Age affects treatment',
      options:[{id:'young',label:'3-5 saal',icon:'🌱'},{id:'adult',label:'6-15 saal',icon:'🌴'},
               {id:'mature',label:'15-30 saal',icon:'🎋'},{id:'old',label:'30+ saal',icon:'🧓'}]},
  ],
  'Healthy': [
    {id:'ch1', text:'Yeh check kyun kar rahe hain?', hint:'Preventive monitoring reason',
      options:[{id:'routine',label:'Routine check',icon:'📋'},{id:'worried',label:'Thoda worried hoon',icon:'😟'},
               {id:'nearby',label:'Paas ke ped mein bimari',icon:'⚠️'},{id:'expert',label:'Expert ne kaha',icon:'👨‍⚕️'}]},
    {id:'ch2', text:'Naariyal ke patte kaisa dikhte hain?', hint:'Leaf health check',
      options:[{id:'bright',label:'Chamakdar hare',icon:'✅'},{id:'dull',label:'Matteese hare',icon:'🌿'},
               {id:'slight_yellow',label:'Thoda peela',icon:'🟡'},{id:'very_green',label:'Bahut ghane hare',icon:'🌳'}]},
    {id:'ch3', text:'Production kaisi hai?', hint:'Yield check',
      options:[{id:'good',label:'Bahut achhi',icon:'✅'},{id:'normal',label:'Normal',icon:'👍'},
               {id:'less',label:'Pehle se kam',icon:'📉'},{id:'none',label:'Bilkul nahi',icon:'❌'}]},
    {id:'ch4', text:'Last fertilizer kab diya?', hint:'Nutrition check',
      options:[{id:'recent',label:'1 mahine mein',icon:'📅'},{id:'months',label:'3-6 mahine mein',icon:'🗓️'},
               {id:'long',label:'6+ mahine pehle',icon:'📆'},{id:'never',label:'Kabhi nahi',icon:'❌'}]},
    {id:'ch5', text:'Koi keede ya pest dikhe hain?', hint:'Pest monitoring',
      options:[{id:'rhinoceros',label:'Bada kala beetle',icon:'🦏'},{id:'red_palm',label:'Lal rang ka ghun',icon:'🐛'},
               {id:'none',label:'Koi nahi dikha',icon:'✅'},{id:'small',label:'Chote keede',icon:'🔬'}]},
  ],
};

const COCONUT_Q1_DATA = {
  id:'cq1',
  text:'Naariyal ke ped mein kya problem dikh rahi hai?',
  hint:'Jo sabse pehle aapne notice kiya woh select karein',
  options:[
    {id:'coconut_spots', label:'Patte pe daag / Leaf Spots', icon:'🔴', desc:'Gray Leaf Spot ya Leaf Rot ke symptoms'},
    {id:'coconut_wilt',  label:'Tanaa ya Bud mein pareshani', icon:'🥀', desc:'Stem Bleeding ya Bud Rot ke signs'},
    {id:'coconut_yellow',label:'Patte peele ho rahe hain',   icon:'🟡', desc:'Yellowing — Nutrient ya disease'},
    {id:'coconut_none',  label:'Koi dikhi problem nahi',     icon:'✅', desc:'Preventive check — healthy dikhta hai'},
  ]
};

const Q1_DATA = {
  id:'q1',text:'Fasal mein kya problem dikhi hai?',
  hint:'Sabse pehle jo problem dikha woh select karein',
  options:[
    {id:'spots',label:'Daag / Spots',icon:'🔴',desc:'Patte ya tanay pe daag hain'},
    {id:'wilt', label:'Murjhana / Wilting',icon:'🥀',desc:'Paudha murjhaya hua hai'},
    {id:'yellow',label:'Peele Patte',icon:'🟡',desc:'Patte peele ho rahe hain'},
    {id:'none', label:'Koi Problem Nahi',icon:'✅',desc:'Preventive check chahiye'},
  ]
};

const BRANCH_QS = {
  spots:[
    {id:'q2',text:'Daag kahan hain?',hint:'Sabse zyada kahan dikhe?',options:[
      {id:'lower',label:'Neeche ke patte',icon:'⬇️'},{id:'upper',label:'Upar ke patte',icon:'⬆️'},
      {id:'all',label:'Poori fasal',icon:'🌿'},{id:'stem',label:'Tanaa / Stem',icon:'🌵'}
    ]},
    {id:'q3',text:'Daag ka rang kya hai?',hint:'Ek choose karein',options:[
      {id:'brown',label:'Bhoora / Brown',icon:'🟤'},{id:'yellow',label:'Peela / Yellow',icon:'🟡'},
      {id:'black',label:'Kaala / Black',icon:'⚫'},{id:'red',label:'Laal / Red',icon:'🔴'}
    ]},
    {id:'q4',text:'Daag ka aakar kaisa hai?',hint:'Shape describe karein',options:[
      {id:'round',label:'Gol / Round',icon:'⭕'},{id:'irreg',label:'Irregular',icon:'🔶'},
      {id:'stripe',label:'Dhabbedaar',icon:'🫧'},{id:'ring',label:'Ring shape',icon:'🎯'}
    ]},
    {id:'q9',text:'Pichle hafte baarish aayi thi?',hint:'Moisture level samajhna zaroori hai',options:[
      {id:'heavy',label:'Haan, zyada baarish',icon:'🌧️'},{id:'light',label:'Thodi si baarish',icon:'🌦️'},
      {id:'no',label:'Bilkul nahi',icon:'☀️'},{id:'fog',label:'Fog / Kuhasa tha',icon:'🌫️'}
    ]},
    {id:'q20',text:'Pichli spray kab ki thi?',hint:'Fungicide / pesticide schedule',options:[
      {id:'recent',label:'3 din se kam',icon:'💊'},{id:'week',label:'1 hafte mein',icon:'📅'},
      {id:'old',label:'2+ hafte pehle',icon:'⏳'},{id:'never',label:'Kabhi nahi',icon:'❌'}
    ]},
  ],
  wilt:[
    {id:'q7',text:'Paudha kab murjhata hai?',hint:'Timing pattern batayein',options:[
      {id:'morning',label:'Subah theek, shaam murjhata',icon:'🌅'},{id:'always',label:'Hamesha murjhaya',icon:'💧'},
      {id:'heat',label:'Sirf tej dhoop mein',icon:'☀️'},{id:'water',label:'Paani dene ke baad',icon:'💦'}
    ]},
    {id:'q9',text:'Pichle hafte baarish ya zyada paani?',hint:'Waterlogging check',options:[
      {id:'heavy',label:'Zyada baarish',icon:'🌧️'},{id:'normal',label:'Normal',icon:'🌤️'},
      {id:'no',label:'Bilkul sukha',icon:'🏜️'},{id:'fog',label:'Fog / Nami',icon:'🌫️'}
    ]},
    {id:'q15',text:'Paani kitni baar dete hain?',hint:'Irrigation frequency',options:[
      {id:'daily',label:'Roz',icon:'💦'},{id:'alt',label:'Ek din chhodkar',icon:'📆'},
      {id:'twice',label:'Hafte mein 2 baar',icon:'🗓️'},{id:'once',label:'Hafte mein 1 baar',icon:'📅'}
    ]},
    {id:'q16',text:'Khet mein paani ruka rehta hai?',hint:'Drainage problem?',options:[
      {id:'yes',label:'Haan, bahut rukta hai',icon:'🌊'},{id:'sometimes',label:'Kabhi kabhi',icon:'🤔'},
      {id:'no',label:'Nahi, drain ho jaata',icon:'✅'},{id:'dk',label:'Pata nahi',icon:'❓'}
    ]},
    {id:'q30',text:'Problem kab se shuru hui?',hint:'Timeline',options:[
      {id:'today',label:'Aaj se / 1-2 din',icon:'📍'},{id:'week',label:'Is hafte',icon:'📅'},
      {id:'twoweek',label:'2 hafte se',icon:'🗓️'},{id:'month',label:'1+ mahine se',icon:'⏳'}
    ]},
  ],
  yellow:[
    {id:'q5',text:'Patte girr bhi rahe hain?',hint:'Leaf drop pattern',options:[
      {id:'heavy',label:'Bahut zyada gir rahe',icon:'🍂'},{id:'some',label:'Thode gir rahe',icon:'🍁'},
      {id:'no',label:'Nahi gir rahe',icon:'🌿'},{id:'tip',label:'Sirf tip se peele',icon:'🌱'}
    ]},
    {id:'q10',text:'Humidity / Nami kaisi rehti hai?',hint:'',options:[
      {id:'high',label:'Bahut zyada nami',icon:'💧'},{id:'med',label:'Normal',icon:'🌡️'},
      {id:'low',label:'Bahut sukha',icon:'🏜️'},{id:'variable',label:'Badlti rehti hai',icon:'🌤️'}
    ]},
    {id:'q21',text:'Pichla fertilizer konsa diya tha?',hint:'Poshan deficiency check',options:[
      {id:'urea',label:'Urea (Neel)',icon:'🧪'},{id:'dap',label:'DAP',icon:'⚗️'},
      {id:'npk',label:'NPK Mix',icon:'🌱'},{id:'none',label:'Kuch nahi diya',icon:'❌'}
    ]},
    {id:'q22',text:'Fertilizer kab diya tha?',hint:'Timing matters',options:[
      {id:'recent',label:'3 din mein',icon:'📍'},{id:'week',label:'Is hafte',icon:'📅'},
      {id:'old',label:'2+ hafte pehle',icon:'⏳'},{id:'never',label:'Kabhi nahi',icon:'❌'}
    ]},
    {id:'q18',text:'Soil / Mitti ka test karaya?',hint:'pH aur NPK levels',options:[
      {id:'recent',label:'Haan, haal hi mein',icon:'✅'},{id:'old',label:'Puraana test tha',icon:'📋'},
      {id:'no',label:'Nahi karaya',icon:'❌'},{id:'plan',label:'Plan hai',icon:'📝'}
    ]},
  ],
  none:[
    {id:'q25',text:'Fasal kitne din purani hai?',hint:'Growth stage samajhna',options:[
      {id:'seed',label:'0–30 din (Seedling)',icon:'🌱'},{id:'veg',label:'30–60 din (Vegetative)',icon:'🌿'},
      {id:'flower',label:'60–90 din (Flowering)',icon:'🌸'},{id:'fruit',label:'90+ din (Fruiting)',icon:'🍅'}
    ]},
    {id:'q29',text:'Fasal abhi kis stage mein hai?',hint:'',options:[
      {id:'seed',label:'Seedling',icon:'🌱'},{id:'veg',label:'Vegetative',icon:'🌿'},
      {id:'flower',label:'Flowering',icon:'🌸'},{id:'fruit',label:'Fruiting / Maturing',icon:'🍎'}
    ]},
    {id:'q11',text:'Fasal ko poori dhoop milti hai?',hint:'Sunlight requirement',options:[
      {id:'full',label:'Haan, 8+ ghante',icon:'☀️'},{id:'partial',label:'4–6 ghante',icon:'⛅'},
      {id:'shade',label:'Zyada chhhav',icon:'🌥️'},{id:'vary',label:'Badlta rehta hai',icon:'🔄'}
    ]},
    {id:'q27',text:'Konsa beej / variety use kiya?',hint:'Disease resistance check',options:[
      {id:'hybrid',label:'Hybrid variety',icon:'🧬'},{id:'local',label:'Local / Desi beej',icon:'🌾'},
      {id:'certified',label:'Certified seed',icon:'📜'},{id:'dk',label:'Pata nahi',icon:'🤷'}
    ]},
    {id:'q28',text:'Aas-paas ke khet mein bhi koi problem?',hint:'Outbreak detection',options:[
      {id:'yes',label:'Haan, unhe bhi hai',icon:'⚠️'},{id:'no',label:'Nahi, theek hain',icon:'✅'},
      {id:'dk',label:'Check nahi kiya',icon:'🤔'},{id:'some',label:'Kuch khet mein hai',icon:'🔍'}
    ]},
  ],

  /* ── COCONUT-ONLY QUESTIONS (30 total, 5 shown per session) ── */

  /* Branch A: Daag/Spots → Gray Leaf Spot ya Leaf Rot */
  coconut_spots:[
    {id:'cq2',text:'Daag kahan nazar aa rahe hain?',hint:'Patte ka kaunsa hissa zyada prabhavit hai?',
      options:[
        {id:'c_tip',   label:'Patte ki nauk pe',    icon:'📍', desc:'Tip se shuru hota hai'},
        {id:'c_middle',label:'Beech mein',           icon:'🎯', desc:'Patte ke darmiyan'},
        {id:'c_all',   label:'Poore patte pe',       icon:'🍃', desc:'Pura patta prabhavit'},
        {id:'c_frond', label:'Poori frond/shaakh pe',icon:'🌿', desc:'Puri shaakh kharab'},
      ]
    },
    {id:'cq3',text:'Daag ka rang kaisa hai?',hint:'Sabse sahi description choose karein',
      options:[
        {id:'c_gray',   label:'Bhoora-Bhura / Grayish',icon:'⬜', desc:'Gray Leaf Spot ka sign'},
        {id:'c_brown',  label:'Koyla Bhoora / Dark Brown',icon:'🟤', desc:'Leaf rot ka sign'},
        {id:'c_yellow_halo',label:'Peele ring ke saath daag',icon:'🟡', desc:'Fungal infection'},
        {id:'c_black',  label:'Kaale daag',              icon:'⚫', desc:'Advanced infection'},
      ]
    },
    {id:'cq4',text:'Daag ka aakar kaisa hai?',hint:'Shape closely dekho',
      options:[
        {id:'c_oval',    label:'Oval / Lamba',          icon:'⭕', desc:'Pestalotiopsis sign'},
        {id:'c_irreg',   label:'Irregular shape',       icon:'🔶', desc:'Phytophthora sign'},
        {id:'c_stripe',  label:'Lambi pattiyan / Strips',icon:'🫧', desc:'Bacterial lesion'},
        {id:'c_concentric',label:'Gol rings mein',      icon:'🎯', desc:'Fungal bull-eye'},
      ]
    },
    {id:'cq5',text:'Patte ke aandar koi powder ya fungus dikhti hai?',hint:'Patte palat ke dekho',
      options:[
        {id:'c_powder_yes',label:'Haan, safed powder',  icon:'⬜', desc:'Powdery mildew'},
        {id:'c_spores',    label:'Haan, kaale spore',   icon:'⚫', desc:'Fungal sporulation'},
        {id:'c_none',      label:'Nahi, sirf daag',     icon:'✅', desc:'Leaf spot only'},
        {id:'c_wet',       label:'Geela ya water-soaked',icon:'💧',desc:'Bacterial/fungal'},
      ]
    },
    {id:'cq6',text:'Kitne patte prabhavit hain?',hint:'Spread ka andaza lagao',
      options:[
        {id:'c_few',    label:'1-2 patte',         icon:'🍃', desc:'Abhi shuruat'},
        {id:'c_some',   label:'5-10 patte',        icon:'🌿', desc:'Moderate spread'},
        {id:'c_half',   label:'Aadha ped',         icon:'🌴', desc:'Serious problem'},
        {id:'c_all_tree',label:'Poora ped kharab', icon:'🆘', desc:'Critical — turant karein'},
      ]
    },
  ],

  /* Branch B: Murjhana/Wilt → Bud Rot ya Stem Bleeding */
  coconut_wilt:[
    {id:'cq7',text:'Kaunsa hissa sabse zyada prabhavit hai?',hint:'Dhyan se pehchano',
      options:[
        {id:'c_topbud',   label:'Upar wali kali (Bud)',  icon:'🌱', desc:'Bud rot ka sign'},
        {id:'c_trunk',    label:'Tanaa / Trunk',          icon:'🌵', desc:'Stem bleeding'},
        {id:'c_roots',    label:'Jadein / Roots',         icon:'🌳', desc:'Root rot'},
        {id:'c_crown',    label:'Crown — ped ka dil',    icon:'👑', desc:'Crown rot'},
      ]
    },
    {id:'cq8',text:'Tanay ya bud se koi liquid (ras) nikalta hai?',hint:'Tanaa closely check karo',
      options:[
        {id:'c_dark_liquid',label:'Haan, kaala/bhoora liquid',icon:'🖤', desc:'Stem bleeding — serious'},
        {id:'c_sticky',     label:'Haan, chipchipa ras',       icon:'🍯', desc:'Fungal exudate'},
        {id:'c_none_liquid',label:'Nahi, koi liquid nahi',     icon:'❌', desc:'Physical damage'},
        {id:'c_smell',      label:'Badboo aati hai',           icon:'👃', desc:'Bacterial rot'},
      ]
    },
    {id:'cq9',text:'Nayi pattiyaan kaise dikhti hain?',hint:'Sabse upar ke naye patte dekho',
      options:[
        {id:'c_brown_young',label:'Bhoori ho gayi — paghlti hain',icon:'🟤', desc:'Bud rot classic sign'},
        {id:'c_pale',       label:'Pale/Peeli aur kamzor',       icon:'🟡', desc:'Nutrient deficiency'},
        {id:'c_normal_but_wilt',label:'Theek lagti hai phir bhi murjhati',icon:'🥀',desc:'Root/vascular issue'},
        {id:'c_no_new',    label:'Nayi pattiyaan aa hi nahi raheen',icon:'⛔',desc:'Severe bud damage'},
      ]
    },
    {id:'cq10',text:'Kab se yeh problem shuru hui?',hint:'Timeline batao',
      options:[
        {id:'c_days',  label:'2-3 din pehle',    icon:'📅', desc:'Early stage'},
        {id:'c_week',  label:'1-2 hafte pehle',  icon:'🗓️', desc:'Progressed'},
        {id:'c_month', label:'1 mahina pehle',   icon:'📆', desc:'Chronic issue'},
        {id:'c_sudden',label:'Achanak hua',       icon:'⚡', desc:'Acute onset'},
      ]
    },
    {id:'cq11',text:'Kya aas-paas ke naariyal ke ped bhi prabhavit hain?',hint:'Poora bagaan dekho',
      options:[
        {id:'c_spread_yes', label:'Haan, 2-3 ped aur bhi',    icon:'⚠️', desc:'Epidemic risk'},
        {id:'c_spread_many',label:'Haan, bahut saare ped',    icon:'🆘', desc:'Critical spread'},
        {id:'c_only_one',   label:'Sirf yeh ek ped',          icon:'🎯', desc:'Isolated case'},
        {id:'c_nearby',     label:'Paas ka ek ped',           icon:'👀', desc:'Early spread'},
      ]
    },
  ],

  /* Branch C: Peele Patte → Yellowing / Nutrient Issues */
  coconut_yellow:[
    {id:'cq12',text:'Peele patte kahan se shuru ho rahe hain?',hint:'Kaunse patte pehle peele hue?',
      options:[
        {id:'c_old_first', label:'Purane/Neeche ke patte pehle',icon:'⬇️', desc:'Nitrogen deficiency'},
        {id:'c_young_first',label:'Nayi pattiyaan pehle peeli',  icon:'⬆️', desc:'Iron/Zinc deficiency'},
        {id:'c_all_same',  label:'Sab ek saath peele',          icon:'🌿', desc:'Root/stem issue'},
        {id:'c_frond_tip', label:'Frond ki nauk se shuru',      icon:'📍', desc:'Potassium deficiency'},
      ]
    },
    {id:'cq13',text:'Peelapan ka rang kaisa hai?',hint:'Color closely dekho',
      options:[
        {id:'c_bright_y',  label:'Chamkila Peela',     icon:'💛', desc:'Nutrient deficiency'},
        {id:'c_pale_green',label:'Halkaa Peela-Hara',  icon:'🟢', desc:'Mild chlorosis'},
        {id:'c_bronze',    label:'Kaansi/Bronze rang', icon:'🥉', desc:'Potassium/Mn deficiency'},
        {id:'c_necrotic',  label:'Peele mein bhoora/dead',icon:'🟤',desc:'Advanced damage'},
      ]
    },
    {id:'cq14',text:'Kya pichle 3 mahine mein khad (fertilizer) diya tha?',hint:'Nutrient management check',
      options:[
        {id:'c_fert_yes',   label:'Haan, niyamit diya',   icon:'✅', desc:'Rule out deficiency'},
        {id:'c_fert_partial',label:'Kabhi kabhi',          icon:'⚠️', desc:'Partial deficiency'},
        {id:'c_fert_no',    label:'Nahi, bilkul nahi',    icon:'❌', desc:'Likely deficiency'},
        {id:'c_fert_excess',label:'Zyada diya shayad',    icon:'⬆️', desc:'Toxicity possible'},
      ]
    },
    {id:'cq15',text:'Zameen (soil) kaisi hai aapke naariyal ke bagaan ki?',hint:'Soil type identify karo',
      options:[
        {id:'c_sandy',   label:'Reti (Sandy)',          icon:'🏖️', desc:'Low nutrient retention'},
        {id:'c_loamy',   label:'Domatii (Loamy)',       icon:'🌱', desc:'Good soil'},
        {id:'c_clay',    label:'Chiknii Maitti (Clay)', icon:'🧱', desc:'Drainage issue'},
        {id:'c_saline',  label:'Khaari (Saline/Salty)',icon:'🧂', desc:'Salt stress — coastal'},
      ]
    },
    {id:'cq16',text:'Kya tree ki jadein (roots) ko paani mein rehna padta hai?',hint:'Waterlogging check',
      options:[
        {id:'c_water_yes', label:'Haan, barish mein paani rukta hai',icon:'💧', desc:'Root rot risk'},
        {id:'c_water_no',  label:'Nahi, achha drainage hai',        icon:'✅', desc:'Not waterlogging'},
        {id:'c_dry',       label:'Ulta — bahut sukha rehta hai',    icon:'🏜️', desc:'Drought stress'},
        {id:'c_seasonal',  label:'Mausam ke hisaab se',             icon:'🌦️', desc:'Seasonal issue'},
      ]
    },
  ],

  /* Branch D: Koi problem nahi / Preventive → General coconut health */
  coconut_none:[
    {id:'cq17',text:'Naariyal ka ped kitne saal purana hai?',hint:'Age se disease risk samjha jaata hai',
      options:[
        {id:'c_young',  label:'1-3 saal (Navjaat)',    icon:'🌱', desc:'Young tree — more vulnerable'},
        {id:'c_mid',    label:'4-10 saal',             icon:'🌴', desc:'Growing phase'},
        {id:'c_mature', label:'10-20 saal',            icon:'🥥', desc:'Productive age'},
        {id:'c_old',    label:'20+ saal',              icon:'🌳', desc:'Old tree care needed'},
      ]
    },
    {id:'cq18',text:'Pichle season mein koi bimari thi?',hint:'History check karo',
      options:[
        {id:'c_hist_none',  label:'Bilkul nahi tha',        icon:'✅', desc:'Healthy history'},
        {id:'c_hist_minor', label:'Thodi bimari thi',       icon:'⚠️', desc:'Watch carefully'},
        {id:'c_hist_major', label:'Badi bimari thi',        icon:'🆘', desc:'High recurrence risk'},
        {id:'c_hist_spray', label:'Spray kiya tha — theek hua',icon:'💊',desc:'Treatment worked'},
      ]
    },
    {id:'cq19',text:'Aapka bagaan kahan hai?',hint:'Location se disease risk samjha jaata hai',
      options:[
        {id:'c_coastal',   label:'Samundar ke paas (Coastal)',icon:'🌊', desc:'High humidity — fungal risk'},
        {id:'c_inland',    label:'Andar desh (Inland)',       icon:'🏞️', desc:'Moderate risk'},
        {id:'c_hilly',     label:'Pahadi ilaaka',             icon:'⛰️', desc:'Cool — different diseases'},
        {id:'c_backwater', label:'Nadi/taalaab ke paas',     icon:'🌊', desc:'High moisture risk'},
      ]
    },
    {id:'cq20',text:'Abhi kaunsa mausam chal raha hai?',hint:'Season disease pattern affect karta hai',
      options:[
        {id:'c_monsoon',  label:'Barsaat / Monsoon',  icon:'🌧️', desc:'Highest fungal risk'},
        {id:'c_winter',   label:'Sardi / Winter',     icon:'❄️', desc:'Moderate risk'},
        {id:'c_summer',   label:'Garmi / Summer',     icon:'☀️', desc:'Drought stress risk'},
        {id:'c_preM',     label:'Barsaat se pehle',   icon:'⛅', desc:'Prevention time'},
      ]
    },
    {id:'cq21',text:'Aakhri baar spray kab kiya tha?',hint:'Chemical history important hai',
      options:[
        {id:'c_sp_week',  label:'Is hafte',           icon:'✅', desc:'Recent — good'},
        {id:'c_sp_month', label:'Pichle mahine',      icon:'⚠️', desc:'Due soon'},
        {id:'c_sp_season',label:'Pichle season mein', icon:'❌', desc:'Overdue'},
        {id:'c_sp_never', label:'Kabhi nahi kiya',    icon:'🆘', desc:'Immediate action needed'},
      ]
    },
  ],

  /* EXTRA 10 QUESTIONS (cq22-cq31) — Advanced coconut diagnostics */
  coconut_advanced:[
    {id:'cq22',text:'Naariyal (fruit) mein koi problem dikh rahi hai?',hint:'Fruit quality check',
      options:[
        {id:'c_fr_ok',    label:'Nahi, theek lag raha hai',       icon:'✅', desc:'No fruit issue'},
        {id:'c_fr_small', label:'Naariyal chhota reh jaata hai',  icon:'📉', desc:'Poor nutrition'},
        {id:'c_fr_black', label:'Naariyal pe kaale daag',         icon:'⚫', desc:'Fungal/bacterial'},
        {id:'c_fr_drop',  label:'Kacha naariyal girta hai',       icon:'⬇️', desc:'Physiological/pest'},
      ]
    },
    {id:'cq23',text:'Kya koi kida (insect) dikha hai naariyal ke ped pe?',hint:'Pest + disease combo check',
      options:[
        {id:'c_bug_mite',  label:'Haan, chhote kide/mite',       icon:'🐛', desc:'Mite damage'},
        {id:'c_bug_weevil',label:'Haan, bada kaala kida (weevil)',icon:'🪲', desc:'Rhinoceros beetle'},
        {id:'c_bug_scale', label:'Haan, chipke hue kide',        icon:'🔵', desc:'Scale insect'},
        {id:'c_bug_no',    label:'Koi kida nahi dikha',          icon:'❌', desc:'Disease only'},
      ]
    },
    {id:'cq24',text:'Tanae pe koi daag ya ghav (wound) hai?',hint:'Physical damage check',
      options:[
        {id:'c_wound_cut',  label:'Haan, kataa hua daag',         icon:'🔪', desc:'Physical + infection risk'},
        {id:'c_wound_black',label:'Haan, kaala hua hissa',        icon:'⚫', desc:'Stem bleeding sign'},
        {id:'c_wound_crack',label:'Haan, daraar (crack) hai',    icon:'🔓', desc:'Borer entry point'},
        {id:'c_wound_no',   label:'Nahi, tanaa saaf hai',        icon:'✅', desc:'No physical damage'},
      ]
    },
    {id:'cq25',text:'Pani (irrigation) kitna aur kab dete ho?',hint:'Water management crucial for coconut',
      options:[
        {id:'c_irr_regular',label:'Niyamit — hafte mein ek baar',icon:'💧', desc:'Good practice'},
        {id:'c_irr_excess', label:'Zyada paani deta/deti hoon',  icon:'🌊', desc:'Root rot risk'},
        {id:'c_irr_less',   label:'Kam paani milta hai',         icon:'🏜️', desc:'Drought stress'},
        {id:'c_irr_rain',   label:'Sirf barsaat pe dependent',   icon:'🌧️', desc:'Irregular moisture'},
      ]
    },
    {id:'cq26',text:'Naariyal ke ped ke aas-paas kaun si fasal lagaate ho?',hint:'Intercrop pest transfer check',
      options:[
        {id:'c_ic_arecanut',label:'Supari (Arecanut)',       icon:'🌰', desc:'High bud rot transfer risk'},
        {id:'c_ic_banana',  label:'Kela (Banana)',           icon:'🍌', desc:'Leaf spot cross-infection'},
        {id:'c_ic_veg',     label:'Sabziyan (Vegetables)',   icon:'🥦', desc:'Generally safe'},
        {id:'c_ic_none',    label:'Kuch nahi — sirf naariyal',icon:'🥥',desc:'Monocrop'},
      ]
    },
    {id:'cq27',text:'Zameen ki pH kaisi hai?',hint:'Soil acidity affects disease',
      options:[
        {id:'c_ph_acid',   label:'Khaati (Acidic < 6)',     icon:'🔴', desc:'Nutrient lockout'},
        {id:'c_ph_normal', label:'Theek (6-7.5)',           icon:'✅', desc:'Ideal for coconut'},
        {id:'c_ph_alkali', label:'Khaari (Alkaline > 7.5)',icon:'🔵', desc:'Mg/Mn deficiency risk'},
        {id:'c_ph_unknown',label:'Pata nahi',               icon:'❓', desc:'Test recommended'},
      ]
    },
    {id:'cq28',text:'Kya poshan (fertilizer) mein NPK + micronutrients dete ho?',hint:'Complete nutrition check',
      options:[
        {id:'c_npk_complete',label:'Haan, NPK + Mg + Zn + B',  icon:'✅', desc:'Complete nutrition'},
        {id:'c_npk_basic',   label:'Sirf NPK deta/deti hoon',   icon:'⚠️', desc:'Micronutrient gap'},
        {id:'c_npk_organic', label:'Sirf compost/organic',      icon:'🌱', desc:'May be deficient'},
        {id:'c_npk_no',      label:'Bilkul fertilizer nahi',    icon:'❌', desc:'High deficiency risk'},
      ]
    },
    {id:'cq29',text:'Kya pedon mein se mushroom jaisa kuch ugaa hai (basidiomycete)?',hint:'Root/trunk rot sign',
      options:[
        {id:'c_mush_yes',  label:'Haan, jadein/tanay ke paas',  icon:'🍄', desc:'Ganoderma/root rot critical'},
        {id:'c_mush_no',   label:'Nahi, kuch nahi',             icon:'✅', desc:'No basal rot'},
        {id:'c_mush_check',label:'Check nahi kiya abhi tak',   icon:'🔍', desc:'Check immediately'},
        {id:'c_mush_soil', label:'Zameen mein hai, ped pe nahi',icon:'🌱', desc:'Environmental only'},
      ]
    },
    {id:'cq30',text:'Ped ka upar wala hissa (canopy) kaisa dikhta hai?',hint:'Overall health indicator',
      options:[
        {id:'c_can_full',   label:'Ghana — poora hara',         icon:'🌴', desc:'Good health'},
        {id:'c_can_sparse', label:'Virla — patte kam hain',     icon:'🌿', desc:'Moderate stress'},
        {id:'c_can_dead',   label:'Upar se sukh raha hai',      icon:'💀', desc:'Severe — bud/trunk rot'},
        {id:'c_can_lean',   label:'Ped jhuk raha hai ek taraf', icon:'↗️', desc:'Root/structural issue'},
      ]
    },
  ]
};

/* ════════════════════════════════════════════════════════════════
   QUESTION FLOW STYLES
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   CONSULT PAGE  — Crop → Upload → Question Flow → Processing
════════════════════════════════════════════════════════════════ */
/* ════════════════════════════════════════════════════════════════
   FARMER DASHBOARD
════════════════════════════════════════════════════════════════ */
function FarmerDash({user,nav,toast}) {
  const myCrops=CROPS.filter(c=>user?.crops?.includes(c.id));
  const displayCrops=myCrops.length?myCrops:CROPS.slice(0,5);
  const [farmScore,setFarmScore]=useState(78);
  const [farmConsults,setFarmConsults]=useState(CONSULTATIONS.slice(0,3));
  useEffect(()=>{
    if(!user) return;
    API.get('/api/consultations')
      .then(d=>{ if(d.consultations&&d.consultations.length>0){
        const list=d.consultations;
        const recent=list.slice(0,5);
        const avgSev=recent.reduce((s,c)=>s+(c.severity||1),0)/recent.length;
        const completedRatio=list.filter(c=>c.status==='completed').length/list.length;
        const calculated=Math.max(40,Math.min(100,Math.round(100-(avgSev*8)+(completedRatio*10))));
        setFarmScore(calculated);
        setFarmConsults(list.slice(0,3).map(c=>({
        id:c._id, crop:c.cropName, emoji:c.cropEmoji||'🌱',
        issue:`${c.disease} — ${c.confidence}% confidence`,
        date:new Date(c.createdAt).toLocaleDateString('en-IN')+' · '+new Date(c.createdAt).toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit'}),
        expert:c.expertName||'Auto-assign',
        status:c.status==='completed'?'completed':c.status==='expert_assigned'?'expert':'pending',
        statusLabel:c.status==='completed'?'Completed':c.status==='expert_assigned'?'Expert Assigned':'Pending',
        sev:c.severity||1, conf:c.confidence||0,
      })));
      }}).catch(()=>{});
  },[user]);

  return (
    <div className="wrap">
      {/* ── GREETING CARD ── */}
      <div className="greet-card">
        <div className="gc-ring1"/><div className="gc-ring2"/>
        <div className="gc-top">
          <div>
            <div style={{fontSize:12,opacity:.76,marginBottom:4,fontWeight:600}}>🌅 Subah ki shubhkamnayein</div>
            <div className="gc-name">Namaskar, {user?.name?.split(' ')?.[0]||'Kisan'}! 🙏</div>
            <div className="gc-sub">Aaj {displayCrops.length} fasalein active hain aapke farm mein</div>
          </div>
          <div style={{textAlign:'right'}}>
            <div className="gc-score-n">{farmScore}</div>
            <div className="gc-score-l">Farm Health Score 🏆</div>
          </div>
        </div>
        <div style={{marginBottom:16}}>
          <div style={{display:'flex',justifyContent:'space-between',fontSize:12,fontWeight:600,opacity:.82,marginBottom:5}}>
            <span>Farm Health</span><span>{farmScore}/100</span>
          </div>
          <div className="gc-prog"><div className="gc-prog-f" style={{width:`${farmScore}%`}}/></div>
        </div>
        <div className="gc-btns">
          <button className="gc-btn prim" onClick={()=>nav('consultation')}>🔬 Fasal Check Karo</button>
          <button className="gc-btn" onClick={()=>nav('voice')}>🎤 Voice Input</button>
          <button className="gc-btn" onClick={()=>nav('forecast')}>📊 Forecast</button>
          <button className="gc-btn" onClick={()=>nav('satellite')}>🛰️ Satellite</button>
          <button className="gc-btn" onClick={()=>nav('soil-sensors')}>🌱 Soil Sensors</button>
          <button className="gc-btn" onClick={()=>nav('robot-dashboard')}>🤖 Robots</button>
          <button className="gc-btn" onClick={()=>nav('insurance')}>🏦 Insurance</button>
          <button className="gc-btn" onClick={()=>nav('marketplace')}>📦 Market</button>
        </div>
      </div>

      {/* ── URGENT ALERT BANNER ── */}
      <div className="alert-bar">
        <div className="alert-txt">🔴 <span>Disease Outbreak:</span> Late Blight — 8 cases within 5km of your farm! Preventive spray karein.</div>
        <button className="btn btn-red btn-sm" onClick={()=>nav('notifications')}>Dekhna Hai →</button>
      </div>

      {/* ── MY CROPS (Horizontal scroll) ── */}
      <div style={{marginBottom:24}}>
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:13}}>
          <div style={{fontSize:16,fontWeight:800,color:'var(--g1)'}}>🌾 Meri Fasalein</div>
          <button className="btn btn-ghost btn-sm" onClick={()=>nav('my-farm')}>Sab Dekho →</button>
        </div>
        <div className="crops-scroll">
          {displayCrops.map(c=>(
            <div key={c.id} className="crop-sc-card" onClick={()=>{nav('consultation');}}>
              <div className="crop-sc-em">{c.emoji}</div>
              <div className="crop-sc-nm">{c.name}</div>
              <div className="crop-sc-hl" style={{color:c.health>80?'var(--g4)':c.health>60?'var(--a2)':'var(--r2)'}}>● {c.health}%</div>
              <div className="crop-sc-st">{c.stage}</div>
            </div>
          ))}
          <div className="crop-sc-card" style={{background:'var(--gp)',border:'2px dashed var(--br2)'}} onClick={()=>nav('consultation')}>
            <div style={{fontSize:26,marginBottom:7}}>➕</div>
            <div style={{fontSize:12.5,fontWeight:700,color:'var(--g3)'}}>Nayi Fasal</div>
            <div style={{fontSize:11,color:'var(--tx3)',marginTop:3}}>Add karo</div>
          </div>
        </div>
      </div>

      {/* ── MAIN 2-COL GRID ── */}
      <div className="dash-2">
        {/* LEFT: Recent Consultations */}
        <div>
          <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:13}}>
            <div style={{fontSize:16,fontWeight:800,color:'var(--g1)'}}>📋 Recent Consultations</div>
            <button className="btn btn-ghost btn-sm" onClick={()=>nav('my-consultations')}>Sab →</button>
          </div>
          {farmConsults.map(c=>(
            <div key={c.id} className="cons-card card-hov" style={{marginBottom:11}} onClick={()=>nav('ai-report')}>
              <div style={{display:'flex',gap:11,padding:14}}>
                <div style={{width:56,height:56,borderRadius:10,flexShrink:0,fontSize:28,display:'flex',alignItems:'center',justifyContent:'center',background:'linear-gradient(135deg,var(--gp),var(--gpb))'}}>{c.emoji}</div>
                <div style={{flex:1,minWidth:0}}>
                  <div style={{display:'flex',justifyContent:'space-between',marginBottom:3,alignItems:'flex-start',gap:6}}>
                    <div className="cons-nm">{c.crop}</div>
                    <span className={`badge ${c.status==='completed'?'bg-g':c.status==='expert'?'bg-b':c.status==='pending'?'bg-a':'bg-p'}`} style={{flexShrink:0,fontSize:11}}>{c.status==='completed'?'✅':c.status==='expert'?'👨‍⚕️':c.status==='pending'?'⏳':'🔵'} {c.statusLabel}</span>
                  </div>
                  <div className="cons-issue">{c.issue}</div>
                  <div style={{fontSize:11.5,color:'var(--tx3)',marginBottom:8}}>👨‍⚕️ {c.expert} • {c.date.split('•')[0].trim()}</div>
                  <div className="cons-acts">
                    <button className="ca-rep" onClick={e=>{e.stopPropagation();nav('ai-report');}}>📄 Report</button>
                    <button className="ca-chat" onClick={e=>{e.stopPropagation();nav('chat');}}>💬 Chat</button>
                  </div>
                </div>
              </div>
            </div>
          ))}
          <button className="btn btn-out btn-sm" style={{width:'100%',marginTop:4}} onClick={()=>nav('my-consultations')}>
            Sari Consultations Dekho →
          </button>
        </div>

        {/* RIGHT: Weather + Mandi + Quick Actions */}
        <div className="dash-r">
          {/* Weather Widget */}
          <div className="weather-card">
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:11}}>
              <div style={{fontSize:15,fontWeight:800,color:'var(--g1)'}}>🌤️ Aaj Ka Mausam</div>
              <div style={{fontSize:11,color:'var(--tx3)',fontWeight:600}}>📍 {user?.district||'Pune'}, MH</div>
            </div>
            <div className="wt-main">
              <div>
                <div className="wt-temp">28°C</div>
                <div style={{fontSize:13,color:'var(--tx2)',marginTop:2}}>🌤️ Partly Cloudy</div>
              </div>
              <div style={{textAlign:'right'}}>
                <span className="wt-risk risk-med">🟡 Medium Risk</span>
                <div style={{fontSize:12,color:'var(--tx2)',marginTop:5}}>Humidity: 68%</div>
                <div style={{fontSize:12,color:'var(--tx3)',marginTop:2}}>Wind: 12 km/h</div>
              </div>
            </div>
            <div style={{display:'flex',gap:8,marginBottom:10}}>
              {[{d:'Kal',t:'26°C',i:'🌧️'},{d:'Parson',t:'24°C',i:'🌧️'},{d:'3 din',t:'29°C',i:'⛅'}].map(w=>(
                <div key={w.d} style={{flex:1,textAlign:'center',padding:'7px 4px',background:'rgba(255,255,255,.6)',borderRadius:8}}>
                  <div style={{fontSize:14}}>{w.i}</div>
                  <div style={{fontSize:10.5,fontWeight:700,color:'var(--tx2)',marginTop:2}}>{w.d}</div>
                  <div style={{fontSize:11,color:'var(--tx3)'}}>{w.t}</div>
                </div>
              ))}
            </div>
            <div style={{padding:'9px 13px',background:'rgba(240,165,0,.12)',borderRadius:8,fontSize:12.5,color:'var(--a1)',fontWeight:600}}>
              ⚠️ Agle 2 din baarish — spray band rakhein
            </div>
          </div>

          {/* Mandi Prices */}
          <div className="card" style={{padding:18}}>
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:13}}>
              <div style={{fontSize:15,fontWeight:800,color:'var(--g1)'}}>📈 Mandi Prices</div>
              <span style={{fontSize:11,color:'var(--tx3)',fontWeight:600}}>Live</span>
            </div>
            {[{nm:'🍅 Tomato',pr:'₹1,240/qtl',ch:'+12%',d:'up',mspr:'MSP: N/A'},{nm:'🌾 Wheat',pr:'₹2,180/qtl',ch:'Stable',d:'st',mspr:'MSP: ₹2,275'},{nm:'🥔 Potato',pr:'₹780/qtl',ch:'-3%',d:'dn',mspr:'MSP: N/A'}].map(m=>(
              <div key={m.nm} className="mandi-row">
                <div>
                  <div className="mandi-crop">{m.nm}</div>
                  <div style={{fontSize:11,color:'var(--tx3)'}}>{m.mspr}</div>
                </div>
                <div style={{textAlign:'right'}}>
                  <div className="mandi-price">{m.pr}</div>
                  <div className={`mandi-ch ${m.d==='up'?'ch-up':m.d==='dn'?'ch-dn':'ch-st'}`}>{m.d==='up'?'↑':m.d==='dn'?'↓':'→'} {m.ch}</div>
                </div>
              </div>
            ))}
            <div style={{marginTop:11,padding:'9px 13px',background:'var(--gp)',borderRadius:8,fontSize:12.5,color:'var(--g2)',fontWeight:700}}>
              🍅 Tomato price ↑ 18% — Bechne ka sahi waqt!
            </div>
          </div>

          {/* Quick Stats */}
          <div className="card" style={{padding:18}}>
            <div style={{fontSize:15,fontWeight:800,color:'var(--g1)',marginBottom:13}}>💰 Is Season Ka Kharch</div>
            {[['🌱 Seeds','₹4,500'],['🧪 Fertilizer','₹6,200'],['💊 Medicines','₹1,840'],['👨‍⚕️ Consultations','₹984']].map(([l,v])=>(
              <div key={l} style={{display:'flex',justifyContent:'space-between',padding:'7px 0',borderBottom:'1px solid var(--gp)',fontSize:13}}>
                <span style={{color:'var(--tx2)',fontWeight:600}}>{l}</span>
                <span style={{fontWeight:800,color:'var(--tx)'}}>{v}</span>
              </div>
            ))}
            <div style={{display:'flex',justifyContent:'space-between',paddingTop:10,fontSize:14}}>
              <span style={{fontWeight:700,color:'var(--tx)'}}>Total Kharch</span>
              <span style={{fontWeight:900,color:'var(--r2)'}}>₹13,524</span>
            </div>
            <div style={{display:'flex',justifyContent:'space-between',fontSize:13}}>
              <span style={{fontWeight:600,color:'var(--tx3)'}}>Expected Yield</span>
              <span style={{fontWeight:800,color:'var(--g3)'}}>₹85,000</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   EXPERT DASHBOARD
════════════════════════════════════════════════════════════════ */
/* ════════════════════════════════════════════════════════════════
   EXPERTS PAGE
════════════════════════════════════════════════════════════════ */
function ExpertsPage({user,nav,toast}) {
  const [avail,setAvail]=useState(false);
  const [spec,setSpec]=useState('');
  const [apiExperts,setApiExperts]=useState([]);
  const [loading,setLoading]=useState(true);
  const [profileExpert,setProfileExpert]=useState(null);
  useEffect(()=>{
    fetch('/api/experts').then(r=>r.json()).then(d=>{
      if(d.experts&&d.experts.length>0) setApiExperts(d.experts);
      setLoading(false);
    }).catch(()=>setLoading(false));
  },[]);
  const baseList=apiExperts.length>0?apiExperts.map(e=>({
    id:e._id, name:e.name, spec:e.spec||'Agricultural Expert',
    exp:'5+ yrs', langs:(e.langs||'Hindi').split(','),
    price:e.fee||500, rating:e.rating||4.5, reviews:e.totalCases||0,
    online:e.available||false, emoji:'👨‍🔬', crops:e.crops||'All Crops',
    cases:e.totalCases||0, response:'45 min', success:95
  })):EXPERTS;
  const list=baseList.filter(e=>{if(avail&&!e.online)return false;if(spec&&e.spec!==spec)return false;return true;});
  return (
    <div className="wrap">
      <div style={{textAlign:'center',marginBottom:32}}>
        <div style={{fontFamily:"'Baloo 2',cursive",fontSize:32,fontWeight:900,color:'var(--g1)'}}>👨‍⚕️ Certified Agricultural Experts</div>
        <div style={{fontSize:15,color:'var(--tx2)',marginTop:6}}>Sahi specialist dhundho — disease, soil ya crop ke liye</div>
      </div>
      <div className="exp-filters">
        <select className="flt-sel" value={spec} onChange={e=>setSpec(e.target.value)}>
          <option value="">All Specializations</option>
          <option>Plant Pathologist</option><option>Horticulture Expert</option>
          <option>Soil Scientist</option><option>Crop Scientist</option>
        </select>
        {['Crop Type','Language','Min Rating'].map(f=><select key={f} className="flt-sel"><option>{f}</option></select>)}
        <div style={{marginLeft:'auto',display:'flex',alignItems:'center',gap:9,fontSize:13.5,fontWeight:600,color:'var(--tx2)'}}>
          <label className="sw"><input type="checkbox" checked={avail} onChange={e=>setAvail(e.target.checked)}/><span className="sw-sl"/></label>
          Available Now Only
        </div>
      </div>
      <div className="experts-grid">
        {list.map(e=>(
          <div key={e.id} className="exp-card">
            <div style={{display:'flex',gap:13,marginBottom:14}}>
              <div className="exp-av">{e.emoji}{e.online&&<div className="on-dot"/>}</div>
              <div style={{flex:1}}>
                <div className="exp-nm">{e.name}</div>
                <div className="exp-sp">{e.spec}</div>
                <div className="exp-rat">⭐ {e.rating} <span style={{color:'var(--tx3)',fontWeight:400}}>({e.reviews})</span></div>
              </div>
            </div>
            <div className="exp-det">⏱️ {e.exp} experience</div>
            <div className="exp-det">🗣️ {e.langs.join(', ')}</div>
            <div className="exp-det">🌾 {e.crops}</div>
            <div className="exp-det">✅ {e.cases} cases • ⚡ {e.response} avg</div>
            <div className="exp-pr">₹{e.price} <span>/ consultation</span></div>
            <div style={{display:'flex',gap:9}}>
              <button className="btn btn-out btn-sm" style={{flex:1}} onClick={()=>setProfileExpert(e)}>👤 Profile</button>
              <button className="btn btn-g btn-sm" style={{flex:2}} onClick={()=>{if(!user){toast('Pehle login karein!','err');return;}localStorage.setItem('bh_sel_expert',JSON.stringify({id:e.id,name:e.name,spec:e.spec,fee:e.price,rating:e.rating}));nav('booking');}}>✅ Select Expert</button>
            </div>
          </div>
        ))}
      </div>
      <div style={{textAlign:'center',marginTop:30}}>
        <button className="btn btn-out btn-md">Load More Experts</button>
      </div>

      {/* Expert Profile Modal */}
      {profileExpert&&(
        <div style={{position:'fixed',inset:0,background:'rgba(0,0,0,.5)',zIndex:999,display:'flex',alignItems:'center',justifyContent:'center',padding:20}} onClick={()=>setProfileExpert(null)}>
          <div style={{background:'white',borderRadius:'var(--rad)',padding:28,maxWidth:420,width:'100%',boxShadow:'var(--sh2)'}} onClick={e=>e.stopPropagation()}>
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',marginBottom:18}}>
              <div style={{display:'flex',gap:13,alignItems:'center'}}>
                <div style={{width:56,height:56,borderRadius:'50%',background:'linear-gradient(135deg,var(--g5),var(--g6))',display:'flex',alignItems:'center',justifyContent:'center',fontSize:26,position:'relative'}}>
                  {profileExpert.emoji||'👨‍🔬'}
                  {profileExpert.online&&<div className="on-dot"/>}
                </div>
                <div>
                  <div style={{fontSize:18,fontWeight:900,color:'var(--tx)'}}>{profileExpert.name}</div>
                  <div style={{fontSize:13,color:'var(--g3)',fontWeight:700}}>{profileExpert.spec}</div>
                  <div style={{fontSize:12.5,color:'var(--tx3)'}}>⭐ {profileExpert.rating} • {profileExpert.reviews||profileExpert.cases||0} cases</div>
                </div>
              </div>
              <button style={{background:'none',border:'none',fontSize:20,cursor:'pointer',color:'var(--tx3)'}} onClick={()=>setProfileExpert(null)}>✕</button>
            </div>
            <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:10,marginBottom:16}}>
              {[['💰 Fee',`₹${profileExpert.price||profileExpert.fee||500}/consult`],['⏱️ Experience',profileExpert.exp||'5+ yrs'],['🗣️ Languages',(profileExpert.langs||['Hindi']).join?.(', ')||profileExpert.langs],['✅ Success Rate',`${profileExpert.success||95}%`]].map(([k,v])=>(
                <div key={k} style={{padding:'10px 12px',background:'var(--gp)',borderRadius:9}}>
                  <div style={{fontSize:11,color:'var(--tx3)',fontWeight:700}}>{k}</div>
                  <div style={{fontSize:13.5,color:'var(--tx)',fontWeight:800,marginTop:2}}>{v}</div>
                </div>
              ))}
            </div>
            <div style={{padding:'12px 14px',background:'var(--gp)',borderRadius:9,fontSize:13,color:'var(--tx2)',lineHeight:1.65,marginBottom:16}}>
              {profileExpert.bio||`${profileExpert.name} ek certified agricultural expert hain jo ${profileExpert.spec||'crop diseases'} mein specialization rakhte hain. Aapki fasal ki problems ke liye expert guidance milegi.`}
            </div>
            <div style={{display:'flex',gap:10}}>
              <button className="btn btn-out btn-md" style={{flex:1}} onClick={()=>setProfileExpert(null)}>Wapas</button>
              <button className="btn btn-g btn-md" style={{flex:2}} onClick={()=>{
                if(!user){toast('Pehle login karein!','err');return;}
                localStorage.setItem('bh_sel_expert',JSON.stringify({id:profileExpert.id,name:profileExpert.name,spec:profileExpert.spec,fee:profileExpert.price||profileExpert.fee||500,rating:profileExpert.rating}));
                setProfileExpert(null);nav('booking');
              }}>✅ Select Expert →</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   BOOKING PAGE
════════════════════════════════════════════════════════════════ */
function BookingPage({user,nav,toast}) {
  const [type,setType]=useState('text');
  const [selExpert]=useState(()=>{
    try{ return JSON.parse(localStorage.getItem('bh_sel_expert')||'null'); }catch{ return null; }
  });
  const [booking,setBooking]=useState(false);

  const confirmBooking=async()=>{
    if(!user){ if(toast) toast('Pehle login karein!','err'); return; }
    setBooking(true);
    try{
      const consultId=localStorage.getItem('bh_latest_consult');
      if(consultId && selExpert){
        await API.patch('/api/consultations/'+consultId+'/status',{status:'expert_assigned',report:null});
      }
      if(toast) toast('Booking confirmed! Expert jald hi contact karega. ✅');
      setTimeout(()=>nav('my-consultations'),1500);
    }catch(e){ if(toast) toast(e.message,'err'); }
    setBooking(false);
  };

  const types=[{id:'text',l:'💬 Written Report',p:800,t:'24–48 hours',d:'Expert detailed report likhega'},{id:'voice',l:'📞 Voice Call',p:600,t:'15–30 min',d:'Direct call expert se'},{id:'video',l:'📹 Video Call',p:1200,t:'30–45 min',d:'Live field guidance'},{id:'emergency',l:'🚨 Emergency',p:2000,t:'< 2 hours',d:'Critical disease — instant'}];
  const sel=types.find(t=>t.id===type);
  const total=Math.round(sel.p*1.18*1.05);
  return (
    <div className="wrap-sm">
      <button className="btn btn-ghost btn-sm" style={{marginBottom:18}} onClick={()=>nav('experts')}>← Wapas</button>
      <div style={{fontFamily:"'Baloo 2',cursive",fontSize:26,fontWeight:900,color:'var(--g1)',marginBottom:5}}>📅 Consultation Book Karo</div>
      <div style={{fontSize:14,color:'var(--tx2)',marginBottom:24}}>Dr. Rajesh Kumar ke saath type select karein</div>
      <div className="book-types">
        {types.map(t=>(
          <div key={t.id} className={`book-type${type===t.id?' sel':''}`} onClick={()=>setType(t.id)}>
            <div style={{fontSize:17,fontWeight:800,color:'var(--g1)',marginBottom:3}}>{t.l}</div>
            <div style={{fontFamily:"'Baloo 2',cursive",fontSize:21,fontWeight:900,color:'var(--g4)',marginBottom:3}}>₹{t.p}</div>
            <div style={{fontSize:12,color:'var(--tx3)',marginBottom:3}}>⏱️ {t.t}</div>
            <div style={{fontSize:12.5,color:'var(--tx2)'}}>{t.d}</div>
          </div>
        ))}
      </div>
      <div className="card" style={{padding:22,marginBottom:18}}>
        <div style={{fontSize:15,fontWeight:800,color:'var(--g1)',marginBottom:14}}>💰 Payment Summary</div>
        {[['Consultation Fee',`₹${sel.p}`],['Platform Fee (5%)',`₹${Math.round(sel.p*.05)}`],['GST (18%)',`₹${Math.round(sel.p*.18)}`]].map(([k,v])=>(
          <div key={k} style={{display:'flex',justifyContent:'space-between',padding:'9px 0',borderBottom:'1px solid var(--gp)',fontSize:13.5,color:'var(--tx2)'}}>
            <span>{k}</span><span style={{fontWeight:700,color:'var(--tx)'}}>{v}</span>
          </div>
        ))}
        <div style={{display:'flex',justifyContent:'space-between',padding:'13px 0',fontSize:16,fontWeight:900,color:'var(--g1)'}}>
          <span>TOTAL</span><span style={{fontFamily:"'Baloo 2',cursive",fontSize:22}}>₹{total}</span>
        </div>
        <div style={{fontSize:12,color:'var(--tx3)'}}>🔒 Razorpay secured • 24hr refund policy</div>
      </div>
      <div className="card" style={{padding:20,marginBottom:18}}>
        <div style={{fontSize:14,fontWeight:800,color:'var(--g1)',marginBottom:13}}>💳 Payment Method</div>
        {['UPI (Google Pay, PhonePe, Paytm)','Debit / Credit Card','Net Banking','Kisan Credit Card'].map((m,i)=>(
          <div key={m} style={{display:'flex',alignItems:'center',gap:10,padding:'9px 0',borderBottom:'1px solid var(--gp)',cursor:'pointer'}}>
            <div style={{width:17,height:17,borderRadius:'50%',border:'2px solid var(--g4)',background:i===0?'var(--g4)':'none'}}/>
            <span style={{fontSize:13.5,fontWeight:600,color:'var(--tx)'}}>{m}</span>
          </div>
        ))}
      </div>
      <button className="btn btn-g" style={{width:'100%',padding:'14px',fontSize:16,borderRadius:12}} onClick={()=>{toast(`Payment ₹${total} successful! Consultation booked ✅`);nav('chat');}}>
        🔐 Secure Pay ₹{total} →
      </button>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   CHAT PAGE
════════════════════════════════════════════════════════════════ */
function ChatPage({user,nav}) {
  const [msgs,setMsgs]=useState(MESSAGES_DATA);
  const [txt,setTxt]=useState('');
  const endRef=useRef(null);
  const isEx=user?.type==='expert';
  useEffect(()=>endRef.current?.scrollIntoView({behavior:'smooth'}),[msgs]);
  const [consultId,setConsultId]=useState(null);
  useEffect(()=>{
    // Load most recent consultation for chat
    if(!user) return;
    // Check localStorage first for just-created consultation
    const latestId = localStorage.getItem('bh_latest_consult');
    if(latestId){ setConsultId(latestId); }
    API.get('/api/consultations').then(d=>{
      if(d.consultations&&d.consultations.length>0){
        const c=d.consultations[0];
        if(!latestId) setConsultId(c._id);
        // Load existing messages
        API.get('/api/consultations/'+c._id+'/messages').then(m=>{
          if(m.messages&&m.messages.length>0){
            setMsgs(m.messages.map(msg=>({
              id:msg._id,
              from:msg.senderType,
              text:msg.text,
              senderName:msg.senderName,
              time:new Date(msg.createdAt).toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'})
            })));
          }
        }).catch(()=>{});
      }
    }).catch(()=>{});
  },[user]);

  // Auto-refresh messages every 5 seconds when consultId available
  useEffect(()=>{
    if(!consultId) return;
    const poll=setInterval(()=>{
      API.get('/api/consultations/'+consultId+'/messages').then(m=>{
        if(m.messages&&m.messages.length>0){
          setMsgs(prev=>{
            if(prev.length===m.messages.length) return prev;
            return m.messages.map(msg=>({
              id:msg._id, from:msg.senderType, text:msg.text,
              senderName:msg.senderName,
              time:new Date(msg.createdAt).toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit'})
            }));
          });
        }
      }).catch(()=>{});
    },5000);
    return ()=>clearInterval(poll);
  },[consultId]);

  const send=async()=>{
    if(!txt.trim()) return;
    const msgTxt=txt.trim(); setTxt('');
    const nm={id:Date.now(),from:isEx?'expert':'farmer',text:msgTxt,time:new Date().toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'})};
    setMsgs(p=>[...p,nm]);
    if(consultId){
      try{ await API.post('/api/consultations/'+consultId+'/messages',{text:msgTxt}); }
      catch(e){ console.warn('Message send warn:',e.message); }
    }
  };
  return (
    <div className="chat-wrap">
      <div className="chat-hd">
        <button className="btn btn-ghost btn-sm" onClick={()=>nav(isEx?'expert-dashboard':'farmer-dashboard')}>←</button>
        <div style={{width:40,height:40,borderRadius:'50%',background:'linear-gradient(135deg,var(--g5),var(--g6))',display:'flex',alignItems:'center',justifyContent:'center',fontSize:19}}>👨‍🔬</div>
        <div>
          <div style={{fontSize:15,fontWeight:800,color:'var(--tx)'}}>Dr. Rajesh Kumar</div>
          <div style={{fontSize:12,color:'var(--g4)',fontWeight:600}}>🟢 Online • Plant Pathologist</div>
        </div>
        <div style={{marginLeft:'auto',display:'flex',gap:8}}>
          <button className="btn btn-ghost btn-sm">📞</button>
          <button className="btn btn-ghost btn-sm">📹</button>
          <button className="btn btn-ghost btn-sm">📄 Report</button>
        </div>
      </div>
      <div style={{padding:'8px 22px',background:'var(--gp)',fontSize:13,color:'var(--tx3)',textAlign:'center',fontWeight:600,borderBottom:'1px solid var(--br)'}}>
        🍅 Case #CONS-0001 — Tomato Early Blight • AI Report Attached
      </div>
      <div className="chat-msgs">
        <div style={{textAlign:'center',fontSize:11.5,color:'var(--tx3)',margin:'6px 0'}}>Oct 24, 2026 • 2:15 PM</div>
        {msgs.map(m=>(
          <div key={m.id} className={`chat-msg${(isEx?m.from==='expert':m.from==='farmer')?' mine':' theirs'}`}>
            <div className="msg-bbl">{m.text}</div>
            <div className="msg-time">{m.time}{(isEx?m.from==='expert':m.from==='farmer')&&' ✓✓'}</div>
          </div>
        ))}
        <div ref={endRef}/>
      </div>
      <div className="chat-input-bar">
        <button style={{fontSize:20,background:'none',border:'none',padding:'0 3px',cursor:'pointer'}}>📷</button>
        <button style={{fontSize:20,background:'none',border:'none',padding:'0 3px',cursor:'pointer'}}>🎤</button>
        <input className="chat-inp" placeholder="Message ya voice note..." value={txt} onChange={e=>setTxt(e.target.value)} onKeyDown={e=>e.key==='Enter'&&send()}/>
        <button className="chat-send" onClick={send}>➤</button>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   CASE DETAIL (Expert)
════════════════════════════════════════════════════════════════ */
function CaseDetailPage({user,nav,toast}) {
  const [cases,setCases]=useState([]);
  const [selCase,setSelCase]=useState(null);
  const [tab,setTab]=useState('overview');
  const [msgs,setMsgs]=useState([]);
  const [msgTxt,setMsgTxt]=useState('');
  const [sending,setSending]=useState(false);
  const [reportTxt,setReportTxt]=useState('');
  const [submitting,setSubmitting]=useState(false);
  const [loadingCases,setLoadingCases]=useState(true);

  const [pollRef,setPollRef]=useState(null);

  const selectCase=(c)=>{
    setSelCase(c);
    setMsgs([]);
    if(pollRef) clearInterval(pollRef);
    const loadMsgs=()=>{
      API.get('/api/consultations/'+c._id+'/messages').then(m=>{
        if(m.messages) setMsgs(m.messages);
      }).catch(()=>{});
    };
    loadMsgs();
    const iv=setInterval(loadMsgs, 5000);
    setPollRef(iv);
  };

  useEffect(()=>{
    if(!user) return;
    API.get('/api/consultations').then(d=>{
      const list=d.consultations||[];
      setCases(list);
      if(list.length>0) selectCase(list[0]);
      setLoadingCases(false);
    }).catch(()=>setLoadingCases(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  },[user]);

  // Cleanup on unmount
  useEffect(()=>()=>{ if(pollRef) clearInterval(pollRef); },[pollRef]);

  const sendMsg=async()=>{
    if(!msgTxt.trim()||!selCase) return;
    setSending(true);
    try{
      const res=await API.post('/api/consultations/'+selCase._id+'/messages',{text:msgTxt.trim()});
      if(res.message) setMsgs(p=>[...p,res.message]);
      setMsgTxt('');
    }catch(e){if(toast) toast(e.message,'err');}
    setSending(false);
  };

  const submitReport=async()=>{
    if(!reportTxt.trim()||!selCase) return;
    setSubmitting(true);
    try{
      await API.patch('/api/consultations/'+selCase._id+'/status',{status:'completed',report:reportTxt.trim()});
      if(toast) toast('Report submit ho gayi! Farmer ko notify kiya. ✅');
      setSelCase(p=>p?{...p,status:'completed',report:reportTxt.trim()}:p);
      setReportTxt('');
    }catch(e){if(toast) toast(e.message,'err');}
    setSubmitting(false);
  };

  if(!user) return (
    <div className="wrap" style={{textAlign:'center',padding:'80px 20px'}}>
      <div style={{fontSize:60,marginBottom:16}}>🔒</div>
      <div style={{fontFamily:"'Baloo 2',cursive",fontSize:22,fontWeight:900,color:'var(--g1)'}}>Login Karein</div>
      <div style={{fontSize:14,color:'var(--tx3)',marginTop:8}}>Cases dekhne ke liye login zaroor hai</div>
    </div>
  );

  return (
    <div className="wrap">
      <div style={{fontFamily:"'Baloo 2',cursive",fontSize:26,fontWeight:900,color:'var(--g1)',marginBottom:20}}>
        📋 {user?.type==='expert'?'Assigned Cases':'My Consultations'}
      </div>
      {loadingCases&&<div style={{textAlign:'center',padding:40,color:'var(--tx3)'}}>⏳ Loading cases...</div>}
      {!loadingCases&&cases.length===0&&(
        <div style={{textAlign:'center',padding:'60px 20px',color:'var(--tx3)'}}>
          <div style={{fontSize:50,marginBottom:12}}>📭</div>
          <div style={{fontSize:16,fontWeight:700}}>Abhi koi case nahi hai</div>
          {user?.type!=='expert'&&<button className="btn btn-g btn-md" style={{marginTop:16}} onClick={()=>nav('consultation')}>🔬 Nayi Consultation</button>}
        </div>
      )}
      <div style={{display:'grid',gridTemplateColumns:cases.length>0?'300px 1fr':'1fr',gap:20}}>
        {/* Case List */}
        {cases.length>0&&(
          <div style={{display:'flex',flexDirection:'column',gap:10}}>
            {cases.map(c=>(
              <div key={c._id} className={`card${selCase?._id===c._id?' card-hov':' card-hov'}`}
                style={{padding:'14px 16px',cursor:'pointer',borderLeft:selCase?._id===c._id?'3px solid var(--g4)':'3px solid transparent'}}
                onClick={()=>selectCase(c)}>
                <div style={{display:'flex',alignItems:'center',gap:10,marginBottom:6}}>
                  <span style={{fontSize:24}}>{c.cropEmoji||'🌱'}</span>
                  <div>
                    <div style={{fontSize:14,fontWeight:800,color:'var(--tx)'}}>{c.cropName}</div>
                    <div style={{fontSize:11,color:'var(--tx3)'}}>{new Date(c.createdAt).toLocaleDateString('en-IN')}</div>
                  </div>
                </div>
                <div style={{fontSize:12,color:'var(--tx2)',marginBottom:4}}>{c.disease}</div>
                <span className={`badge ${c.status==='completed'?'bg-g':c.status==='expert_assigned'?'bg-b':'bg-a'}`} style={{fontSize:10}}>
                  {c.status==='completed'?'✅ Completed':c.status==='expert_assigned'?'🔵 Expert Assigned':'🟠 Pending'}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Case Detail */}
        {selCase&&(
          <div className="card" style={{padding:24}}>
            <div style={{display:'flex',alignItems:'center',gap:12,marginBottom:20,paddingBottom:16,borderBottom:'1.5px solid var(--br)'}}>
              <span style={{fontSize:36}}>{selCase.cropEmoji||'🌱'}</span>
              <div>
                <div style={{fontFamily:"'Baloo 2',cursive",fontSize:20,fontWeight:900,color:'var(--g1)'}}>{selCase.cropName} — {selCase.disease}</div>
                <div style={{fontSize:13,color:'var(--tx3)'}}>
                  Expert: {selCase.expertName} · {new Date(selCase.createdAt).toLocaleDateString('en-IN')}
                </div>
              </div>
            </div>

            <div style={{display:'flex',gap:8,marginBottom:20}}>
              {['overview','chat','report'].map(t=>(
                <button key={t} onClick={()=>setTab(t)} className="btn btn-sm"
                  style={{background:tab===t?'var(--g4)':'var(--gp)',color:tab===t?'white':'var(--g2)',border:'none'}}>
                  {t==='overview'?'📊 Overview':t==='chat'?'💬 Chat':'📄 Report'}
                </button>
              ))}
            </div>

            {tab==='overview'&&(
              <div>
                <div style={{display:'grid',gridTemplateColumns:'1fr 1fr 1fr',gap:12,marginBottom:16}}>
                  {[['AI Confidence',selCase.confidence+'%','var(--b3)'],
                    ['Severity','Stage '+selCase.severity+'/5',selCase.severity>=3?'var(--r2)':selCase.severity===2?'var(--a2)':'var(--g4)'],
                    ['Status',selCase.status,selCase.status==='completed'?'var(--g4)':'var(--a2)']
                  ].map(([l,v,c])=>(
                    <div key={l} style={{background:'var(--gb)',borderRadius:10,padding:'12px 14px',textAlign:'center'}}>
                      <div style={{fontFamily:"'Baloo 2',cursive",fontSize:20,fontWeight:900,color:c}}>{v}</div>
                      <div style={{fontSize:11,color:'var(--tx3)',marginTop:3}}>{l}</div>
                    </div>
                  ))}
                </div>
                <div style={{background:'var(--gp)',borderRadius:10,padding:'14px 16px',marginBottom:12}}>
                  <div style={{fontSize:13,fontWeight:700,color:'var(--g1)',marginBottom:6}}>📋 Answers from Farmer:</div>
                  {Object.entries(selCase.answers||{}).map(([k,v])=>(
                    <div key={k} style={{fontSize:12,color:'var(--tx2)',padding:'3px 0'}}>
                      <strong>{k}:</strong> {v?.label||String(v)}
                    </div>
                  ))}
                </div>
                {selCase.report&&(
                  <div style={{background:'var(--tp)',borderRadius:10,padding:'14px 16px',border:'1.5px solid var(--t3)'}}>
                    <div style={{fontSize:13,fontWeight:700,color:'var(--t1)',marginBottom:6}}>✅ Expert Report:</div>
                    <div style={{fontSize:13,color:'var(--tx)',lineHeight:1.6}}>{selCase.report}</div>
                  </div>
                )}
              </div>
            )}

            {tab==='chat'&&(
              <div>
                <div style={{height:300,overflowY:'auto',marginBottom:14,padding:'8px 0'}}>
                  {msgs.length===0&&<div style={{textAlign:'center',color:'var(--tx4)',padding:30,fontSize:13}}>
                    Abhi koi message nahi. Pehla message bhejo!
                  </div>}
                  {msgs.map((m,i)=>(
                    <div key={m._id||i} style={{
                      display:'flex',justifyContent:m.senderType===user?.type?'flex-end':'flex-start',
                      marginBottom:10
                    }}>
                      <div style={{
                        maxWidth:'75%',background:m.senderType===user?.type?'var(--g4)':'white',
                        color:m.senderType===user?.type?'white':'var(--tx)',
                        borderRadius:12,padding:'10px 14px',
                        border:m.senderType===user?.type?'none':'1.5px solid var(--br)',
                        fontSize:13,lineHeight:1.5,
                      }}>
                        <div style={{fontSize:10,opacity:.7,marginBottom:4,fontWeight:600}}>
                          {m.senderName} · {new Date(m.createdAt).toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit'})}
                        </div>
                        {m.text}
                      </div>
                    </div>
                  ))}
                </div>
                <div style={{display:'flex',gap:10}}>
                  <input className="finp" style={{flex:1}} value={msgTxt} onChange={e=>setMsgTxt(e.target.value)}
                    placeholder="Message type karo..." onKeyDown={e=>e.key==='Enter'&&sendMsg()}/>
                  <button className="btn btn-g btn-md" onClick={sendMsg} disabled={sending||!msgTxt.trim()}>
                    {sending?'..':'Bhejo →'}
                  </button>
                </div>
              </div>
            )}

            {tab==='report'&&(
              <div>
                {selCase.report?(
                  <div style={{background:'var(--gp)',borderRadius:12,padding:'16px 18px'}}>
                    <div style={{fontWeight:800,color:'var(--g1)',marginBottom:8}}>✅ Submitted Report:</div>
                    <div style={{fontSize:13,color:'var(--tx)',lineHeight:1.6,whiteSpace:'pre-wrap'}}>{selCase.report}</div>
                  </div>
                ):(
                  user?.type==='expert'?(
                    <div>
                      <div style={{fontSize:14,fontWeight:700,color:'var(--tx)',marginBottom:8}}>
                        📝 Farmer ke liye report likhein:
                      </div>
                      <textarea className="ftxt" rows={6} value={reportTxt}
                        onChange={e=>setReportTxt(e.target.value)}
                        placeholder="Disease analysis, treatment plan, medicines, follow-up instructions..."/>
                      <button className="btn btn-g btn-full" style={{marginTop:12}} onClick={submitReport} disabled={submitting||!reportTxt.trim()}>
                        {submitting?'Submitting...':'✅ Report Submit Karo'}
                      </button>
                    </div>
                  ):(
                    <div style={{textAlign:'center',padding:'40px 20px',color:'var(--tx3)'}}>
                      <div style={{fontSize:40,marginBottom:12}}>⏳</div>
                      Expert abhi report likh raha hai. Thodi der mein check karein.
                    </div>
                  )
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
function MyFarmPage({user,nav,toast}) {
  const [farmConsults,setFarmConsults]=useState([]);
  const [loadingFarm,setLoadingFarm]=useState(true);

  useEffect(()=>{
    if(!user) return;
    API.get('/api/consultations').then(d=>{
      setFarmConsults(d.consultations||[]);
      setLoadingFarm(false);
    }).catch(()=>setLoadingFarm(false));
  },[user]);

  // Derive farm health from consultations
  const avgConf = farmConsults.length > 0
    ? Math.round(farmConsults.reduce((s,c)=>s+(100-c.confidence*0.3),0)/farmConsults.length)
    : 78;
  const farmHealth = Math.min(99, Math.max(40, avgConf));
  const months=['Jul','Aug','Sep','Oct','Nov'];
  const health=[98,94,85,78,72];
  return (
    <div className="wrap">
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',marginBottom:26}}>
        <div>
          <div style={{fontFamily:"'Baloo 2',cursive",fontSize:28,fontWeight:900,color:'var(--g1)'}}>🗺️ My Farm — Digital Twin</div>
          <div style={{fontSize:14,color:'var(--tx2)',marginTop:5}}>📍 {user?.district||'Pune'}, Maharashtra • 4.5 Acres</div>
        </div>
        <button className="btn btn-g btn-sm">+ Add Field</button>
      </div>
      <div className="farm-map">
        <div style={{fontSize:56,opacity:.25}}>🗺️</div>
        <div className="farm-mark" style={{left:80,top:80}}>Field 1 — 🍅 Tomato (2 ac)</div>
        <div className="farm-mark" style={{right:80,top:100}}>Field 2 — 🌾 Wheat (1.5 ac)</div>
        <div className="farm-mark" style={{left:'38%',bottom:55}}>Field 3 — 🥔 Potato (1 ac)</div>
      </div>
      <div className="farm-stats">
        {[['4.5 ac','Total Area'],['78/100','Health Score'],['3','Active Crops'],['₹13,524','Season Cost']].map(([n,l])=>(
          <div key={l} className="fs-item"><div className="fs-n">{n}</div><div className="fs-l">{l}</div></div>
        ))}
      </div>
      <div className="dash-2">
        <div className="card" style={{padding:20}}>
          <div style={{fontSize:15,fontWeight:800,color:'var(--g1)'}}>📊 Tomato Health Timeline</div>
          <div style={{fontSize:12,color:'var(--tx3)',marginTop:2,marginBottom:10}}>Last 5 months</div>
          <div className="tl-bars">
            {health.map((h,i)=>(
              <div key={i} className="tl-bw">
                <div style={{fontSize:11,color:h>85?'var(--g4)':h>70?'var(--a2)':'var(--r2)',fontWeight:700,marginBottom:3}}>{h}%</div>
                <div className="tl-bar" style={{height:`${h*.72}px`,background:h>85?'var(--g5)':h>70?'var(--a2)':'var(--r3)',opacity:.85}}/>
                <div className="tl-bl">{months[i]}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="card" style={{padding:20}}>
          <div style={{fontSize:15,fontWeight:800,color:'var(--g1)',marginBottom:14}}>📋 Active Tasks</div>
          {[['✅','Oct 24: Mancozeb spray done'],['⏳','Oct 31: Follow-up photo bhejo'],['📅','Nov 5: Second consultation'],['💊','Nov 7: Fertilizer application']].map(([i,t])=>(
            <div key={t} style={{display:'flex',gap:11,padding:'9px 0',borderBottom:'1px solid var(--gp)',alignItems:'flex-start'}}>
              <span style={{fontSize:18,flexShrink:0}}>{i}</span>
              <span style={{fontSize:13.5,color:'var(--tx)',fontWeight:600}}>{t}</span>
            </div>
          ))}
          <button className="btn btn-ghost btn-sm" style={{width:'100%',marginTop:12}}>+ Task Add Karo</button>
        </div>
      </div>
      <div className="card" style={{padding:22,marginTop:20}}>
        <div style={{fontSize:15,fontWeight:800,color:'var(--g1)',marginBottom:14}}>💰 Season Cost Tracker</div>
        <div style={{display:'grid',gridTemplateColumns:'repeat(5,1fr)',gap:11}}>
          {[['Seeds','₹4,500'],['Fertilizer','₹6,200'],['Medicine','₹1,840'],['Consultations','₹984'],['Labour','₹8,000']].map(([k,v])=>(
            <div key={k} style={{textAlign:'center',padding:13,background:'var(--gp)',borderRadius:10}}>
              <div style={{fontSize:15,fontWeight:800,color:'var(--g3)'}}>{v}</div>
              <div style={{fontSize:12,color:'var(--tx3)',marginTop:2}}>{k}</div>
            </div>
          ))}
        </div>
        <div style={{marginTop:14,padding:'13px 16px',background:'linear-gradient(135deg,var(--g3),var(--g1))',borderRadius:10,color:'white',display:'flex',justifyContent:'space-between',alignItems:'center'}}>
          <span style={{fontWeight:700}}>Expected Revenue:</span>
          <span style={{fontFamily:"'Baloo 2',cursive",fontSize:20,fontWeight:900}}>₹85,000</span>
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   NOTIFICATIONS PAGE
════════════════════════════════════════════════════════════════ */
function NotifPage({nav,user}) {
  const [notifs,setNotifs]=useState(NOTIFICATIONS);
  const [loading,setLoading]=useState(false);
  useEffect(()=>{
    if(!user) return;
    setLoading(true);
    API.get('/api/notifications')
      .then(d=>{
        if(d.notifications&&d.notifications.length>0){
          setNotifs(d.notifications.map(n=>({
            id:n._id, type:n.type||'info', icon:n.icon||'🔔',
            col:n.type==='welcome'?'#eaf7ef':n.type==='consultation'?'#e3f2fd':n.type==='message'?'#fff8e1':n.type==='report_ready'?'#eaf7ef':'#f3e5f5',
            title:n.title, desc:n.body, time:new Date(n.createdAt).toLocaleDateString('en-IN',{day:'2-digit',month:'short'}),
            unread:!n.read,
          })));
        }
        setLoading(false);
      })
      .catch(()=>setLoading(false));
  },[user]);
  return (
    <div className="wrap-sm">
      <div style={{fontFamily:"'Baloo 2',cursive",fontSize:26,fontWeight:900,color:'var(--g1)',marginBottom:5}}>🔔 Notifications</div>
      <div style={{fontSize:14,color:'var(--tx2)',marginBottom:22}}>Disease alerts, weather warnings, expert replies</div>
      {loading&&<div style={{textAlign:'center',padding:20,fontSize:13,color:'var(--tx3)'}}>⏳ Loading...</div>}
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:16}}>
        <div style={{fontSize:13,color:'var(--tx3)'}}>{notifs.filter(n=>n.unread).length} unread</div>
        <button className="btn btn-ghost btn-sm" onClick={async()=>{
          setNotifs(p=>p.map(n=>({...n,unread:false})));
          try{await API.patch('/api/notifications/read-all');}catch(e){console.warn(e);}
        }}>✓ Sab Read Karo</button>
      </div>
      {notifs.map(n=>(
        <div key={n.id} className={`notif-item${n.unread?' unread':''}`}>
          <div className="notif-icon" style={{background:n.col}}>{n.icon}</div>
          <div style={{flex:1}}>
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
              <div className="notif-t">{n.title}</div>
              {n.unread&&<div style={{width:7,height:7,background:'var(--g4)',borderRadius:'50%',flexShrink:0}}/>}
            </div>
            <div className="notif-d">{n.desc}</div>
            <div className="notif-time">{n.time}</div>
          </div>
        </div>
      ))}
      {notifs.length===0&&<div style={{textAlign:'center',padding:'40px 20px',color:'var(--tx4)',fontSize:14}}>
        🔔 Koi notification nahi hai
      </div>}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   SUPPORT PAGE
════════════════════════════════════════════════════════════════ */
function SupportPage({toast}) {
  const [ticket,setTicket]=useState({name:'',mobile:'',issue:'',desc:''});
  const [submitting,setSubmitting]=useState(false);
  const [submitted,setSubmitted]=useState(false);

  const submitTicket=async()=>{
    if(!ticket.issue.trim()||!ticket.desc.trim()){
      toast('Topic aur message zaroor bharo','err'); return;
    }
    setSubmitting(true);
    try{
      const res = await API.post('/api/support',{
        name: ticket.name,
        mobile: ticket.mobile,
        issue: ticket.issue,
        desc: ticket.desc,
      });
      setSubmitted(true);
      toast(`Support ticket #${res.ticketId?.slice(-6).toUpperCase()||'XXXXXX'} submit ho gaya! 24 ghante mein reply milegi. ✅`);
    }catch(e){ toast(e.message||'Submit fail hua','err'); }
    setSubmitting(false);
  };
  const [openFaq,setOpenFaq]=useState(null);
  const faqs=[
    {q:'Crop photos kaise upload karein?',a:'Crop Consultation page par jao, "Photo Upload" select karo, aur photo lo ya gallery se choose karo.'},
    {q:'AI report kab milegi?',a:'Photo upload ke baad 30–60 seconds mein AI analysis complete hoti hai.'},
    {q:'Expert se kaise connect karein?',a:'Experts page par specialist dhundho, "Select Expert" click karo aur consultation book karo.'},
    {q:'Payment refund policy kya hai?',a:'Consultation ke 24 ghante ke andar refund request ki ja sakti hai. support@beejhealth.com contact karein.'},
    {q:'App offline kaam karta hai kya?',a:'Basic features aur last AI report offline accessible hain. Full functionality ke liye internet chahiye.'},
  ];
  return (
    <div className="wrap-md">
      <div style={{fontFamily:"'Baloo 2',cursive",fontSize:26,fontWeight:900,color:'var(--g1)',marginBottom:5}}>🆘 Support Center</div>
      <div style={{fontSize:14,color:'var(--tx2)',marginBottom:28}}>Kisi bhi sawaal ya issue ke liye — hum yahan hain</div>
      <div style={{display:'grid',gridTemplateColumns:'1fr 300px',gap:22}}>
        <div className="card" style={{padding:26}}>
          <div style={{fontSize:16,fontWeight:800,color:'var(--g1)',marginBottom:20}}>📝 Message Bhejein</div>
          <div className="frow">
            <div className="fgrp"><label className="flbl">Naam</label><input className="finp" placeholder="Aapka naam" value={ticket.name} onChange={e=>setTicket(p=>({...p,name:e.target.value}))}/></div>
            <div className="fgrp"><label className="flbl">Mobile</label><input className="finp" placeholder="Mobile number" value={ticket.mobile} onChange={e=>setTicket(p=>({...p,mobile:e.target.value}))}/></div>
          </div>
          <div className="fgrp">
            <label className="flbl">Topic</label>
            <select className="fsel" value={ticket.issue} onChange={e=>setTicket(p=>({...p,issue:e.target.value}))}>
              <option value="">Select topic</option>
              <option>Technical Issue</option><option>Payment Problem</option><option>Expert Related</option><option>AI Report Query</option><option>Other</option>
            </select>
          </div>
          <div className="fgrp">
            <label className="flbl">Message</label>
            <textarea className="ftxt" rows={4} placeholder="Apna issue ya sawaal describe karein..." value={ticket.desc} onChange={e=>setTicket(p=>({...p,desc:e.target.value}))}/>
          </div>
          <button className="btn btn-g btn-full" onClick={submitTicket} disabled={submitting}>
            {submitting?<><div className="spin"/>Bhej raha hoon...</>:'📨 Submit Request'}
          </button>
          {submitted&&<div style={{marginTop:14,padding:'13px 16px',background:'var(--gp)',borderRadius:10,fontSize:13,color:'var(--g2)',fontWeight:700,textAlign:'center'}}>✅ Ticket #{Math.floor(Math.random()*90000+10000)} submit ho gaya! 24 ghante mein reply milegi.</div>}
        </div>
        <div>
          <div className="card" style={{padding:20,marginBottom:18}}>
            <div style={{fontSize:14,fontWeight:800,color:'var(--g1)',marginBottom:14}}>📞 Contact Info</div>
            {[['📧','Email','support@beejhealth.com'],['📞','Call','+91 123 456 7890'],['💬','WhatsApp','+91 98765 43210'],['⏰','Hours','Mon–Sat: 9AM–6PM']].map(([i,l,v])=>(
              <div key={l} style={{display:'flex',gap:11,padding:'11px 0',borderBottom:'1px solid var(--gp)'}}>
                <div style={{width:34,height:34,background:'var(--gp)',borderRadius:9,display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0,fontSize:15}}>{i}</div>
                <div><div style={{fontSize:11,fontWeight:700,color:'var(--tx3)',textTransform:'uppercase',letterSpacing:.5}}>{l}</div><div style={{fontSize:13.5,fontWeight:600,color:'var(--tx)'}}>{v}</div></div>
              </div>
            ))}
          </div>
          <div className="card" style={{padding:20}}>
            <div style={{fontSize:14,fontWeight:800,color:'var(--g1)',marginBottom:13}}>❓ FAQs</div>
            {faqs.map((f,i)=>(
              <div key={i} style={{borderBottom:'1px solid var(--gp)',cursor:'pointer'}} onClick={()=>setOpenFaq(openFaq===i?null:i)}>
                <div style={{padding:'11px 0',display:'flex',justifyContent:'space-between',fontSize:13,fontWeight:700,color:'var(--tx)'}}>
                  <span>{f.q}</span><span style={{fontSize:11,marginLeft:8}}>{openFaq===i?'▲':'▼'}</span>
                </div>
                {openFaq===i&&<div style={{fontSize:13,color:'var(--tx2)',paddingBottom:11,lineHeight:1.65}}>{f.a}</div>}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   PROFILE PAGE
════════════════════════════════════════════════════════════════ */
function ProfilePage({user,nav,toast,setUser}) {
  const isEx = user?.type==='expert';
  const [editing,setEditing] = useState(false);
  const [nm,setNm]   = useState(user?.name||'');
  const [em,setEm]   = useState(user?.email||'');
  const [dt,setDt]   = useState(user?.district||'');
  const [vl,setVl]   = useState(user?.village||'');
  const [bio,setBio] = useState(user?.bio||'');
  const [saving,setSaving] = useState(false);

  const saveProfile = async () => {
    if(!nm.trim()){ toast('Naam required hai','err'); return; }
    setSaving(true);
    try {
      const res = await API.patch('/api/auth/profile',{name:nm.trim(),email:em.trim(),district:dt,village:vl,bio});
      if(res.user){
        saveSession(localStorage.getItem('bh_token'), res.user);
        if(setUser) setUser(res.user);
      }
      setEditing(false);
      toast('Profile update ho gayi! ✅');
    } catch(e){ toast(e.message,'err'); }
    setSaving(false);
  };

  const cancelEdit = () => {
    setNm(user?.name||''); setEm(user?.email||'');
    setDt(user?.district||''); setVl(user?.village||'');
    setBio(user?.bio||''); setEditing(false);
  };

  return (
    <div className="wrap-md">
      {/* Header */}
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:24}}>
        <div style={{fontFamily:"'Baloo 2',cursive",fontSize:26,fontWeight:900,color:'var(--g1)'}}>👤 My Profile</div>
        {!editing&&(
          <button className="btn btn-g btn-md" onClick={()=>setEditing(true)}>✏️ Edit Profile</button>
        )}
      </div>

      {/* Avatar Card */}
      <div className="card" style={{padding:24,marginBottom:20}}>
        <div style={{display:'flex',gap:20,alignItems:'center',marginBottom:20}}>
          <div style={{width:72,height:72,borderRadius:'50%',background:isEx?'linear-gradient(135deg,var(--b3),var(--b4))':'linear-gradient(135deg,var(--g4),var(--g5))',display:'flex',alignItems:'center',justifyContent:'center',color:'white',fontSize:26,fontWeight:900,flexShrink:0,boxShadow:'0 4px 16px rgba(0,0,0,.15)'}}>
            {(nm||user?.name||'U').split(' ').map(w=>w[0]).join('').slice(0,2).toUpperCase()}
          </div>
          <div>
            <div style={{fontFamily:"'Baloo 2',cursive",fontSize:22,fontWeight:900,color:'var(--tx)'}}>{user?.name}</div>
            <div style={{fontSize:13,color:isEx?'var(--b3)':'var(--g4)',fontWeight:700,marginTop:3}}>{isEx?'👨‍⚕️ Agricultural Expert':'🌾 Farmer'}</div>
            <div style={{display:'flex',gap:8,marginTop:8,flexWrap:'wrap'}}>
              <span className="badge bg-g">✅ Verified</span>
              {user?.district&&<span className="badge" style={{background:'var(--gp)',color:'var(--g2)'}}>📍 {user?.district}</span>}
              {isEx&&user?.spec&&<span className="badge bg-b">{user?.spec}</span>}
            </div>
          </div>
        </div>

        {/* View Mode */}
        {!editing&&(
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:12}}>
            {[
              ['📱 Mobile',user?.mobile||'—'],
              ['📧 Email',user?.email||'Not set'],
              ['📍 District',user?.district||'Not set'],
              ['🏘️ Village',user?.village||'Not set'],
              ...(isEx?[['🎓 Specialization',user?.spec||'—'],['💰 Fee','₹'+(user?.fee||0)+' / consultation']]:
                       [['🌾 Crops',(user?.crops||[]).join(', ')||'Not set'],['🗣️ Languages',user?.langs||'Hindi']]),
            ].map(([k,v])=>(
              <div key={k} style={{background:'var(--gb)',borderRadius:10,padding:'12px 14px'}}>
                <div style={{fontSize:11,color:'var(--tx3)',fontWeight:600,marginBottom:3}}>{k}</div>
                <div style={{fontSize:13,color:'var(--tx)',fontWeight:700}}>{v}</div>
              </div>
            ))}
          </div>
        )}

        {/* Edit Mode */}
        {editing&&(
          <div>
            <div style={{fontSize:14,fontWeight:800,color:'var(--g1)',marginBottom:16}}>✏️ Profile Edit Karo</div>
            <div className="frow">
              <div className="fgrp">
                <label className="flbl">Naam *</label>
                <input className="finp" value={nm} onChange={e=>setNm(e.target.value)} placeholder="Poora naam"/>
              </div>
              <div className="fgrp">
                <label className="flbl">Email</label>
                <input className="finp" type="email" value={em} onChange={e=>setEm(e.target.value)} placeholder="email@example.com"/>
              </div>
            </div>
            <div className="frow">
              <div className="fgrp">
                <label className="flbl">District</label>
                <input className="finp" value={dt} onChange={e=>setDt(e.target.value)} placeholder="Aapka district"/>
              </div>
              <div className="fgrp">
                <label className="flbl">Village / Gaon</label>
                <input className="finp" value={vl} onChange={e=>setVl(e.target.value)} placeholder="Gaon ka naam"/>
              </div>
            </div>
            <div className="fgrp">
              <label className="flbl">Bio / About</label>
              <textarea className="ftxt" rows={3} value={bio} onChange={e=>setBio(e.target.value)} placeholder="Apne baare mein kuch likhein..."/>
            </div>
            <div style={{display:'flex',gap:10,marginTop:4}}>
              <button className="btn btn-out btn-md" style={{flex:1}} onClick={cancelEdit} disabled={saving}>✕ Cancel</button>
              <button className="btn btn-g btn-md" style={{flex:2}} onClick={saveProfile} disabled={saving}>
                {saving?<><div className="spin"/>Saving...</>:'💾 Save Changes'}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Stats */}
      <div className="card" style={{padding:20,marginBottom:20}}>
        <div style={{fontSize:14,fontWeight:800,color:'var(--g1)',marginBottom:14}}>📊 Activity Stats</div>
        <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:12}}>
          {(isEx?[
            ['Total Cases',user?.totalCases||0,'var(--b3)'],
            ['Rating',user?.rating?user?.rating.toFixed(1)+'⭐':'N/A','var(--a2)'],
            ['Status',user?.available?'🟢 Online':'🔴 Offline',user?.available?'var(--g4)':'var(--r2)'],
          ]:[
            ['Consultations','—','var(--g4)'],
            ['Crops',user?.crops?.length||0,'var(--a2)'],
            ['Member Since',user?.createdAt?new Date(user.createdAt).getFullYear():'—','var(--b3)'],
          ]).map(([l,v,c])=>(
            <div key={l} style={{background:'var(--gb)',borderRadius:10,padding:'14px 12px',textAlign:'center'}}>
              <div style={{fontFamily:"'Baloo 2',cursive",fontSize:22,fontWeight:900,color:c}}>{v}</div>
              <div style={{fontSize:11,color:'var(--tx3)',marginTop:3,fontWeight:600}}>{l}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card" style={{padding:20}}>
        <div style={{fontSize:14,fontWeight:800,color:'var(--g1)',marginBottom:14}}>⚡ Quick Actions</div>
        <div style={{display:'flex',gap:10,flexWrap:'wrap'}}>
          <button className="btn btn-ghost btn-md" onClick={()=>nav('settings')}>⚙️ Settings</button>
          <button className="btn btn-ghost btn-md" onClick={()=>nav('notifications')}>🔔 Notifications</button>
          {isEx&&<button className="btn btn-ghost btn-md" onClick={()=>nav('earnings')}>💰 Earnings</button>}
          {!isEx&&<button className="btn btn-ghost btn-md" onClick={()=>nav('my-farm')}>🌾 My Farm</button>}
          <button className="btn btn-red btn-md" onClick={()=>{
            clearSession();
            ['bh_latest_consult','bh_latest_crop','bh_view_consult','bh_sel_expert'].forEach(k=>localStorage.removeItem(k));
            if(setUser) setUser(null); nav('home');
            toast('Logout successful 👋','inf');
          }}>🚪 Logout</button>
        </div>
      </div>
    </div>
  );
}
function EarningsPage({user,nav,toast}) {
  const [stats,setStats]=useState({total:0,pending:0,completed:0,thisMonth:0,cases:[]});
  const [loading,setLoading]=useState(true);

  useEffect(()=>{
    if(!user||user?.type!=='expert') return;
    API.get('/api/earnings').then(d=>{
      setStats({
        total:      d.total||0,
        pending:    d.pending||0,
        completed:  d.completed||0,
        thisMonth:  d.thisMonth||0,
        totalCases: d.totalCases||0,
        cases:      d.recentCases||[],
        feePerCase: d.feePerCase||500,
      });
      setLoading(false);
    }).catch(()=>{
      // Fallback to consultations API
      API.get('/api/consultations').then(d=>{
        const cases=d.consultations||[];
        const completed=cases.filter(c=>c.status==='completed');
        const now=new Date();
        const thisMonthCases=completed.filter(c=>{
          const dd=new Date(c.createdAt);
          return dd.getMonth()===now.getMonth()&&dd.getFullYear()===now.getFullYear();
        });
        const fee=user?.fee||500;
        setStats({total:completed.length*fee,pending:cases.filter(c=>c.status!=='completed').length*fee,
          completed:completed.length,thisMonth:thisMonthCases.length*fee,
          totalCases:cases.length,cases:cases.slice(0,10),feePerCase:fee});
        setLoading(false);
      }).catch(()=>setLoading(false));
    });
  },[user]);
  return (
    <div className="wrap-sm">
      <button className="btn btn-ghost btn-sm" style={{marginBottom:18}} onClick={()=>nav('expert-dashboard')}>← Back</button>
      <div style={{fontFamily:"'Baloo 2',cursive",fontSize:26,fontWeight:900,color:'var(--b1)',marginBottom:5}}>💰 Earnings & Payouts</div>
      <div style={{fontSize:14,color:'var(--tx2)',marginBottom:24}}>Aapki kamaai ka complete breakdown</div>

      {/* Summary Cards */}
      <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:12,marginBottom:22}}>
        {[{l:'Is Mahine',v:`₹${(stats.thisMonth||0).toLocaleString('en-IN')}`,s:`${stats.completed||0} completed cases`,c:'var(--g3)'},{l:'Pending',v:`₹${(stats.pending||0).toLocaleString('en-IN')}`,s:`${(stats.totalCases||0)-(stats.completed||0)} cases pending`,c:'var(--a2)'},{l:'Total Earned',v:`₹${(stats.total||0).toLocaleString('en-IN')}`,s:`₹${stats.feePerCase||500}/case`,c:'var(--b3)'}].map(({l,v,s,c})=>(
          <div key={l} style={{padding:18,borderRadius:'var(--rad)',background:'white',border:'1.5px solid var(--br)',textAlign:'center',boxShadow:'var(--sh)'}}>
            <div style={{fontSize:11,fontWeight:700,color:'var(--tx3)',textTransform:'uppercase',letterSpacing:'.6px',marginBottom:7}}>{l}</div>
            <div style={{fontFamily:"'Baloo 2',cursive",fontSize:24,fontWeight:900,color:c}}>{v}</div>
            <div style={{fontSize:11.5,color:'var(--tx3)',marginTop:4}}>{s}</div>
          </div>
        ))}
      </div>

      {/* Payment Breakdown */}
      <div className="card" style={{padding:22,marginBottom:18}}>
        <div style={{fontSize:15,fontWeight:800,color:'var(--b1)',marginBottom:14}}>📊 Is Mahine Ka Breakdown</div>
        {[['Gross Earnings',`₹${(stats.total||0).toLocaleString('en-IN')}`,'var(--tx)'],['Platform Fee (15%)',`-₹${Math.round((stats.total||0)*0.15).toLocaleString('en-IN')}`,'var(--r2)'],['TDS (10%)',`-₹${Math.round((stats.total||0)*0.10).toLocaleString('en-IN')}`,'var(--r2)']].map(([l,v,c])=>(
          <div key={l} style={{display:'flex',justifyContent:'space-between',padding:'10px 0',borderBottom:'1px solid var(--gp)'}}>
            <span style={{fontSize:13.5,color:'var(--tx2)',fontWeight:600}}>{l}</span>
            <span style={{fontSize:14.5,fontWeight:800,color:c}}>{v}</span>
          </div>
        ))}
        <div style={{display:'flex',justifyContent:'space-between',padding:'14px 0',fontSize:16,fontWeight:900}}>
          <span style={{color:'var(--g2)'}}>✅ Net Payout</span>
          <span style={{fontFamily:"'Baloo 2',cursive",fontSize:22,color:'var(--g3)'}}>₹{Math.round((stats.total||0)*0.75).toLocaleString('en-IN')}</span>
        </div>
        <div style={{padding:'9px 14px',background:'var(--ap)',borderRadius:8,fontSize:12.5,color:'var(--a1)',fontWeight:600}}>
          ⏳ Pending: ₹6,400 (3 cases settled — processing)
        </div>
      </div>

      {/* Consultation Type Breakdown */}
      <div className="card" style={{padding:22,marginBottom:18}}>
        <div style={{fontSize:15,fontWeight:800,color:'var(--b1)',marginBottom:14}}>📋 Consultation Breakdown</div>
        {[{t:'💬 Text Reports',n:28,amt:'₹22,400'},{t:'📞 Voice Calls',n:12,amt:'₹9,600'},{t:'📹 Video Calls',n:8,amt:'₹9,600'},{t:'🏠 Field Visits',n:0,amt:'₹0'}].map(({t,n,amt})=>(
          <div key={t} style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'9px 0',borderBottom:'1px solid var(--gp)'}}>
            <div>
              <div style={{fontSize:13.5,fontWeight:700,color:'var(--tx)'}}>{t}</div>
              <div style={{fontSize:11.5,color:'var(--tx3)'}}>{n} cases</div>
            </div>
            <div style={{fontFamily:"'Baloo 2',cursive",fontSize:17,fontWeight:900,color:n>0?'var(--b3)':'var(--tx4)'}}>{amt}</div>
          </div>
        ))}
      </div>

      {/* Withdraw + Invoice */}
      <div style={{display:'flex',gap:11}}>
        <button className="btn btn-b btn-lg" style={{flex:2}} onClick={()=>toast('Bank withdrawal initiated! 2-3 din mein credit hoga','inf')}>
          💸 Bank Mein Withdraw Karo
        </button>
        <button className="btn btn-out-b btn-lg" style={{flex:1}} onClick={()=>toast('GST Invoice download ho raha hai...','inf')}>
          🧾 Invoice
        </button>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   SETTINGS PAGE
════════════════════════════════════════════════════════════════ */
function SettingsPage({user,setUser,nav,toast}) {
  // Profile fields
  const [nm,setNm]   = useState(user?.name||'');
  const [em,setEm]   = useState(user?.email||'');
  const [dt,setDt]   = useState(user?.district||'');
  const [vl,setVl]   = useState(user?.village||'');
  const [sp,setSp]   = useState(user?.spec||'');
  const [fee,setFee] = useState(user?.fee||'');
  const [lang,setLang] = useState(user?.langs||'Hindi');
  const [savingProfile,setSavingProfile] = useState(false);

  // Password fields
  const [oldPwd,setOldPwd] = useState('');
  const [newPwd,setNewPwd] = useState('');
  const [cnfPwd,setCnfPwd] = useState('');
  const [savingPwd,setSavingPwd] = useState(false);

  // Notification prefs
  const [notifs,setNotifs] = useState({disease:true,weather:true,market:true,expert:true});

  const [activeTab,setActiveTab] = useState('profile');
  const isEx = user?.type==='expert';

  const saveProfile = async () => {
    if(!nm.trim()){ toast('Naam required hai','err'); return; }
    setSavingProfile(true);
    try {
      const payload = {name:nm.trim(),email:em.trim(),district:dt,village:vl,langs:lang};
      if(isEx){ payload.spec=sp; payload.fee=Number(fee)||0; }
      const res = await API.patch('/api/auth/profile', payload);
      if(res.user){
        saveSession(localStorage.getItem('bh_token'), res.user);
        if(setUser) setUser(res.user);
      }
      toast('Profile save ho gayi! ✅');
    } catch(e){ toast(e.message,'err'); }
    setSavingProfile(false);
  };

  const savePassword = async () => {
    if(!oldPwd){ toast('Purana password daalein','err'); return; }
    if(newPwd.length<8){ toast('Naya password 8+ characters ka hona chahiye','err'); return; }
    if(newPwd!==cnfPwd){ toast('Passwords match nahi kar rahe','err'); return; }
    setSavingPwd(true);
    try {
      await API.patch('/api/auth/password',{oldPassword:oldPwd,newPassword:newPwd});
      setOldPwd(''); setNewPwd(''); setCnfPwd('');
      toast('Password successfully change ho gaya! ✅');
    } catch(e){ toast(e.message,'err'); }
    setSavingPwd(false);
  };

  const doLogout = () => {
    clearSession();
    localStorage.removeItem('bh_latest_consult');
    localStorage.removeItem('bh_latest_crop');
    localStorage.removeItem('bh_view_consult');
    localStorage.removeItem('bh_sel_expert');
    if(setUser) setUser(null);
    nav('home');
    toast('Aap logout ho gaye 👋','inf');
  };

  const tabs = [{id:'profile',l:'👤 Profile'},{id:'password',l:'🔑 Password'},{id:'notifs',l:'🔔 Notifications'},{id:'account',l:'⚙️ Account'}];

  return (
    <div className="wrap-sm">
      <div style={{fontFamily:"'Baloo 2',cursive",fontSize:26,fontWeight:900,color:'var(--g1)',marginBottom:22}}>⚙️ Settings</div>

      {/* Tab bar */}
      <div style={{display:'flex',gap:6,marginBottom:24,flexWrap:'wrap'}}>
        {tabs.map(t=>(
          <button key={t.id} onClick={()=>setActiveTab(t.id)}
            className="btn btn-sm"
            style={{background:activeTab===t.id?'var(--g4)':'white',color:activeTab===t.id?'white':'var(--tx2)',border:`1.5px solid ${activeTab===t.id?'var(--g4)':'var(--br)'}`,fontWeight:700}}>
            {t.l}
          </button>
        ))}
      </div>

      {/* ── PROFILE TAB ── */}
      {activeTab==='profile'&&(
        <div className="card" style={{padding:24}}>
          <div style={{fontSize:15,fontWeight:800,color:'var(--g1)',marginBottom:18}}>👤 Profile Information</div>

          {/* Avatar */}
          <div style={{display:'flex',alignItems:'center',gap:16,marginBottom:22,padding:'14px 16px',background:'var(--gp)',borderRadius:12}}>
            <div style={{width:56,height:56,borderRadius:'50%',background:isEx?'var(--b3)':'var(--g4)',display:'flex',alignItems:'center',justifyContent:'center',color:'white',fontSize:20,fontWeight:900,flexShrink:0}}>
              {(nm||user?.name||'U').split(' ').map(w=>w[0]).join('').slice(0,2).toUpperCase()}
            </div>
            <div>
              <div style={{fontSize:15,fontWeight:800,color:'var(--tx)'}}>{nm||user?.name}</div>
              <div style={{fontSize:12,color:'var(--tx3)'}}>{isEx?'👨‍⚕️ Agricultural Expert':'🌾 Farmer'} • {user?.mobile}</div>
            </div>
          </div>

          <div className="frow">
            <div className="fgrp">
              <label className="flbl">Naam *</label>
              <input className="finp" value={nm} onChange={e=>setNm(e.target.value)} placeholder="Aapka poora naam"/>
            </div>
            <div className="fgrp">
              <label className="flbl">Email</label>
              <input className="finp" type="email" value={em} onChange={e=>setEm(e.target.value)} placeholder="email@example.com"/>
            </div>
          </div>
          <div className="frow">
            <div className="fgrp">
              <label className="flbl">District</label>
              <input className="finp" value={dt} onChange={e=>setDt(e.target.value)} placeholder="Aapka district"/>
            </div>
            <div className="fgrp">
              <label className="flbl">Village / Gaon</label>
              <input className="finp" value={vl} onChange={e=>setVl(e.target.value)} placeholder="Gaon ka naam"/>
            </div>
          </div>
          {isEx&&(
            <div className="frow">
              <div className="fgrp">
                <label className="flbl">Specialization</label>
                <select className="fsel" value={sp} onChange={e=>setSp(e.target.value)}>
                  <option value="">Select...</option>
                  <option>Plant Pathologist</option>
                  <option>Horticulture Expert</option>
                  <option>Soil Scientist</option>
                  <option>Crop Scientist</option>
                </select>
              </div>
              <div className="fgrp">
                <label className="flbl">Consultation Fee (₹)</label>
                <input className="finp" type="number" value={fee} onChange={e=>setFee(e.target.value)} placeholder="e.g. 500"/>
              </div>
            </div>
          )}
          <div className="fgrp">
            <label className="flbl">Preferred Language</label>
            <select className="fsel" value={lang} onChange={e=>setLang(e.target.value)}>
              {['Hindi','English','Marathi','Punjabi','Gujarati','Tamil','Telugu'].map(l=><option key={l}>{l}</option>)}
            </select>
          </div>
          <button className="btn btn-g btn-full" style={{marginTop:8}} onClick={saveProfile} disabled={savingProfile}>
            {savingProfile?<><div className="spin"/>Saving...</>:'💾 Profile Save Karo'}
          </button>
        </div>
      )}

      {/* ── PASSWORD TAB ── */}
      {activeTab==='password'&&(
        <div className="card" style={{padding:24}}>
          <div style={{fontSize:15,fontWeight:800,color:'var(--g1)',marginBottom:18}}>🔑 Password Change Karo</div>
          <div className="fgrp">
            <label className="flbl">Purana Password *</label>
            <input className="finp" type="password" value={oldPwd} onChange={e=>setOldPwd(e.target.value)} placeholder="Current password"/>
          </div>
          <div className="fgrp">
            <label className="flbl">Naya Password * (8+ characters)</label>
            <input className="finp" type="password" value={newPwd} onChange={e=>setNewPwd(e.target.value)} placeholder="New password"/>
          </div>
          <div className="fgrp">
            <label className="flbl">Naya Password Confirm *</label>
            <input className="finp" type="password" value={cnfPwd} onChange={e=>setCnfPwd(e.target.value)} placeholder="Confirm new password"/>
          </div>
          {newPwd&&cnfPwd&&newPwd!==cnfPwd&&(
            <div className="ferr" style={{marginBottom:12}}>⚠️ Passwords match nahi kar rahe</div>
          )}
          {newPwd&&newPwd.length<8&&(
            <div className="ferr" style={{marginBottom:12}}>⚠️ Password 8+ characters ka hona chahiye</div>
          )}
          <button className="btn btn-g btn-full" onClick={savePassword} disabled={savingPwd||!oldPwd||!newPwd||!cnfPwd}>
            {savingPwd?<><div className="spin"/>Changing...</>:'🔑 Password Change Karo'}
          </button>
        </div>
      )}

      {/* ── NOTIFICATIONS TAB ── */}
      {activeTab==='notifs'&&(
        <div className="card" style={{padding:24}}>
          <div style={{fontSize:15,fontWeight:800,color:'var(--g1)',marginBottom:18}}>🔔 Notification Preferences</div>
          {[
            ['disease','🦠 Disease Alerts','Aapke area mein outbreak alerts'],
            ['weather','🌦️ Weather Warnings','Spray timing aur mausam alerts'],
            ['market','📈 Market Price Updates','Jab bhav mein badlaav aaye'],
            ['expert','💬 Expert Replies','Consultation aur report updates'],
          ].map(([k,l,s])=>(
            <div key={k} className="sett-row" style={{padding:'12px 0',borderBottom:'1px solid var(--gp)'}}>
              <div>
                <div className="sett-lbl" style={{fontSize:14,fontWeight:700}}>{l}</div>
                <div style={{fontSize:12,color:'var(--tx3)',marginTop:2}}>{s}</div>
              </div>
              <label className="sw">
                <input type="checkbox" checked={notifs[k]} onChange={e=>setNotifs(p=>({...p,[k]:e.target.checked}))}/>
                <span className="sw-sl"/>
              </label>
            </div>
          ))}
          <button className="btn btn-g btn-full" style={{marginTop:16}}
            onClick={()=>toast('Notification preferences save ho gayi ✅')}>
            💾 Save Preferences
          </button>
        </div>
      )}

      {/* ── ACCOUNT TAB ── */}
      {activeTab==='account'&&(
        <div style={{display:'flex',flexDirection:'column',gap:14}}>
          <div className="card" style={{padding:20}}>
            <div style={{fontSize:14,fontWeight:800,color:'var(--g1)',marginBottom:12}}>📋 Account Info</div>
            {[
              ['Mobile',user?.mobile||'—'],
              ['Account Type',isEx?'👨‍⚕️ Expert':'🌾 Farmer'],
              ['Member Since',user?.createdAt?new Date(user.createdAt).toLocaleDateString('en-IN',{day:'2-digit',month:'short',year:'numeric'}):'—'],
              ['Status',user?.verified?'✅ Verified':'⏳ Pending'],
            ].map(([k,v])=>(
              <div key={k} style={{display:'flex',justifyContent:'space-between',padding:'9px 0',borderBottom:'1px solid var(--gp)',fontSize:13}}>
                <span style={{color:'var(--tx3)',fontWeight:600}}>{k}</span>
                <span style={{color:'var(--tx)',fontWeight:700}}>{v}</span>
              </div>
            ))}
          </div>

          <div className="card" style={{padding:20}}>
            <div style={{fontSize:14,fontWeight:800,color:'var(--g1)',marginBottom:12}}>🗑️ Danger Zone</div>
            <div style={{fontSize:13,color:'var(--tx3)',marginBottom:14}}>
              Yeh actions reversible nahi hain. Dhyan se proceed karein.
            </div>
            <button className="btn btn-out btn-md" style={{width:'100%',marginBottom:10,color:'var(--r2)',borderColor:'var(--r2)'}}
              onClick={()=>toast('Data export 24 ghante mein email pe milega','inf')}>
              📤 My Data Export Karo
            </button>
            <button className="btn btn-red btn-md" style={{width:'100%'}} onClick={doLogout}>
              🚪 Logout
            </button>
          </div>

          <div style={{textAlign:'center',fontSize:12,color:'var(--tx4)',padding:'8px 0'}}>
            BeejHealth v3.0 • Made with 💚 for Indian Farmers
          </div>
        </div>
      )}
    </div>
  );
}
function VoiceInputPage({user,nav,toast}) {
  const [listening,setListening]=useState(false);
  const [transcript,setTranscript]=useState('');
  const [processing,setProcessing]=useState(false);
  const [result,setResult]=useState(null);
  const recognitionRef=useRef(null);
  // Fix: use a ref to hold the latest transcript so onend closure isn't stale
  const transcriptRef=useRef('');

  const startListening=()=>{
    const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
    if(!SR){ toast('Aapka browser voice input support nahi karta. Chrome use karein.','err'); return; }
    transcriptRef.current='';
    const rec=new SR();
    rec.lang='hi-IN';
    rec.continuous=false;
    rec.interimResults=true;
    rec.onstart=()=>setListening(true);
    rec.onresult=(e)=>{
      const t=Array.from(e.results).map(r=>r[0].transcript).join('');
      transcriptRef.current=t;
      setTranscript(t);
    };
    rec.onerror=(e)=>{ toast('Voice error: '+e.error,'err'); setListening(false); };
    rec.onend=()=>{
      setListening(false);
      const latest=transcriptRef.current;
      if(latest.trim()) processVoice(latest);
    };
    recognitionRef.current=rec;
    rec.start();
  };

  const stopListening=()=>{
    if(recognitionRef.current) recognitionRef.current.stop();
    setListening(false);
  };

  const processVoice=async(text)=>{
    setProcessing(true);
    // Detect crop and disease keywords from Hindi transcript
    const crops={'tamatar':'tomato','gehu':'wheat','aalu':'potato','kapas':'cotton','naariyal':'coconut','makka':'corn','aam':'mango'};
    const diseases={'daag':'spots','peela':'yellow','murjhana':'wilt','sukh':'dry','rot':'spots'};
    let detectedCrop='tomato', detectedIssue='spots';
    Object.entries(crops).forEach(([hi,en])=>{ if(text.toLowerCase().includes(hi)) detectedCrop=en; });
    Object.entries(diseases).forEach(([hi,en])=>{ if(text.toLowerCase().includes(hi)) detectedIssue=en; });
    await new Promise(r=>setTimeout(r,1200));
    setResult({ crop:detectedCrop, issue:detectedIssue, text, confidence:82 });
    setProcessing(false);
    toast('Voice input process ho gaya! ✅');
  };

  const SAMPLE_TRANSCRIPTS=[
    'Mere tamatar ke patte peele ho rahe hain aur neeche gir rahe hain',
    'Gehun ki fasal mein lal rang ke daag aa rahe hain patto pe',
    'Alu ke paudhon mein kaale rang ke dhabb dikhe hain'
  ];



  const analyze=async()=>{
    if(!transcript)return;
    setProcessing(true);
    await new Promise(r=>setTimeout(r,1800));
    setProcessing(false);
    setResult({crop:'Tomato 🍅',disease:'Early Blight',conf:87,action:'Mancozeb 75% WP spray karein — 2.5g/L'});
  };

  return (
    <div className="wrap-sm">
      <button className="btn btn-ghost btn-sm" style={{marginBottom:18}} onClick={()=>nav('consultation')}>← Wapas</button>
      <div style={{fontFamily:"'Baloo 2',cursive",fontSize:26,fontWeight:900,color:'var(--g1)',marginBottom:5}}>🎤 Voice Se Bolo</div>
      <div style={{fontSize:14,color:'var(--tx2)',marginBottom:28}}>Hindi mein apni fasal ki problem batao — AI samjhega</div>

      {/* Mic Button */}
      <div style={{textAlign:'center',marginBottom:32}}>
        <div style={{marginBottom:18,fontSize:13,color:'var(--tx3)',fontWeight:600}}>
          {listening?'🔴 Sun raha hoon...':transcript?'✅ Suna gaya':'Mic dabao aur bolna shuru karo'}
        </div>
        <button className={`voice-btn${listening?' listening':' idle'}`} onClick={startListening} disabled={processing}>
          {listening?'🎙️':'🎤'}
        </button>
        {listening&&(
          <div style={{display:'flex',justifyContent:'center',marginTop:18}}>
            <div className="voice-wave">
              {[28,14,36,20,32,16,30].map((h,i)=>(
                <div key={i} className="vw-bar" style={{height:h,animationDelay:`${i*0.12}s`}}/>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Transcript Box */}
      {(transcript||listening)&&(
        <div className="card" style={{padding:20,marginBottom:18}} key="trans">
          <div style={{fontSize:11,fontWeight:700,color:'var(--tx3)',textTransform:'uppercase',letterSpacing:'.6px',marginBottom:9}}>📝 Aapne Kaha:</div>
          <div style={{fontSize:15,lineHeight:1.7,fontStyle:listening?'italic':'normal',color:listening?'var(--tx3)':'var(--tx)'}}>
            {listening?'...' : `"${transcript}"`}
          </div>
          {transcript&&!listening&&(
            <div style={{display:'flex',gap:9,marginTop:14}}>
              <button className="btn btn-ghost btn-sm" style={{flex:1}} onClick={()=>{setTranscript('');setResult(null);}}>🔄 Dobara Bolo</button>
              <button className="btn btn-g btn-sm" style={{flex:2}} onClick={analyze} disabled={processing}>
                {processing?<><div className="spin"/>Analyze ho raha hai...</>:'🤖 AI Se Analyze Karao →'}
              </button>
            </div>
          )}
        </div>
      )}

      {/* AI Result */}
      {result&&(
        <div className="card" style={{padding:22,border:'2px solid var(--g4)'}} key="result">
          <div style={{fontSize:13,fontWeight:700,color:'var(--g3)',marginBottom:12}}>✅ AI Analysis Complete</div>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:10,marginBottom:14}}>
            {[['Crop',result.crop],['Disease',result.disease],['Confidence',`${result.conf}%`],['Severity','Stage 2/5']].map(([k,v])=>(
              <div key={k} style={{padding:12,background:'var(--gp)',borderRadius:10}}>
                <div style={{fontSize:11,color:'var(--tx3)',fontWeight:700,marginBottom:3}}>{k}</div>
                <div style={{fontSize:14,fontWeight:800,color:'var(--g1)'}}>{v}</div>
              </div>
            ))}
          </div>
          <div style={{padding:13,background:'var(--ap)',borderRadius:10,fontSize:13.5,color:'var(--a1)',fontWeight:600,marginBottom:14}}>
            💊 {result.action}
          </div>
          <div style={{display:'flex',gap:9}}>
            <button className="btn btn-out btn-sm" style={{flex:1}} onClick={()=>nav('ai-report')}>📄 Full Report</button>
            <button className="btn btn-g btn-sm" style={{flex:2}} onClick={()=>nav('experts')}>👨‍⚕️ Expert Se Confirm →</button>
          </div>
        </div>
      )}

      {/* Language Support */}
      <div style={{marginTop:24,padding:16,background:'var(--gp)',borderRadius:'var(--rad)',fontSize:13,color:'var(--g2)',fontWeight:600}}>
        🗣️ Supported Languages: Hindi • Marathi • Punjabi • Gujarati • Telugu • Tamil
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   2. SATELLITE FIELD MONITOR 🛰️
════════════════════════════════════════════════════════════════ */
function SatellitePage({user,nav,toast}) {
  const [mapLoaded,setMapLoaded]=useState(false);
  const mapRef=useRef(null);

  useEffect(()=>{
    // Load Leaflet dynamically
    const script=document.createElement('script');
    script.src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
    script.onload=()=>{
      const link=document.createElement('link');
      link.rel='stylesheet'; link.href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
      document.head.appendChild(link);
      setTimeout(()=>{
        if(mapRef.current&&!mapRef.current._leaflet_id){
          const map=window.L.map(mapRef.current).setView([18.52,73.85],12);
          window.L.tileLayer('https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',{
            maxZoom:20, subdomains:['mt0','mt1','mt2','mt3'],
            attribution:'Google Satellite'
          }).addTo(map);
          // Add farm marker
          window.L.marker([18.52,73.85]).addTo(map)
            .bindPopup(`<b>${user?.name||'My Farm'}</b><br>${user?.district||'Pune'}`).openPopup();
          setMapLoaded(true);
        }
      },300);
    };
    document.head.appendChild(script);
    return ()=>{ if(mapRef.current&&mapRef.current._leaflet_id) mapRef.current._leaflet_id=null; };
  },[]);
  const [selField,setSelField]=useState(null);
  const [view,setView]=useState('ndvi');

  const fields=[
    {id:1,name:'Field 1',crop:'🍅 Tomato',acres:2,ndvi:.61,status:'warn',x:15,y:20,w:34,h:28,color:'#ffc940'},
    {id:2,name:'Field 2',crop:'🌾 Wheat',acres:1.5,ndvi:.78,status:'ok',x:55,y:15,w:28,h:32,color:'#4dbd7a'},
    {id:3,name:'Field 3',crop:'🥔 Potato',acres:1,ndvi:.82,status:'ok',x:20,y:58,w:22,h:26,color:'#7dd4a0'},
  ];

  const viewColors={'ndvi':'NDVI Health','rgb':'True Color','temp':'Temperature'};

  return (
    <div className="wrap-md">
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:22}}>
        <div>
          <div style={{fontFamily:"'Baloo 2',cursive",fontSize:26,fontWeight:900,color:'var(--g1)'}}>🛰️ Satellite Monitor</div>
          <div style={{fontSize:13,color:'var(--tx2)',marginTop:3}}>ISRO Resourcesat + Sentinel-2 • Last updated: 2 hours ago</div>
        </div>
        <div style={{display:'flex',gap:7}}>
          {Object.keys(viewColors).map(v=>(
            <button key={v} onClick={()=>setView(v)} style={{padding:'7px 14px',borderRadius:8,fontSize:12.5,fontWeight:700,border:`2px solid ${view===v?'var(--g4)':'var(--br)'}`,background:view===v?'var(--gp)':'none',color:view===v?'var(--g3)':'var(--tx2)',cursor:'pointer',fontFamily:"'Outfit',sans-serif"}}>
              {v.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Satellite Map */}
      <div className="sat-map" onClick={()=>setSelField(null)}>
        <div className="sat-grid"/>
        <div className="sat-overlay">📡 {viewColors[view]} • Wagholi Farm</div>
        {fields.map(f=>(
          <div key={f.id} className="sat-field" onClick={e=>{e.stopPropagation();setSelField(f);}}
            style={{left:`${f.x}%`,top:`${f.y}%`,width:`${f.w}%`,height:`${f.h}%`,background:f.color+'55',borderColor:selField?.id===f.id?'white':f.color+'88'}}>
            <div style={{textAlign:'center'}}>
              <div style={{fontSize:14}}>{f.crop.split(' ')[0]}</div>
              <div style={{fontSize:10,fontWeight:700,color:'white',marginTop:2,textShadow:'0 1px 3px rgba(0,0,0,.8)'}}>{f.name}</div>
            </div>
          </div>
        ))}
        {/* NDVI Legend */}
        <div style={{position:'absolute',bottom:12,right:12,background:'rgba(0,0,0,.7)',borderRadius:8,padding:'8px 12px',backdropFilter:'blur(8px)'}}>
          <div style={{fontSize:10,color:'rgba(255,255,255,.7)',marginBottom:5,fontWeight:600}}>NDVI Index</div>
          <div className="ndvi-bar" style={{width:120}}/>
          <div style={{display:'flex',justifyContent:'space-between',fontSize:9,color:'rgba(255,255,255,.6)',marginTop:2}}><span>0 Low</span><span>1 High</span></div>
        </div>
      </div>

      {/* Field Detail */}
      {selField?(
        <div className="card" style={{padding:22,marginBottom:18,border:`2px solid ${selField.color}`}} key={selField.id}>
          <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:14}}>
            <div style={{fontSize:17,fontWeight:900,color:'var(--g1)'}}>{selField.crop} — {selField.name}</div>
            <span className={`badge ${selField.status==='ok'?'bg-g':'bg-a'}`}>{selField.status==='ok'?'✅ Healthy':'⚠️ Monitor'}</span>
          </div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:10,marginBottom:14}}>
            {[['NDVI Score',(selField.ndvi*100).toFixed(0)+'%','🌿'],['Area',selField.acres+' Acres','📐'],['Last Scan','2 hrs ago','📡']].map(([l,v,i])=>(
              <div key={l} style={{padding:12,background:'var(--gp)',borderRadius:10,textAlign:'center'}}>
                <div style={{fontSize:18,marginBottom:4}}>{i}</div>
                <div style={{fontSize:16,fontWeight:900,color:'var(--g2)'}}>{v}</div>
                <div style={{fontSize:11,color:'var(--tx3)',marginTop:2}}>{l}</div>
              </div>
            ))}
          </div>
          <div className="ndvi-bar" style={{marginBottom:8}}>
            <div className="ndvi-marker" style={{left:`${selField.ndvi*100-2}%`}}/>
          </div>
          {selField.status==='warn'&&(
            <div style={{padding:12,background:'var(--ap)',borderRadius:10,fontSize:13,color:'var(--a1)',fontWeight:600,marginBottom:12}}>
              ⚠️ NDVI {(selField.ndvi*100).toFixed(0)}% — Below optimal (75%). Possible early stress detected. Inspection recommended.
            </div>
          )}
          <div style={{display:'flex',gap:9}}>
            <button className="btn btn-ghost btn-sm" style={{flex:1}} onClick={()=>toast('Historical comparison loading...','inf')}>📊 History</button>
            <button className="btn btn-g btn-sm" style={{flex:2}} onClick={()=>nav('consultation')}>🔬 Consultation Book Karo →</button>
          </div>
        </div>
      ):(
        <div style={{padding:14,background:'var(--gp)',borderRadius:10,fontSize:13,color:'var(--g2)',fontWeight:600,marginBottom:18,textAlign:'center'}}>
          👆 Map pe koi field click karein details dekhne ke liye
        </div>
      )}

      {/* NDVI Legend */}
      <div className="sat-legend">
        {[['#ef5350','Stress (0-40%)'],['#ffc940','Monitor (40-65%)'],['#4dbd7a','Healthy (65-80%)'],['#1e7e42','Excellent (80%+)']].map(([c,l])=>(
          <div key={l} className="sat-leg-item"><div className="sat-leg-dot" style={{background:c}}/>{l}</div>
        ))}
      </div>

      {/* All Fields Summary */}
      <div className="card" style={{padding:20}}>
        <div style={{fontSize:15,fontWeight:800,color:'var(--g1)',marginBottom:14}}>📊 Farm Summary</div>
        {fields.map(f=>(
          <div key={f.id} style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'10px 0',borderBottom:'1px solid var(--gp)',cursor:'pointer'}} onClick={()=>setSelField(f)}>
            <div style={{display:'flex',gap:10,alignItems:'center'}}>
              <div style={{width:10,height:10,borderRadius:2,background:f.color,flexShrink:0}}/>
              <div>
                <div style={{fontSize:13.5,fontWeight:700}}>{f.crop} — {f.name}</div>
                <div style={{fontSize:11.5,color:'var(--tx3)'}}>{f.acres} Acres</div>
              </div>
            </div>
            <div style={{textAlign:'right'}}>
              <div style={{fontSize:14,fontWeight:800,color:f.status==='ok'?'var(--g3)':'var(--a2)'}}>{(f.ndvi*100).toFixed(0)}% NDVI</div>
              <span className={`badge ${f.status==='ok'?'bg-g':'bg-a'}`} style={{fontSize:10}}>{f.status==='ok'?'✅ OK':'⚠️ Watch'}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   3. PREDICTIVE DISEASE FORECAST 📊
════════════════════════════════════════════════════════════════ */
function ForecastPage({user,nav,toast}) {
  const [weather,setWeather]=useState(null);
  const [loadingWx,setLoadingWx]=useState(true);
  const [wxErr,setWxErr]=useState(false);

  useEffect(()=>{
    const district = user?.district || 'Pune';
    // OpenWeatherMap free tier API
    const API_KEY = '4d8fb5b93d4af21d66a2948a6e8a74a0'; // Demo key — replace with yours
    fetch(`https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(district)},IN&appid=${API_KEY}&units=metric&lang=hi`)
      .then(r=>r.json())
      .then(d=>{
        if(d.cod===200){
          setWeather({
            temp:    Math.round(d.main.temp),
            feels:   Math.round(d.main.feels_like),
            humidity:d.main.humidity,
            desc:    d.weather[0].description,
            wind:    Math.round(d.wind.speed*3.6), // m/s to km/h
            city:    d.name,
            icon:    d.weather[0].icon,
            pressure:d.main.pressure,
          });
        } else { setWxErr(true); }
        setLoadingWx(false);
      })
      .catch(()=>{ setWxErr(true); setLoadingWx(false); });
  },[user?.district]);
  const forecasts=[
    {disease:'Late Blight',crop:'🥔 Potato',risk:78,level:'high',days:5,reason:'High humidity (72%) + cool temp expected',action:'Preventive spray karein aaj'},
    {disease:'Powdery Mildew',crop:'🍇 Grape',risk:52,level:'med',days:8,reason:'Dry warm weather forecast',action:'Monitor karein — spray ready rakhein'},
    {disease:'Leaf Rust',crop:'🌾 Wheat',risk:31,level:'low',days:14,reason:'Conditions unfavorable currently',action:'No action needed abhi'},
    {disease:'Early Blight',crop:'🍅 Tomato',risk:67,level:'med',days:6,reason:'Humidity spike + rain forecast',action:'Copper-based fungicide consider karein'},
  ];

  const timeline=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
  const riskData=[22,35,48,61,78,72,55];

  return (
    <div className="wrap-md">
      {/* Live Weather from OpenWeatherMap */}
      {loadingWx&&<div style={{textAlign:'center',padding:'16px',fontSize:13,color:'var(--tx3)'}}>🌐 Live mausam load ho raha hai...</div>}
      {!loadingWx&&!wxErr&&weather&&(
        <div className="card" style={{padding:20,marginBottom:20,background:'linear-gradient(135deg,var(--bp),var(--bpb))',border:'1.5px solid var(--bpb)'}}>
          <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:10}}>
            <div style={{fontFamily:"'Baloo 2',cursive",fontSize:16,fontWeight:800,color:'var(--b1)'}}>🌤️ Live Mausam — {weather.city}</div>
            <div style={{fontSize:10,color:'var(--tx3)'}}>OpenWeatherMap Live</div>
          </div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:8}}>
            {[['🌡️','Temp',weather.temp+'°C'],['💧','Humidity',weather.humidity+'%'],['💨','Wind',weather.wind+' km/h'],['📊','Pressure',weather.pressure+' hPa']].map(([ic,l,v])=>(
              <div key={l} style={{background:'rgba(255,255,255,.6)',borderRadius:8,padding:'10px 8px',textAlign:'center'}}>
                <div style={{fontSize:18,marginBottom:3}}>{ic}</div>
                <div style={{fontFamily:"'Baloo 2',cursive",fontSize:15,fontWeight:800,color:'var(--b1)'}}>{v}</div>
                <div style={{fontSize:10,color:'var(--tx3)'}}>{l}</div>
              </div>
            ))}
          </div>
          <div style={{marginTop:8,fontSize:12,color:'var(--b2)',fontWeight:600,textTransform:'capitalize'}}>Aaj: {weather.desc} · Feels like {weather.feels}°C</div>
        </div>
      )}
      {!loadingWx&&wxErr&&(
        <div style={{background:'var(--ap)',borderRadius:8,padding:'10px 14px',marginBottom:12,fontSize:12,color:'var(--a1)'}}>⚠️ Live weather load nahi hua — internet ya API key check karein</div>
      )}

      <div style={{fontFamily:"'Baloo 2',cursive",fontSize:26,fontWeight:900,color:'var(--g1)',marginBottom:5}}>📊 Disease Forecast</div>
      <div style={{fontSize:13,color:'var(--tx2)',marginBottom:22}}>AI + IMD Weather Data • {user?.district||'Pune'}, Maharashtra</div>

      {/* 7-Day Risk Timeline */}
      <div className="card" style={{padding:22,marginBottom:22}}>
        <div style={{fontSize:15,fontWeight:800,color:'var(--g1)',marginBottom:16}}>📅 7-Day Disease Risk Timeline</div>
        <div style={{display:'flex',gap:0,alignItems:'flex-end',height:100,marginBottom:8}}>
          {riskData.map((r,i)=>(
            <div key={i} style={{flex:1,display:'flex',flexDirection:'column',alignItems:'center',gap:4}}>
              <div style={{fontSize:10,fontWeight:700,color:r>60?'var(--r2)':r>40?'var(--a2)':'var(--g4)'}}>{r}%</div>
              <div style={{width:'70%',borderRadius:'4px 4px 0 0',background:r>60?'var(--r3)':r>40?'var(--a3)':'var(--g5)',transition:'height .7s ease',height:`${r}%`,minHeight:6}}/>
            </div>
          ))}
        </div>
        <div style={{display:'flex',gap:0}}>
          {timeline.map((d,i)=>(
            <div key={d} style={{flex:1,textAlign:'center',fontSize:11,fontWeight:700,color:i===4?'var(--r2)':'var(--tx3)',paddingTop:6,borderTop:`2px solid ${i===4?'var(--r2)':'var(--br)'}`}}>{d}</div>
          ))}
        </div>
        <div style={{marginTop:12,padding:'9px 13px',background:'var(--rp)',borderRadius:8,fontSize:13,color:'var(--r2)',fontWeight:700}}>
          🔴 Peak Risk: Friday — Late Blight probability 78%
        </div>
      </div>

      {/* Disease Forecasts */}
      <div style={{fontSize:16,fontWeight:800,color:'var(--g1)',marginBottom:14}}>🦠 Crop-wise Forecast</div>
      {forecasts.map((f,i)=>(
        <div className="forecast-card" key={i}>
          <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',marginBottom:10}}>
            <div>
              <div style={{fontSize:15,fontWeight:800,color:'var(--tx)'}}>{f.crop} — {f.disease}</div>
              <div style={{fontSize:12,color:'var(--tx3)',marginTop:2}}>⏱️ Peak risk in {f.days} days</div>
            </div>
            <div style={{textAlign:'right'}}>
              <div style={{fontFamily:"'Baloo 2',cursive",fontSize:22,fontWeight:900,color:f.level==='high'?'var(--r2)':f.level==='med'?'var(--a2)':'var(--g4)'}}>{f.risk}%</div>
              <span className={`badge ${f.level==='high'?'bg-r':f.level==='med'?'bg-a':'bg-g'}`} style={{fontSize:10}}>{f.level==='high'?'🔴 High':f.level==='med'?'🟡 Medium':'🟢 Low'}</span>
            </div>
          </div>
          <div className="risk-meter">
            <div className={`risk-fill ${f.level}`} style={{width:`${f.risk}%`}}/>
          </div>
          <div style={{fontSize:12.5,color:'var(--tx2)',margin:'8px 0',fontStyle:'italic'}}>📌 {f.reason}</div>
          <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
            <div style={{fontSize:13,fontWeight:700,color:f.level==='high'?'var(--r2)':f.level==='med'?'var(--a1)':'var(--g3)'}}>💡 {f.action}</div>
            {f.level!=='low'&&<button className="btn btn-g btn-sm" onClick={()=>nav('marketplace')}>🛒 Medicine Order</button>}
          </div>
        </div>
      ))}

      {/* AI Insight */}
      <div style={{padding:18,background:'linear-gradient(135deg,var(--gp),var(--gpb))',borderRadius:'var(--rad)',border:'1.5px solid var(--br2)',marginTop:8}}>
        <div style={{fontSize:14,fontWeight:800,color:'var(--g1)',marginBottom:7}}>🤖 AI Insight</div>
        <div style={{fontSize:13.5,color:'var(--tx2)',lineHeight:1.7}}>
          Is hafta weather pattern ke hisaab se <strong>fungal diseases ka risk 40% zyada</strong> hai normal se. Humidity 68-75% range mein hai aur temperature drop forecast hai — yeh Late Blight ke liye ideal conditions hain. Preventive action Thursday tak le lein.
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   4. IoT SOIL SENSORS 🌱
════════════════════════════════════════════════════════════════ */
function SoilSensorPage({user,nav,toast}) {
  const [selField,setSelField]=useState('field1');
  const sensors={
    field1:[
      {label:'Soil Moisture',val:62,unit:'%',status:'ok',color:'var(--b3)',hist:[55,58,61,65,63,60,62],min:0,max:100,optimal:'55-75%'},
      {label:'Temperature',val:24,unit:'°C',status:'ok',color:'var(--g3)',hist:[22,23,24,25,24,23,24],min:0,max:50,optimal:'20-28°C'},
      {label:'pH Level',val:6.2,unit:'pH',status:'warn',color:'var(--a2)',hist:[6.5,6.4,6.3,6.2,6.2,6.1,6.2],min:0,max:14,optimal:'6.0-7.0'},
      {label:'Nitrogen (N)',val:38,unit:'mg/kg',status:'bad',color:'var(--r2)',hist:[52,48,44,41,39,37,38],min:0,max:100,optimal:'50-80 mg/kg'},
      {label:'Phosphorus (P)',val:28,unit:'mg/kg',status:'ok',color:'var(--g4)',hist:[24,25,27,26,28,27,28],min:0,max:60,optimal:'20-40 mg/kg'},
      {label:'Potassium (K)',val:145,unit:'mg/kg',status:'ok',color:'var(--pu)',hist:[130,135,140,142,144,143,145],min:0,max:300,optimal:'120-200 mg/kg'},
    ]
  };
  const data=sensors[selField]||sensors.field1;

  return (
    <div className="wrap-md">
      <div style={{fontFamily:"'Baloo 2',cursive",fontSize:26,fontWeight:900,color:'var(--g1)',marginBottom:5}}>🌱 Soil Sensors</div>
      <div style={{fontSize:13,color:'var(--tx2)',marginBottom:16}}>Real-time soil health monitoring • IoT sensors</div>

      {/* Field Selector */}
      <div style={{display:'flex',gap:8,marginBottom:22}}>
        {[{k:'field1',l:'🍅 Field 1 — Tomato'},{k:'field2',l:'🌾 Field 2 — Wheat'},{k:'field3',l:'🥔 Field 3 — Potato'}].map(f=>(
          <button key={f.k} onClick={()=>setSelField(f.k)} style={{flex:1,padding:'9px 8px',borderRadius:9,fontSize:12.5,fontWeight:700,border:`2px solid ${selField===f.k?'var(--g4)':'var(--br)'}`,background:selField===f.k?'var(--gp)':'none',color:selField===f.k?'var(--g3)':'var(--tx2)',cursor:'pointer',fontFamily:"'Outfit',sans-serif",transition:'all .18s'}}>
            {f.l}
          </button>
        ))}
      </div>

      {/* AI Recommendation Banner */}
      <div style={{padding:16,background:'linear-gradient(135deg,var(--rp),#fff5f5)',border:'2px solid var(--rpb)',borderRadius:'var(--rad)',marginBottom:22,display:'flex',gap:13,alignItems:'flex-start'}}>
        <div style={{fontSize:26,flexShrink:0}}>🤖</div>
        <div>
          <div style={{fontSize:14,fontWeight:800,color:'var(--r1)',marginBottom:4}}>AI Soil Alert</div>
          <div style={{fontSize:13,color:'var(--tx2)',lineHeight:1.65}}>
            <strong>Nitrogen deficiency detected!</strong> NPK 19:19:19 — 5kg/acre apply karein is hafte. pH bhi thoda low hai — agricultural lime 2kg/acre se correct karein.
          </div>
          <button className="btn btn-red btn-sm" style={{marginTop:10}} onClick={()=>nav('marketplace')}>🛒 Fertilizer Order Karo</button>
        </div>
      </div>

      {/* Sensor Cards Grid */}
      <div className="sensor-grid">
        {data.map((s,i)=>{
          const pct=((s.val-s.min)/(s.max-s.min)*100).toFixed(0);
          return (
            <div key={i} className={`sensor-card ${s.status}`}>
              <div className="sensor-lbl">{s.label}</div>
              <div style={{display:'flex',alignItems:'baseline',gap:5}}>
                <div className="sensor-val" style={{color:s.color}}>{s.val}</div>
                <div className="sensor-unit">{s.unit}</div>
              </div>
              <div style={{fontSize:11,color:'var(--tx3)',marginTop:2}}>Optimal: {s.optimal}</div>
              <div className="sensor-gauge">
                <div className="sensor-gauge-fill" style={{width:`${pct}%`,background:s.color}}/>
              </div>
              <div className="sensor-hist">
                {s.hist.map((v,j)=>{
                  const hp=((v-s.min)/(s.max-s.min)*100);
                  return <div key={j} className="sh-bar" style={{height:`${Math.max(hp,8)}%`,background:j===s.hist.length-1?s.color:'var(--br2)'}}/>;
                })}
              </div>
              <div style={{fontSize:10,color:'var(--tx4)',marginTop:3}}>Last 7 readings</div>
              {s.status==='bad'&&<div style={{marginTop:7,fontSize:11,fontWeight:700,color:'var(--r2)'}}>⚠️ Action needed!</div>}
              {s.status==='warn'&&<div style={{marginTop:7,fontSize:11,fontWeight:700,color:'var(--a2)'}}>⚡ Monitor closely</div>}
            </div>
          );
        })}
      </div>

      {/* Schedule Next Reading */}
      <div className="card" style={{padding:18,marginTop:6}}>
        <div style={{fontSize:14,fontWeight:800,color:'var(--g1)',marginBottom:11}}>⏰ Sensor Schedule</div>
        {[['Last Reading','Aaj 8:30 AM','✅'],['Next Auto-Reading','Aaj 2:30 PM','⏳'],['Weekly Report','Sunday','📊']].map(([l,v,i])=>(
          <div key={l} style={{display:'flex',justifyContent:'space-between',padding:'8px 0',borderBottom:'1px solid var(--gp)',fontSize:13}}>
            <span style={{color:'var(--tx2)',fontWeight:600}}>{i} {l}</span>
            <span style={{fontWeight:800,color:'var(--tx)'}}>{v}</span>
          </div>
        ))}
        <button className="btn btn-g btn-sm" style={{width:'100%',marginTop:12}} onClick={()=>toast('Manual reading triggered! 2 min mein results aayenge','inf')}>
          🔄 Manual Reading Trigger Karo
        </button>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   5. B2B DATA INTELLIGENCE 💼
════════════════════════════════════════════════════════════════ */
function B2BPage({nav,toast}) {
  const [lead,setLead]=useState({company:'',contact:'',mobile:'',email:'',type:'',msg:''});
  const [submitting,setSubmitting]=useState(false);
  const [submitted,setSubmitted]=useState(false);

  const submitLead=async()=>{
    if(!lead.company||!lead.mobile){toast('Company aur mobile zaroor bharo','err');return;}
    setSubmitting(true);
    try{
      await API.post('/api/consultations',{
        cropId:'b2b', cropName:'B2B Lead: '+lead.company, cropEmoji:'💼',
        method:'manual', disease:'B2B Enquiry — '+lead.type,
        confidence:100, severity:1,
        answers:{company:lead.company,contact:lead.contact,mobile:lead.mobile,email:lead.email,type:lead.type,msg:lead.msg},
      });
      setSubmitted(true);
      toast('B2B enquiry submit ho gayi! Team 24 ghante mein contact karegi. ✅');
    }catch(e){toast('Submit fail — dobara try karo','err');}
    setSubmitting(false);
  };
  const districts=['Pune','Nashik','Aurangabad','Nagpur','Solapur','Kolhapur','Satara','Sangli'];
  const diseases=['Early Blight','Late Blight','Powdery Mildew','Leaf Rust','Bacterial Wilt','Downy Mildew','Fusarium','Anthracnose'];
  const heatData=Array.from({length:56},()=>Math.floor(Math.random()*5));
  const heatColors=['#eaf7ef','#b8d9c2','#7dd4a0','#4dbd7a','#1e7e42'];

  return (
    <div className="wrap-md">
      <div style={{display:'flex',gap:10,alignItems:'center',marginBottom:5}}>
        <div style={{padding:'4px 12px',background:'var(--bp)',borderRadius:100,fontSize:12,fontWeight:700,color:'var(--b2)'}}>💼 B2B Portal</div>
        <div style={{fontSize:12,color:'var(--tx3)'}}>Agri companies & government access</div>
      </div>
      <div style={{fontFamily:"'Baloo 2',cursive",fontSize:26,fontWeight:900,color:'var(--b1)',marginBottom:18}}>Disease Intelligence Dashboard</div>

      {/* KPI Row */}
      <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:14,marginBottom:24}}>
        {[{n:'47,320',l:'Active Farmers',i:'👨‍🌾'},{n:'1,24,580',l:'Disease Cases',i:'🦠'},{n:'94.2%',l:'AI Accuracy',i:'🎯'},{n:'58',l:'Districts Covered',i:'📍'}].map(s=>(
          <div key={s.l} className="b2b-stat">
            <div style={{fontSize:24,marginBottom:6}}>{s.i}</div>
            <div className="b2b-n">{s.n}</div>
            <div className="b2b-l">{s.l}</div>
          </div>
        ))}
      </div>

      <div className="dash-2">
        <div>
          {/* Disease Heatmap */}
          <div className="card" style={{padding:20,marginBottom:18}}>
            <div style={{fontSize:15,fontWeight:800,color:'var(--b1)',marginBottom:5}}>🗺️ District Disease Heatmap</div>
            <div style={{fontSize:12,color:'var(--tx3)',marginBottom:13}}>Maharashtra — Last 30 days</div>
            <div style={{display:'flex',gap:6,marginBottom:8,flexWrap:'wrap'}}>
              {districts.map(d=><span key={d} style={{fontSize:11,padding:'2px 8px',background:'var(--bp)',borderRadius:100,fontWeight:600,color:'var(--b2)'}}>{d}</span>)}
            </div>
            <div className="heatmap-grid">
              {heatData.map((v,i)=>(
                <div key={i} className="hm-cell" style={{background:heatColors[v]}} title={`Cases: ${v*23}`}/>
              ))}
            </div>
            <div style={{display:'flex',gap:8,alignItems:'center',marginTop:8}}>
              {heatColors.map((c,i)=><div key={i} style={{width:16,height:10,background:c,borderRadius:2}}/>)}
              <span style={{fontSize:11,color:'var(--tx3)'}}>Low → High cases</span>
            </div>
          </div>

          {/* Top Diseases */}
          <div className="card" style={{padding:20}}>
            <div style={{fontSize:15,fontWeight:800,color:'var(--b1)',marginBottom:14}}>📊 Top Diseases This Season</div>
            {diseases.slice(0,5).map((d,i)=>{
              const pct=Math.floor(85-i*12);
              return (
                <div key={d} className="disease-bar-row">
                  <div style={{width:120,fontSize:12.5,fontWeight:600,color:'var(--tx)',flexShrink:0}}>{d}</div>
                  <div className="disease-bar-track"><div className="disease-bar-fill" style={{width:`${pct}%`}}/></div>
                  <div style={{fontSize:12,fontWeight:800,color:'var(--b3)',width:40,textAlign:'right'}}>{pct}%</div>
                </div>
              );
            })}
          </div>
        </div>

        <div style={{display:'flex',flexDirection:'column',gap:18}}>
          {/* Revenue opportunity */}
          <div className="card" style={{padding:20,background:'linear-gradient(135deg,#eef5ff,#dbeafe)',border:'1.5px solid var(--bpb)'}}>
            <div style={{fontSize:15,fontWeight:800,color:'var(--b1)',marginBottom:13}}>💰 Data API Pricing</div>
            {[['Basic Plan','₹15,000/mo','District-level data'],['Pro Plan','₹45,000/mo','Real-time + crop-wise'],['Enterprise','Custom','Full API + white-label']].map(([p,pr,d])=>(
              <div key={p} style={{padding:'11px 13px',background:'white',borderRadius:10,marginBottom:9,display:'flex',justifyContent:'space-between',alignItems:'center'}}>
                <div>
                  <div style={{fontSize:13.5,fontWeight:700,color:'var(--b1)'}}>{p}</div>
                  <div style={{fontSize:11,color:'var(--tx3)'}}>{d}</div>
                </div>
                <div style={{fontFamily:"'Baloo 2',cursive",fontSize:16,fontWeight:900,color:'var(--b3)'}}>{pr}</div>
              </div>
            ))}
            <button className="btn btn-b btn-sm" style={{width:'100%',marginTop:4}} onClick={()=>toast('Sales team se contact kiya jayega! 📧','inf')}>
              📧 Contact Sales Team
            </button>
          </div>

          {/* Recent Alerts */}
          <div className="card" style={{padding:20}}>
            <div style={{fontSize:15,fontWeight:800,color:'var(--b1)',marginBottom:13}}>🚨 Recent Outbreak Alerts</div>
            {[{d:'Late Blight',loc:'Nashik (5 blocks)',n:124,sev:'High'},{d:'Early Blight',loc:'Pune (3 blocks)',n:87,sev:'Med'},{d:'Powdery Mildew',loc:'Satara',n:43,sev:'Low'}].map(a=>(
              <div key={a.d} style={{padding:'10px 0',borderBottom:'1px solid var(--gp)',display:'flex',justifyContent:'space-between',alignItems:'center'}}>
                <div>
                  <div style={{fontSize:13,fontWeight:700,color:'var(--tx)'}}>{a.d}</div>
                  <div style={{fontSize:11.5,color:'var(--tx3)'}}>📍 {a.loc}</div>
                </div>
                <div style={{textAlign:'right'}}>
                  <div style={{fontSize:13,fontWeight:800,color:'var(--b3)'}}>{a.n} cases</div>
                  <span className={`badge ${a.sev==='High'?'bg-r':a.sev==='Med'?'bg-a':'bg-g'}`} style={{fontSize:10}}>{a.sev}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   6. INPUT MARKETPLACE 📦
════════════════════════════════════════════════════════════════ */
function MarketplacePage({user,nav,toast}) {
  // Static market prices (updated weekly from Agmarknet)
  // Replace with live API: https://agmarknet.gov.in or data.gov.in
  const MARKET_PRICES = [
    {crop:'Tomato🍅',    price:'₹18-28/kg',  trend:'+12%', market:'Pune APMC',    updated:'Aaj'},
    {crop:'Wheat🌾',     price:'₹22-26/kg',  trend:'-3%',  market:'Delhi Mandi',  updated:'Aaj'},
    {crop:'Potato🥔',    price:'₹12-18/kg',  trend:'+8%',  market:'Agra Mandi',   updated:'Kal'},
    {crop:'Cotton🌸',    price:'₹55-65/kg',  trend:'+5%',  market:'Akola APMC',   updated:'Aaj'},
    {crop:'Onion🧅',     price:'₹15-22/kg',  trend:'-8%',  market:'Nashik APMC',  updated:'Aaj'},
    {crop:'Coconut🥥',   price:'₹28-40/piece',trend:'+15%',market:'Palakkad',     updated:'Kal'},
    {crop:'Soybean🫘',   price:'₹42-48/kg',  trend:'+2%',  market:'Indore APMC',  updated:'Aaj'},
    {crop:'Corn🌽',      price:'₹18-24/kg',  trend:'+6%',  market:'Nizamabad',    updated:'Aaj'},
    {crop:'Sugarcane',   price:'₹3.5-4.2/kg',trend:'+1%',  market:'UP Govt Rate', updated:'Mahina'},
    {crop:'Grape🍇',     price:'₹35-55/kg',  trend:'+20%', market:'Nashik APMC',  updated:'Kal'},
    {crop:'Apple🍎',     price:'₹80-150/kg', trend:'-5%',  market:'Delhi Market', updated:'Aaj'},
    {crop:'Rice/Paddy',  price:'₹22-28/kg',  trend:'+3%',  market:'Punjab Mandi', updated:'Aaj'},
  ];
  const [search,setSearch]=useState('');
  const [sortBy,setSortBy]=useState('crop');
  const [cart,setCart]=useState([]);
  const [cat,setCat]=useState('all');

  const products=[
    {id:1,name:'Mancozeb 75% WP',type:'Fungicide',price:280,unit:'kg',emoji:'🧪',ai:true,rating:4.8,stock:'In Stock',desc:'Early & Late Blight'},
    {id:2,name:'Copper Oxychloride',type:'Fungicide',price:320,unit:'kg',emoji:'⚗️',ai:false,rating:4.6,stock:'In Stock',desc:'Bacterial diseases'},
    {id:3,name:'Neem Oil Extract',type:'Organic',price:180,unit:'L',emoji:'🌿',ai:false,rating:4.5,stock:'In Stock',desc:'Eco-friendly option'},
    {id:4,name:'NPK 19:19:19',type:'Fertilizer',price:680,unit:'kg',emoji:'🌱',ai:true,rating:4.9,stock:'In Stock',desc:'Balanced nutrition'},
    {id:5,name:'Imidacloprid 17.8%',type:'Insecticide',price:420,unit:'100ml',emoji:'🐛',ai:false,rating:4.4,stock:'Low Stock',desc:'Sucking pests control'},
    {id:6,name:'BT Hybrid Tomato Seeds',type:'Seeds',price:850,unit:'50g',emoji:'🍅',ai:true,rating:4.9,stock:'In Stock',desc:'Disease resistant variety'},
    {id:7,name:'Potassium Humate',type:'Fertilizer',price:540,unit:'kg',emoji:'💧',ai:false,rating:4.7,stock:'In Stock',desc:'Soil conditioner'},
    {id:8,name:'Trichoderma Viride',type:'Bio-pesticide',price:240,unit:'kg',emoji:'🦠',ai:false,rating:4.6,stock:'In Stock',desc:'Soil-borne disease control'},
    {id:9,name:'Chlorpyrifos 20% EC',type:'Insecticide',price:380,unit:'500ml',emoji:'🧴',ai:false,rating:4.3,stock:'In Stock',desc:'Broad-spectrum insect control'},
  ];

  const cats=['all','Fungicide','Fertilizer','Seeds','Insecticide','Organic','Bio-pesticide'];
  const filtered=products.filter(p=>(cat==='all'||p.type===cat)&&(p.name.toLowerCase().includes(search.toLowerCase())||p.desc.toLowerCase().includes(search.toLowerCase())));
  const addCart=(p)=>{setCart(c=>[...c.filter(x=>x.id!==p.id),{...p,qty:1}]);toast(`${p.name} cart mein add kiya ✅`);};

  return (
    <div className="wrap">
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',marginBottom:18}}>
        <div>
          <div style={{fontFamily:"'Baloo 2',cursive",fontSize:26,fontWeight:900,color:'var(--g1)'}}>📦 AgriMart</div>
          <div style={{fontSize:13,color:'var(--tx2)',marginTop:3}}>AI-recommended medicines, seeds, fertilizers • Village delivery</div>
        </div>
        {cart.length>0&&<div style={{padding:'9px 18px',background:'var(--g3)',color:'white',borderRadius:10,fontSize:14,fontWeight:700,cursor:'pointer'}} onClick={()=>toast(`Cart mein ${cart.length} items — Checkout coming soon!`,'inf')}>🛒 Cart ({cart.length})</div>}
      </div>

      {/* AI Banner */}
      <div style={{padding:15,background:'linear-gradient(135deg,var(--gp),var(--gpb))',border:'1.5px solid var(--br2)',borderRadius:'var(--rad)',marginBottom:20,display:'flex',gap:12,alignItems:'center'}}>
        <span style={{fontSize:26}}>🤖</span>
        <div>
          <div style={{fontSize:13.5,fontWeight:800,color:'var(--g1)'}}>AI Recommended for You</div>
          <div style={{fontSize:12.5,color:'var(--tx2)'}}>Aapki tomato Early Blight case ke hisaab se — Mancozeb 75% WP + NPK 19:19:19 recommended hai</div>
        </div>
      </div>

      {/* Search + Filter */}
      <div style={{display:'flex',gap:11,marginBottom:18,flexWrap:'wrap'}}>
        <input className="finp" style={{flex:1,minWidth:200}} placeholder="🔍 Medicine ya fertilizer search karein..." value={search} onChange={e=>setSearch(e.target.value)}/>
        <div style={{display:'flex',gap:6,flexWrap:'wrap'}}>
          {cats.map(c=>(
            <button key={c} onClick={()=>setCat(c)} style={{padding:'8px 13px',borderRadius:8,fontSize:12.5,fontWeight:700,border:`2px solid ${cat===c?'var(--g4)':'var(--br)'}`,background:cat===c?'var(--gp)':'none',color:cat===c?'var(--g3)':'var(--tx2)',cursor:'pointer',fontFamily:"'Outfit',sans-serif",whiteSpace:'nowrap'}}>
              {c}
            </button>
          ))}
        </div>
      </div>

      {/* Products */}
      <div className="mkt-grid">
        {filtered.map(p=>(
          <div key={p.id} className="mkt-card">
            <div className="mkt-img">{p.emoji}</div>
            <div className="mkt-body">
              {p.ai&&<div className="mkt-ai">🤖 AI Recommended</div>}
              <div className="mkt-nm">{p.name}</div>
              <div className="mkt-type">{p.type} • ⭐ {p.rating}</div>
              <div style={{fontSize:12,color:'var(--tx3)',marginBottom:8}}>{p.desc}</div>
              <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:10}}>
                <div className="mkt-pr">₹{p.price}<span style={{fontFamily:'Outfit',fontSize:11,color:'var(--tx3)',fontWeight:500}}>/{p.unit}</span></div>
                <span style={{fontSize:11,fontWeight:700,color:p.stock==='In Stock'?'var(--g4)':'var(--a2)'}}>{p.stock==='In Stock'?'✅':'⚠️'} {p.stock}</span>
              </div>
              <button className="btn btn-g btn-sm" style={{width:'100%'}} onClick={()=>addCart(p)}>🛒 Cart Mein Add</button>
            </div>
          </div>
        ))}
      </div>

      {cart.length>0&&(
        <div className="cart-badge" onClick={()=>toast(`Cart: ${cart.length} items — Checkout coming soon!`,'inf')}>
          🛒 {cart.length} items • ₹{cart.reduce((a,b)=>a+b.price,0)} — Checkout →
        </div>
      )}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   7. INSURANCE CLAIM 🏦
════════════════════════════════════════════════════════════════ */
function InsurancePage({user,nav,toast}) {
  const [applying,setApplying]=useState(false);
  const [applied,setApplied]=useState(false);
  const [form,setForm]=useState({scheme:'PMFBY',crop:'',area:'',bank:'',accountNo:'',aadhaar:''});

  const applyInsurance=async()=>{
    if(!form.crop||!form.area){toast('Crop aur area fill karo','err');return;}
    setApplying(true);
    try{
      await API.post('/api/consultations',{
        cropId:'insurance', cropName:'Insurance: '+form.crop, cropEmoji:'🏦',
        method:'manual', disease:'PMFBY Application',
        confidence:100, severity:1,
        answers:{...form, userName:user?.name, district:user?.district},
      });
      setApplied(true);
      toast('Insurance application submit ho gayi! Krishi vibhag se confirmation aayegi. ✅');
    }catch(e){toast('Submit fail hua','err');}
    setApplying(false);
  };

  const SCHEMES=[
    {id:'pmfby',name:'PMFBY — Pradhan Mantri Fasal Bima Yojana',premium:'2% (Kharif)',cover:'Full crop loss',apply:'pm-kisan.gov.in'},
    {id:'wbcis',name:'WBCIS — Weather Based Crop Insurance',premium:'2% (Kharif)',cover:'Rainfall based',apply:'aicofindia.com'},
    {id:'unified',name:'Unified Package Insurance Scheme',premium:'Nominal',cover:'Life + Assets + Crop',apply:'agricoop.nic.in'},
    {id:'coconut',name:'Coconut Palm Insurance Scheme',premium:'₹14.83/palm/year',cover:'Per palm damage',apply:'cpcri.res.in'},
  ];
  const [step,setStep]=useState(1);
  const [claimData,setClaimData]=useState({crop:'',date:'',damage:'',area:''});

  const steps=[
    {n:1,l:'AI Verification',s:'Disease confirm karein',done:step>1,active:step===1},
    {n:2,l:'Photo Evidence',s:'3-5 photos upload',done:step>2,active:step===2},
    {n:3,l:'Field Assessment',s:'GPS location + area',done:step>3,active:step===3},
    {n:4,l:'Insurance Co.',s:'Claim submit karein',done:step>4,active:step===4},
    {n:5,l:'Payout',s:'Bank transfer',done:step>5,active:step===5},
  ];

  return (
    <div className="wrap-sm">
      <div style={{fontFamily:"'Baloo 2',cursive",fontSize:26,fontWeight:900,color:'var(--g1)',marginBottom:5}}>🏦 Insurance Claim</div>
      <div style={{fontSize:13,color:'var(--tx2)',marginBottom:20}}>PMFBY — Pradhan Mantri Fasal Bima Yojana • AI-verified claims</div>

      {/* Claim Status Header */}
      <div className="claim-status" style={{background:step>=4?'linear-gradient(135deg,var(--gp),var(--gpb))':'linear-gradient(135deg,var(--ap),#fff8e1)',border:`2px solid ${step>=4?'var(--g4)':'var(--a2)'}`}}>
        <div style={{fontSize:36,marginBottom:8}}>{step>=5?'🎉':step>=4?'⏳':'📋'}</div>
        <div style={{fontFamily:"'Baloo 2',cursive",fontSize:20,fontWeight:900,color:step>=4?'var(--g2)':'var(--a1)'}}>
          {step>=5?'Claim Approved! ✅':step>=4?'Review Under Process':'Claim Filing in Progress'}
        </div>
        <div style={{fontSize:13,color:'var(--tx2)',marginTop:5}}>
          {step>=5?'₹42,000 aapke account mein 3-5 din mein aayega':step>=4?'Insurance company review kar rahi hai — 5-7 working days':`Step ${step}/5 complete`}
        </div>
        {step>=4&&<div style={{marginTop:12,fontFamily:"'Baloo 2',cursive",fontSize:28,fontWeight:900,color:'var(--g3)'}}>₹42,000</div>}
      </div>

      {/* Progress Steps */}
      <div style={{marginBottom:22}}>
        {steps.map(s=>(
          <div key={s.n} className={`ins-step${s.active?' active':s.done?' done':''}`}>
            <div className={`ins-step-num ${s.done?'done':s.active?'active':'wait'}`}>
              {s.done?'✓':s.n}
            </div>
            <div style={{flex:1}}>
              <div style={{fontSize:14,fontWeight:700,color:s.active?'var(--g2)':s.done?'var(--g4)':'var(--tx)'}}>{s.l}</div>
              <div style={{fontSize:12,color:'var(--tx3)',marginTop:2}}>{s.s}</div>
            </div>
            {s.active&&<span className="badge bg-g" style={{fontSize:11}}>Current</span>}
            {s.done&&<span className="badge bg-g" style={{fontSize:11}}>Done ✅</span>}
          </div>
        ))}
      </div>

      {/* Active Step Content */}
      {step===1&&(
        <div className="card" style={{padding:22,marginBottom:18}}>
          <div style={{fontSize:15,fontWeight:800,color:'var(--g1)',marginBottom:14}}>🤖 AI Disease Verification</div>
          <div style={{padding:14,background:'var(--gp)',borderRadius:10,marginBottom:14}}>
            <div style={{fontSize:13.5,fontWeight:700,color:'var(--g2)',marginBottom:4}}>✅ AI Report Available</div>
            <div style={{fontSize:13,color:'var(--tx2)'}}>🍅 Tomato — Early Blight • Conf: 94% • Severity: Stage 2/5 • Affected Area: 23%</div>
          </div>
          <div className="frow">
            <div className="fgrp"><label className="flbl">Crop</label>
              <select className="fsel" value={claimData.crop} onChange={e=>setClaimData(p=>({...p,crop:e.target.value}))}>
                <option value="">Select</option><option>Tomato</option><option>Wheat</option><option>Cotton</option>
              </select>
            </div>
            <div className="fgrp"><label className="flbl">Damage Date</label>
              <input className="finp" type="date" value={claimData.date} onChange={e=>setClaimData(p=>({...p,date:e.target.value}))}/>
            </div>
          </div>
          <div className="frow">
            <div className="fgrp"><label className="flbl">Damage %</label>
              <select className="fsel" value={claimData.damage} onChange={e=>setClaimData(p=>({...p,damage:e.target.value}))}>
                <option value="">Select</option><option>10-25%</option><option>25-50%</option><option>50-75%</option><option>75%+</option>
              </select>
            </div>
            <div className="fgrp"><label className="flbl">Affected Area (Acres)</label>
              <input className="finp" type="number" placeholder="e.g. 1.5" value={claimData.area} onChange={e=>setClaimData(p=>({...p,area:e.target.value}))}/>
            </div>
          </div>
          <button className="btn btn-g btn-full" onClick={()=>{setStep(2);toast('Step 1 complete! ✅');}}>Next: Photo Upload →</button>
        </div>
      )}
      {step===2&&(
        <div className="card" style={{padding:22,marginBottom:18}}>
          <div style={{fontSize:15,fontWeight:800,color:'var(--g1)',marginBottom:14}}>📸 Photo Evidence Upload</div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:10,marginBottom:14}}>
            {['Affected Area 1','Affected Area 2','Field Overview'].map((l,i)=>(
              <div key={l} onClick={()=>toast(`${l} uploaded ✅`)} style={{height:80,background:'linear-gradient(135deg,var(--gp),var(--gpb))',borderRadius:10,border:'2px dashed var(--br2)',display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',cursor:'pointer',gap:4}}>
                <div style={{fontSize:20}}>📷</div>
                <div style={{fontSize:10,fontWeight:600,color:'var(--tx3)',textAlign:'center'}}>{l}</div>
              </div>
            ))}
          </div>
          <div style={{padding:12,background:'var(--ap)',borderRadius:9,fontSize:12.5,color:'var(--a1)',fontWeight:600,marginBottom:14}}>
            💡 Minimum 3 photos required: Affected leaves, field view, GPS timestamp
          </div>
          <div style={{display:'flex',gap:9}}>
            <button className="btn btn-ghost btn-md" style={{flex:1}} onClick={()=>setStep(1)}>← Wapas</button>
            <button className="btn btn-g btn-md" style={{flex:2}} onClick={()=>{setStep(3);toast('Photos uploaded! ✅');}}>Next: GPS Location →</button>
          </div>
        </div>
      )}
      {step===3&&(
        <div className="card" style={{padding:22,marginBottom:18}}>
          <div style={{fontSize:15,fontWeight:800,color:'var(--g1)',marginBottom:14}}>📍 Field GPS Assessment</div>
          <div style={{height:120,background:'linear-gradient(135deg,#c8e6c9,#a5d6a7)',borderRadius:10,display:'flex',alignItems:'center',justifyContent:'center',fontSize:40,marginBottom:14,cursor:'pointer'}} onClick={()=>toast('GPS location captured: 18.5912°N, 73.7389°E ✅')}>
            🗺️
          </div>
          <button className="btn btn-ghost btn-sm" style={{width:'100%',marginBottom:14}} onClick={()=>toast('GPS: 18.5912°N, 73.7389°E — Wagholi Farm ✅')}>📍 Current GPS Location Use Karo</button>
          <div style={{display:'flex',gap:9}}>
            <button className="btn btn-ghost btn-md" style={{flex:1}} onClick={()=>setStep(2)}>← Wapas</button>
            <button className="btn btn-g btn-md" style={{flex:2}} onClick={()=>{setStep(4);toast('GPS location saved! ✅');}}>Next: Submit Claim →</button>
          </div>
        </div>
      )}
      {step===4&&(
        <div className="card" style={{padding:22,marginBottom:18}}>
          <div style={{fontSize:15,fontWeight:800,color:'var(--g1)',marginBottom:14}}>📋 Claim Summary</div>
          {[['Policy Number','PMFBY-MH-2026-47821'],['Crop','Tomato — Early Blight'],['Damage','25-50% (Stage 2/5)'],['Affected Area','1.5 Acres'],['AI Confidence','94% — Verified'],['Estimated Payout','₹38,000 – ₹48,000']].map(([k,v])=>(
            <div key={k} style={{display:'flex',justifyContent:'space-between',padding:'9px 0',borderBottom:'1px solid var(--gp)',fontSize:13}}>
              <span style={{color:'var(--tx3)',fontWeight:600}}>{k}</span>
              <span style={{fontWeight:700,color:'var(--tx)'}}>{v}</span>
            </div>
          ))}
          <button className="btn btn-g btn-full" style={{marginTop:16}} onClick={()=>{setStep(5);toast('Claim submitted! Insurance company ko bhej diya gaya ✅');}}>
            ✅ Final Submit — Insurance Company Ko Bhejo
          </button>
        </div>
      )}
      {step===5&&(
        <div style={{textAlign:'center',padding:20}}>
          <div style={{fontSize:64,animation:'bounce 1.2s infinite',marginBottom:14}}>🎉</div>
          <div style={{fontFamily:"'Baloo 2',cursive",fontSize:22,fontWeight:900,color:'var(--g2)',marginBottom:8}}>Claim Approved!</div>
          <div style={{fontSize:14,color:'var(--tx2)',marginBottom:20,lineHeight:1.7}}>₹42,000 aapke Kisan Credit Card mein 3-5 working days mein transfer hoga.</div>
          <button className="btn btn-g btn-lg" onClick={()=>nav('farmer-dashboard')}>🏠 Dashboard Par Jao</button>
        </div>
      )}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   8. GOVT DISEASE SURVEILLANCE MAP 📈
════════════════════════════════════════════════════════════════ */
function GovtMapPage({nav,toast}) {
  const [selDist,setSelDist]=useState(null);
  const [hoverDist,setHoverDist]=useState(null);
  const [filter,setFilter]=useState('all');

  const districts=[
    {id:'pune',name:'Pune',cases:342,disease:'Early Blight',top:'Tomato',sev:'high',x:38,y:52,w:12,h:10},
    {id:'nashik',name:'Nashik',cases:287,disease:'Late Blight',top:'Grape',sev:'high',x:28,y:22,w:11,h:9},
    {id:'aurangabad',name:'Aurangabad',cases:198,disease:'Bacterial Wilt',top:'Cotton',sev:'med',x:52,y:38,w:10,h:8},
    {id:'nagpur',name:'Nagpur',cases:156,disease:'Leaf Rust',top:'Wheat',sev:'med',x:72,y:30,w:10,h:9},
    {id:'solapur',name:'Solapur',cases:89,disease:'Powdery Mildew',top:'Pomegranate',sev:'low',x:45,y:65,w:9,h:8},
    {id:'kolhapur',name:'Kolhapur',cases:67,disease:'Downy Mildew',top:'Grape',sev:'low',x:22,y:70,w:8,h:8},
    {id:'amravati',name:'Amravati',cases:201,disease:'Bollworm',top:'Cotton',sev:'high',x:62,y:22,w:9,h:8},
    {id:'jalgaon',name:'Jalgaon',cases:134,disease:'Fusarium',top:'Banana',sev:'med',x:32,y:12,w:9,h:8},
  ];

  const sevColor={high:'#ef5350',med:'#ffc940',low:'#4dbd7a'};
  const filtered=filter==='all'?districts:districts.filter(d=>d.sev===filter);

  return (
    <div className="wrap-md">
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',marginBottom:18,flexWrap:'wrap',gap:12}}>
        <div>
          <div style={{fontFamily:"'Baloo 2',cursive",fontSize:26,fontWeight:900,color:'var(--b1)'}}>📈 Disease Surveillance</div>
          <div style={{fontSize:13,color:'var(--tx2)',marginTop:3}}>Maharashtra — Government Portal • Real-time data</div>
        </div>
        <div style={{display:'flex',gap:7}}>
          {['all','high','med','low'].map(f=>(
            <button key={f} onClick={()=>setFilter(f)} style={{padding:'7px 13px',borderRadius:8,fontSize:12.5,fontWeight:700,border:`2px solid ${filter===f?'var(--b3)':'var(--br)'}`,background:filter===f?'var(--bp)':'none',color:filter===f?'var(--b3)':'var(--tx2)',cursor:'pointer',fontFamily:"'Outfit',sans-serif",transition:'all .18s'}}>
              {f==='all'?'All':f==='high'?'🔴 High':f==='med'?'🟡 Med':'🟢 Low'}
            </button>
          ))}
        </div>
      </div>

      {/* State Summary Cards */}
      <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:12,marginBottom:20}}>
        {[{l:'Total Cases',v:'1,574',i:'🦠',c:'var(--r2)'},{l:'Districts Affected',v:'22/36',i:'📍',c:'var(--b3)'},{l:'High Risk Areas',v:'8',i:'🔴',c:'var(--r2)'},{l:'Farmers Alerted',v:'47K+',i:'👨‍🌾',c:'var(--g3)'}].map(s=>(
          <div key={s.l} style={{padding:16,borderRadius:'var(--rad)',background:'white',border:'1.5px solid var(--br)',textAlign:'center',boxShadow:'var(--sh)'}}>
            <div style={{fontSize:22,marginBottom:5}}>{s.i}</div>
            <div style={{fontFamily:"'Baloo 2',cursive",fontSize:22,fontWeight:900,color:s.c}}>{s.v}</div>
            <div style={{fontSize:11,color:'var(--tx3)',fontWeight:600,marginTop:3}}>{s.l}</div>
          </div>
        ))}
      </div>

      {/* Map */}
      <div className="gov-map">
        <div style={{position:'absolute',inset:0,background:'linear-gradient(160deg,#e8f4fd,#dbedf8)'}}/>
        <div style={{position:'absolute',top:10,left:12,fontSize:13,fontWeight:700,color:'var(--b2)',opacity:.7}}>Maharashtra State Map</div>
        {filtered.map(d=>(
          <div key={d.id} className="map-district"
            style={{left:`${d.x}%`,top:`${d.y}%`,width:`${d.w}%`,height:`${d.h}%`,background:sevColor[d.sev]+(selDist?.id===d.id?'ee':'99'),zIndex:selDist?.id===d.id?10:1}}
            onClick={()=>setSelDist(selDist?.id===d.id?null:d)}
            onMouseEnter={()=>setHoverDist(d)}
            onMouseLeave={()=>setHoverDist(null)}>
            {d.name}
            {(hoverDist?.id===d.id)&&(
              <div className="map-tooltip" style={{bottom:'110%',left:'50%',transform:'translateX(-50%)'}}>
                {d.cases} cases • {d.disease}
              </div>
            )}
          </div>
        ))}
        <div style={{position:'absolute',bottom:12,right:12,display:'flex',gap:7,background:'rgba(255,255,255,.85)',padding:'8px 12px',borderRadius:8,backdropFilter:'blur(6px)'}}>
          {[['#ef5350','High'],['#ffc940','Medium'],['#4dbd7a','Low']].map(([c,l])=>(
            <div key={l} style={{display:'flex',alignItems:'center',gap:5,fontSize:11,fontWeight:600}}>
              <div style={{width:10,height:10,borderRadius:2,background:c}}/>{l}
            </div>
          ))}
        </div>
      </div>

      {/* District Detail */}
      {selDist&&(
        <div className="card" style={{padding:22,marginBottom:18,border:`2px solid ${sevColor[selDist.sev]}`}}>
          <div style={{display:'flex',justifyContent:'space-between',marginBottom:14}}>
            <div style={{fontFamily:"'Baloo 2',cursive",fontSize:20,fontWeight:900,color:'var(--b1)'}}>{selDist.name} District</div>
            <span className={`badge ${selDist.sev==='high'?'bg-r':selDist.sev==='med'?'bg-a':'bg-g'}`}>{selDist.sev==='high'?'🔴 High Risk':selDist.sev==='med'?'🟡 Medium':'🟢 Low Risk'}</span>
          </div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:10,marginBottom:14}}>
            {[['Total Cases',selDist.cases,'🦠'],['Main Disease',selDist.disease,'🔬'],['Top Crop',selDist.top,'🌾']].map(([l,v,i])=>(
              <div key={l} style={{padding:12,background:'var(--gp)',borderRadius:10,textAlign:'center'}}>
                <div style={{fontSize:18,marginBottom:4}}>{i}</div>
                <div style={{fontSize:14,fontWeight:900,color:'var(--g1)'}}>{v}</div>
                <div style={{fontSize:11,color:'var(--tx3)',marginTop:2}}>{l}</div>
              </div>
            ))}
          </div>
          <div style={{display:'flex',gap:9}}>
            <button className="btn btn-ghost btn-sm" style={{flex:1}} onClick={()=>toast('Full district report download ho raha hai...','inf')}>📊 Full Report</button>
            <button className="btn btn-b btn-sm" style={{flex:2}} onClick={()=>toast('Advisory bheja ja raha hai district farmers ko...','inf')}>📢 Send Advisory to Farmers</button>
          </div>
        </div>
      )}

      {/* Outbreak List */}
      <div style={{fontSize:16,fontWeight:800,color:'var(--b1)',marginBottom:13}}>🚨 Active Outbreaks</div>
      {filtered.sort((a,b)=>b.cases-a.cases).map(d=>(
        <div key={d.id} className="outbreak-row" onClick={()=>setSelDist(d)}>
          <div className="outbreak-sev" style={{background:sevColor[d.sev]+'22'}}>
            <span style={{fontSize:17}}>{d.sev==='high'?'🔴':d.sev==='med'?'🟡':'🟢'}</span>
          </div>
          <div style={{flex:1}}>
            <div style={{fontSize:14,fontWeight:800,color:'var(--tx)'}}>{d.name} — {d.disease}</div>
            <div style={{fontSize:12.5,color:'var(--tx3)',marginTop:2}}>Top crop: {d.top} • {d.cases} cases reported</div>
          </div>
          <div style={{textAlign:'right'}}>
            <div style={{fontFamily:"'Baloo 2',cursive",fontSize:18,fontWeight:900,color:sevColor[d.sev]}}>{d.cases}</div>
            <div style={{fontSize:11,color:'var(--tx3)'}}>cases</div>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   ROBOT DASHBOARD 🤖
════════════════════════════════════════════════════════════════ */
const ROBOTS=[
  {id:'R01',name:'DroneBot Alpha',type:'Drone',model:'DJI Agras T40',status:'online',battery:87,signal:94,field:'Field 1',task:'Spray in progress',spray:68,flights:124,area:'2.1 Acres',lastSeen:'Just now',emoji:'🚁'},
  {id:'R02',name:'GroundBot Beta',type:'Ground',model:'TartanSense TG-1',status:'busy',battery:62,signal:88,field:'Field 2',task:'Soil scanning',spray:0,flights:0,area:'1.5 Acres',lastSeen:'3 min ago',emoji:'🤖'},
  {id:'R03',name:'DroneBot Gamma',type:'Drone',model:'ideaForge RYNO',status:'offline',battery:12,signal:0,field:'Charging',task:'Charging station',spray:0,flights:89,area:'—',lastSeen:'2 hrs ago',emoji:'🚁'},
  {id:'R04',name:'SensorBot Delta',type:'Sensor',model:'Custom IoT v2',status:'online',battery:91,signal:99,field:'All Fields',task:'Monitoring',spray:0,flights:0,area:'4.5 Acres',lastSeen:'Live',emoji:'📡'},
];

function RobotDashboard({user,nav,toast}) {
  const [robots,setRobots]=useState(ROBOTS);
  const [selRobot,setSelRobot]=useState(ROBOTS[0]||null);
  const [logs,setLogs]=useState([]);
  const [tick,setTick]=useState(0);

  // Load real robots from API
  useEffect(()=>{
    if(!user) return;
    API.get('/api/robots').then(d=>{
      if(d.robots&&d.robots.length>0){
        setRobots(d.robots.map(r=>({...r, id:r.robotId, lastSeen:r.status==='online'?'Just now':'2 hrs ago'})));
        setSelRobot(r=>d.robots[0]||r);
      }
    }).catch(()=>{});
    API.get('/api/robots/R01/logs').then(d=>{
      if(d.logs) setLogs(d.logs);
    }).catch(()=>{});
  },[user]);

  // Poll every 10s for live updates
  useEffect(()=>{
    if(!user) return;
    const iv=setInterval(()=>{
      API.get('/api/robots').then(d=>{
        if(d.robots) setRobots(d.robots.map(r=>({...r,id:r.robotId,lastSeen:r.status==='online'?'Just now':'2 hrs ago'})));
      }).catch(()=>{});
    },10000);
    return ()=>clearInterval(iv);
  },[user]);

  useEffect(()=>{const t=setInterval(()=>setTick(p=>p+1),2000);return()=>clearInterval(t);},[]);

  const sendCommand=async(robotId,command)=>{
    try{
      await API.post(`/api/robots/${robotId}/command`,{command});
      toast(`${command} command bheja gaya! ✅`);
      // Refresh
      API.get('/api/robots').then(d=>{
        if(d.robots) setRobots(d.robots.map(r=>({...r,id:r.robotId,lastSeen:r.status==='online'?'Just now':'2 hrs ago'})));
      }).catch(()=>{});
    }catch(e){toast(e.message,'err');}
  };

  const statusColor={online:'#00ff9d',busy:'#ffd700',offline:'rgba(255,255,255,.3)',error:'#ff4444'};

  return (
    <div className="rob-shell">
      <div className="rob-wrap">
        {/* Header */}
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',marginBottom:28}}>
          <div>
            <div style={{display:'flex',alignItems:'center',gap:10,marginBottom:6}}>
              <span style={{fontSize:28}}>🤖</span>
              <div style={{fontFamily:"'Baloo 2',cursive",fontSize:28,fontWeight:900,color:'#00d4ff'}}>Robot Command Center</div>
            </div>
            <div style={{fontSize:13,color:'rgba(255,255,255,.5)'}}>BeejHealth Robot Fleet — {user?.district||'Wagholi Farm'} • Live</div>
          </div>
          <div style={{display:'flex',gap:8}}>
            <button className="rob-btn ghost" onClick={()=>nav('robot-control')}>🎮 Manual Control</button>
            <button className="rob-btn primary" onClick={()=>nav('robot-spray')}>💊 Schedule Spray</button>
          </div>
        </div>

        {/* Fleet Stats */}
        <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:14,marginBottom:28}}>
          {[{n:robots.length,l:'Total Robots',c:'#00d4ff',i:'🤖'},{n:robots.filter(r=>r.status==='online').length,l:'Active Now',c:'#00ff9d',i:'✅'},{n:robots.filter(r=>r.status==='busy').length,l:'Busy',c:'#ffd700',i:'⚡'},{n:robots.filter(r=>r.status==='offline').length,l:'Offline',c:'rgba(255,255,255,.35)',i:'🔴'}].map(s=>(
            <div key={s.l} className="rob-stat">
              <div style={{fontSize:22,marginBottom:6}}>{s.i}</div>
              <div className="rob-stat-n" style={{color:s.c}}>{s.n}</div>
              <div className="rob-stat-l">{s.l}</div>
            </div>
          ))}
        </div>

        <div style={{display:'grid',gridTemplateColumns:'1fr 1.4fr',gap:20}}>
          {/* Robot List */}
          <div>
            <div style={{fontSize:14,fontWeight:700,color:'rgba(255,255,255,.6)',marginBottom:13,textTransform:'uppercase',letterSpacing:'.8px'}}>Fleet</div>
            {robots.map(r=>(
              <div key={r.robotId||r.id} className={`robot-row${(selRobot.robotId||selRobot.id)===(r.robotId||r.id)?' sel':''}`} onClick={()=>setSelRobot(r)} style={{cursor:'pointer'}}>
                <div className="robot-av" style={{background:r.status==='online'?'rgba(0,255,157,.12)':r.status==='busy'?'rgba(255,215,0,.12)':'rgba(255,255,255,.05)'}}>
                  {r.emoji}
                </div>
                <div style={{flex:1,minWidth:0}}>
                  <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:4}}>
                    <div style={{fontSize:14,fontWeight:800,color:'white',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{r.name}</div>
                    <span className={`rob-badge ${r.status}`}><span className={`rob-dot ${r.status}`}/>{r.status}</span>
                  </div>
                  <div style={{fontSize:12,color:'rgba(255,255,255,.45)'}}>{r.type} • {r.field}</div>
                  <div style={{marginTop:7}}>
                    <div style={{display:'flex',justifyContent:'space-between',fontSize:10,color:'rgba(255,255,255,.35)',marginBottom:3}}>
                      <span>🔋 {r.battery}%</span><span>📶 {r.signal}%</span>
                    </div>
                    <div className="rob-prog"><div className={`rob-prog-fill ${r.battery>50?'green':r.battery>20?'yellow':'red'}`} style={{width:`${r.battery}%`}}/></div>
                  </div>
                </div>
              </div>
            ))}
            <button className="rob-btn ghost" style={{width:'100%',marginTop:8}} onClick={()=>toast('New robot pairing — hardware connection required','inf')}>
              + Add New Robot
            </button>
          </div>

          {/* Selected Robot Detail */}
          <div style={{display:'flex',flexDirection:'column',gap:16}}>
            <div className="rob-card" style={{padding:20}}>
              <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',marginBottom:16}}>
                <div>
                  <div style={{fontSize:20,fontWeight:900,color:'white',marginBottom:4}}>{selRobot.emoji} {selRobot.name}</div>
                  <div style={{fontSize:12,color:'rgba(255,255,255,.45)'}}>{selRobot.model} • ID: {selRobot.id}</div>
                </div>
                <span className={`rob-badge ${selRobot.status}`}><span className={`rob-dot ${selRobot.status}`}/>{selRobot.status.toUpperCase()}</span>
              </div>

              <div style={{padding:13,background:'rgba(0,212,255,.07)',borderRadius:10,marginBottom:14,fontSize:13,color:'#00d4ff',fontWeight:600}}>
                📋 Current Task: {selRobot.task}
              </div>

              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:10,marginBottom:14}}>
                {[['🔋 Battery',`${selRobot.battery}%`,selRobot.battery>50?'#00ff9d':selRobot.battery>20?'#ffd700':'#ff4444'],
                  ['📶 Signal',`${selRobot.signal}%`,'#00d4ff'],
                  ['📍 Location',selRobot.field,'rgba(255,255,255,.7)'],
                  ['⏱️ Last Seen',selRobot.lastSeen,'rgba(255,255,255,.7)']].map(([l,v,c])=>(
                  <div key={l} style={{padding:11,background:'rgba(255,255,255,.04)',borderRadius:9}}>
                    <div style={{fontSize:11,color:'rgba(255,255,255,.4)',marginBottom:3}}>{l}</div>
                    <div style={{fontSize:14,fontWeight:800,color:c}}>{v}</div>
                  </div>
                ))}
              </div>

              {selRobot.status!=='offline'&&(
                <div style={{display:'flex',gap:9}}>
                  <button className="rob-btn ghost" style={{flex:1}} onClick={()=>nav('robot-camera')}>📡 Camera</button>
                  <button className="rob-btn ghost" style={{flex:1}} onClick={()=>nav('robot-map')}>🗺️ Navigate</button>
                  <button className="rob-btn danger" style={{flex:1}} onClick={()=>sendCommand(selRobot.robotId||selRobot.id,'emergency_stop')}>🛑 Stop</button>
                </div>
              )}
              {selRobot.status==='offline'&&(
                <button className="rob-btn primary" style={{width:'100%'}} onClick={()=>sendCommand(selRobot.robotId||selRobot.id,'wake_up')}>⚡ Wake Up Robot</button>
              )}
            </div>

            {/* Live Activity Log */}
            <div className="rob-card" style={{padding:18}}>
              <div style={{fontSize:13,fontWeight:700,color:'rgba(255,255,255,.55)',marginBottom:11,textTransform:'uppercase',letterSpacing:'.6px'}}>Live Log</div>
              {(logs.length>0?logs:[
                {createdAt:new Date().toISOString(),event:'System online — all sensors nominal',level:'info'},
                {createdAt:new Date(Date.now()-300000).toISOString(),event:'GPS lock acquired — 18.59°N 73.74°E',level:'info'},
                {createdAt:new Date(Date.now()-600000).toISOString(),event:'Battery charge complete — 87%',level:'info'},
              ]).slice(0,5).map((log,i)=>(
                <div key={i} style={{display:'flex',gap:9,padding:'7px 0',borderBottom:'1px solid rgba(255,255,255,.05)',fontSize:12}}>
                  <span style={{color:'rgba(255,255,255,.3)',flexShrink:0,fontFamily:'monospace'}}>{new Date(log.createdAt).toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit'})}</span>
                  <span style={{color:log.level==='warning'?'#ffd700':log.level==='error'?'#ff4444':'#00d4ff'}}>{log.event}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        {/* Quick Nav */}
        <div style={{marginTop:24,display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:12}}>
          {[{id:'robot-camera',l:'📡 Live Camera',d:'All feeds'},{id:'robot-map',l:'🗺️ Navigation',d:'Auto-pilot'},{id:'robot-control',l:'🎮 Manual Control',d:'Joystick'},{id:'robot-spray',l:'💊 Spray Scheduler',d:'Plan missions'},{id:'robot-maintenance',l:'🔋 Maintenance',d:'Battery & alerts'},{id:'robot-analytics',l:'📊 Analytics',d:'Reports'},{id:'satellite',l:'🛰️ Satellite Map',d:'NDVI view'},{id:'soil-sensors',l:'🌱 Soil Sensors',d:'IoT data'}].map(q=>(
            <div key={q.id} className="rob-card" style={{padding:16,cursor:'pointer',transition:'all .18s'}} onClick={()=>nav(q.id)}
              onMouseEnter={e=>{e.currentTarget.style.borderColor='#00d4ff';e.currentTarget.style.background='rgba(0,212,255,.07)';}}
              onMouseLeave={e=>{e.currentTarget.style.borderColor='rgba(0,212,255,.2)';e.currentTarget.style.background='rgba(255,255,255,.04)';}}>
              <div style={{fontSize:20,marginBottom:7}}>{q.l.split(' ')[0]}</div>
              <div style={{fontSize:13,fontWeight:700,color:'white'}}>{q.l.slice(3)}</div>
              <div style={{fontSize:11,color:'rgba(255,255,255,.4)',marginTop:3}}>{q.d}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
function RobotSprayPage({nav,toast}) {
  const [zones,setZones]=useState([]);
  const [loading,setLoading]=useState(true);

  useEffect(()=>{
    API.get('/api/spray-jobs').then(d=>{
      if(d.jobs&&d.jobs.length>0){
        setZones(d.jobs.map((j,i)=>({
          id:j.jobId||i, name:`${j.field} — ${j.crop}`, area:j.area,
          disease:j.disease, chem:j.chemical, dose:j.dose,
          scheduled:new Date(j.scheduledAt).toLocaleString('en-IN',{day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit'}),
          priority:j.priority, sel:j.status!=='completed', status:j.status, jobId:j.jobId,
        })));
      }
      setLoading(false);
    }).catch(()=>setLoading(false));
  },[]);

  const cancelJob=async(jobId)=>{
    try{
      await API.delete(`/api/spray-jobs/${jobId}`);
      setZones(p=>p.filter(z=>z.jobId!==jobId));
      toast('Spray job cancelled ✅');
    }catch(e){toast('Cancel fail','err');}
  };
  const [running,setRunning]=useState(false);
  const [progress,setProgress]=useState(0);

  const startSpray=async()=>{
    setRunning(true);setProgress(0);
    for(let i=1;i<=10;i++){await new Promise(r=>setTimeout(r,400));setProgress(i*10);}
    setRunning(false);toast('✅ Spray mission complete! DroneBot Alpha finished Field 1');
  };

  const prioColor={high:'#ff4444',med:'#ffd700',low:'#00ff9d'};

  return (
    <div className="rob-shell">
      <div className="rob-wrap-sm">
        <div style={{display:'flex',gap:10,alignItems:'center',marginBottom:22}}>
          <button className="rob-btn ghost" onClick={()=>nav('robot-dashboard')}>← Back</button>
          <div style={{fontFamily:"'Baloo 2',cursive",fontSize:24,fontWeight:900,color:'#00d4ff'}}>💊 Precision Spray Scheduler</div>
        </div>

        {/* Active Mission */}
        {running&&(
          <div className="rob-card rob-card-glow" style={{padding:22,marginBottom:20}}>
            <div style={{fontSize:15,fontWeight:800,color:'#00d4ff',marginBottom:14}}>🚁 Mission Active — DroneBot Alpha</div>
            <div style={{display:'flex',justifyContent:'space-between',fontSize:13,color:'rgba(255,255,255,.6)',marginBottom:8}}>
              <span>Field 1 — Tomato • Mancozeb spray</span><span>{progress}%</span>
            </div>
            <div className="rob-prog" style={{height:10,marginBottom:14}}>
              <div className="rob-prog-fill cyan" style={{width:`${progress}%`}}/>
            </div>
            <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:10,marginBottom:14}}>
              {[['Area Done',`${(progress/100*2).toFixed(1)} Acres`,'#00d4ff'],['Chemical Used',`${(progress/100*5).toFixed(1)}L`,'#00ff9d'],['ETA',`${Math.ceil((100-progress)/10)*2} min`,'#ffd700']].map(([l,v,c])=>(
                <div key={l} style={{padding:11,background:'rgba(255,255,255,.04)',borderRadius:9,textAlign:'center'}}>
                  <div style={{fontSize:11,color:'rgba(255,255,255,.4)',marginBottom:3}}>{l}</div>
                  <div style={{fontSize:16,fontWeight:900,color:c}}>{v}</div>
                </div>
              ))}
            </div>
            <button className="rob-btn danger" style={{width:'100%'}} onClick={()=>{setRunning(false);toast('Mission aborted! 🛑','err');}}>🛑 Abort Mission</button>
          </div>
        )}

        {/* AI Recommendations */}
        <div className="rob-card" style={{padding:18,marginBottom:20,borderColor:'rgba(0,255,157,.3)'}}>
          <div style={{fontSize:13,fontWeight:700,color:'#00ff9d',marginBottom:10}}>🤖 AI Spray Recommendation</div>
          <div style={{fontSize:13,color:'rgba(255,255,255,.7)',lineHeight:1.7}}>
            Weather forecast ke hisaab se <strong style={{color:'white'}}>aaj 4-6 PM ideal window</strong> hai spray ke liye. Wind speed 8 km/h (optimal), humidity 65% (good). Kal baarish expected — delay mat karein.
          </div>
        </div>

        {/* Spray Zones */}
        <div style={{fontSize:13,fontWeight:700,color:'rgba(255,255,255,.5)',marginBottom:12,textTransform:'uppercase',letterSpacing:'.7px'}}>Spray Queue</div>
        {zones.map(z=>(
          <div key={z.id} className={`spray-zone${z.sel?' active':''}`} onClick={()=>setZones(p=>p.map(x=>({...x,sel:x.id===z.id})))}>
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',marginBottom:10}}>
              <div>
                <div style={{fontSize:15,fontWeight:800,color:'white'}}>{z.name}</div>
                <div style={{fontSize:12,color:'rgba(255,255,255,.45)',marginTop:2}}>📐 {z.area} Acres • 🦠 {z.disease}</div>
              </div>
              <div style={{textAlign:'right'}}>
                <div style={{fontSize:11,fontWeight:700,color:prioColor[z.priority],textTransform:'uppercase'}}>{z.priority} priority</div>
                <div style={{fontSize:11,color:'rgba(255,255,255,.4)',marginTop:2}}>⏰ {z.scheduled}</div>
              </div>
            </div>
            <div style={{display:'flex',gap:14,fontSize:12.5}}>
              <span style={{color:'rgba(255,255,255,.6)'}}>💊 {z.chem}</span>
              <span style={{color:'rgba(255,255,255,.6)'}}>💧 {z.dose}g/L</span>
              <span style={{color:'rgba(255,255,255,.6)'}}>🤖 DroneBot Alpha</span>
            </div>
            {z.sel&&(
              <div style={{display:'flex',gap:9,marginTop:12}}>
                <button className="rob-btn ghost" style={{flex:1}} onClick={e=>{e.stopPropagation();toast('Schedule updated ✅');}}>✏️ Edit</button>
                <button className="rob-btn primary" style={{flex:2}} onClick={e=>{e.stopPropagation();startSpray();}} disabled={running}>
                  {running?'🚁 In Progress...':'🚀 Start Spray Now'}
                </button>
              </div>
            )}
          </div>
        ))}

        {/* Chemical Tank Status */}
        <div className="rob-card" style={{padding:20,marginTop:8}}>
          <div style={{fontSize:14,fontWeight:700,color:'rgba(255,255,255,.6)',marginBottom:14,textTransform:'uppercase',letterSpacing:'.7px'}}>Chemical Tank Status</div>
          {[{name:'Mancozeb 75% WP',level:72,cap:'10L'},{name:'Copper Oxychloride',level:45,cap:'5L'},{name:'Water Tank',level:88,cap:'40L'}].map(t=>(
            <div key={t.name} style={{marginBottom:13}}>
              <div style={{display:'flex',justifyContent:'space-between',fontSize:13,marginBottom:5}}>
                <span style={{color:'rgba(255,255,255,.7)',fontWeight:600}}>{t.name}</span>
                <span style={{color:t.level>50?'#00ff9d':t.level>25?'#ffd700':'#ff4444',fontWeight:700}}>{t.level}% ({t.cap})</span>
              </div>
              <div className="rob-prog"><div className={`rob-prog-fill ${t.level>50?'green':t.level>25?'yellow':'red'}`} style={{width:`${t.level}%`}}/></div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   LIVE CAMERA FEED 📡
════════════════════════════════════════════════════════════════ */
function RobotCameraPage({nav,toast}) {
  const [activeCam,setActiveCam]=useState('R01');
  const [zoom,setZoom]=useState(1);
  const [detect,setDetect]=useState(true);
  const [tick,setTick]=useState(0);
  const [cameraFeeds,setCameraFeeds]=useState([]);
  const [cameraInfo,setCameraInfo]=useState(null);
  const [detections,setDetections]=useState([
    {x:35,y:25,w:15,h:18,label:'Early Blight',conf:94,color:'#ff4444'},
    {x:62,y:48,w:12,h:14,label:'Healthy Leaf',conf:98,color:'#00ff9d'},
    {x:18,y:60,w:10,h:12,label:'Pest Damage',conf:79,color:'#ffd700'},
  ]);
  const [gpsInfo,setGpsInfo]=useState({lat:'18.591°N',lng:'73.741°E',alt:'12m',speed:'3.2m/s'});

  useEffect(()=>{
    // Load all camera feeds
    API.get('/api/camera/all').then(d=>{
      if(d.feeds&&d.feeds.length>0) setCameraFeeds(d.feeds);
    }).catch(()=>{});
    // Load active camera info
    API.get('/api/robots/R01/camera').then(d=>{
      if(d) setCameraInfo(d);
      if(d.detections&&d.detections.length>0) setDetections(d.detections);
      if(d.gps) setGpsInfo({lat:`${d.gps.lat.toFixed(3)}°N`,lng:`${d.gps.lng.toFixed(3)}°E`,alt:d.altitude,speed:d.speed});
    }).catch(()=>{});
  },[]);

  useEffect(()=>{
    if(!activeCam) return;
    API.get(`/api/robots/${activeCam}/camera`).then(d=>{
      if(d.detections) setDetections(d.detections);
      if(d.gps) setGpsInfo({lat:`${d.gps.lat.toFixed(3)}°N`,lng:`${d.gps.lng.toFixed(3)}°E`,alt:d.altitude,speed:d.speed});
    }).catch(()=>{});
  },[activeCam]);

  // Poll camera info every 5s for live updates
  useEffect(()=>{
    const iv=setInterval(()=>{
      API.get(`/api/robots/${activeCam}/camera`).then(d=>{
        if(d.detections) setDetections(d.detections);
        if(d.gps) setGpsInfo({lat:`${d.gps.lat.toFixed(3)}°N`,lng:`${d.gps.lng.toFixed(3)}°E`,alt:d.altitude,speed:d.speed});
      }).catch(()=>{});
    },5000);
    return()=>clearInterval(iv);
  },[activeCam]);

  useEffect(()=>{const t=setInterval(()=>setTick(p=>p+1),1500);return()=>clearInterval(t);},[]);

  const cameras=cameraFeeds.length>0
    ?cameraFeeds.map(f=>({id:f.robotId,name:f.robotName,type:f.primaryCam,status:f.isLive?'live':'offline',field:f.field,emoji:f.emoji}))
    :[
      {id:'R01',name:'DroneBot Alpha',type:'Drone Cam',status:'live',field:'Field 1',emoji:'🚁'},
      {id:'R02',name:'GroundBot Beta',type:'Front Cam',status:'live',field:'Field 2',emoji:'🤖'},
      {id:'R04',name:'SensorBot Delta',type:'Wide Cam',status:'live',field:'All Fields',emoji:'📡'},
    ];

  const takeSnapshot=async()=>{
    try{
      const res=await API.post(`/api/robots/${activeCam}/camera/snapshot`,{cameraId:'front'});
      toast(`Screenshot saved! ID: ${res.snapId} 📸`);
    }catch(e){toast('Screenshot save nahi hua','err');}
  };

  return (
    <div className="rob-shell">
      <div className="rob-wrap">
        <div style={{display:'flex',gap:10,alignItems:'center',marginBottom:22,flexWrap:'wrap'}}>
          <button className="rob-btn ghost" onClick={()=>nav('robot-dashboard')}>← Back</button>
          <div style={{fontFamily:"'Baloo 2',cursive",fontSize:24,fontWeight:900,color:'#00d4ff'}}>📡 Live Camera Feed</div>
          <span className="rob-badge online" style={{marginLeft:8}}><span className="rob-dot online"/>LIVE</span>
        </div>

        <div style={{display:'grid',gridTemplateColumns:'1fr 280px',gap:18}}>
          {/* Main Feed */}
          <div>
            <div className="cam-feed" style={{height:320}}>
              {/* Simulated camera view */}
              <div style={{position:'absolute',inset:0,background:`linear-gradient(${135+tick*2}deg,#0d2b0d,#1a3a1a,#0d2b1a)`,display:'flex',alignItems:'center',justifyContent:'center'}}>
                <div style={{fontSize:80,opacity:.15,filter:'blur(2px)'}}>🌿</div>
              </div>
              <div className="cam-scanline"/>
              <div className="cam-overlay-tl"/><div className="cam-overlay-tr"/>
              <div className="cam-overlay-bl"/><div className="cam-overlay-br"/>
              <div className="cam-rec"><span className="cam-rec-dot"/>REC • {cameras.find(c=>c.id===activeCam)?.name}</div>

              {/* AI Detection Boxes */}
              {detect&&detections.map((d,i)=>(
                <div key={i} style={{position:'absolute',left:`${d.x}%`,top:`${d.y}%`,width:`${d.w}%`,height:`${d.h}%`,border:`2px solid ${d.color}`,borderRadius:4,zIndex:6}}>
                  <div style={{position:'absolute',top:-20,left:0,background:d.color+'cc',padding:'2px 6px',borderRadius:4,fontSize:10,fontWeight:700,color:d.color==='#ff4444'?'white':'#0a0f1e',whiteSpace:'nowrap'}}>
                    {d.label} {d.conf}%
                  </div>
                </div>
              ))}

              {/* HUD Overlay */}
              <div style={{position:'absolute',bottom:12,left:12,right:12,display:'flex',justifyContent:'space-between',zIndex:7}}>
                <div style={{fontSize:11,color:'#00d4ff',fontFamily:'monospace',background:'rgba(0,0,0,.55)',padding:'4px 8px',borderRadius:6}}>
                  ALT: {gpsInfo.alt} | SPD: {gpsInfo.speed} | GPS: {gpsInfo.lat}
                </div>
                <div style={{fontSize:11,color:'#00ff9d',fontFamily:'monospace',background:'rgba(0,0,0,.55)',padding:'4px 8px',borderRadius:6}}>
                  {new Date().toLocaleTimeString()}
                </div>
              </div>
            </div>

            {/* Controls */}
            <div style={{display:'flex',gap:10,marginTop:12,flexWrap:'wrap'}}>
              <button className={`rob-btn ${detect?'primary':'ghost'}`} onClick={()=>setDetect(v=>!v)}>
                🎯 AI Detection {detect?'ON':'OFF'}
              </button>
              <button className="rob-btn ghost" onClick={()=>setZoom(v=>Math.min(v+0.5,4))}>🔍 Zoom In ({zoom}x)</button>
              <button className="rob-btn ghost" onClick={()=>setZoom(v=>Math.max(v-0.5,1))}>🔎 Zoom Out</button>
              <button className="rob-btn ghost" onClick={takeSnapshot}>📸 Screenshot</button>
              <button className="rob-btn ghost" onClick={()=>toast('Recording started 🔴')}>⏺️ Record</button>
            </div>

            {/* Detection Log */}
            {detect&&(
              <div className="rob-card" style={{padding:16,marginTop:14}}>
                <div style={{fontSize:13,fontWeight:700,color:'rgba(255,255,255,.55)',marginBottom:11,textTransform:'uppercase',letterSpacing:'.6px'}}>AI Detections ({detections.length})</div>
                {detections.map((d,i)=>(
                  <div key={i} style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'8px 0',borderBottom:'1px solid rgba(255,255,255,.06)'}}>
                    <div style={{display:'flex',gap:9,alignItems:'center'}}>
                      <div style={{width:10,height:10,borderRadius:2,background:d.color,flexShrink:0}}/>
                      <span style={{fontSize:13,color:'white',fontWeight:600}}>{d.label}</span>
                    </div>
                    <div style={{display:'flex',gap:10,alignItems:'center'}}>
                      <span style={{fontSize:13,fontWeight:800,color:d.color}}>{d.conf}%</span>
                      {d.label!=='Healthy Leaf'&&<button className="rob-btn ghost" style={{padding:'4px 10px',fontSize:11}} onClick={()=>nav('robot-spray')}>Spray →</button>}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Camera List */}
          <div>
            <div style={{fontSize:13,fontWeight:700,color:'rgba(255,255,255,.5)',marginBottom:12,textTransform:'uppercase',letterSpacing:'.7px'}}>All Cameras</div>
            {cameras.map(c=>(
              <div key={c.id} className={`robot-row${activeCam===c.id?' sel':''}`} onClick={()=>setActiveCam(c.id)} style={{marginBottom:8,cursor:'pointer'}}>
                <div style={{fontSize:22}}>{c.emoji}</div>
                <div style={{flex:1}}>
                  <div style={{fontSize:13,fontWeight:700,color:'white'}}>{c.name}</div>
                  <div style={{fontSize:11,color:'rgba(255,255,255,.4)'}}>{c.type} • {c.field}</div>
                  <span className="rob-badge online" style={{marginTop:5,fontSize:10}}><span className="rob-dot online"/>LIVE</span>
                </div>
              </div>
            ))}

            {/* Quick Stats from active camera */}
            <div className="rob-card" style={{padding:16,marginTop:14}}>
              <div style={{fontSize:12,fontWeight:700,color:'rgba(255,255,255,.45)',marginBottom:11,textTransform:'uppercase',letterSpacing:'.6px'}}>Stream Info</div>
              {[['Resolution','4K / 30fps'],['Bitrate','12 Mbps'],['Latency','~220ms'],['Storage','128GB (68% free)']].map(([l,v])=>(
                <div key={l} style={{display:'flex',justifyContent:'space-between',padding:'6px 0',borderBottom:'1px solid rgba(255,255,255,.05)',fontSize:12}}>
                  <span style={{color:'rgba(255,255,255,.4)'}}>{l}</span>
                  <span style={{color:'rgba(255,255,255,.8)',fontWeight:700}}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   AUTO FIELD NAVIGATION & MAPPING 🗺️
════════════════════════════════════════════════════════════════ */
function RobotMapPage({nav,toast}) {
  const [mode,setMode]=useState('auto');
  const [mission,setMission]=useState(null);
  const [robotLoc,setRobotLoc]=useState(null);
  const [selRobotId,setSelRobotId]=useState('R01');
  useEffect(()=>{
    API.get(`/api/robots/${selRobotId}/location`).then(d=>setRobotLoc(d)).catch(()=>{});
    const iv=setInterval(()=>{
      API.get(`/api/robots/${selRobotId}/location`).then(d=>setRobotLoc(d)).catch(()=>{});
    },8000);
    return()=>clearInterval(iv);
  },[selRobotId]);

  const startMission=async(modeType)=>{
    try{
      const res=await API.post(`/api/robots/${selRobotId}/navigate`,{mode:modeType,field:'Field 1'});
      setMission({mode:modeType,eta:res.eta,started:new Date().toLocaleTimeString()});
      toast(`Mission started! ETA: ${res.eta} ✅`);
    }catch(e){toast('Mission start fail','err');}
  };
  const [tick,setTick]=useState(0);
  useEffect(()=>{const t=setInterval(()=>setTick(p=>p+1),800);return()=>clearInterval(t);},[]);

  const robotX=30+Math.sin(tick*0.3)*8;
  const robotY=40+Math.cos(tick*0.25)*6;

  const fields=[
    {name:'F1',x:12,y:15,w:30,h:25,color:'rgba(255,68,68,.25)',border:'#ff4444',crop:'🍅'},
    {name:'F2',x:50,y:12,w:26,h:28,color:'rgba(0,255,157,.15)',border:'#00ff9d',crop:'🌾'},
    {name:'F3',x:18,y:52,w:22,h:24,color:'rgba(0,212,255,.15)',border:'#00d4ff',crop:'🥔'},
  ];

  const waypoints=[{x:25,y:28},{x:40,y:35},{x:55,y:42},{x:68,y:30},{x:72,y:55}];

  return (
    <div className="rob-shell">
      <div className="rob-wrap">
        <div style={{display:'flex',gap:10,alignItems:'center',marginBottom:22,flexWrap:'wrap'}}>
          <button className="rob-btn ghost" onClick={()=>nav('robot-dashboard')}>← Back</button>
          <div style={{fontFamily:"'Baloo 2',cursive",fontSize:24,fontWeight:900,color:'#00d4ff'}}>🗺️ Field Navigation</div>
          <div style={{marginLeft:'auto',display:'flex',gap:7}}>
            {['auto','manual','scan'].map(m=>(
              <button key={m} onClick={()=>setMode(m)} className={`rob-btn ${mode===m?'primary':'ghost'}`} style={{padding:'7px 14px',fontSize:12}}>
                {m==='auto'?'🤖 Auto':m==='manual'?'🎮 Manual':'🔍 Scan'}
              </button>
            ))}
          </div>
        </div>

        {/* Field Map */}
        <div className="rob-field-map" style={{marginBottom:18}}>
          <div className="rob-grid-bg"/>

          {/* Fields */}
          {fields.map(f=>(
            <div key={f.name} style={{position:'absolute',left:`${f.x}%`,top:`${f.y}%`,width:`${f.w}%`,height:`${f.h}%`,background:f.color,border:`1.5px solid ${f.border}`,borderRadius:8,display:'flex',alignItems:'center',justifyContent:'center',flexDirection:'column',gap:3,cursor:'pointer'}} onClick={()=>toast(`${f.name} selected — mission set karein`)}>
              <div style={{fontSize:20}}>{f.crop}</div>
              <div style={{fontSize:10,fontWeight:700,color:f.border}}>{f.name}</div>
            </div>
          ))}

          {/* Waypoints */}
          {waypoints.map((w,i)=>(
            <div key={i} style={{position:'absolute',left:`${w.x}%`,top:`${w.y}%`,width:10,height:10,background:'rgba(255,215,0,.7)',border:'2px solid #ffd700',borderRadius:'50%',transform:'translate(-50%,-50%)',zIndex:3}}/>
          ))}

          {/* Robot Icon (animated) */}
          <div className="rob-robot-icon" style={{left:`${robotX}%`,top:`${robotY}%`,transform:'translate(-50%,-50%)'}}>🚁</div>
          <div className="rob-ping" style={{left:`${robotX}%`,top:`${robotY}%`,transform:'translate(-50%,-50%)'}}/>

          {/* HUD */}
          <div style={{position:'absolute',top:10,left:12,background:'rgba(0,0,0,.65)',borderRadius:8,padding:'8px 12px',backdropFilter:'blur(8px)'}}>
            <div style={{fontSize:11,color:'#00d4ff',fontFamily:'monospace'}}>GPS: 18.5912°N, 73.7389°E</div>
            <div style={{fontSize:11,color:'#00ff9d',marginTop:2}}>SPD: 2.8 m/s | ALT: 8m</div>
          </div>

          {/* Legend */}
          <div style={{position:'absolute',bottom:10,right:10,background:'rgba(0,0,0,.65)',borderRadius:8,padding:'7px 10px',fontSize:10,color:'rgba(255,255,255,.6)'}}>
            🚁 DroneBot Alpha<br/>🟡 Waypoints ({waypoints.length})<br/>📐 4.5 Acres Total
          </div>
        </div>

        {/* Mission Planner */}
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:16}}>
          <div className="rob-card" style={{padding:20}}>
            <div style={{fontSize:14,fontWeight:700,color:'rgba(255,255,255,.6)',marginBottom:14,textTransform:'uppercase',letterSpacing:'.7px'}}>Mission Planner</div>
            <div style={{display:'flex',flexDirection:'column',gap:10,marginBottom:16}}>
              {[{l:'Mission Type',opts:['Spray','Scan','Survey','Monitor']},{l:'Robot',opts:['DroneBot Alpha','GroundBot Beta']},{l:'Pattern',opts:['Grid','Spiral','Custom','Perimeter']}].map(({l,opts})=>(
                <div key={l}>
                  <div style={{fontSize:11,color:'rgba(255,255,255,.4)',marginBottom:5,fontWeight:600}}>{l}</div>
                  <select style={{width:'100%',padding:'9px 12px',borderRadius:9,background:'rgba(255,255,255,.07)',border:'1px solid rgba(0,212,255,.2)',color:'white',fontSize:13,fontFamily:'Outfit,sans-serif',outline:'none'}}>
                    {opts.map(o=><option key={o} style={{background:'#0d1b3e'}}>{o}</option>)}
                  </select>
                </div>
              ))}
            </div>
            <button className="rob-btn primary" style={{width:'100%'}} onClick={()=>{setMission({status:'running',pct:0});toast('Mission started! 🚀');}}>
              🚀 Start Mission
            </button>
          </div>

          <div className="rob-card" style={{padding:20}}>
            <div style={{fontSize:14,fontWeight:700,color:'rgba(255,255,255,.6)',marginBottom:14,textTransform:'uppercase',letterSpacing:'.7px'}}>Mission Status</div>
            {mission?(
              <>
                <div style={{textAlign:'center',marginBottom:14}}>
                  <div style={{fontSize:36,marginBottom:7}}>🚁</div>
                  <div style={{fontSize:15,fontWeight:800,color:'#00d4ff'}}>Mission Running</div>
                  <div style={{fontSize:12,color:'rgba(255,255,255,.45)',marginTop:3}}>Waypoint 3/5 • ETA 8 min</div>
                </div>
                <div className="rob-prog" style={{height:8,marginBottom:14}}>
                  <div className="rob-prog-fill cyan" style={{width:'60%'}}/>
                </div>
                <button className="rob-btn danger" style={{width:'100%'}} onClick={()=>{setMission(null);toast('Mission cancelled 🛑','err');}}>🛑 Cancel Mission</button>
              </>
            ):(
              <div style={{textAlign:'center',padding:20,color:'rgba(255,255,255,.3)'}}>
                <div style={{fontSize:40,marginBottom:10}}>🗺️</div>
                <div style={{fontSize:13}}>No active mission.<br/>Plan karein aur start karein.</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   MANUAL ROBOT CONTROL 🎮
════════════════════════════════════════════════════════════════ */
function RobotControlPage({nav,toast}) {
  const [pressed,setPressed]=useState({});
  const [speed,setSpeed]=useState(50);
  const [sprayOn,setSprayOn]=useState(false);
  const [camTilt,setCamTilt]=useState(0);
  const [selRobot,setSelRobot]=useState('R01');
  const [altitude,setAltitude]=useState(8);
  const [cmdLog,setCmdLog]=useState([]);

  const sendCmd=async(command,params={})=>{
    try{
      const res=await API.post(`/api/robots/${selRobot}/command`,{command,params});
      const entry=`${new Date().toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit'})} ← ${command}`;
      setCmdLog(p=>[entry,...p].slice(0,8));
      if(res.acknowledged) toast(`${command} sent ✅`,'inf');
    }catch(e){toast(`Command fail: ${e.message}`,'err');}
  };

  const press=(key)=>setPressed(p=>({...p,[key]:true}));
  const release=(key)=>setPressed(p=>({...p,[key]:false}));

  const dirs=[
    {k:'up',l:'↑',r:0,c:1},{k:'left',l:'←',r:1,c:0},
    {k:'down',l:'↓',r:2,c:1},{k:'right',l:'→',r:1,c:2}
  ];

  return (
    <div className="rob-shell">
      <div className="rob-wrap-sm">
        <div style={{display:'flex',gap:10,alignItems:'center',marginBottom:22}}>
          <button className="rob-btn ghost" onClick={()=>nav('robot-dashboard')}>← Back</button>
          <div style={{fontFamily:"'Baloo 2',cursive",fontSize:24,fontWeight:900,color:'#00d4ff'}}>🎮 Manual Control</div>
        </div>

        {/* Robot Selector */}
        <div style={{display:'flex',gap:9,marginBottom:22}}>
          {ROBOTS.filter(r=>r.status!=='offline').map(r=>(
            <button key={r.id} onClick={()=>setSelRobot(r.id)} className={`rob-btn ${selRobot===r.id?'primary':'ghost'}`} style={{flex:1,padding:'10px 8px',fontSize:12.5}}>
              {r.emoji} {r.name.split(' ')[0]}
            </button>
          ))}
        </div>

        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:18}}>
          {/* Joystick */}
          <div className="rob-card" style={{padding:24,textAlign:'center'}}>
            <div style={{fontSize:13,fontWeight:700,color:'rgba(255,255,255,.5)',marginBottom:20,textTransform:'uppercase',letterSpacing:'.7px'}}>Movement</div>
            <div style={{display:'grid',gridTemplateColumns:'repeat(3,44px)',gap:8,justifyContent:'center',margin:'0 auto 20px'}}>
              {Array.from({length:9},(_,i)=>{
                const r=Math.floor(i/3),c=i%3;
                const dir=dirs.find(d=>d.r===r&&d.c===c);
                if(!dir&&!(r===1&&c===1)) return <div key={i}/>;
                if(r===1&&c===1) return (
                  <div key={i} style={{width:44,height:44,borderRadius:10,background:'rgba(0,212,255,.08)',border:'1px solid rgba(0,212,255,.2)',display:'flex',alignItems:'center',justifyContent:'center',fontSize:16}}>🤖</div>
                );
                return (
                  <button key={i} className={`joy-dir-btn${pressed[dir.k]?' pressed':''}`}
                    onMouseDown={()=>{press(dir.k);sendCmd('move',{direction:dir.k,speed});}}
                    onMouseUp={()=>release(dir.k)}
                    onTouchStart={()=>press(dir.k)}
                    onTouchEnd={()=>release(dir.k)}>
                    {dir.l}
                  </button>
                );
              })}
            </div>

            <div style={{marginBottom:14}}>
              <div style={{display:'flex',justifyContent:'space-between',fontSize:12,color:'rgba(255,255,255,.5)',marginBottom:6}}>
                <span>Speed</span><span style={{color:'#00d4ff',fontWeight:700}}>{speed}%</span>
              </div>
              <input type="range" min={10} max={100} value={speed} onChange={e=>setSpeed(+e.target.value)}
                style={{width:'100%',accentColor:'#00d4ff'}}/>
            </div>

            {ROBOTS.find(r=>(r.robotId||r.id)===selRobot)?.type==='Drone'&&(
              <div style={{marginBottom:14}}>
                <div style={{display:'flex',justifyContent:'space-between',fontSize:12,color:'rgba(255,255,255,.5)',marginBottom:6}}>
                  <span>Altitude</span><span style={{color:'#ffd700',fontWeight:700}}>{altitude}m</span>
                </div>
                <input type="range" min={2} max={30} value={altitude} onChange={e=>setAltitude(+e.target.value)}
                  style={{width:'100%',accentColor:'#ffd700'}}/>
              </div>
            )}

            <button className={`rob-btn ${sprayOn?'danger':'green'}`} style={{width:'100%'}} onClick={()=>{setSprayOn(v=>!v);sendCmd(sprayOn?'spray_stop':'spray_start');}}>
              {sprayOn?'🛑 Stop Spray':'💧 Start Spray'}
            </button>
          </div>

          {/* Status Panel */}
          <div style={{display:'flex',flexDirection:'column',gap:14}}>
            <div className="rob-card" style={{padding:18}}>
              <div style={{fontSize:13,fontWeight:700,color:'rgba(255,255,255,.5)',marginBottom:12,textTransform:'uppercase',letterSpacing:'.7px'}}>Robot Status</div>
              {[['Robot',ROBOTS.find(r=>(r.robotId||r.id)===selRobot)?.name||'—','white'],['Battery',`${ROBOTS.find(r=>(r.robotId||r.id)===selRobot)?.battery}%`,'#00ff9d'],['Speed',`${speed}%`,'#00d4ff'],['Spray',sprayOn?'Active':'Idle',sprayOn?'#00ff9d':'rgba(255,255,255,.4)'],['Status','Connected','#00ff9d']].map(([l,v,c])=>(
                <div key={l} style={{display:'flex',justifyContent:'space-between',padding:'8px 0',borderBottom:'1px solid rgba(255,255,255,.05)',fontSize:13}}>
                  <span style={{color:'rgba(255,255,255,.45)'}}>{l}</span>
                  <span style={{fontWeight:700,color:c}}>{v}</span>
                </div>
              ))}
            </div>

            <div className="rob-card" style={{padding:18}}>
              <div style={{fontSize:13,fontWeight:700,color:'rgba(255,255,255,.5)',marginBottom:12,textTransform:'uppercase',letterSpacing:'.7px'}}>Camera</div>
              <div style={{marginBottom:10}}>
                <div style={{display:'flex',justifyContent:'space-between',fontSize:12,color:'rgba(255,255,255,.5)',marginBottom:6}}>
                  <span>Tilt</span><span style={{color:'#00d4ff',fontWeight:700}}>{camTilt}°</span>
                </div>
                <input type="range" min={-90} max={0} value={camTilt} onChange={e=>setCamTilt(+e.target.value)} style={{width:'100%',accentColor:'#00d4ff'}}/>
              </div>
              <div style={{display:'flex',gap:9}}>
                <button className="rob-btn ghost" style={{flex:1,fontSize:12}} onClick={()=>nav('robot-camera')}>📡 Live Feed</button>
                <button className="rob-btn ghost" style={{flex:1,fontSize:12}} onClick={()=>toast('Screenshot saved 📸')}>📸 Snap</button>
              </div>
            </div>

            <button className="rob-btn danger" style={{width:'100%',padding:'13px'}} onClick={()=>toast('EMERGENCY STOP! All robots halted 🛑','err')}>
              🛑 EMERGENCY STOP
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   BATTERY & MAINTENANCE 🔋
════════════════════════════════════════════════════════════════ */
function RobotMaintenancePage({nav,toast}) {
  const [selBot,setSelBot]=useState('R01');
  const [maintData,setMaintData]=useState(null);
  useEffect(()=>{
    API.get(`/api/robots/${selBot}/maintenance`).then(d=>setMaintData(d)).catch(()=>{});
  },[selBot]);
  return (
    <div className="rob-shell">
      <div className="rob-wrap-sm">
        <div style={{display:'flex',gap:10,alignItems:'center',marginBottom:22}}>
          <button className="rob-btn ghost" onClick={()=>nav('robot-dashboard')}>← Back</button>
          <div style={{fontFamily:"'Baloo 2',cursive",fontSize:24,fontWeight:900,color:'#00d4ff'}}>🔋 Battery & Maintenance</div>
        </div>

        {/* Battery Overview */}
        <div style={{display:'grid',gridTemplateColumns:'repeat(2,1fr)',gap:14,marginBottom:24}}>
          {ROBOTS.map(r=>({...r, battery: r.robotId===selBot&&maintData ? maintData.battery?.level||r.battery : r.battery})).map(r=>(
            <div key={r.id||r.robotId} onClick={()=>setSelBot(r.robotId||r.id)} className="rob-card" style={{padding:18,cursor:'pointer',borderColor:(r.robotId||r.id)===selBot?'rgba(0,212,255,.6)':r.battery<20?'rgba(255,68,68,.4)':r.battery<50?'rgba(255,215,0,.3)':'rgba(0,212,255,.2)'}}>
              <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:12}}>
                <div style={{display:'flex',gap:9,alignItems:'center'}}>
                  <span style={{fontSize:20}}>{r.emoji}</span>
                  <div>
                    <div style={{fontSize:13.5,fontWeight:800,color:'white'}}>{r.name}</div>
                    <div style={{fontSize:11,color:'rgba(255,255,255,.4)'}}>{r.model}</div>
                  </div>
                </div>
                <span className={`rob-badge ${r.status}`}><span className={`rob-dot ${r.status}`}/>{r.status}</span>
              </div>
              <div style={{display:'flex',justifyContent:'space-between',fontSize:12,marginBottom:6}}>
                <span style={{color:'rgba(255,255,255,.5)'}}>🔋 Battery</span>
                <span style={{fontWeight:800,color:r.battery>50?'#00ff9d':r.battery>20?'#ffd700':'#ff4444'}}>{r.battery}%</span>
              </div>
              <div className="rob-prog" style={{marginBottom:10}}>
                <div className={`rob-prog-fill ${r.battery>50?'green':r.battery>20?'yellow':'red'}`} style={{width:`${r.battery}%`}}/>
              </div>
              <div style={{fontSize:11.5,color:'rgba(255,255,255,.45)'}}>
                {r.battery<20?'⚠️ Charge immediately!':r.battery<50?'⚡ Charge soon':'✅ Battery good'}
              </div>
              {r.status==='offline'&&r.battery<20&&(
                <button className="rob-btn primary" style={{width:'100%',marginTop:10,fontSize:12}} onClick={()=>toast(`${r.name} charging initiated ⚡`)}>
                  ⚡ Start Charging
                </button>
              )}
            </div>
          ))}
        </div>

        {/* Maintenance Schedule */}
        <div className="rob-card" style={{padding:22,marginBottom:18}}>
          <div style={{fontSize:15,fontWeight:700,color:'rgba(255,255,255,.6)',marginBottom:16,textTransform:'uppercase',letterSpacing:'.7px'}}>Maintenance Schedule</div>
          {[
            {robot:'DroneBot Alpha',task:'Propeller inspection',due:'In 3 days',status:'upcoming',color:'#ffd700'},
            {robot:'DroneBot Alpha',task:'Spray nozzle clean',due:'Today',status:'urgent',color:'#ff4444'},
            {robot:'GroundBot Beta',task:'Wheel bearing check',due:'In 7 days',status:'upcoming',color:'#ffd700'},
            {robot:'All Robots',task:'Firmware update v2.4.1',due:'Available now',status:'available',color:'#00d4ff'},
            {robot:'DroneBot Gamma',task:'Battery replacement',due:'Overdue',status:'critical',color:'#ff4444'},
            {robot:'SensorBot Delta',task:'Sensor calibration',due:'In 14 days',status:'ok',color:'#00ff9d'},
          ].map((m,i)=>(
            <div key={i} style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'12px 0',borderBottom:'1px solid rgba(255,255,255,.06)'}}>
              <div>
                <div style={{fontSize:13.5,fontWeight:700,color:'white'}}>{m.task}</div>
                <div style={{fontSize:11.5,color:'rgba(255,255,255,.4)',marginTop:2}}>🤖 {m.robot}</div>
              </div>
              <div style={{textAlign:'right'}}>
                <div style={{fontSize:12,fontWeight:700,color:m.color}}>{m.due}</div>
                <button className="rob-btn ghost" style={{marginTop:6,padding:'4px 10px',fontSize:11}} onClick={()=>toast(`${m.task} — scheduled ✅`)}>
                  {m.status==='available'?'Update Now':'Schedule'}
                </button>
              </div>
            </div>
          ))}
        </div>

        {/* Alerts */}
        <div className="rob-card" style={{padding:20,borderColor:'rgba(255,68,68,.3)'}}>
          <div style={{fontSize:14,fontWeight:700,color:'#ff4444',marginBottom:13}}>⚠️ Active Alerts</div>
          {(maintData?.alerts?.length>0 ? maintData.alerts.map(a=>({l:a.msg,c:a.level==='critical'?'#ff4444':a.level==='warning'?'#ffd700':'#00d4ff'})) : [
            {l:'DroneBot Gamma battery critically low (12%) — charge immediately',c:'#ff4444'},
            {l:'DroneBot Alpha spray nozzle needs cleaning — affects spray quality',c:'#ffd700'},
            {l:'Firmware v2.4.1 available — performance improvements + bug fixes',c:'#00d4ff'},
          ]).map((a,i)=>(
            <div key={i} style={{display:'flex',gap:9,padding:'10px 0',borderBottom:'1px solid rgba(255,255,255,.05)',fontSize:12.5}}>
              <div style={{width:8,height:8,borderRadius:'50%',background:a.c,flexShrink:0,marginTop:4}}/>
              <span style={{color:'rgba(255,255,255,.75)',lineHeight:1.6}}>{a.l}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   ROBOT ANALYTICS 📊
════════════════════════════════════════════════════════════════ */
function RobotAnalyticsPage({nav,toast}) {
  const [analytics,setAnalytics]=useState(null);
  useEffect(()=>{
    API.get('/api/robots/analytics/summary').then(d=>setAnalytics(d)).catch(()=>{});
  },[]);

  const days=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
  const sprayData=analytics?.weeklyData?.map(d=>d.spray/150)||[1.2,2.1,1.8,2.4,1.9,0.8,2.3];
  const areaData=analytics?.weeklyData?.map(d=>d.area)||[1.0,1.8,1.5,2.0,1.7,0.6,2.1];
  const maxSpray=Math.max(...sprayData);

  return (
    <div className="rob-shell">
      <div className="rob-wrap">
        <div style={{display:'flex',gap:10,alignItems:'center',marginBottom:22,flexWrap:'wrap'}}>
          <button className="rob-btn ghost" onClick={()=>nav('robot-dashboard')}>← Back</button>
          <div style={{fontFamily:"'Baloo 2',cursive",fontSize:24,fontWeight:900,color:'#00d4ff'}}>📊 Robot Analytics</div>
        </div>

        {/* KPIs */}
        <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:14,marginBottom:24}}>
          {[{n:analytics?.sprayVolume||'—',l:'Chemical Used (Week)',c:'#00d4ff',i:'💊'},{n:analytics?.areaCovered||'—',l:'Acres Covered',c:'#00ff9d',i:'📐'},{n:`${analytics?.totalFlights||0} flights`,l:'Total Flights',c:'#ffd700',i:'🛫'},{n:`${analytics?.avgBattery||0}%`,l:'Avg Battery',c:'#b347ff',i:'🔋'}].map(s=>(
            <div key={s.l} className="rob-stat">
              <div style={{fontSize:22,marginBottom:6}}>{s.i}</div>
              <div className="rob-stat-n" style={{color:s.c}}>{s.n}</div>
              <div className="rob-stat-l">{s.l}</div>
            </div>
          ))}
        </div>

        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:18,marginBottom:18}}>
          {/* Spray Chart */}
          <div className="rob-card" style={{padding:20}}>
            <div style={{fontSize:14,fontWeight:700,color:'rgba(255,255,255,.6)',marginBottom:18,textTransform:'uppercase',letterSpacing:'.6px'}}>Chemical Usage (L/day)</div>
            <div className="rob-chart-bar">
              {sprayData.map((v,i)=>(
                <div key={i} className="rcb-col">
                  <div style={{fontSize:9,color:'rgba(255,255,255,.4)',marginBottom:3}}>{v}L</div>
                  <div className="rcb-bar" style={{height:`${(v/maxSpray)*100}%`,background:`linear-gradient(180deg,#00d4ff,#0088aa)`}}/>
                  <div className="rcb-lbl">{days[i]}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Area Chart */}
          <div className="rob-card" style={{padding:20}}>
            <div style={{fontSize:14,fontWeight:700,color:'rgba(255,255,255,.6)',marginBottom:18,textTransform:'uppercase',letterSpacing:'.6px'}}>Area Covered (Acres/day)</div>
            <div className="rob-chart-bar">
              {areaData.map((v,i)=>(
                <div key={i} className="rcb-col">
                  <div style={{fontSize:9,color:'rgba(255,255,255,.4)',marginBottom:3}}>{v}A</div>
                  <div className="rcb-bar" style={{height:`${(v/Math.max(...areaData))*100}%`,background:'linear-gradient(180deg,#00ff9d,#00aa66)'}}/>
                  <div className="rcb-lbl">{days[i]}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Per Robot Stats */}
        <div className="rob-card" style={{padding:22,marginBottom:18}}>
          <div style={{fontSize:14,fontWeight:700,color:'rgba(255,255,255,.6)',marginBottom:16,textTransform:'uppercase',letterSpacing:'.7px'}}>Per Robot Performance</div>
          <div style={{overflowX:'auto'}}>
            <table style={{width:'100%',borderCollapse:'collapse',fontSize:13}}>
              <thead>
                <tr style={{borderBottom:'1px solid rgba(255,255,255,.1)'}}>
                  {['Robot','Missions','Area (Acres)','Chemical (L)','Accuracy','Uptime'].map(h=>(
                    <th key={h} style={{padding:'8px 12px',textAlign:'left',color:'rgba(255,255,255,.4)',fontWeight:700,fontSize:11,textTransform:'uppercase',letterSpacing:'.5px'}}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[
                  ['🚁 DroneBot Alpha','18','8.2','9.1','97.4%','94%'],
                  ['🤖 GroundBot Beta','12','3.5','4.3','95.1%','88%'],
                  ['📡 SensorBot Delta','—','4.5 (scan)','—','99.8%','99%'],
                ].map((row,i)=>(
                  <tr key={i} style={{borderBottom:'1px solid rgba(255,255,255,.05)'}}>
                    {row.map((cell,j)=>(
                      <td key={j} style={{padding:'11px 12px',color:j===0?'white':'rgba(255,255,255,.7)',fontWeight:j===0?700:400}}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* AI Insights */}
        <div className="rob-card" style={{padding:20,borderColor:'rgba(0,255,157,.25)'}}>
          <div style={{fontSize:14,fontWeight:700,color:'#00ff9d',marginBottom:13}}>🤖 AI Fleet Insights</div>
          {[
            {i:'📈','msg':'Is hafte spray efficiency 12% improve hui — nozzle cleaning ka positive impact.'},
            {i:'⚡','msg':'DroneBot Alpha ko Tuesday 6 AM mission pe deploy karein — weather optimal rahega.'},
            {i:'💊','msg':'Mancozeb usage 23% zyada tha estimate se — Early Blight severity recalibrate karein.'},
            {i:'🔋','msg':'DroneBot Gamma ka battery health 78% — 6 mahine mein replacement recommend.'},
          ].map((ins,i)=>(
            <div key={i} style={{display:'flex',gap:10,padding:'10px 0',borderBottom:'1px solid rgba(255,255,255,.05)',fontSize:13,color:'rgba(255,255,255,.7)',lineHeight:1.6}}>
              <span style={{fontSize:16,flexShrink:0}}>{ins.i}</span>
              <span>{ins.msg}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   ROOT APP
════════════════════════════════════════════════════════════════ */
function BeejHealthApp() {
  const {user: savedUser} = loadSession();
  const [page,setPage]=useState(savedUser ? (savedUser.type==='expert' ? 'expert-dashboard' : 'farmer-dashboard') : 'home');
  const [user,setUser]=useState(savedUser||null);
  const [showAuth,setShowAuth]=useState(false);
  const [authMode,setAuthMode]=useState('login');
  const [onboarding,setOnboarding]=useState(false);
  const [selCrop,setSelCrop]=useState(null);
  const [qAnswers,setQAnswers]=useState({});
  const [showDD,setShowDD]=useState(false);
  const {toasts,add:toast}=useToasts();
  const isEx=user?.type==='expert';
  const [unreadCount,setUnreadCount]=useState(0);
  useEffect(()=>{
    if(!user) return;
    const checkUnread=()=>{
      API.get('/api/notifications')
        .then(d=>{
          if(d.notifications) setUnreadCount(d.notifications.filter(n=>!n.read).length);
        }).catch(()=>{});
    };
    checkUnread();
    const iv=setInterval(checkUnread,30000);
    return ()=>clearInterval(iv);
  },[user]);

  const nav=useCallback((pg)=>{
    const locked=['farmer-dashboard','expert-dashboard','my-consultations','my-farm','chat','notifications','profile','settings','booking','case-detail','ai-report','earnings','voice','satellite','forecast','soil-sensors','insurance','marketplace','robot-dashboard','robot-spray','robot-camera','robot-map','robot-control','robot-maintenance','robot-analytics'];
    if(locked.includes(pg)&&!user){setShowAuth(true);setAuthMode('login');return;}
    setPage(pg);
    window.scrollTo(0,0);
  },[user]);

  const handleAuthDone=(u)=>{
    setUser(u);setShowAuth(false);
    if(u.fresh){setOnboarding(true);}
    else{
      setPage(u.type==='expert'?'expert-dashboard':'farmer-dashboard');
      toast(`Swagat Hai, ${u.name.split(' ')[0]}! 🌱`);
      // Reset consultation state on new login
      setSelCrop(null); setQAnswers({});
    }
  };

  const handleObDone=()=>{
    setUser(p=>({...p,fresh:false}));setOnboarding(false);
    setPage(isEx?'expert-dashboard':'farmer-dashboard');toast('Setup complete! 🎉');
  };

  // Onboarding screens override
  if(onboarding&&user){
    return (<><style>{APP_CSS}</style>{isEx?<ExpertOnboarding user={user} onDone={handleObDone} setUser={setUser}/>:<FarmerOnboarding user={user} onDone={handleObDone} setUser={setUser}/>}</>);
  }

  // Nav config
  const publicLinks=[{id:'consultation',l:'Crop Consultation'},{id:'experts',l:'Experts'},{id:'support',l:'Support'}];
  const farmerLinks=[{id:'farmer-dashboard',l:'🏠 Home'},{id:'consultation',l:'🔬 Consult'},{id:'forecast',l:'📊 Forecast'},{id:'marketplace',l:'📦 Market'},{id:'robot-dashboard',l:'🤖 Robots'},{id:'experts',l:'👨‍⚕️ Experts'}];
  const expertLinks=[{id:'expert-dashboard',l:'🏠 Dashboard'},{id:'case-detail',l:'📋 Cases'},{id:'earnings',l:'💰 Earnings'},{id:'robot-dashboard',l:'🤖 Robots'},{id:'b2b',l:'💼 B2B'}];
  const links=user?(isEx?expertLinks:farmerLinks):publicLinks;

  const renderPage=()=>{
    try { switch(page){
      case 'home':             return <HomePage nav={nav} setAuth={setShowAuth} setAuthMode={setAuthMode} user={user}/>;
      case 'farmer-dashboard': return <FarmerDash user={user} nav={nav} toast={toast}/>;
      case 'expert-dashboard': return <ExpertDash user={user} nav={nav} toast={toast}/>;
      case 'consultation':     return <ConsultPage user={user} nav={nav} toast={toast} selCrop={selCrop} setSelCrop={setSelCrop} qAnswers={qAnswers} setQAnswers={setQAnswers}/>;
      case 'ai-report':        return <AIReportPage selCrop={selCrop||CROPS[0]} nav={nav} toast={toast} qAnswers={qAnswers} viewConsultId={localStorage.getItem('bh_view_consult')||localStorage.getItem('bh_latest_consult')}/>;
      case 'my-consultations': return <MyConsultPage user={user} nav={nav} toast={toast}/>;
      case 'experts':          return <ExpertsPage user={user} nav={nav} toast={toast}/>;
      case 'chat':             return <ChatPage user={user} nav={nav}/>;
      case 'my-farm':          return <MyFarmPage user={user} nav={nav} toast={toast}/>;
      case 'notifications':    return <NotifPage nav={nav} user={user}/>;
      case 'earnings':          return <EarningsPage user={user} nav={nav} toast={toast}/>;
      case 'voice':             return <VoiceInputPage user={user} nav={nav} toast={toast}/>;
      case 'satellite':         return <SatellitePage user={user} nav={nav} toast={toast}/>;
      case 'forecast':          return <ForecastPage user={user} nav={nav} toast={toast}/>;
      case 'soil-sensors':      return <SoilSensorPage user={user} nav={nav} toast={toast}/>;
      case 'b2b':               return <B2BPage nav={nav} toast={toast}/>;
      case 'marketplace':       return <MarketplacePage user={user} nav={nav} toast={toast}/>;
      case 'insurance':         return <InsurancePage user={user} nav={nav} toast={toast}/>;
      case 'govt-map':          return <GovtMapPage nav={nav} toast={toast}/>;
      case 'robot-dashboard':   return <RobotDashboard user={user} nav={nav} toast={toast}/>;
      case 'robot-spray':       return <RobotSprayPage nav={nav} toast={toast}/>;
      case 'robot-camera':      return <RobotCameraPage nav={nav} toast={toast}/>;
      case 'robot-map':         return <RobotMapPage nav={nav} toast={toast}/>;
      case 'robot-control':     return <RobotControlPage nav={nav} toast={toast}/>;
      case 'robot-maintenance': return <RobotMaintenancePage nav={nav} toast={toast}/>;
      case 'robot-analytics':   return <RobotAnalyticsPage nav={nav} toast={toast}/>;
      case 'support':          return <SupportPage toast={toast}/>;
      case 'profile':          return <ProfilePage user={user} nav={nav} toast={toast} setUser={setUser}/>;
      case 'settings':         return <SettingsPage user={user} setUser={setUser} nav={nav} toast={toast}/>;
      case 'case-detail':      return <CaseDetailPage user={user} nav={nav} toast={toast}/>;
      case 'booking':          return <BookingPage user={user} nav={nav} toast={toast}/>;
      default:                 return <HomePage nav={nav} setAuth={setShowAuth} setAuthMode={setAuthMode} user={user}/>;
    }
    } catch(err) {
      console.error('Page render error:', err);
      return (
        <div style={{textAlign:'center',padding:'60px 20px',fontFamily:'sans-serif'}}>
          <div style={{fontSize:48,marginBottom:16}}>🌿</div>
          <div style={{fontSize:18,fontWeight:700,color:'#166534',marginBottom:8}}>Page load mein error</div>
          <div style={{fontSize:13,color:'#6b7280',marginBottom:20}}>{err?.message}</div>
          <button onClick={()=>setPage('home')} style={{padding:'10px 24px',background:'#16a34a',color:'white',border:'none',borderRadius:8,fontSize:14,fontWeight:600,cursor:'pointer'}}>
            🏠 Home Par Jao
          </button>
        </div>
      );
    }
  };

  return (
    <>
      <style>{APP_CSS}</style>
      <div className="shell" onClick={()=>setShowDD(false)}>

        {/* NAVBAR */}
        <nav className="nav">
          <div className="nav-logo" onClick={()=>setPage(user?(isEx?'expert-dashboard':'farmer-dashboard'):'home')}>
            <div className="nav-logo-mark">🌱</div>
            <span className="nav-logo-txt">BeejHealth</span>
          </div>
          <div className="nav-links">
            {links.map(l=>(
              <button key={l.id} className={`nav-a${page===l.id?' on':''}`} onClick={()=>nav(l.id)}>{l.l}</button>
            ))}
          </div>
          <div className="nav-right">
            {user ? (
              <>
                <div className="nav-bell" onClick={()=>{nav('notifications');setUnreadCount(0);}}>
                  🔔{unreadCount>0&&<div className="nav-bell-dot"/>}
                </div>
                <div style={{position:'relative'}}>
                  <div className={`nav-av${isEx?' ex-av':''}`} onClick={e=>{e.stopPropagation();setShowDD(v=>!v);}}>
                    {user?.initials}
                  </div>
                  {showDD&&(
                    <div className="dd-menu" onClick={e=>e.stopPropagation()}>
                      <div className="dd-head">
                        <div className="dd-name">{user?.name||''}</div>
                        <div className="dd-sub">{isEx?'👨‍⚕️ Expert':'🌾 Farmer'} • ✅ Verified</div>
                      </div>
                      {[['👤','My Profile','profile'],['📋',isEx?'My Cases':'My Consultations',isEx?'case-detail':'my-consultations'],[isEx?'💰':'🗺️',isEx?'Earnings':'My Farm',isEx?'earnings':'my-farm'],['🔔','Notifications','notifications'],['⚙️','Settings','settings']].map(([ic,l,p])=>(
                        <div key={l} className="dd-row" onClick={()=>{nav(p);setShowDD(false);}}><span>{ic}</span>{l}</div>
                      ))}
                      <div className="dd-div"/>
                      <div className="dd-row red-row" onClick={()=>{
                        clearSession();
                        localStorage.removeItem('bh_latest_consult');
                        localStorage.removeItem('bh_latest_crop');
                        localStorage.removeItem('bh_view_consult');
                        localStorage.removeItem('bh_sel_expert');
                        setUser(null);setPage('home');setShowDD(false);
                        toast('Aap successfully logout ho gaye 👋','inf');
                      }}>🚪 Logout</div>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <>
                <button className="nav-btn-login" onClick={()=>{setAuthMode('login');setShowAuth(true);}}>Login</button>
                <button className="nav-btn-reg" onClick={()=>{setAuthMode('register');setShowAuth(true);}}>Register</button>
              </>
            )}
          </div>
        </nav>

        {/* MAIN CONTENT */}
        <main className={page==='chat'?'pg-chat':'pg'}>
          {renderPage()}
        </main>
      </div>

      {/* AUTH MODAL */}
      {showAuth&&<AuthModal mode={authMode} setMode={setAuthMode} onClose={()=>setShowAuth(false)} onDone={handleAuthDone} initType={isEx?'expert':'farmer'}/>}

      {/* TOASTS */}
      <div className="toast-wrap">
        {toasts.map(t=>(
          <div key={t.id} className={`toast${t.type==='err'?' err':t.type==='inf'?' inf':t.type==='warn'?' warn':''}`}>
            <span style={{fontSize:17}}>{t.type==='err'?'❌':t.type==='inf'?'ℹ️':t.type==='warn'?'⚠️':'✅'}</span>
            {t.msg}
          </div>
        ))}
      </div>
    </>
  );
}

export default function SafeBeejHealth() {
  return (
    <ErrorBoundary>
      <BeejHealthApp />
    </ErrorBoundary>
  );
}
