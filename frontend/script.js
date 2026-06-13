const fs = require('fs');

let content = fs.readFileSync('src/BeejHealth.jsx', 'utf-8');

const css_replacements = [
    [/\.rob-shell\{background:#0a0f1e;min-height:calc\(100vh - 62px\);color:white;\}/g, 
     '.rob-shell{background:var(--gb);min-height:calc(100vh - 62px);color:var(--tx);}'],
    [/\.rob-card\{background:rgba\(255,255,255,\.04\);border:1px solid rgba\(0,212,255,\.2\);border-radius:16px;\}/g,
     '.rob-card{background:white;border:1.5px solid var(--br);border-radius:16px;box-shadow:var(--sh);}'],
    [/\.rob-card-glow\{border-color:#00d4ff;box-shadow:0 0 20px rgba\(0,212,255,\.15\);\}/g,
     '.rob-card-glow{border-color:var(--g4);box-shadow:var(--sh2);}'],
    [/\.rob-badge\.online\{background:rgba\(0,255,157,\.12\);color:#00ff9d;border:1px solid rgba\(0,255,157,\.3\);\}/g,
     '.rob-badge.online{background:var(--gp);color:var(--g3);border:1px solid var(--g4);}'],
    [/\.rob-badge\.offline\{background:rgba\(255,255,255,\.06\);color:rgba\(255,255,255,\.4\);border:1px solid rgba\(255,255,255,\.1\);\}/g,
     '.rob-badge.offline{background:var(--bp);color:var(--b1);border:1px solid var(--bpb);}'],
    [/\.rob-badge\.busy\{background:rgba\(255,215,0,\.12\);color:#ffd700;border:1px solid rgba\(255,215,0,\.3\);\}/g,
     '.rob-badge.busy{background:var(--ap);color:var(--a1);border:1px solid var(--a3);}'],
    [/\.rob-badge\.error\{background:rgba\(255,68,68,\.15\);color:#ff4444;border:1px solid rgba\(255,68,68,\.3\);\}/g,
     '.rob-badge.error{background:var(--rp);color:var(--r2);border:1px solid var(--rpb);}'],
    [/\.rob-dot\.online\{background:#00ff9d;animation:roboBlink 1\.8s infinite;\}/g,
     '.rob-dot.online{background:var(--g4);animation:roboBlink 1.8s infinite;}'],
    [/\.rob-dot\.offline\{background:rgba\(255,255,255,\.3\);\}/g,
     '.rob-dot.offline{background:var(--tx3);}'],
    [/\.rob-dot\.busy\{background:#ffd700;animation:roboBlink 1s infinite;\}/g,
     '.rob-dot.busy{background:var(--a2);animation:roboBlink 1s infinite;}'],
    [/\.rob-dot\.error\{background:#ff4444;animation:roboBlink \.6s infinite;\}/g,
     '.rob-dot.error{background:var(--r2);animation:roboBlink .6s infinite;}'],
    [/\.rob-stat\{padding:18px 20px;border-radius:14px;border:1px solid rgba\(0,212,255,\.15\);background:rgba\(0,212,255,\.04\);transition:all \.2s;\}/g,
     '.rob-stat{padding:18px 20px;border-radius:14px;border:1.5px solid var(--br);background:white;transition:all .2s;box-shadow:var(--sh);}'],
    [/\.rob-stat:hover\{border-color:#00d4ff;background:rgba\(0,212,255,\.08\);\}/g,
     '.rob-stat:hover{border-color:var(--br2);transform:translateY(-2px);box-shadow:var(--sh2);}']
];

for (const [old, new_str] of css_replacements) {
    content = content.replace(old, new_str);
}

const robot_idx = content.indexOf('ROBOT DASHBOARD');
if (robot_idx !== -1) {
    let before = content.substring(0, robot_idx);
    let after = content.substring(robot_idx);
    
    after = after.replace(/color:\s*'white'/g, "color:'var(--tx)'");
    after = after.replace(/color:\s*"white"/g, "color:'var(--tx)'");
    
    after = after.replace(/rgba\(\s*255\s*,\s*255\s*,\s*255\s*,\s*\.([0-9]+)\)/g, (match, p1) => {
        return "rgba(0,0,0,." + Math.min(9, parseInt(p1) + 2) + ")";
    });
    
    after = after.replace(/#00d4ff/g, "var(--g3)");
    after = after.replace(/#00ff9d/g, "var(--g4)");
    after = after.replace(/rgba\(0,212,255,/g, "rgba(30,126,66,");
    after = after.replace(/rgba\(0,255,157,/g, "rgba(77,189,122,");

    content = before + after;
}

fs.writeFileSync('src/BeejHealth.jsx', content, 'utf-8');
console.log('Done');
