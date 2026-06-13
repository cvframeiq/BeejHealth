import re

with open('src/BeejHealth.jsx', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix CSS
css_replacements = [
    (r'\.rob-shell\{background:#0a0f1e;min-height:calc\(100vh - 62px\);color:white;\}', 
     r'.rob-shell{background:var(--gb);min-height:calc(100vh - 62px);color:var(--tx);}'),
    (r'\.rob-card\{background:rgba\(255,255,255,\.04\);border:1px solid rgba\(0,212,255,\.2\);border-radius:16px;\}',
     r'.rob-card{background:white;border:1.5px solid var(--br);border-radius:16px;box-shadow:var(--sh);}'),
    (r'\.rob-card-glow\{border-color:#00d4ff;box-shadow:0 0 20px rgba\(0,212,255,\.15\);\}',
     r'.rob-card-glow{border-color:var(--g4);box-shadow:var(--sh2);}'),
    (r'\.rob-badge\.online\{background:rgba\(0,255,157,\.12\);color:#00ff9d;border:1px solid rgba\(0,255,157,\.3\);\}',
     r'.rob-badge.online{background:var(--gp);color:var(--g3);border:1px solid var(--g4);}'),
    (r'\.rob-badge\.offline\{background:rgba\(255,255,255,\.06\);color:rgba\(255,255,255,\.4\);border:1px solid rgba\(255,255,255,\.1\);\}',
     r'.rob-badge.offline{background:var(--bp);color:var(--b1);border:1px solid var(--bpb);}'),
    (r'\.rob-badge\.busy\{background:rgba\(255,215,0,\.12\);color:#ffd700;border:1px solid rgba\(255,215,0,\.3\);\}',
     r'.rob-badge.busy{background:var(--ap);color:var(--a1);border:1px solid var(--a3);}'),
    (r'\.rob-badge\.error\{background:rgba\(255,68,68,\.15\);color:#ff4444;border:1px solid rgba\(255,68,68,\.3\);\}',
     r'.rob-badge.error{background:var(--rp);color:var(--r2);border:1px solid var(--rpb);}'),
    (r'\.rob-dot\.online\{background:#00ff9d;animation:roboBlink 1\.8s infinite;\}',
     r'.rob-dot.online{background:var(--g4);animation:roboBlink 1.8s infinite;}'),
    (r'\.rob-dot\.offline\{background:rgba\(255,255,255,\.3\);\}',
     r'.rob-dot.offline{background:var(--tx3);}'),
    (r'\.rob-dot\.busy\{background:#ffd700;animation:roboBlink 1s infinite;\}',
     r'.rob-dot.busy{background:var(--a2);animation:roboBlink 1s infinite;}'),
    (r'\.rob-dot\.error\{background:#ff4444;animation:roboBlink \.6s infinite;\}',
     r'.rob-dot.error{background:var(--r2);animation:roboBlink .6s infinite;}'),
    (r'\.rob-stat\{padding:18px 20px;border-radius:14px;border:1px solid rgba\(0,212,255,\.15\);background:rgba\(0,212,255,\.04\);transition:all \.2s;\}',
     r'.rob-stat{padding:18px 20px;border-radius:14px;border:1.5px solid var(--br);background:white;transition:all .2s;box-shadow:var(--sh);}'),
    (r'\.rob-stat:hover\{border-color:#00d4ff;background:rgba\(0,212,255,\.08\);\}',
     r'.rob-stat:hover{border-color:var(--br2);transform:translateY(-2px);box-shadow:var(--sh2);}')
]

for old, new in css_replacements:
    content = re.sub(old, new, content)

robot_idx = content.find('ROBOT DASHBOARD')
if robot_idx != -1:
    before = content[:robot_idx]
    after = content[robot_idx:]
    
    after = after.replace("color:'white'", "color:'var(--tx)'")
    after = after.replace("color: 'white'", "color:'var(--tx)'")
    after = after.replace('color:"white"', "color:'var(--tx)'")
    
    # fix inline rgba(255,255,255, .x) to be dark grays
    after = re.sub(r'rgba\(255,\s*255,\s*255,\s*\.([0-9]+)\)', lambda m: f'rgba(0,0,0,.{min(9, int(m.group(1))+2)})', after)
    
    after = after.replace("#00d4ff", "var(--g3)")
    after = after.replace("#00ff9d", "var(--g4)")
    after = after.replace("rgba(0,212,255,", "rgba(30,126,66,")
    after = after.replace("rgba(0,255,157,", "rgba(77,189,122,")

    content = before + after

with open('src/BeejHealth.jsx', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')
