const fs = require('fs');
let constants = fs.readFileSync('src/utils/constants.jsx', 'utf8');
const old = fs.readFileSync('src/BeejHealth_old__.jsx', 'utf8');
const logoMatch = old.match(/const LOGO_FULL_B64 = "[A-Za-z0-9+/=]+";/);
if (logoMatch) {
  constants += '\n\nexport ' + logoMatch[0] + '\n';
  console.log('Added LOGO_FULL_B64');
} else {
  console.log('Logo not found in old file');
}
constants = constants.replace(/function buildReportAiRows/g, 'export function buildReportAiRows');
constants = constants.replace(/function buildReportSnapshot/g, 'export function buildReportSnapshot');
fs.writeFileSync('src/utils/constants.jsx', constants);
console.log('Done.');
