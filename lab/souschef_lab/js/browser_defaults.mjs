// Dumps the launcher's own defaults for a never-saved site, as the browser
// computes them, into a JSON file (docs/LAB.md §2 "Browser defaults"):
//   pages[template][page] = getDefaultPageSections(template, page)   (pageSections.jsx;
//                           what targets.resolveSectionSlot — and so the turn
//                           context — uses for a non-home page nothing was saved for)
//   faqCategories[template] = PreviewPage.jsx getDefaultFaqForTemplate(template)
//                             (defaultBasicFaqCategories for every basic template)
// With --contexts it instead runs turnContext.buildTurnContext on lab states
// (the browser's own context, for the lab's parity tests).
// Run with the cookie-cutter launcher client as CLIENT_ROOT:
//   node browser_defaults.mjs <launcher/client> <pages.json> <out.json>
// <pages.json> = ["home", "faq", …] — the page slugs to dump. The result goes
// to <out.json>, never a pipe (a piped stdout is cut at 64 KiB when node exits
// right after the write — the headless.mjs gap, docs/LAB.md §12.3). The modules are
// loaded through launcher/client's own vite.config.js, as headless.mjs loads
// the engine (the same pipeline vitest uses). Read-only: nothing but <out.json> is written.
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { pathToFileURL } from 'node:url';

const BASIC = ['default', 'bold', 'boutique', 'cozy', 'editorial', 'elegant', 'playful', 'pluto', 'retro', 'sweet', 'vintage'];

function findVite(from) {
  let dir = path.resolve(from);
  for (;;) {
    const pkg = path.join(dir, 'node_modules', 'vite', 'package.json');
    if (fs.existsSync(pkg)) {
      const meta = JSON.parse(fs.readFileSync(pkg, 'utf8'));
      const exp = meta.exports?.['.'];
      const rel = typeof exp === 'string' ? exp : (exp?.import?.default || exp?.import || exp?.default || meta.module || meta.main);
      return pathToFileURL(path.join(path.dirname(pkg), typeof rel === 'string' ? rel : rel.default)).href;
    }
    const up = path.dirname(dir);
    if (up === dir) throw new Error(`no node_modules/vite above ${from}`);
    dir = up;
  }
}

async function main() {
  const argv = process.argv.slice(2);
  const contexts = argv[0] === '--contexts';
  const [clientRoot, inFile, outFile] = contexts ? argv.slice(1) : argv;
  if (!clientRoot || !inFile || !outFile) {
    process.stderr.write('usage: browser_defaults.mjs [--contexts] <launcher/client> <pages.json | cases.json> <out.json>\n');
    process.exit(2);
  }
  const root = path.resolve(clientRoot);
  const input = JSON.parse(fs.readFileSync(inFile, 'utf8'));
  const { createServer, createServerModuleRunner } = await import(findVite(root));
  const server = await createServer({
    configFile: path.join(root, 'vite.config.js'),
    root,
    appType: 'custom',
    logLevel: 'silent',
    server: { middlewareMode: true, hmr: false, ws: false, watch: null },
    optimizeDeps: { noDiscovery: true, include: [] },
  });
  const runner = createServerModuleRunner(server.environments.ssr, { hmr: false });
  try {
    if (contexts) {
      // --contexts: turnContext.buildTurnContext(state, ctx, extras) for each
      // {state, ctx, extras} of <cases.json> — the browser's own context for
      // a lab state (tests/test_browser_defaults.py compares state.context_of).
      const tc = await runner.import('/src/sousChef/turnContext.js');
      const outs = input.map((c) => tc.buildTurnContext(c.state, c.ctx, c.extras || {}));
      fs.writeFileSync(outFile, JSON.stringify(outs));
      return;
    }
    const pages = input;
    const ps = await runner.import('/src/data/pageSections.jsx');
    const ui = await runner.import('@breadstack/ui');
    const out = { templates: BASIC, pages: {}, faqCategories: {} };
    for (const t of BASIC) {
      out.pages[t] = {};
      for (const p of pages) out.pages[t][p] = ps.getDefaultPageSections(t, p) ?? null;
      // PreviewPage.jsx getDefaultFaqForTemplate: every basic id → defaultBasicFaqCategories.
      out.faqCategories[t] = ui.defaultBasicFaqCategories;
    }
    // A template outside the map: getDefaultPageSections' own fallback.
    out.pages['*'] = {};
    for (const p of pages) out.pages['*'][p] = ps.getDefaultPageSections('__none__', p) ?? null;
    fs.writeFileSync(outFile, JSON.stringify(out));
  } finally {
    await runner.close();
    await server.close();
  }
}

main().then(() => process.exit(0), (e) => { process.stderr.write(String(e?.stack || e) + '\n'); process.exit(1); });
