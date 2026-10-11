// A long-lived twin of the cookie-cutter engine runner
// (launcher/client/src/sousChef/ops/headless.mjs) for the lab's collection
// and verification: the SAME engine load (Vite's SSR module runner over
// launcher/client's own vite.config.js, the pipeline headless.mjs and vitest
// use) and the SAME per-request steps (host.withTargetDefaults(ops, ctx) →
// engine.applyOps(state, filled, manifest, ctx) → engine.leafDiff per change),
// but ONE boot for many requests (headless.mjs boots Vite per call: ~10 s
// wall, ~7 s CPU — the collection's bottleneck with 8 workers).
//
//   node engine_server.mjs <launcher/client> <manifest.json>
//
// Protocol (souschef_lab/engine.py PersistentEngine): one JSON request per
// stdin line, {id, state, ops, ctx, out}; the output (headless.mjs's shape
// with `state`) is written to the file `out` — never the pipe, which node
// cuts at 64 KiB when it exits right after a write (docs/LAB.md §12.3) — and
// one short line goes to stdout: {id, exit} (0 every op applied, 2 at least
// one failed) or {id, exit: 1, error} (the request crashed the engine: what
// headless.mjs reports as exit 1). stdout carries nothing else. Read-only
// towards the checkout: nothing but the `out` files is written.
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import readline from 'node:readline';
import { pathToFileURL } from 'node:url';

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

const DEFAULT_TEMPLATE_ID = 'editorial'; // headless.mjs R8: a real basic template id

async function main() {
  const [clientRoot, manifestFile] = process.argv.slice(2);
  if (!clientRoot || !manifestFile) {
    process.stderr.write('usage: engine_server.mjs <launcher/client> <manifest.json>\n');
    process.exit(2);
  }
  const root = path.resolve(clientRoot);
  const manifest = JSON.parse(fs.readFileSync(manifestFile, 'utf8'));
  process.chdir(root);
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
  const engine = await runner.import('/src/sousChef/ops/engine.js');
  const host = await runner.import('/src/sousChef/host/previewApi.js');
  const say = (o) => fs.writeSync(1, `${JSON.stringify(o)}\n`);
  say({ ready: true });

  const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line.trim()) continue;
    let req;
    try {
      req = JSON.parse(line);
    } catch (e) {
      say({ id: null, exit: 1, error: `bad request line: ${e}` });
      continue;
    }
    try {
      const ctx = req.ctx || { templateId: (req.state && req.state.template && req.state.template.id) || DEFAULT_TEMPLATE_ID };
      const ops = Array.isArray(req.ops) ? req.ops : [req.ops];
      const filled = host.withTargetDefaults(ops, ctx);
      const r = engine.applyOps(req.state || {}, filled, manifest, ctx);
      const output = {
        ok: r.ok,
        results: r.results,
        changes: r.changes.map((c) => ({ slice: c.slice, written: c.written, diff: engine.leafDiff(c.before, c.after, c.slice) })),
        ...(r.claims ? { claims: r.claims } : {}),
        state: r.state,
      };
      fs.writeFileSync(req.out, JSON.stringify(output));
      say({ id: req.id, exit: output.ok ? 0 : 2 });
    } catch (e) {
      say({ id: req.id, exit: 1, error: String(e && e.stack ? e.stack : e).slice(0, 4000) });
    }
  }
  await runner.close();
  await server.close();
  process.exit(0);
}

main().catch((e) => {
  process.stderr.write(`${e && e.stack ? e.stack : e}\n`);
  process.exit(1);
});
