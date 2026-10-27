// Loads the built wasm with the runtime CodeBoard ships (@vscode/tree-sitter-wasm)
// and fails if it cannot be loaded or misparses a kernel launch.
// Usage: node script/check-wasm.mjs <tree-sitter-cuda.wasm>
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";

const require = createRequire(join(process.cwd(), "package.json"));
const TS = require("@vscode/tree-sitter-wasm");
const runtime = dirname(require.resolve("@vscode/tree-sitter-wasm/wasm/tree-sitter.wasm"));

await TS.Parser.init({ locateFile: (f) => join(runtime, f) });
const lang = await TS.Language.load(resolve(process.argv[2]));
const parser = new TS.Parser();
parser.setLanguage(lang);

const src = `template <int N>
__global__ void __launch_bounds__(256) k(float* x) { x[threadIdx.x] = N; }
void run(float* x, cudaStream_t s) {
  ns::k<8>
      <<<1, 256, 0, s>>>(x);
}
`;
const tree = parser.parse(src);
const launches = tree.rootNode.descendantsOfType("kernel_call_syntax");
const fail = (m) => { console.error(`check-wasm: ${m}\n${tree.rootNode.toString()}`); process.exit(1); };
if (tree.rootNode.hasError) fail("parse error");
if (launches.length !== 1) fail(`expected 1 kernel launch, got ${launches.length}`);
console.log(`ok: ABI ${lang.abiVersion}, ${lang.types?.length ?? "?"} node types`);
