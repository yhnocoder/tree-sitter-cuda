# @yhnocoder/tree-sitter-cuda

`tree-sitter-cuda.wasm`, the CUDA grammar (a superset of C++) from
[yhnocoder/tree-sitter-cuda](https://github.com/yhnocoder/tree-sitter-cuda), a fork of
[tree-sitter-grammars/tree-sitter-cuda](https://github.com/tree-sitter-grammars/tree-sitter-cuda)
that also parses CUDA as written in sglang, DeepEP and CUTLASS-based code: `__cluster_dims__`,
launch bounds behind `#if`, modifier macros such as `SGL_DEVICE`, and more.

Built with tree-sitter CLI 0.25.1 (ABI 15) for web-tree-sitter 0.25 and
`@vscode/tree-sitter-wasm` 0.3. Only the wasm and the highlight queries are in this package; for
the native bindings use the upstream `tree-sitter-cuda` package.

```js
import { Parser, Language } from "web-tree-sitter";
await Parser.init();
const cuda = await Language.load(require.resolve("@yhnocoder/tree-sitter-cuda/tree-sitter-cuda.wasm"));
```
