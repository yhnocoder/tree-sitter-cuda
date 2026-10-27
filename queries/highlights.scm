; inherits: cpp

[
  "<<<"
  ">>>"
] @punctuation.bracket

[
  "__host__"
  "__device__"
  "__global__"
  "__managed__"
  "__forceinline__"
  "__noinline__"
] @keyword.modifier

"__launch_bounds__" @keyword.modifier

[
  "__cluster_dims__"
  "__maxnreg__"
  "__align__"
] @keyword.modifier

(macro_modifier) @keyword.modifier

(pragma_macro) @keyword.directive
