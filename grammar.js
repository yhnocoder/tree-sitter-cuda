const CPP = require("tree-sitter-cpp/grammar")
const { preprocessor } = require("tree-sitter-c/grammar")

// sglang, vLLM, PyTorch (C10_*), CUTLASS / CuTe and similar CUDA code bases
const MODIFIER_MACROS = [
    'SGL_DEVICE', 'SGL_DEVICE_HOST', 'SGL_HIP_INLINE', 'DINLINE',
    'DEVICE_INLINE', 'ALWAYS_INLINE', 'FORCE_INLINE', 'HOST_DEVICE', 'HOST_DEVICE_INLINE',
    'CUTLASS_DEVICE', 'CUTLASS_HOST', 'CUTLASS_HOST_DEVICE', 'CUTLASS_GLOBAL',
    'CUTE_DEVICE', 'CUTE_HOST', 'CUTE_HOST_DEVICE', 'CUTE_HOST_RTC',
    'C10_DEVICE', 'C10_HOST', 'C10_HOST_DEVICE', 'C10_ALWAYS_INLINE',
]
const PRAGMA_MACROS = ['CUTE_UNROLL', 'CUTE_NO_UNROLL', 'CUTLASS_PRAGMA_UNROLL', 'CUTLASS_PRAGMA_NO_UNROLL', 'C10_UNROLL']
const ALIGNAS_MACROS = ['CUTE_ALIGNAS', 'CUTLASS_ALIGNAS']

module.exports = grammar(CPP, {
    name: 'cuda',

    conflicts: ($, original) => original.concat([
        [$._declaration_modifiers, $._launch_config],
        [$._block_item, $.pragma_statement],
    ]),

    rules: {
        _top_level_item: (_, original) => original,

        _declaration_modifiers: ($, original) =>
            choice(
                $.launch_bounds,
                $.cluster_dims,
                $.maxnreg,
                $.macro_modifier,
                alias($.preproc_if_in_declaration_modifiers, $.preproc_if),
                alias($.preproc_ifdef_in_declaration_modifiers, $.preproc_ifdef),
                '__host__',
                prec(10, choice('__device__', '__tile__')),
                prec(10, choice('__global__', '__tile_global__')),
                '__forceinline__',
                '__noinline__',
                original
            ),

        // `__global__ void\n#if X\n__launch_bounds__(N)\n#endif\nkernel(...)`
        // Only launch configuration, and never empty, so that a plain `#if` around whole
        // declarations stays a top-level preproc_if.
        preproc_if_in_declaration_modifiers: $ => seq(
            preprocessor('if'),
            field('condition', $._preproc_expression),
            '\n',
            repeat1($._launch_config),
            field('alternative', optional(alias($.preproc_else_in_declaration_modifiers, $.preproc_else))),
            preprocessor('endif'),
        ),

        preproc_ifdef_in_declaration_modifiers: $ => seq(
            choice(preprocessor('ifdef'), preprocessor('ifndef')),
            field('name', $.identifier),
            repeat1($._launch_config),
            field('alternative', optional(alias($.preproc_else_in_declaration_modifiers, $.preproc_else))),
            preprocessor('endif'),
        ),

        preproc_else_in_declaration_modifiers: $ => seq(preprocessor('else'), repeat($._launch_config)),

        _launch_config: $ => choice($.launch_bounds, $.cluster_dims, $.maxnreg),

        _block_item: ($, original) => choice(original, $.pragma_macro),

        // `for (...)\n#pragma unroll\nfor (...)`: a pragma in front of a loop body without braces.
        // In a block the pragma stays a sibling of the statement.
        _non_case_statement: ($, original) => choice(original, $.pragma_statement),

        pragma_statement: $ => prec.dynamic(-1, seq(
            field('pragma', choice($.preproc_call, $.pragma_macro)),
            field('body', $.statement),
        )),

        alignas_qualifier: ($, original) => choice(
            original,
            seq(choice('__align__', $._alignas_macro), '(', choice($.expression, $.type_descriptor), ')'),
        ),

        delete_expression: (_, original) => prec.left(original),

        expression: ($, original) => choice(
            original,
            alias(prec(10, $.kernel_call_expression), $.call_expression),
        ),

        kernel_call_expression: ($) => prec(1,seq(
            field('function', $.expression),
            $.kernel_call_syntax,
            field('arguments', $.argument_list),
        )),

        kernel_call_syntax: $ => seq(alias(rep3('<'), '<<<'), $.expression, repeat(seq(",", $.expression)), alias(rep3('>'), '>>>')),

        type_qualifier: (_, original) => choice(
            original,
            '__shared__',
            '__global__',
            '__local__',
            '__device__',
            '__tile__',
            '__constant__',
            '__managed__',
            '__grid_constant__',
        ),

        launch_bounds: $ => seq("__launch_bounds__", "(", $.expression, optional(seq(",", $.expression, optional(seq(",", $.expression)))), ")"),

        cluster_dims: $ => seq("__cluster_dims__", "(", $.expression, repeat(seq(",", $.expression)), ")"),

        maxnreg: $ => seq("__maxnreg__", "(", $.expression, ")"),

        // Common macros that expand to declaration modifiers. They are matched by exact
        // name and only where a modifier may appear; elsewhere they stay identifiers.
        // A name pattern (`*_DEVICE`) would also swallow calls such as CHECK_DEVICE(x).
        macro_modifier: _ => choice(
            ...MODIFIER_MACROS,
            token(prec(1, /__[a-z0-9_]*_inline__/)),
        ),

        // `#pragma unroll` behind a macro, on its own line before a loop
        pragma_macro: _ => choice(...PRAGMA_MACROS),

        _alignas_macro: _ => choice(...ALIGNAS_MACROS),
    }
});

function rep3(obj) {
    return token(seq(obj, /\s*/, obj, /\s*/, obj));
}
