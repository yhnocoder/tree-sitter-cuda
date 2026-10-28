const CPP = require("tree-sitter-cpp/grammar")
const { preprocessor } = require("tree-sitter-c/grammar")

// Symbol visibility in PyTorch; they also go between `class` and the class name.
const EXPORT_MACROS = [
    'TORCH_API', 'TORCH_PYTHON_API', 'TORCH_CUDA_CPP_API', 'TORCH_CUDA_CU_API', 'TORCH_XPU_API',
    'C10_API', 'C10_EXPORT', 'C10_IMPORT', 'C10_HIDDEN', 'C10_CUDA_API', 'C10_XPU_API',
    'AOTI_API', 'AOTI_TORCH_EXPORT', 'VISIBILITY_HIDDEN', 'PYBIND11_EXPORT',
]

// sglang, vLLM, PyTorch (C10_*), CUTLASS / CuTe and similar CUDA code bases
const MODIFIER_MACROS = [
    'SGL_DEVICE', 'SGL_DEVICE_HOST', 'SGL_HIP_INLINE', 'DINLINE',
    'DEVICE_INLINE', 'ALWAYS_INLINE', 'FORCE_INLINE', 'HOST_DEVICE', 'HOST_DEVICE_INLINE',
    'CUTLASS_DEVICE', 'CUTLASS_HOST', 'CUTLASS_HOST_DEVICE', 'CUTLASS_GLOBAL',
    'CUTE_DEVICE', 'CUTE_HOST', 'CUTE_HOST_DEVICE', 'CUTE_HOST_RTC',
    'C10_DEVICE', 'C10_HOST', 'C10_HOST_DEVICE', 'C10_ALWAYS_INLINE',
    'C10_ALWAYS_INLINE_UNLESS_MOBILE', 'C10_NOINLINE', 'GPU_LAMBDA', 'JITERATOR_HOST_DEVICE', 'MAYBE_GLOBAL',
    'METAL_FUNC', 'PYTORCH_QNNP_INTERNAL',
    ...EXPORT_MACROS,
]
const PRAGMA_MACROS = ['CUTE_UNROLL', 'CUTE_NO_UNROLL', 'CUTLASS_PRAGMA_UNROLL', 'CUTLASS_PRAGMA_NO_UNROLL', 'C10_UNROLL']
const RESTRICT_MACROS = ['C10_RESTRICT', 'RESTRICT']
const LAUNCH_BOUNDS_MACROS = ['C10_LAUNCH_BOUNDS_1', 'C10_LAUNCH_BOUNDS_2']
const ALIGNAS_MACROS = ['CUTE_ALIGNAS', 'CUTLASS_ALIGNAS']

module.exports = grammar(CPP, {
    name: 'cuda',

    externals: ($, original) => original.concat([$._macro_line_end, $._macro_line_start]),

    conflicts: ($, original) => original.concat([
        [$._block_item, $.pragma_statement],
        [$._declarator, $.type_specifier, $.expression, $._macro_statement_call],
    ]),

    rules: {
        _top_level_item: ($, original) => choice(original, alias($._macro_call_line, $.expression_statement)),

        // PyTorch's `REGISTER_DISPATCH(abs_stub, &abs_kernel_cuda)` with no `;`: the macro expands to
        // a whole declaration. The external scanner decides by the line break, not by the name.
        _macro_call_line: $ => seq(
            choice($.call_expression, alias($._macro_statement_call, $.call_expression)),
            $._macro_line_end,
        ),

        // `TORCH_ARG(double, p) = 0.5;` in PyTorch's options structs, `VECTORIZEDN_DEFINE_UNARY_OP(abs)` lines
        _field_declaration_list_item: ($, original) => choice(
            original,
            alias($._macro_field_declaration, $.field_declaration),
            alias($._macro_field_line, $.expression_statement),
        ),

        _macro_field_line: $ => seq($._macro_line_start, alias($._macro_statement_call, $.call_expression)),

        _macro_field_declaration: $ => seq(
            field('declarator', $.function_declarator),
            '=',
            field('default_value', choice($.expression, $.initializer_list)),
            ';',
        ),

        // Empty arguments and operator names too, as in `TYPED_TEST_SUITE(SmallVectorTest, SmallVectorTestTypes, );`
        // and `VECTORIZEDN_DEFINE_BINARY_OP(operator==)`
        macro_argument_list: $ => {
            const argument = choice($.expression, $.type_descriptor, $.operator_name)
            return seq('(', optional(seq(argument, repeat(seq(',', optional(argument))))), ')')
        },

        _declaration_modifiers: ($, original) =>
            choice(
                $.launch_bounds,
                $.cluster_dims,
                $.maxnreg,
                $.macro_modifier,
                '__host__',
                prec(10, choice('__device__', '__tile__')),
                prec(10, choice('__global__', '__tile_global__')),
                '__forceinline__',
                '__noinline__',
                original
            ),

        // `__global__ void\n#if X\n__launch_bounds__(N)\n#endif\nkernel(...)`: only after a modifier or the
        // type, since at the start an `#if` can also close a `case` or open a block-level #if.
        _declaration_specifiers: $ => prec.right(seq(
            optional(seq($._declaration_modifiers, repeat(choice($._declaration_modifiers, $._preproc_launch_config)))),
            field('type', $.type_specifier),
            repeat(choice($._declaration_modifiers, $._preproc_launch_config)),
        )),

        _preproc_launch_config: $ => choice(
            alias($.preproc_if_in_declaration_modifiers, $.preproc_if),
            alias($.preproc_ifdef_in_declaration_modifiers, $.preproc_ifdef),
        ),

        // `template <...>\n#if X\nC10_LAUNCH_BOUNDS_2(...)\n#endif\n__global__ void k(...)`: right after a
        // template header no other #if can start.
        template_declaration: ($, original) => seq(
            ...original.members.slice(0, -1),
            choice(...original.members.at(-1).members, alias($._kernel_definition_behind_preproc, $.function_definition)),
        ),

        _kernel_definition_behind_preproc: $ => seq(
            repeat1($._preproc_launch_config),
            $._declaration_specifiers,
            field('declarator', $._declarator),
            field('body', $.compound_statement),
        ),

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

        _block_item: ($, original) => choice(
            original,
            $.pragma_macro,
            alias($._macro_call_line, $.expression_statement),
        ),

        // `SMALL_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2);`, `DEFINE_SVE_CAST(int64_t, s64, double, f64)`:
        // only a macro takes types as arguments, so a plain call or a declaration wins whenever it parses.
        _macro_statement_call: $ => prec.dynamic(-10, seq(
            field('function', $.identifier),
            field('arguments', alias($.macro_argument_list, $.argument_list)),
        )),

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

        // `struct C10_API TensorImpl : ...`: without this the class parses as a function definition.
        _class_declaration: $ => seq(
            repeat(choice($.attribute_specifier, $.alignas_qualifier)),
            optional($.ms_declspec_modifier),
            repeat($.attribute_declaration),
            optional(alias($._export_macro, $.macro_modifier)),
            $._class_declaration_item,
        ),

        // CUDA extended lambdas: `[=] __device__ (int i) {...}`, `[] GPU_LAMBDA(scalar_t x) -> scalar_t {...}`
        lambda_expression: $ => seq(
            field('captures', $.lambda_capture_specifier),
            repeat(choice('__host__', '__device__', $.macro_modifier)),
            optional(seq(
                field('template_parameters', $.template_parameter_list),
                optional(field('constraint', $.requires_clause)),
            )),
            optional(field('declarator', $.lambda_declarator)),
            field('body', $.compound_statement),
        ),

        // PyTorch's `__ubsan_ignore_float_divide_by_zero__` and friends expand to
        // `__attribute__((no_sanitize(...)))`, also after a parameter list; C10_DEPRECATED_MESSAGE("...")
        // to `[[deprecated("...")]]`.
        attribute_specifier: ($, original) => choice(
            original,
            token(prec(1, /__ubsan_ignore_[a-z0-9_]*__/)),
            seq('C10_DEPRECATED_MESSAGE', $.argument_list),
        ),

        lambda_specifier: ($, original) => choice(original, $.attribute_specifier),

        // `TORCH_API friend bool operator==(...)`
        friend_declaration: ($, original) => choice(
            original,
            seq($.macro_modifier, 'friend', choice($.declaration, $.function_definition)),
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
            ...RESTRICT_MACROS,
        ),

        launch_bounds: $ => seq(choice("__launch_bounds__", ...LAUNCH_BOUNDS_MACROS), "(", $.expression, optional(seq(",", $.expression, optional(seq(",", $.expression)))), ")"),

        cluster_dims: $ => seq("__cluster_dims__", "(", $.expression, repeat(seq(",", $.expression)), ")"),

        maxnreg: $ => seq("__maxnreg__", "(", $.expression, ")"),

        // Common macros that expand to declaration modifiers. They are matched by exact
        // name and only where a modifier may appear; elsewhere they stay identifiers.
        // A name pattern (`*_DEVICE`) would also swallow calls such as CHECK_DEVICE(x).
        macro_modifier: _ => choice(
            ...MODIFIER_MACROS,
            token(prec(1, /__[a-z0-9_]*_inline__/)),
        ),

        _export_macro: _ => choice(...EXPORT_MACROS),

        // `#pragma unroll` behind a macro, on its own line before a loop
        pragma_macro: _ => choice(...PRAGMA_MACROS),

        _alignas_macro: _ => choice(...ALIGNAS_MACROS),
    }
});

function rep3(obj) {
    return token(seq(obj, /\s*/, obj, /\s*/, obj));
}
