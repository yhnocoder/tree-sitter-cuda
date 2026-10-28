#include "tree_sitter/alloc.h"
#include "tree_sitter/parser.h"

#include <assert.h>
#include <string.h>
#include <wctype.h>

enum TokenType { RAW_STRING_DELIMITER, RAW_STRING_CONTENT, MACRO_LINE_END, MACRO_LINE_START };

/// The spec limits delimiters to 16 chars
#define MAX_DELIMITER_LENGTH 16

typedef struct {
    uint8_t delimiter_length;
    wchar_t delimiter[MAX_DELIMITER_LENGTH];
} Scanner;

static inline void advance(TSLexer *lexer) { lexer->advance(lexer, false); }

static inline void reset(Scanner *scanner) {
    scanner->delimiter_length = 0;
    memset(scanner->delimiter, 0, sizeof scanner->delimiter);
}

/// Scan the raw string delimiter in R"delimiter(content)delimiter"
static bool scan_raw_string_delimiter(Scanner *scanner, TSLexer *lexer) {
    if (scanner->delimiter_length > 0) {
        // Closing delimiter: must exactly match the opening delimiter.
        // We already checked this when scanning content, but this is how we
        // know when to stop. We can't stop at ", because R"""hello""" is valid.
        for (int i = 0; i < scanner->delimiter_length; ++i) {
            if (lexer->lookahead != scanner->delimiter[i]) {
                return false;
            }
            advance(lexer);
        }
        reset(scanner);
        return true;
    }

    // Opening delimiter: record the d-char-sequence up to (.
    // d-char is any basic character except parens, backslashes, and spaces.
    for (;;) {
        if (scanner->delimiter_length >= MAX_DELIMITER_LENGTH || lexer->eof(lexer) || lexer->lookahead == '\\' ||
            iswspace(lexer->lookahead)) {
            return false;
        }
        if (lexer->lookahead == '(') {
            // Rather than create a token for an empty delimiter, we fail and
            // let the grammar fall back to a delimiter-less rule.
            return scanner->delimiter_length > 0;
        }
        scanner->delimiter[scanner->delimiter_length++] = lexer->lookahead;
        advance(lexer);
    }
}

/// Scan the raw string content in R"delimiter(content)delimiter"
static bool scan_raw_string_content(Scanner *scanner, TSLexer *lexer) {
    // The progress made through the delimiter since the last ')'.
    // The delimiter may not contain ')' so a single counter suffices.
    for (int delimiter_index = -1;;) {
        // If we hit EOF, consider the content to terminate there.
        // This forms an incomplete raw_string_literal, and models the code
        // well.
        if (lexer->eof(lexer)) {
            lexer->mark_end(lexer);
            return true;
        }

        if (delimiter_index >= 0) {
            if (delimiter_index == scanner->delimiter_length) {
                if (lexer->lookahead == '"') {
                    return true;
                }
                delimiter_index = -1;
            } else {
                if (lexer->lookahead == scanner->delimiter[delimiter_index]) {
                    delimiter_index += 1;
                } else {
                    delimiter_index = -1;
                }
            }
        }

        if (delimiter_index == -1 && lexer->lookahead == ')') {
            // The content doesn't include the )delimiter" part.
            // We must still scan through it, but exclude it from the token.
            lexer->mark_end(lexer);
            delimiter_index = 0;
        }

        advance(lexer);
    }
}

void *tree_sitter_cuda_external_scanner_create() {
    Scanner *scanner = (Scanner *)ts_calloc(1, sizeof(Scanner));
    memset(scanner, 0, sizeof(Scanner));
    return scanner;
}

static bool skip_comment(TSLexer *lexer) {
    advance(lexer);
    if (lexer->lookahead == '/') {
        while (lexer->lookahead != '\n' && !lexer->eof(lexer)) {
            advance(lexer);
        }
        return true;
    }
    if (lexer->lookahead != '*') {
        return false;
    }
    advance(lexer);
    while (!lexer->eof(lexer)) {
        if (lexer->lookahead == '*') {
            advance(lexer);
            if (lexer->lookahead == '/') {
                advance(lexer);
                return true;
            }
        } else {
            advance(lexer);
        }
    }
    return false;
}

/// After the `)` of a macro call: true when only a line break, blanks and comments follow before
/// something new starts, so that a call continued on the next line (`.bar()`, `+ x`, `{`, `: a(x)`,
/// `= default`) is never taken for a macro.
static bool ends_line_before_new_item(TSLexer *lexer) {
    bool newline = false;
    for (;;) {
        if (lexer->lookahead == '\n') {
            newline = true;
            advance(lexer);
        } else if (iswspace(lexer->lookahead)) {
            advance(lexer);
        } else if (lexer->lookahead == '/') {
            if (!skip_comment(lexer)) {
                return false;
            }
        } else {
            break;
        }
    }
    if (lexer->eof(lexer)) {
        return true;
    }
    int32_t c = lexer->lookahead;
    return newline && (iswalpha(c) || c == '_' || c == '#' || c == '}' || c == '[' || c == '~');
}

/// Zero-width end of a macro call that stands on its own line without a `;`, such as
/// `REGISTER_DISPATCH(abs_stub, &abs_kernel_cuda)`.
static bool scan_macro_line_end(TSLexer *lexer) {
    lexer->mark_end(lexer);
    return ends_line_before_new_item(lexer);
}

static bool skip_quoted(TSLexer *lexer) {
    int32_t quote = lexer->lookahead;
    advance(lexer);
    while (lexer->lookahead != quote) {
        if (lexer->eof(lexer) || lexer->lookahead == '\n') {
            return false;
        }
        if (lexer->lookahead == '\\') {
            advance(lexer);
        }
        advance(lexer);
    }
    advance(lexer);
    return true;
}

/// Zero-width start of the same in a class body, such as `VECTORIZEDN_DEFINE_UNARY_OP(abs)`. It
/// looks ahead over the whole call because there, unlike at namespace scope, a call expression
/// may not start, and letting one start everywhere costs real declarations their parse.
static bool scan_macro_line_start(TSLexer *lexer) {
    while (iswspace(lexer->lookahead)) {
        lexer->advance(lexer, true);
    }
    lexer->mark_end(lexer);
    if (!iswalpha(lexer->lookahead) && lexer->lookahead != '_') {
        return false;
    }
    while (iswalnum(lexer->lookahead) || lexer->lookahead == '_') {
        advance(lexer);
    }
    while (lexer->lookahead == ' ' || lexer->lookahead == '\t') {
        advance(lexer);
    }
    if (lexer->lookahead != '(') {
        return false;
    }
    unsigned depth = 0;
    do {
        if (lexer->eof(lexer)) {
            return false;
        }
        int32_t c = lexer->lookahead;
        if (c == '"' || c == '\'') {
            if (!skip_quoted(lexer)) {
                return false;
            }
            continue;
        }
        if (c == '(') {
            depth++;
        } else if (c == ')') {
            depth--;
        } else if (c == ';' || c == '{' || c == '}') {
            return false;
        }
        advance(lexer);
    } while (depth > 0);
    return ends_line_before_new_item(lexer);
}

bool tree_sitter_cuda_external_scanner_scan(void *payload, TSLexer *lexer, const bool *valid_symbols) {
    Scanner *scanner = (Scanner *)payload;

    if (valid_symbols[RAW_STRING_DELIMITER] && valid_symbols[RAW_STRING_CONTENT]) {
        // we're in error recovery
        return false;
    }

    // Both look ahead past the token, so a failed scan must not fall through to another one.
    if (valid_symbols[MACRO_LINE_START]) {
        lexer->result_symbol = MACRO_LINE_START;
        return scan_macro_line_start(lexer);
    }

    if (valid_symbols[MACRO_LINE_END]) {
        lexer->result_symbol = MACRO_LINE_END;
        return scan_macro_line_end(lexer);
    }

    // No skipping leading whitespace: raw-string grammar is space-sensitive.
    if (valid_symbols[RAW_STRING_DELIMITER]) {
        lexer->result_symbol = RAW_STRING_DELIMITER;
        return scan_raw_string_delimiter(scanner, lexer);
    }

    if (valid_symbols[RAW_STRING_CONTENT]) {
        lexer->result_symbol = RAW_STRING_CONTENT;
        return scan_raw_string_content(scanner, lexer);
    }

    return false;
}

unsigned tree_sitter_cuda_external_scanner_serialize(void *payload, char *buffer) {
    static_assert(MAX_DELIMITER_LENGTH * sizeof(wchar_t) < TREE_SITTER_SERIALIZATION_BUFFER_SIZE,
                  "Serialized delimiter is too long!");

    Scanner *scanner = (Scanner *)payload;
    size_t size = scanner->delimiter_length * sizeof(wchar_t);
    memcpy(buffer, scanner->delimiter, size);
    return (unsigned)size;
}

void tree_sitter_cuda_external_scanner_deserialize(void *payload, const char *buffer, unsigned length) {
    assert(length % sizeof(wchar_t) == 0 && "Can't decode serialized delimiter!");

    Scanner *scanner = (Scanner *)payload;
    scanner->delimiter_length = length / sizeof(wchar_t);
    if (length > 0) {
        memcpy(&scanner->delimiter[0], buffer, length);
    }
}

void tree_sitter_cuda_external_scanner_destroy(void *payload) {
    Scanner *scanner = (Scanner *)payload;
    ts_free(scanner);
}
