# zrtl_string

String manipulation functions for Zyntax-based languages.

## Overview

Provides comprehensive string operations including concatenation, case conversion, trimming, search/replace, splitting, padding, and number parsing/formatting.

## Exported Symbols

### Basic Operations

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$String$length` | `(StringPtr) -> i64` | Get string length in bytes |
| `$String$char_count` | `(StringPtr) -> i64` | Get length in Unicode characters |
| `$String$is_empty` | `(StringPtr) -> i32` | Check if empty (1=yes, 0=no) |
| `$String$concat` | `(StringPtr, StringPtr) -> StringPtr` | Concatenate two strings |
| `$String$repeat` | `(StringPtr, i64) -> StringPtr` | Repeat string n times |
| `$String$join` | `(ArrayPtr, StringPtr) -> StringPtr` | Join string array with separator |

### Case Conversion

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$String$to_upper` | `(StringPtr) -> StringPtr` | Convert to uppercase |
| `$String$to_lower` | `(StringPtr) -> StringPtr` | Convert to lowercase |
| `$String$capitalize` | `(StringPtr) -> StringPtr` | Capitalize first character |
| `$String$to_title` | `(StringPtr) -> StringPtr` | Title case (capitalize each word) |

### Trimming

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$String$trim` | `(StringPtr) -> StringPtr` | Trim whitespace from both ends |
| `$String$trim_start` | `(StringPtr) -> StringPtr` | Trim from start only |
| `$String$trim_end` | `(StringPtr) -> StringPtr` | Trim from end only |

### Search

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$String$contains` | `(StringPtr, StringPtr) -> i32` | Check if contains substring |
| `$String$starts_with` | `(StringPtr, StringPtr) -> i32` | Check if starts with prefix |
| `$String$ends_with` | `(StringPtr, StringPtr) -> i32` | Check if ends with suffix |
| `$String$index_of` | `(StringPtr, StringPtr) -> i64` | Find first index (-1 if not found) |
| `$String$last_index_of` | `(StringPtr, StringPtr) -> i64` | Find last index (-1 if not found) |
| `$String$count` | `(StringPtr, StringPtr) -> i64` | Count occurrences of substring |

### Replace

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$String$replace` | `(StringPtr, StringPtr, StringPtr) -> StringPtr` | Replace first occurrence |
| `$String$replace_all` | `(StringPtr, StringPtr, StringPtr) -> StringPtr` | Replace all occurrences |
| `$String$remove` | `(StringPtr, StringPtr) -> StringPtr` | Remove all occurrences |

### Extraction

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$String$substring` | `(StringPtr, i64, i64) -> StringPtr` | Get substring [start, end) |
| `$String$char_at` | `(StringPtr, i64) -> StringPtr` | Get character at index |
| `$String$char_code_at` | `(StringPtr, i64) -> i32` | Get Unicode codepoint at index |
| `$String$from_char_code` | `(i32) -> StringPtr` | Create string from codepoint |
| `$String$split` | `(StringPtr, StringPtr) -> ArrayPtr` | Split by delimiter |
| `$String$lines` | `(StringPtr) -> ArrayPtr` | Split into lines |

### Padding

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$String$pad_start` | `(StringPtr, i64, StringPtr) -> StringPtr` | Pad on left to target length |
| `$String$pad_end` | `(StringPtr, i64, StringPtr) -> StringPtr` | Pad on right to target length |

### Python str methods

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$String$check` | `(StringPtr, i64) -> i32` | An `is*` predicate: alpha, alnum, digit, decimal, numeric, space, upper, lower, title, ascii, identifier, printable |
| `$String$case_map` | `(StringPtr, i64) -> StringPtr` | `swapcase` (0) or `casefold` (1) |
| `$String$justify` | `(StringPtr, i64, StringPtr, i64) -> StringPtr` | `ljust`, `rjust`, `center` or `zfill` to a width |
| `$String$find_range` | `(StringPtr, StringPtr, i64, i64, i64) -> i64` | `find`, `rfind` or `count` over a character window |
| `$String$edge_range` | `(StringPtr, StringPtr, i64, i64, i64) -> i32` | `startswith` or `endswith` over a character window |
| `$String$replace_n` | `(StringPtr, StringPtr, StringPtr, i64) -> StringPtr` | Replace the first n matches, all when n is negative |

### Parsing

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$String$parse_int` | `(StringPtr) -> i64` | Parse as integer (0 on failure) |
| `$String$parse_int_radix` | `(StringPtr, i32) -> i64` | Parse with radix (2-36) |
| `$String$parse_float` | `(StringPtr) -> f64` | Parse as float (0.0 on failure) |

### Formatting

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$String$from_int` | `(i64) -> StringPtr` | Convert integer to string |
| `$String$from_int_radix` | `(i64, i32) -> StringPtr` | Convert with radix (2, 8, 16) |
| `$String$from_float` | `(f64) -> StringPtr` | Convert float to string |
| `$String$from_float_precision` | `(f64, i32) -> StringPtr` | Convert with decimal precision |

### Comparison

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$String$compare` | `(StringPtr, StringPtr) -> i32` | Compare (-1, 0, or 1) |
| `$String$compare_ignore_case` | `(StringPtr, StringPtr) -> i32` | Compare ignoring case |
| `$String$equals` | `(StringPtr, StringPtr) -> i32` | Check equality |
| `$String$equals_ignore_case` | `(StringPtr, StringPtr) -> i32` | Check equality ignoring case |

### Other

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `$String$reverse` | `(StringPtr) -> StringPtr` | Reverse the string |

## Usage Example

```zig
// Basic operations
const hello = "Hello";
const world = "World";
const greeting = $String$concat(hello, world);  // "HelloWorld"
const repeated = $String$repeat("ab", 3);       // "ababab"

// Case conversion
const upper = $String$to_upper("hello");        // "HELLO"
const title = $String$to_title("hello world");  // "Hello World"

// Trimming
const trimmed = $String$trim("  hello  ");      // "hello"

// Search
if ($String$contains("hello world", "world") == 1) {
    // Found!
}
const idx = $String$index_of("hello", "l");     // 2

// Replace
const result = $String$replace_all("aaa", "a", "b");  // "bbb"

// Split and join
const parts = $String$split("a,b,c", ",");      // ["a", "b", "c"]
const joined = $String$join(parts, "-");        // "a-b-c"

// Padding
const padded = $String$pad_start("42", 5, "0"); // "00042"

// Number conversion
const num = $String$parse_int("42");            // 42
const hex = $String$from_int_radix(255, 16);    // "ff"
const formatted = $String$from_float_precision(3.14159, 2);  // "3.14"
```

## Unicode Support

- `$String$length` returns byte count (UTF-8 length)
- `$String$char_count` returns actual character count
- `$String$char_at` and `$String$substring` work with character indices, not bytes

## Memory Management

All functions returning `StringPtr` or `ArrayPtr` allocate new memory. Caller must free using SDK functions.

## Dependencies

- `zrtl` - Core ZRTL SDK
