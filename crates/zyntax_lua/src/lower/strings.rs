//! Strings for Lua lowering.

use super::*;

/// The bytes of a string literal token, escapes decoded, and a byte the
/// source could not spell as text restored from its stand-in.
pub(crate) fn string_bytes(token: &TokenReference) -> std::result::Result<Vec<u8>, String> {
    let bytes = literal_bytes(token)?;
    Ok(restore_bytes(bytes))
}

/// Private-use characters standing for bytes (see `source_text`) back
/// to the bytes.
pub(super) fn restore_bytes(bytes: Vec<u8>) -> Vec<u8> {
    // U+F700..U+F7FF are EF 9C 80 .. EF 9F BF.
    if !bytes
        .windows(2)
        .any(|w| w[0] == 0xEF && (0x9C..=0x9F).contains(&w[1]))
    {
        return bytes;
    }
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if i + 2 < bytes.len()
            && bytes[i] == 0xEF
            && (0x9C..=0x9F).contains(&bytes[i + 1])
            && (0x80..=0xBF).contains(&bytes[i + 2])
        {
            let code = ((bytes[i] as u32 & 0x0F) << 12)
                | ((bytes[i + 1] as u32 & 0x3F) << 6)
                | (bytes[i + 2] as u32 & 0x3F);
            if (crate::ESCAPED_BYTES..crate::ESCAPED_BYTES + 0x100).contains(&code) {
                out.push((code - crate::ESCAPED_BYTES) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    out
}

pub(super) fn literal_bytes(token: &TokenReference) -> std::result::Result<Vec<u8>, String> {
    use full_moon::tokenizer::{StringLiteralQuoteType, TokenType};
    let TokenType::StringLiteral {
        literal,
        quote_type,
        ..
    } = token.token().token_type()
    else {
        return Ok(token.token().to_string().into_bytes());
    };
    match quote_type {
        StringLiteralQuoteType::Brackets => {
            // A long string starts after its first line break.
            let s = literal.as_str();
            let s = s
                .strip_prefix("\r\n")
                .or_else(|| s.strip_prefix('\n'))
                .or_else(|| s.strip_prefix('\r'))
                .unwrap_or(s);
            Ok(s.as_bytes().to_vec())
        }
        _ => decode_escapes(literal.as_str()),
    }
}

/// A string literal as a node: text when it is UTF-8, otherwise built
/// at run time from its bytes spelled in hex, since a literal in the
/// typed AST is text.
pub(super) fn string_literal(bytes: Vec<u8>, span: Span) -> Node {
    match String::from_utf8(bytes) {
        Ok(text) => str_lit(&text, span),
        Err(e) => {
            let hex: String = e.as_bytes().iter().map(|b| format!("{b:02x}")).collect();
            call(
                "zl_bytes",
                vec![str_lit(&hex, span)],
                prim(PrimitiveType::String),
                span,
            )
        }
    }
}

/// Lua's escapes: `\n` and the others, `\ddd`, `\xXX`, `\u{XXX}`, `\z`,
/// and a backslash before a line break.
pub(super) fn decode_escapes(s: &str) -> std::result::Result<Vec<u8>, String> {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b != b'\\' {
            out.push(b);
            i += 1;
            continue;
        }
        i += 1;
        let Some(&c) = bytes.get(i) else {
            return Err("unfinished string".to_string());
        };
        i += 1;
        match c {
            b'n' => out.push(b'\n'),
            b't' => out.push(b'\t'),
            b'r' => out.push(b'\r'),
            b'a' => out.push(7),
            b'b' => out.push(8),
            b'f' => out.push(12),
            b'v' => out.push(11),
            b'\\' => out.push(b'\\'),
            b'"' => out.push(b'"'),
            b'\'' => out.push(b'\''),
            b'\n' => {
                out.push(b'\n');
                if bytes.get(i) == Some(&b'\r') {
                    i += 1;
                }
            }
            b'\r' => {
                out.push(b'\n');
                if bytes.get(i) == Some(&b'\n') {
                    i += 1;
                }
            }
            b'x' => {
                let hex = s.get(i..i + 2).ok_or("hexadecimal digit expected")?;
                let v = u8::from_str_radix(hex, 16).map_err(|_| "hexadecimal digit expected")?;
                out.push(v);
                i += 2;
            }
            b'z' => {
                // Lua's whitespace: a vertical tab counts.
                while i < bytes.len()
                    && matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
                {
                    i += 1;
                }
            }
            b'u' => {
                if bytes.get(i) != Some(&b'{') {
                    return Err("missing '{' in \\u{xxxx}".to_string());
                }
                let end = s[i..].find('}').ok_or("missing '}' in \\u{xxxx}")? + i;
                let digits = &s[i + 1..end];
                if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Err("hexadecimal digit expected".to_string());
                }
                // Lua's own UTF-8 reaches 0x7FFFFFFF in six bytes.
                let code = u64::from_str_radix(digits, 16).unwrap_or(u64::MAX);
                if code > 0x7FFF_FFFF {
                    return Err("UTF-8 value too large".to_string());
                }
                crate::host::utf8_encode(code as u32, &mut out);
                i = end + 1;
            }
            d if d.is_ascii_digit() => {
                let mut v: u32 = (d - b'0') as u32;
                let mut n = 1;
                while n < 3 && i < bytes.len() && bytes[i].is_ascii_digit() {
                    v = v * 10 + (bytes[i] - b'0') as u32;
                    i += 1;
                    n += 1;
                }
                if v > 255 {
                    return Err("decimal escape too large".to_string());
                }
                out.push(v as u8);
            }
            other => return Err(format!("invalid escape sequence '\\{}'", other as char)),
        }
    }
    Ok(out)
}
