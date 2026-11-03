//! What the host hands a program natively: number formatting and
//! parsing as Lua does them, `string.format`, the clocks, and the
//! arguments the program was started with. Reached from the library
//! as `$Lua$…` symbols.

mod errors;
mod format;
mod loading;
mod numbers;
mod packing;
mod patterns;
mod plugin;
mod process;
mod random;
mod strings;
mod utf8;

use errors::*;
pub(crate) use format::*;
use loading::*;
pub(crate) use numbers::*;
use packing::*;
use patterns::*;
pub(crate) use plugin::*;
pub use process::*;
use random::*;
pub(crate) use strings::*;
pub(crate) use utf8::*;

use std::sync::OnceLock;

use zrtl::{DynamicBox, StringPtr, TypeCategory, TypeTag};

use crate::{host_io, host_os};

static ARGS: OnceLock<Vec<String>> = OnceLock::new();
static IGNORE_ENV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Where `require` looks for C libraries when nothing says otherwise.
#[cfg(windows)]
const CPATH_DEFAULT: &str = "!\\?.dll;!\\..\\lib\\lua\\5.4\\?.dll;!\\loadall.dll;.\\?.dll";
#[cfg(not(windows))]
const CPATH_DEFAULT: &str = "/usr/local/lib/lua/5.4/?.so;/usr/local/lib/lua/5.4/loadall.so;./?.so";

// ─── numbers ────────────────────────────────────────────────────────

/// What a string converts to as a number: nothing, an integer, or a
/// float, by Lua's rules for numerals: surrounding whitespace, a sign,
/// decimal or hexadecimal digits, a fraction and an exponent. A
/// decimal integer that does not fit is a float; a hexadecimal one
/// wraps.
pub enum Numeral {
    None,
    Int(i64),
    Float(f64),
}

// ─── string.format ──────────────────────────────────────────────────

/// A list of dynamic values as the compiler lays it out: the header
/// holds the data pointer, the length and the capacity.
#[repr(C)]
struct ListHeader {
    data: *const *const DynamicBox,
    len: i64,
    capacity: i64,
}

/// A dynamic value read for formatting. A string is its bytes, any
/// of them; a string or an object carries its address, for `%p`. An
/// object's text is what `tostring` gives it, metamethods included,
/// which the library works out before the call.
pub(crate) enum Arg<'a> {
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(&'a [u8], usize),
    Other(Vec<u8>, usize),
}

// ─── string.pack ────────────────────────────────────────────────────

thread_local! {
    static PACK_ERROR: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
    static UNPACKED: std::cell::RefCell<(Vec<crate::pack::Unpacked>, usize)> = const { std::cell::RefCell::new((Vec::new(), 0)) };
}

// ─── strings by the byte ────────────────────────────────────────────

// ─── random ─────────────────────────────────────────────────────────

/// xoshiro256**, as Lua 5.4 uses; seeded from the clock unless
/// `math.randomseed` says otherwise.
struct Xoshiro([u64; 4]);

impl Xoshiro {
    fn seeded(n: u64) -> Self {
        Self::seeded_with(n, 0)
    }
    /// The reference's seeding: the two numbers and a constant in the
    /// state, then sixteen draws discarded.
    fn seeded_with(n1: u64, n2: u64) -> Self {
        let mut r = Xoshiro([n1, 0xff, n2, 0]);
        for _ in 0..16 {
            r.next();
        }
        r
    }
    fn next(&mut self) -> u64 {
        let s = &mut self.0;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }
}

thread_local! {
    static RANDOM: std::cell::RefCell<Xoshiro> = std::cell::RefCell::new(Xoshiro::seeded(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(7)
    ));
}

// ─── patterns ───────────────────────────────────────────────────────
// The matcher keeps the last match's captures per thread; the library
// reads them back after a successful call. Errors come back as -2 with
// the message held for `$Lua$pat_error`.

use crate::pattern;

// A stack of byte buffers for `gsub`: a replacement function may run
// a `gsub` of its own.
thread_local! {
    static BUFFERS: std::cell::RefCell<Vec<Vec<u8>>> = const { std::cell::RefCell::new(Vec::new()) };
}

// ─── utf8 ───────────────────────────────────────────────────────────
// Lua's own UTF-8: sequences up to six bytes for codes to 0x7FFFFFFF,
// strict by default (no surrogates, nothing past 0x10FFFF).

const MAXUTF: u32 = 0x7FFF_FFFF;
const MAXUNICODE: u32 = 0x10_FFFF;

thread_local! {
    static UTF8_CODES: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
}

// ─── load ───────────────────────────────────────────────────────────

thread_local! {
    static LOAD_ERROR: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
    static LOADS: std::cell::Cell<i64> = const { std::cell::Cell::new(0) };
    /// The chunk the last load compiled returns the function it loaded.
    static LOAD_WRAPS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Chunks `load` compiles carry numbers from here, past any file the
/// program was compiled with.
const LOAD_CHUNKS_FROM: i64 = 1 << 20;

// ─── uncaught errors ────────────────────────────────────────────────

// ─── the plugin ─────────────────────────────────────────────────────

static INFO: zrtl::ZrtlInfo = zrtl::ZrtlInfo::new(c"lua_host".as_ptr());
static SYMBOLS: [zrtl::ZrtlSymbol; 106] = [
    zrtl::ZrtlSymbol::new(c"$Lua$cpath".as_ptr(), host_cpath as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$argc".as_ptr(), host_argc as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$argv".as_ptr(), host_argv as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$clock".as_ptr(), host_clock as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$time".as_ptr(), host_time as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$float_str".as_ptr(), host_float_str as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$number_kind".as_ptr(), host_number_kind as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$parse_int".as_ptr(), host_parse_int as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$parse_float".as_ptr(), host_parse_float as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$float_is_int".as_ptr(),
        host_float_is_int as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$format".as_ptr(), host_format as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$format_g".as_ptr(), host_format_g as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$word".as_ptr(), host_word as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$sub".as_ptr(), host_sub as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$byte_at".as_ptr(), host_byte_at as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$len_any".as_ptr(), host_len_any as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$sub_any".as_ptr(), host_sub_any as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$byte_at_any".as_ptr(), host_byte_at_any as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$unpack_any".as_ptr(), host_unpack_any as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$from_byte".as_ptr(), host_from_byte as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$reverse".as_ptr(), host_reverse as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$rep".as_ptr(), host_rep as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$find_plain".as_ptr(), host_find_plain as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$bytes".as_ptr(), host_bytes as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$upper".as_ptr(), host_upper as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$lower".as_ptr(), host_lower as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$random_seed".as_ptr(), host_random_seed as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$random_next".as_ptr(), host_random_next as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$random_seed_now".as_ptr(),
        host_random_seed_now as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$random_seed_pid".as_ptr(),
        host_random_seed_pid as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$random_float".as_ptr(),
        host_random_float as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$random_int".as_ptr(), host_random_int as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$pat_specials".as_ptr(),
        host_pat_specials as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$pat_find".as_ptr(), host_pat_find as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$pat_match_at".as_ptr(),
        host_pat_match_at as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$pat_gmatch_next".as_ptr(),
        host_pat_gmatch_next as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$pat_end".as_ptr(), host_pat_end as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$pat_level".as_ptr(), host_pat_level as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$pat_cap_kind".as_ptr(),
        host_pat_cap_kind as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$pat_cap_str".as_ptr(), host_pat_cap_str as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$pat_cap_pos".as_ptr(), host_pat_cap_pos as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$pat_error".as_ptr(), host_pat_error as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$buf_open".as_ptr(), host_buf_open as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$buf_push".as_ptr(), host_buf_push as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$buf_push_range".as_ptr(),
        host_buf_push_range as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$buf_expand".as_ptr(), host_buf_expand as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$buf_close".as_ptr(), host_buf_close as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$utf8_char".as_ptr(), host_utf8_char as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$utf8_len".as_ptr(), host_utf8_len as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$utf8_offset".as_ptr(), host_utf8_offset as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$utf8_codepoint".as_ptr(),
        host_utf8_codepoint as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$utf8_code_at".as_ptr(),
        host_utf8_code_at as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$utf8_next".as_ptr(), host_utf8_next as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$load".as_ptr(), host_load as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$load_error".as_ptr(), host_load_error as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$load_wraps".as_ptr(), host_load_wraps as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$dump".as_ptr(), host_dump as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$report_pending".as_ptr(),
        host_report_pending as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$exit_uncaught".as_ptr(),
        host_exit_uncaught as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$searchpath".as_ptr(), host_searchpath as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$read_file".as_ptr(), host_read_file as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$replace_dots".as_ptr(),
        host_replace_dots as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$os_error".as_ptr(),
        host_os::host_os_error as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$date_field".as_ptr(),
        host_os::host_date_field as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$time_of".as_ptr(), host_os::host_time_of as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$date_check".as_ptr(),
        host_os::host_date_check as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$date".as_ptr(), host_os::host_date as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$getenv".as_ptr(), host_os::host_getenv as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$tmpname".as_ptr(), host_os::host_tmpname as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$remove".as_ptr(), host_os::host_remove as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$rename".as_ptr(), host_os::host_rename as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$execute".as_ptr(), host_os::host_execute as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$exec_result".as_ptr(),
        host_os::host_exec_result as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$setlocale".as_ptr(), host_setlocale as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$pack".as_ptr(), host_pack as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$pack_error".as_ptr(), host_pack_error as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$packsize".as_ptr(), host_packsize as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$unpack".as_ptr(), host_unpack as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$unpack_kind".as_ptr(), host_unpack_kind as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$unpack_int".as_ptr(), host_unpack_int as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$unpack_float".as_ptr(),
        host_unpack_float as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$unpack_str".as_ptr(), host_unpack_str as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$unpack_next".as_ptr(), host_unpack_next as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_error".as_ptr(),
        host_io::host_io_error as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_reason".as_ptr(),
        host_io::host_io_reason as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_errno".as_ptr(),
        host_io::host_io_errno as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_failed".as_ptr(),
        host_io::host_io_failed as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$io_std".as_ptr(), host_io::host_io_std as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$stdout_write".as_ptr(),
        host_io::host_stdout_write as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$io_open".as_ptr(), host_io::host_io_open as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_popen".as_ptr(),
        host_io::host_io_popen as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_tmpfile".as_ptr(),
        host_io::host_io_tmpfile as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_is_open".as_ptr(),
        host_io::host_io_is_open as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_is_pipe".as_ptr(),
        host_io::host_io_is_pipe as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_close".as_ptr(),
        host_io::host_io_close as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_read_line".as_ptr(),
        host_io::host_io_read_line as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_read_all".as_ptr(),
        host_io::host_io_read_all as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_read_bytes".as_ptr(),
        host_io::host_io_read_bytes as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_read_number".as_ptr(),
        host_io::host_io_read_number as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_number_int".as_ptr(),
        host_io::host_io_number_int as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_number_float".as_ptr(),
        host_io::host_io_number_float as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_write".as_ptr(),
        host_io::host_io_write as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$io_seek".as_ptr(), host_io::host_io_seek as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_flush".as_ptr(),
        host_io::host_io_flush as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_flush_all".as_ptr(),
        host_io::host_io_flush_all as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_setvbuf".as_ptr(),
        host_io::host_io_setvbuf as *const u8,
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpath_follows_the_environment_as_the_reference_does() {
        let d = cpath_default();
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |name: &str| {
                pairs
                    .iter()
                    .find(|(k, _)| *k == name)
                    .map(|(_, v)| v.to_string())
            }
        };
        assert_eq!(cpath_value(false, env(&[])), d);
        assert_eq!(
            cpath_value(false, env(&[("LUA_CPATH", "a/?.so")])),
            "a/?.so"
        );
        assert_eq!(
            cpath_value(
                false,
                env(&[("LUA_CPATH", "a/?.so"), ("LUA_CPATH_5_4", "b/?.so")])
            ),
            "b/?.so"
        );
        assert_eq!(cpath_value(false, env(&[("LUA_CPATH", ";;")])), d);
        assert_eq!(
            cpath_value(false, env(&[("LUA_CPATH", "a/?.so;;")])),
            format!("a/?.so;{d}")
        );
        assert_eq!(
            cpath_value(false, env(&[("LUA_CPATH", ";;b/?.so")])),
            format!("{d};b/?.so")
        );
        assert_eq!(
            cpath_value(false, env(&[("LUA_CPATH", "a;;b")])),
            format!("a;{d};b")
        );
        assert_eq!(cpath_value(true, env(&[("LUA_CPATH", "a/?.so")])), d);
    }

    #[test]
    fn floats_print_as_lua_prints_them() {
        assert_eq!(float_text(3.0), "3.0");
        assert_eq!(float_text(0.1), "0.1");
        assert_eq!(float_text(1e15), "1e+15");
        assert_eq!(float_text(1e14), "1e+14");
        assert_eq!(float_text(2f64.powi(53)), "9.007199254741e+15");
        assert_eq!(float_text(-0.5), "-0.5");
        assert_eq!(float_text(f64::INFINITY), "inf");
        assert_eq!(float_text(f64::NAN), "nan");
        assert_eq!(float_text(100.0), "100.0");
        assert_eq!(float_text(1.0 / 3.0), "0.33333333333333");
        assert_eq!(float_text(123456.789), "123456.789");
    }

    #[test]
    fn numerals_parse_as_lua_reads_them() {
        assert!(matches!(parse_numeral("42"), Numeral::Int(42)));
        assert!(matches!(parse_numeral("  -7 "), Numeral::Int(-7)));
        assert!(matches!(parse_numeral("0x10"), Numeral::Int(16)));
        assert!(matches!(
            parse_numeral("0xffffffffffffffff"),
            Numeral::Int(-1)
        ));
        assert!(matches!(parse_numeral("1e2"), Numeral::Float(v) if v == 100.0));
        assert!(matches!(parse_numeral("0x1p4"), Numeral::Float(v) if v == 16.0));
        assert!(matches!(parse_numeral(".5"), Numeral::Float(v) if v == 0.5));
        assert!(matches!(parse_numeral("5."), Numeral::Float(v) if v == 5.0));
        assert!(matches!(parse_numeral("abc"), Numeral::None));
        assert!(matches!(parse_numeral(""), Numeral::None));
        assert!(matches!(parse_numeral("1 2"), Numeral::None));
        assert!(matches!(
            parse_numeral("9223372036854775808"),
            Numeral::Float(v) if v == 9223372036854775808.0
        ));
        assert!(matches!(
            parse_numeral("-9223372036854775808"),
            Numeral::Int(i64::MIN)
        ));
    }

    #[test]
    fn format_covers_the_common_conversions() {
        let f = |fmt: &str, args: &[Arg<'_>]| {
            String::from_utf8(format(fmt.as_bytes(), args).unwrap()).unwrap()
        };
        assert_eq!(f("%d items", &[Arg::Int(3)]), "3 items");
        assert_eq!(
            f(
                "%5d|%-5d|%05d",
                &[Arg::Int(42), Arg::Int(42), Arg::Int(-42)]
            ),
            "   42|42   |-0042"
        );
        assert_eq!(f("%.2f", &[Arg::Float(1.23456)]), "1.23");
        assert_eq!(f("%g", &[Arg::Float(100000.0)]), "100000");
        assert_eq!(f("%g", &[Arg::Float(1e20)]), "1e+20");
        assert_eq!(
            f("%x %X %o", &[Arg::Int(255), Arg::Int(255), Arg::Int(8)]),
            "ff FF 10"
        );
        assert_eq!(
            f("%s and %s", &[Arg::Str(b"a", 0), Arg::Float(1.5)]),
            "a and 1.5"
        );
        assert_eq!(f("%q", &[Arg::Str(b"a\"b\n", 0)]), "\"a\\\"b\\\n\"");
        assert_eq!(f("%5.1s|", &[Arg::Str(b"abc", 0)]), "    a|");
        assert_eq!(f("%%", &[]), "%");
        assert_eq!(f("%e", &[Arg::Float(12345.678)]), "1.234568e+04");
        assert_eq!(f("%c%c", &[Arg::Int(72), Arg::Int(105)]), "Hi");
        assert!(format(b"%d", &[Arg::Str(b"x", 0)]).is_err());
        assert!(format(b"%d", &[]).is_err());
    }
}
