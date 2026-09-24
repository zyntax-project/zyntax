//! The `io` library's host side: files as the reference's `FILE*`
//! streams, by handle. A handle names a stream in a table of this
//! thread's; the standard streams share the process's, so what
//! `print` writes and what `io.write` writes come out in order.
//! Reached from the library as `$Lua$io_…` symbols.

use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};

use zrtl::StringPtr;

use crate::host::Numeral;

unsafe fn bytes_of(s: zrtl::StringConstPtr) -> &'static [u8] {
    unsafe { zrtl::string_as_bytes(s) }
}

/// What a stream reads from and writes to.
enum Inner {
    Stdin,
    Stdout,
    Stderr,
    Disk(std::fs::File),
    /// A command's output or input, with the command to wait for on
    /// closing.
    Pipe {
        child: std::process::Child,
        read: bool,
    },
}

/// How writes to a disk file are held back, as `setvbuf` sets it:
/// not at all, until a line ends, or until `size` bytes are waiting.
/// Nothing is held back until asked: a file dropped without being
/// closed has no collector to flush it here.
#[derive(Clone, Copy)]
enum Buffering {
    No,
    Line,
    Full(usize),
}

/// A stream with the read-ahead `read` needs: bytes taken from the
/// inner stream and not yet consumed sit in `ahead` from `at` on.
/// Writes to a disk file wait in `pending` as the buffering says.
struct Stream {
    inner: Inner,
    ahead: Vec<u8>,
    at: usize,
    pending: Vec<u8>,
    buffering: Buffering,
}

const CHUNK: usize = 8192;

impl Stream {
    fn new(inner: Inner) -> Stream {
        Stream {
            inner,
            ahead: Vec::new(),
            at: 0,
            pending: Vec::new(),
            buffering: Buffering::No,
        }
    }

    /// Pending writes go to the file.
    fn drain(&mut self) -> std::io::Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let bytes = std::mem::take(&mut self.pending);
        match &mut self.inner {
            Inner::Disk(f) => f.write_all(&bytes),
            _ => Ok(()),
        }
    }

    /// More bytes ahead, or none at the end.
    fn fill(&mut self) -> std::io::Result<bool> {
        if self.at < self.ahead.len() {
            return Ok(true);
        }
        self.drain()?;
        self.ahead.clear();
        self.at = 0;
        let mut buf = vec![0u8; CHUNK];
        let n = match &mut self.inner {
            Inner::Stdin => std::io::stdin().lock().read(&mut buf)?,
            Inner::Disk(f) => f.read(&mut buf)?,
            Inner::Pipe { child, read: true } => match child.stdout.as_mut() {
                Some(out) => out.read(&mut buf)?,
                None => 0,
            },
            _ => 0,
        };
        if n == 0 {
            return Ok(false);
        }
        buf.truncate(n);
        self.ahead = buf;
        Ok(true)
    }

    fn next_byte(&mut self) -> std::io::Result<Option<u8>> {
        if !self.fill()? {
            return Ok(None);
        }
        let b = self.ahead[self.at];
        self.at += 1;
        Ok(Some(b))
    }

    fn peek_byte(&mut self) -> std::io::Result<Option<u8>> {
        if !self.fill()? {
            return Ok(None);
        }
        Ok(Some(self.ahead[self.at]))
    }

    /// A line, up to and including its newline when `keep`; none at
    /// the end with nothing read.
    fn read_line(&mut self, keep: bool) -> std::io::Result<Option<Vec<u8>>> {
        let mut out = Vec::new();
        loop {
            if !self.fill()? {
                return Ok(if out.is_empty() { None } else { Some(out) });
            }
            let rest = &self.ahead[self.at..];
            match rest.iter().position(|&b| b == b'\n') {
                Some(i) => {
                    out.extend_from_slice(&rest[..i]);
                    if keep {
                        out.push(b'\n');
                    }
                    self.at += i + 1;
                    return Ok(Some(out));
                }
                None => {
                    out.extend_from_slice(rest);
                    self.at = self.ahead.len();
                }
            }
        }
    }

    fn read_all(&mut self) -> std::io::Result<Vec<u8>> {
        let mut out = Vec::new();
        while self.fill()? {
            out.extend_from_slice(&self.ahead[self.at..]);
            self.at = self.ahead.len();
        }
        Ok(out)
    }

    /// Up to `n` bytes; none at the end with nothing read.
    fn read_bytes(&mut self, n: usize) -> std::io::Result<Option<Vec<u8>>> {
        let mut out = Vec::new();
        while out.len() < n && self.fill()? {
            let take = (n - out.len()).min(self.ahead.len() - self.at);
            out.extend_from_slice(&self.ahead[self.at..self.at + take]);
            self.at += take;
        }
        Ok(if out.is_empty() && n > 0 {
            None
        } else {
            Some(out)
        })
    }

    fn write(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        if matches!(self.inner, Inner::Disk(_)) {
            self.discard_ahead()?;
            self.pending.extend_from_slice(bytes);
            let due = match self.buffering {
                Buffering::No => true,
                Buffering::Line => bytes.contains(&b'\n'),
                Buffering::Full(size) => self.pending.len() >= size,
            };
            return if due { self.drain() } else { Ok(()) };
        }
        match &mut self.inner {
            Inner::Stdout => std::io::stdout().lock().write_all(bytes),
            Inner::Stderr => std::io::stderr().lock().write_all(bytes),
            Inner::Pipe { child, read: false } => match child.stdin.as_mut() {
                Some(stdin) => stdin.write_all(bytes),
                None => Ok(()),
            },
            _ => Err(std::io::Error::from_raw_os_error(libc::EBADF)),
        }
    }

    /// Bytes read ahead put back: the file's position becomes the
    /// logical one.
    fn discard_ahead(&mut self) -> std::io::Result<()> {
        let unread = self.ahead.len() - self.at;
        if unread > 0
            && let Inner::Disk(f) = &mut self.inner
        {
            f.seek(SeekFrom::Current(-(unread as i64)))?;
        }
        self.ahead.clear();
        self.at = 0;
        Ok(())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.drain()?;
        match &mut self.inner {
            Inner::Stdout => std::io::stdout().lock().flush(),
            Inner::Stderr => std::io::stderr().lock().flush(),
            Inner::Disk(f) => f.flush(),
            Inner::Pipe { child, read: false } => match child.stdin.as_mut() {
                Some(stdin) => stdin.flush(),
                None => Ok(()),
            },
            _ => Ok(()),
        }
    }

    fn seek(&mut self, whence: i64, offset: i64) -> std::io::Result<u64> {
        self.drain()?;
        let Inner::Disk(f) = &mut self.inner else {
            return Err(std::io::Error::from_raw_os_error(libc::ESPIPE));
        };
        let unread = (self.ahead.len() - self.at) as i64;
        let from = match whence {
            0 => SeekFrom::Start(offset.max(0) as u64),
            1 => SeekFrom::Current(offset - unread),
            _ => SeekFrom::End(offset),
        };
        let pos = f.seek(from)?;
        self.ahead.clear();
        self.at = 0;
        Ok(pos)
    }
}

thread_local! {
    static FILES: RefCell<HashMap<i64, Stream>> = RefCell::new(HashMap::new());
    static NEXT: std::cell::Cell<i64> = const { std::cell::Cell::new(4) };
    static IO_ERROR: RefCell<(String, i64, String)> = const { RefCell::new((String::new(), 0, String::new())) };
    static FAILED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static NUMBER: std::cell::Cell<(i64, f64)> = const { std::cell::Cell::new((0, 0.0)) };
}

fn note(err: &std::io::Error, name: Option<&str>) -> i64 {
    let code = err.raw_os_error().unwrap_or(0) as i64;
    let reason = err
        .to_string()
        .split(" (os error")
        .next()
        .unwrap_or("")
        .to_string();
    let message = match name {
        Some(name) => format!("{name}: {reason}"),
        None => reason.clone(),
    };
    IO_ERROR.with(|e| *e.borrow_mut() = (message, code, reason));
    FAILED.with(|f| f.set(true));
    if code == 0 { -1 } else { code }
}

fn register(stream: Stream) -> i64 {
    let h = NEXT.with(|n| {
        let h = n.get();
        n.set(h + 1);
        h
    });
    FILES.with(|f| f.borrow_mut().insert(h, stream));
    h
}

fn with<R>(h: i64, f: impl FnOnce(&mut Stream) -> R) -> Option<R> {
    FILES.with(|files| files.borrow_mut().get_mut(&h).map(f))
}

pub(crate) extern "C" fn host_io_error() -> StringPtr {
    IO_ERROR.with(|e| zrtl::string::string_from_bytes(e.borrow().0.as_bytes()))
}

pub(crate) extern "C" fn host_io_errno() -> i64 {
    IO_ERROR.with(|e| e.borrow().1)
}

/// The last failure's reason alone, without the file's name.
pub(crate) extern "C" fn host_io_reason() -> StringPtr {
    IO_ERROR.with(|e| zrtl::string::string_from_bytes(e.borrow().2.as_bytes()))
}

/// Whether an operation failed since this was last asked; the answer
/// clears it, so a read tells the end of a file from a failure.
pub(crate) extern "C" fn host_io_failed() -> bool {
    FAILED.with(|f| f.replace(false))
}

/// The standard streams' handles: 1, 2 and 3 for stdin, stdout and
/// stderr, made on first use.
pub(crate) extern "C" fn host_io_std(which: i64) -> i64 {
    let h = which + 1;
    FILES.with(|files| {
        let mut files = files.borrow_mut();
        files.entry(h).or_insert_with(|| {
            Stream::new(match which {
                0 => Inner::Stdin,
                1 => Inner::Stdout,
                _ => Inner::Stderr,
            })
        });
    });
    h
}

/// A file opened as `mode` says; 0 with the error noted, -1 for a
/// mode that is not one.
pub(crate) extern "C" fn host_io_open(
    name: zrtl::StringConstPtr,
    mode: zrtl::StringConstPtr,
) -> i64 {
    let name = String::from_utf8_lossy(unsafe { bytes_of(name) }).into_owned();
    let mode = unsafe { bytes_of(mode) };
    let (kind, plus) = match mode {
        [k @ (b'r' | b'w' | b'a'), rest @ ..] => {
            let rest = rest.strip_prefix(b"+");
            let plus = rest.is_some();
            let rest = rest.unwrap_or(&mode[1..]);
            let rest = rest.strip_prefix(b"b").unwrap_or(rest);
            if !rest.is_empty() {
                return -1;
            }
            (*k, plus)
        }
        _ => return -1,
    };
    let mut options = std::fs::OpenOptions::new();
    match kind {
        b'r' => {
            options.read(true).write(plus);
        }
        b'w' => {
            options.write(true).create(true).truncate(true).read(plus);
        }
        _ => {
            options.append(true).create(true).read(plus);
        }
    }
    match options.open(&name) {
        Ok(f) => register(Stream::new(Inner::Disk(f))),
        Err(e) => {
            note(&e, Some(&name));
            0
        }
    }
}

/// A command run through the shell, its output read or its input
/// written; 0 with the error noted.
pub(crate) extern "C" fn host_io_popen(
    command: zrtl::StringConstPtr,
    mode: zrtl::StringConstPtr,
) -> i64 {
    let command = String::from_utf8_lossy(unsafe { bytes_of(command) }).into_owned();
    let read = unsafe { bytes_of(mode) } != b"w";
    let mut cmd = shell(&command);
    if read {
        cmd.stdout(std::process::Stdio::piped());
    } else {
        cmd.stdin(std::process::Stdio::piped());
    }
    match cmd.spawn() {
        Ok(child) => register(Stream::new(Inner::Pipe { child, read })),
        Err(e) => {
            note(&e, Some(&command));
            0
        }
    }
}

/// `command` as the platform's shell runs it: `sh -c` on Unix, and
/// on Windows `cmd /C` with the command passed as written.
#[cfg(unix)]
fn shell(command: &str) -> std::process::Command {
    let mut cmd = std::process::Command::new("/bin/sh");
    cmd.arg("-c").arg(command);
    cmd
}

#[cfg(windows)]
fn shell(command: &str) -> std::process::Command {
    use std::os::windows::process::CommandExt;
    let mut cmd = std::process::Command::new("cmd");
    cmd.arg("/C").raw_arg(command);
    cmd
}

/// A file under the temporary directory that did not exist before,
/// open for reading and writing, with its path. With `delete_on_close`
/// the file goes when its last handle closes.
#[cfg(windows)]
pub(crate) fn fresh_temp_file(
    delete_on_close: bool,
) -> std::io::Result<(std::path::PathBuf, std::fs::File)> {
    use std::os::windows::fs::OpenOptionsExt;
    use std::sync::atomic::{AtomicU64, Ordering};
    const FILE_ATTRIBUTE_TEMPORARY: u32 = 0x100;
    const FILE_FLAG_DELETE_ON_CLOSE: u32 = 0x0400_0000;
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir();
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    loop {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let tag = (seed ^ (u64::from(std::process::id()) << 32)).wrapping_add(n) & 0xff_ffff;
        let path = dir.join(format!("lua_{tag:06x}"));
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true).create_new(true);
        if delete_on_close {
            options
                .share_mode(0)
                .custom_flags(FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE);
        }
        match options.open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
}

#[cfg(windows)]
pub(crate) extern "C" fn host_io_tmpfile() -> i64 {
    match fresh_temp_file(true) {
        Ok((_, file)) => register(Stream::new(Inner::Disk(file))),
        Err(e) => {
            note(&e, None);
            0
        }
    }
}

#[cfg(unix)]
pub(crate) extern "C" fn host_io_tmpfile() -> i64 {
    let mut template = b"/tmp/lua_XXXXXX\0".to_vec();
    let fd = unsafe { libc::mkstemp(template.as_mut_ptr() as *mut libc::c_char) };
    if fd < 0 {
        note(&std::io::Error::last_os_error(), None);
        return 0;
    }
    // Unlinked at once: the file lives while it is open.
    unsafe { libc::unlink(template.as_ptr() as *const libc::c_char) };
    let file = unsafe { <std::fs::File as std::os::fd::FromRawFd>::from_raw_fd(fd) };
    register(Stream::new(Inner::Disk(file)))
}

/// Whether the handle names an open stream.
pub(crate) extern "C" fn host_io_is_open(h: i64) -> bool {
    with(h, |_| ()).is_some()
}

/// How a file prints, as the library prints it: an address made from
/// the handle, or closed.
pub(crate) fn file_text(h: i64) -> String {
    if host_io_is_open(h) {
        format!("file (0x{:x})", 0x7f00 + h * 16)
    } else {
        "file (closed)".to_string()
    }
}

pub(crate) extern "C" fn host_io_is_pipe(h: i64) -> bool {
    with(h, |s| matches!(s.inner, Inner::Pipe { .. })).unwrap_or(false)
}

/// The status word `system` would give for a command that ended so.
#[cfg(unix)]
fn status_word(status: std::process::ExitStatus) -> i64 {
    use std::os::unix::process::ExitStatusExt;
    status.into_raw() as i64
}

/// On Windows that word is the exit code.
#[cfg(windows)]
fn status_word(status: std::process::ExitStatus) -> i64 {
    status.code().map_or(-1, i64::from)
}

/// Closed: 0, or the error's number; for a command, the status word
/// `system` would give, shifted past the low byte, with bit 0 set.
pub(crate) extern "C" fn host_io_close(h: i64) -> i64 {
    let Some(mut stream) = FILES.with(|files| files.borrow_mut().remove(&h)) else {
        return 0;
    };
    if let Err(e) = stream.flush() {
        return note(&e, None);
    }
    match stream.inner {
        Inner::Pipe { mut child, .. } => {
            drop(child.stdin.take());
            match child.wait() {
                Ok(status) => status_word(status) << 1 | 1,
                Err(e) => note(&e, None),
            }
        }
        _ => 0,
    }
}

/// A line, with or without its newline; null at the end.
pub(crate) extern "C" fn host_io_read_line(h: i64, keep: bool) -> StringPtr {
    match with(h, |s| s.read_line(keep)) {
        Some(Ok(Some(line))) => zrtl::string::string_from_bytes(&line),
        Some(Err(e)) => {
            note(&e, None);
            std::ptr::null_mut()
        }
        _ => std::ptr::null_mut(),
    }
}

pub(crate) extern "C" fn host_io_read_all(h: i64) -> StringPtr {
    match with(h, |s| s.read_all()) {
        Some(Ok(bytes)) => zrtl::string::string_from_bytes(&bytes),
        Some(Err(e)) => {
            note(&e, None);
            std::ptr::null_mut()
        }
        None => zrtl::string::string_from_bytes(b""),
    }
}

/// Up to `n` bytes; null at the end. Zero bytes is a test for the
/// end: empty, or null there.
pub(crate) extern "C" fn host_io_read_bytes(h: i64, n: i64) -> StringPtr {
    if n == 0 {
        return match with(h, |s| s.peek_byte()) {
            Some(Ok(Some(_))) => zrtl::string::string_from_bytes(b""),
            _ => std::ptr::null_mut(),
        };
    }
    match with(h, |s| s.read_bytes(n.max(0) as usize)) {
        Some(Ok(Some(bytes))) => zrtl::string::string_from_bytes(&bytes),
        Some(Err(e)) => {
            note(&e, None);
            std::ptr::null_mut()
        }
        _ => std::ptr::null_mut(),
    }
}

/// A numeral read as the reference reads one: a sign, digits in
/// decimal or hexadecimal, a point, an exponent, up to 200 bytes,
/// then converted. 1 for an integer, 2 for a float, 0 for nothing a
/// numeral; the value waits in `number_int` or `number_float`.
pub(crate) extern "C" fn host_io_read_number(h: i64) -> i64 {
    let text = match with(h, read_numeral) {
        Some(Ok(text)) => text,
        Some(Err(e)) => {
            note(&e, None);
            return 0;
        }
        None => return 0,
    };
    // A numeral that fills the buffer is refused, as the reference
    // refuses one; what was read stays read.
    if text.len() >= MAX_NUMERAL {
        return 0;
    }
    match crate::host::parse_numeral(&String::from_utf8_lossy(&text)) {
        Numeral::Int(v) => {
            NUMBER.with(|n| n.set((v, 0.0)));
            1
        }
        Numeral::Float(v) => {
            NUMBER.with(|n| n.set((0, v)));
            2
        }
        Numeral::None => 0,
    }
}

/// How many bytes a numeral may run to.
const MAX_NUMERAL: usize = 200;

fn read_numeral(s: &mut Stream) -> std::io::Result<Vec<u8>> {
    const MAX: usize = MAX_NUMERAL;
    let mut out = Vec::new();
    // Leading spaces go; then each piece is taken while it fits.
    while let Some(b) = s.peek_byte()? {
        if b.is_ascii_whitespace() {
            s.next_byte()?;
        } else {
            break;
        }
    }
    let take_if = |s: &mut Stream, out: &mut Vec<u8>, set: &[u8]| -> std::io::Result<bool> {
        if out.len() < MAX
            && let Some(b) = s.peek_byte()?
            && set.contains(&b)
        {
            s.next_byte()?;
            out.push(b);
            return Ok(true);
        }
        Ok(false)
    };
    take_if(s, &mut out, b"+-")?;
    let mut hex = false;
    let mut count = 0;
    if take_if(s, &mut out, b"0")? {
        if take_if(s, &mut out, b"xX")? {
            hex = true;
        } else {
            count = 1;
        }
    }
    let digits = |s: &mut Stream, out: &mut Vec<u8>, hex: bool| -> std::io::Result<usize> {
        let mut n = 0;
        while out.len() < MAX
            && let Some(b) = s.peek_byte()?
            && (if hex {
                b.is_ascii_hexdigit()
            } else {
                b.is_ascii_digit()
            })
        {
            s.next_byte()?;
            out.push(b);
            n += 1;
        }
        Ok(n)
    };
    count += digits(s, &mut out, hex)?;
    if take_if(s, &mut out, b".")? {
        count += digits(s, &mut out, hex)?;
    }
    if count > 0 && take_if(s, &mut out, if hex { b"pP" } else { b"eE" })? {
        take_if(s, &mut out, b"+-")?;
        digits(s, &mut out, false)?;
    }
    Ok(out)
}

pub(crate) extern "C" fn host_io_number_int() -> i64 {
    NUMBER.with(|n| n.get().0)
}

pub(crate) extern "C" fn host_io_number_float() -> f64 {
    NUMBER.with(|n| n.get().1)
}

/// Written: 0, or the error's number with its message noted.
pub(crate) extern "C" fn host_io_write(h: i64, s: zrtl::StringConstPtr) -> i64 {
    let bytes = unsafe { bytes_of(s) };
    match with(h, |stream| stream.write(bytes)) {
        Some(Ok(())) => 0,
        Some(Err(e)) => note(&e, None),
        None => libc::EBADF as i64,
    }
}

/// The position after seeking, or -1 with the error noted. `whence`
/// is 0 for the start, 1 for the current position, 2 for the end.
pub(crate) extern "C" fn host_io_seek(h: i64, whence: i64, offset: i64) -> i64 {
    match with(h, |s| s.seek(whence, offset)) {
        Some(Ok(pos)) => pos as i64,
        Some(Err(e)) => {
            note(&e, None);
            -1
        }
        None => -1,
    }
}

/// `setvbuf`: 0 for none, 1 for a line, 2 for `size` bytes.
pub(crate) extern "C" fn host_io_setvbuf(h: i64, mode: i64, size: i64) -> i64 {
    let buffering = match mode {
        0 => Buffering::No,
        1 => Buffering::Line,
        _ => Buffering::Full(size.clamp(1, 1 << 30) as usize),
    };
    match with(h, |s| {
        s.buffering = buffering;
        s.drain()
    }) {
        Some(Ok(())) => 0,
        Some(Err(e)) => note(&e, None),
        None => libc::EBADF as i64,
    }
}

/// Every open stream's buffered output written out, standard output's
/// too, as the process is about to exit. Failures are not reported:
/// nothing is left to report them to.
pub(crate) extern "C" fn host_io_flush_all() {
    FILES.with(|files| {
        for stream in files.borrow_mut().values_mut() {
            let _ = stream.flush();
        }
    });
    let _ = std::io::stdout().lock().flush();
}

pub(crate) extern "C" fn host_io_flush(h: i64) -> i64 {
    match with(h, |s| s.flush()) {
        Some(Ok(())) => 0,
        Some(Err(e)) => note(&e, None),
        None => libc::EBADF as i64,
    }
}
