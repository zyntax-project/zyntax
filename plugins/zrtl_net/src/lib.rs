//! ZRTL Network Plugin
//!
//! Provides TCP and UDP networking for Zyntax-based languages.
//!
//! ## Exported Symbols
//!
//! ### TCP Client
//! - `$Net$tcp_connect` - Connect to a TCP server
//! - `$Net$tcp_read` - Read bytes from TCP connection
//! - `$Net$tcp_write` - Write bytes to TCP connection
//! - `$Net$tcp_close` - Close TCP connection
//!
//! ### TCP Server
//! - `$Net$tcp_listen` - Create TCP listener
//! - `$Net$tcp_accept` - Accept incoming connection
//! - `$Net$tcp_listener_close` - Close listener
//!
//! ### UDP
//! - `$Net$udp_bind` - Create UDP socket
//! - `$Net$udp_send_to` - Send UDP packet
//! - `$Net$udp_recv_from` - Receive UDP packet
//! - `$Net$udp_close` - Close UDP socket
//!
//! ## Async Operations
//! - `$Net$tcp_connect_async` - Connect asynchronously (returns ZrtlPromise)
//! - `$Net$tcp_accept_async` - Accept connection asynchronously
//! - `$Net$tcp_read_async` - Read asynchronously

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, UdpSocket, SocketAddr, ToSocketAddrs};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;
use std::thread;
use zrtl::{
    zrtl_plugin,
    StringConstPtr, ArrayPtr,
    string_length, string_data,
    array_new, array_push, array_length, array_data,
    ZrtlPromise, PollResult, StateMachineHeader,
};

// ============================================================================
// Handle Management
// ============================================================================

static HANDLE_COUNTER: AtomicU64 = AtomicU64::new(1);

type TcpStreamMap = HashMap<u64, TcpStream>;
type TcpListenerMap = HashMap<u64, TcpListener>;
type UdpSocketMap = HashMap<u64, UdpSocket>;

static TCP_STREAMS: Mutex<Option<TcpStreamMap>> = Mutex::new(None);
static TCP_LISTENERS: Mutex<Option<TcpListenerMap>> = Mutex::new(None);
static UDP_SOCKETS: Mutex<Option<UdpSocketMap>> = Mutex::new(None);

fn next_handle() -> u64 {
    HANDLE_COUNTER.fetch_add(1, Ordering::SeqCst)
}

macro_rules! get_map {
    ($static:ident) => {{
        let mut guard = $static.lock().unwrap();
        if guard.is_none() {
            *guard = Some(HashMap::new());
        }
        guard
    }};
}

// ============================================================================
// Helper
// ============================================================================

unsafe fn zrtl_string_to_str<'a>(s: StringConstPtr) -> Option<&'a str> {
    zrtl::string_as_str(s)
}

// ============================================================================
// TCP Client Functions
// ============================================================================

/// Connect to a TCP server
///
/// Address should be in format "host:port"
/// Returns a handle on success, 0 on error.
///
/// # Safety
/// Address must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn net_tcp_connect(address: StringConstPtr) -> u64 {
    let addr_str = match zrtl_string_to_str(address) {
        Some(s) => s,
        None => return 0,
    };

    match TcpStream::connect(addr_str) {
        Ok(stream) => {
            let handle = next_handle();
            let mut streams = get_map!(TCP_STREAMS);
            if let Some(ref mut m) = *streams {
                m.insert(handle, stream);
            }
            handle
        }
        Err(_) => 0,
    }
}

/// Connect with a timeout (in milliseconds)
///
/// # Safety
/// Address must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn net_tcp_connect_timeout(address: StringConstPtr, timeout_ms: u64) -> u64 {
    let addr_str = match zrtl_string_to_str(address) {
        Some(s) => s,
        None => return 0,
    };

    // Resolve address
    let addrs: Vec<SocketAddr> = match addr_str.to_socket_addrs() {
        Ok(iter) => iter.collect(),
        Err(_) => return 0,
    };

    if addrs.is_empty() {
        return 0;
    }

    match TcpStream::connect_timeout(&addrs[0], Duration::from_millis(timeout_ms)) {
        Ok(stream) => {
            let handle = next_handle();
            let mut streams = get_map!(TCP_STREAMS);
            if let Some(ref mut m) = *streams {
                m.insert(handle, stream);
            }
            handle
        }
        Err(_) => 0,
    }
}

/// Read bytes from a TCP connection
///
/// Returns a ZRTL array of bytes, or null on error.
/// Reads up to `max_bytes` bytes.
///
/// # Safety
/// Handle must be valid.
#[no_mangle]
pub extern "C" fn net_tcp_read(handle: u64, max_bytes: u32) -> ArrayPtr {
    let mut streams = get_map!(TCP_STREAMS);
    let stream = match &mut *streams {
        Some(ref mut m) => match m.get_mut(&handle) {
            Some(s) => s,
            None => return std::ptr::null_mut(),
        },
        None => return std::ptr::null_mut(),
    };

    let mut buf = vec![0u8; max_bytes as usize];
    match stream.read(&mut buf) {
        Ok(n) => {
            let arr = array_new::<u8>(n);
            if arr.is_null() {
                return std::ptr::null_mut();
            }
            unsafe {
                for i in 0..n {
                    array_push(arr, buf[i]);
                }
            }
            arr
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Write bytes to a TCP connection
///
/// Returns number of bytes written, or -1 on error.
///
/// # Safety
/// Handle must be valid, data must be a valid ZRTL array.
#[no_mangle]
pub unsafe extern "C" fn net_tcp_write(handle: u64, data: ArrayPtr) -> i64 {
    if data.is_null() {
        return -1;
    }

    let mut streams = get_map!(TCP_STREAMS);
    let stream = match &mut *streams {
        Some(ref mut m) => match m.get_mut(&handle) {
            Some(s) => s,
            None => return -1,
        },
        None => return -1,
    };

    let len = array_length(data) as usize;
    let ptr = array_data::<u8>(data);

    if ptr.is_null() && len > 0 {
        return -1;
    }

    let bytes = if len > 0 {
        std::slice::from_raw_parts(ptr, len)
    } else {
        &[]
    };

    match stream.write(bytes) {
        Ok(n) => n as i64,
        Err(_) => -1,
    }
}

/// Write a ZRTL string to a TCP connection
///
/// Returns number of bytes written, or -1 on error.
///
/// # Safety
/// Handle must be valid, data must be a valid ZRTL string.
#[no_mangle]
pub unsafe extern "C" fn net_tcp_write_string(handle: u64, data: StringConstPtr) -> i64 {
    if data.is_null() {
        return -1;
    }

    let mut streams = get_map!(TCP_STREAMS);
    let stream = match &mut *streams {
        Some(ref mut m) => match m.get_mut(&handle) {
            Some(s) => s,
            None => return -1,
        },
        None => return -1,
    };

    let len = string_length(data) as usize;
    let ptr = string_data(data);

    let bytes = if len > 0 && !ptr.is_null() {
        std::slice::from_raw_parts(ptr, len)
    } else {
        &[]
    };

    match stream.write(bytes) {
        Ok(n) => n as i64,
        Err(_) => -1,
    }
}

/// Close a TCP connection
#[no_mangle]
pub extern "C" fn net_tcp_close(handle: u64) {
    let mut streams = get_map!(TCP_STREAMS);
    if let Some(ref mut m) = *streams {
        m.remove(&handle);
    }
}

/// Set read timeout on TCP connection (milliseconds, 0 = no timeout)
#[no_mangle]
pub extern "C" fn net_tcp_set_read_timeout(handle: u64, timeout_ms: u64) -> i32 {
    let streams = get_map!(TCP_STREAMS);
    if let Some(ref m) = *streams {
        if let Some(stream) = m.get(&handle) {
            let timeout = if timeout_ms == 0 {
                None
            } else {
                Some(Duration::from_millis(timeout_ms))
            };
            match stream.set_read_timeout(timeout) {
                Ok(()) => return 0,
                Err(_) => return -1,
            }
        }
    }
    -1
}

/// Set write timeout on TCP connection (milliseconds, 0 = no timeout)
#[no_mangle]
pub extern "C" fn net_tcp_set_write_timeout(handle: u64, timeout_ms: u64) -> i32 {
    let streams = get_map!(TCP_STREAMS);
    if let Some(ref m) = *streams {
        if let Some(stream) = m.get(&handle) {
            let timeout = if timeout_ms == 0 {
                None
            } else {
                Some(Duration::from_millis(timeout_ms))
            };
            match stream.set_write_timeout(timeout) {
                Ok(()) => return 0,
                Err(_) => return -1,
            }
        }
    }
    -1
}

// ============================================================================
// TCP Server Functions
// ============================================================================

/// Create a TCP listener
///
/// Address should be in format "host:port" (e.g., "127.0.0.1:8080")
/// Returns a handle on success, 0 on error.
///
/// # Safety
/// Address must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn net_tcp_listen(address: StringConstPtr) -> u64 {
    let addr_str = match zrtl_string_to_str(address) {
        Some(s) => s,
        None => return 0,
    };

    match TcpListener::bind(addr_str) {
        Ok(listener) => {
            let handle = next_handle();
            let mut listeners = get_map!(TCP_LISTENERS);
            if let Some(ref mut m) = *listeners {
                m.insert(handle, listener);
            }
            handle
        }
        Err(_) => 0,
    }
}

/// Accept an incoming TCP connection
///
/// Returns a stream handle on success, 0 on error.
/// This call blocks until a connection arrives.
#[no_mangle]
pub extern "C" fn net_tcp_accept(listener_handle: u64) -> u64 {
    let listeners = get_map!(TCP_LISTENERS);
    let listener = match &*listeners {
        Some(ref m) => match m.get(&listener_handle) {
            Some(l) => l,
            None => return 0,
        },
        None => return 0,
    };

    match listener.accept() {
        Ok((stream, _addr)) => {
            drop(listeners); // Release lock before acquiring another
            let handle = next_handle();
            let mut streams = get_map!(TCP_STREAMS);
            if let Some(ref mut m) = *streams {
                m.insert(handle, stream);
            }
            handle
        }
        Err(_) => 0,
    }
}

/// Close a TCP listener
#[no_mangle]
pub extern "C" fn net_tcp_listener_close(handle: u64) {
    let mut listeners = get_map!(TCP_LISTENERS);
    if let Some(ref mut m) = *listeners {
        m.remove(&handle);
    }
}

// ============================================================================
// UDP Functions
// ============================================================================

/// Create and bind a UDP socket
///
/// Address should be in format "host:port"
/// Returns a handle on success, 0 on error.
///
/// # Safety
/// Address must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn net_udp_bind(address: StringConstPtr) -> u64 {
    let addr_str = match zrtl_string_to_str(address) {
        Some(s) => s,
        None => return 0,
    };

    match UdpSocket::bind(addr_str) {
        Ok(socket) => {
            let handle = next_handle();
            let mut sockets = get_map!(UDP_SOCKETS);
            if let Some(ref mut m) = *sockets {
                m.insert(handle, socket);
            }
            handle
        }
        Err(_) => 0,
    }
}

/// Send UDP packet to address
///
/// Returns number of bytes sent, or -1 on error.
///
/// # Safety
/// Handle must be valid, address and data must be valid ZRTL strings/arrays.
#[no_mangle]
pub unsafe extern "C" fn net_udp_send_to(handle: u64, address: StringConstPtr, data: ArrayPtr) -> i64 {
    let addr_str = match zrtl_string_to_str(address) {
        Some(s) => s,
        None => return -1,
    };

    if data.is_null() {
        return -1;
    }

    let sockets = get_map!(UDP_SOCKETS);
    let socket = match &*sockets {
        Some(ref m) => match m.get(&handle) {
            Some(s) => s,
            None => return -1,
        },
        None => return -1,
    };

    let len = array_length(data) as usize;
    let ptr = array_data::<u8>(data);

    let bytes = if len > 0 && !ptr.is_null() {
        std::slice::from_raw_parts(ptr, len)
    } else {
        &[]
    };

    match socket.send_to(bytes, addr_str) {
        Ok(n) => n as i64,
        Err(_) => -1,
    }
}

/// Receive UDP packet
///
/// Returns a ZRTL array of bytes, or null on error.
/// The source address is not returned in this simple API.
#[no_mangle]
pub extern "C" fn net_udp_recv(handle: u64, max_bytes: u32) -> ArrayPtr {
    let sockets = get_map!(UDP_SOCKETS);
    let socket = match &*sockets {
        Some(ref m) => match m.get(&handle) {
            Some(s) => s,
            None => return std::ptr::null_mut(),
        },
        None => return std::ptr::null_mut(),
    };

    let mut buf = vec![0u8; max_bytes as usize];
    match socket.recv(&mut buf) {
        Ok(n) => {
            let arr = array_new::<u8>(n);
            if arr.is_null() {
                return std::ptr::null_mut();
            }
            unsafe {
                for i in 0..n {
                    array_push(arr, buf[i]);
                }
            }
            arr
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Close UDP socket
#[no_mangle]
pub extern "C" fn net_udp_close(handle: u64) {
    let mut sockets = get_map!(UDP_SOCKETS);
    if let Some(ref mut m) = *sockets {
        m.remove(&handle);
    }
}

/// Set UDP socket to non-blocking mode
#[no_mangle]
pub extern "C" fn net_udp_set_nonblocking(handle: u64, nonblocking: i32) -> i32 {
    let sockets = get_map!(UDP_SOCKETS);
    if let Some(ref m) = *sockets {
        if let Some(socket) = m.get(&handle) {
            match socket.set_nonblocking(nonblocking != 0) {
                Ok(()) => return 0,
                Err(_) => return -1,
            }
        }
    }
    -1
}

// ============================================================================
// Plugin Export
// ============================================================================

zrtl_plugin! {
    name: "zrtl_net",
    symbols: [
        // TCP Client
        ("$Net$tcp_connect", net_tcp_connect),
        ("$Net$tcp_connect_timeout", net_tcp_connect_timeout),
        ("$Net$tcp_read", net_tcp_read),
        ("$Net$tcp_write", net_tcp_write),
        ("$Net$tcp_write_string", net_tcp_write_string),
        ("$Net$tcp_close", net_tcp_close),
        ("$Net$tcp_set_read_timeout", net_tcp_set_read_timeout),
        ("$Net$tcp_set_write_timeout", net_tcp_set_write_timeout),

        // TCP Server
        ("$Net$tcp_listen", net_tcp_listen),
        ("$Net$tcp_accept", net_tcp_accept),
        ("$Net$tcp_listener_close", net_tcp_listener_close),

        // UDP
        ("$Net$udp_bind", net_udp_bind),
        ("$Net$udp_send_to", net_udp_send_to),
        ("$Net$udp_recv", net_udp_recv),
        ("$Net$udp_close", net_udp_close),
        ("$Net$udp_set_nonblocking", net_udp_set_nonblocking),

        // Async API
        ("$Net$tcp_connect_async", net_tcp_connect_async),
        ("$Net$tcp_accept_async", net_tcp_accept_async),
        ("$Net$tcp_read_async", net_tcp_read_async),
        ("$Net$async_read_result", net_async_read_result),
        ("$Net$promise_free", net_promise_free),
    ]
}

// ============================================================================
// Async Operations
// ============================================================================

/// Promise type tags for proper cleanup
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum NetPromiseType {
    TcpConnect = 1,
    TcpAccept = 2,
    TcpRead = 3,
}

/// Async future state using thread + Arc<Mutex>
struct NetFuture {
    state: std::sync::Arc<std::sync::Mutex<NetFutureState>>,
}

struct NetFutureState {
    completed: bool,
    result: Option<Result<u64, i32>>,
    data: Option<Vec<u8>>,  // For read operations
}

impl NetFuture {
    fn spawn_tcp_connect(address: String) -> Self {
        let state = std::sync::Arc::new(std::sync::Mutex::new(NetFutureState {
            completed: false,
            result: None,
            data: None,
        }));

        let state_clone = state.clone();
        thread::spawn(move || {
            let result = match TcpStream::connect(&address) {
                Ok(stream) => {
                    let handle = next_handle();
                    let mut streams = get_map!(TCP_STREAMS);
                    if let Some(ref mut m) = *streams {
                        m.insert(handle, stream);
                        Ok(handle)
                    } else {
                        Err(-2)
                    }
                }
                Err(_) => Err(-1),
            };

            if let Ok(mut state) = state_clone.lock() {
                state.completed = true;
                state.result = Some(result);
            }
        });

        NetFuture { state }
    }

    fn spawn_tcp_accept(listener_handle: u64) -> Self {
        let state = std::sync::Arc::new(std::sync::Mutex::new(NetFutureState {
            completed: false,
            result: None,
            data: None,
        }));

        let state_clone = state.clone();
        thread::spawn(move || {
            let result = {
                let listeners = get_map!(TCP_LISTENERS);
                match &*listeners {
                    Some(ref m) => match m.get(&listener_handle) {
                        Some(l) => match l.accept() {
                            Ok((stream, _addr)) => {
                                drop(listeners);
                                let handle = next_handle();
                                let mut streams = get_map!(TCP_STREAMS);
                                if let Some(ref mut m) = *streams {
                                    m.insert(handle, stream);
                                    Ok(handle)
                                } else {
                                    Err(-3)
                                }
                            }
                            Err(_) => Err(-1),
                        },
                        None => Err(-2),
                    },
                    None => Err(-2),
                }
            };

            if let Ok(mut state) = state_clone.lock() {
                state.completed = true;
                state.result = Some(result);
            }
        });

        NetFuture { state }
    }

    fn spawn_tcp_read(stream_handle: u64, max_bytes: u32) -> Self {
        let state = std::sync::Arc::new(std::sync::Mutex::new(NetFutureState {
            completed: false,
            result: None,
            data: None,
        }));

        let state_clone = state.clone();
        thread::spawn(move || {
            let result = {
                let mut streams = get_map!(TCP_STREAMS);
                match &mut *streams {
                    Some(ref mut m) => match m.get_mut(&stream_handle) {
                        Some(stream) => {
                            let mut buf = vec![0u8; max_bytes as usize];
                            match stream.read(&mut buf) {
                                Ok(n) => {
                                    buf.truncate(n);
                                    if let Ok(mut s) = state_clone.lock() {
                                        s.data = Some(buf);
                                    }
                                    Ok(n as u64)
                                }
                                Err(_) => Err(-1),
                            }
                        }
                        None => Err(-2),
                    },
                    None => Err(-2),
                }
            };

            if let Ok(mut state) = state_clone.lock() {
                state.completed = true;
                state.result = Some(result);
            }
        });

        NetFuture { state }
    }

    fn poll(&self) -> PollResult {
        if let Ok(state) = self.state.lock() {
            if state.completed {
                match state.result {
                    Some(Ok(v)) => PollResult::Ready(v as i64),
                    Some(Err(e)) => PollResult::Failed(e),
                    None => PollResult::Failed(-4),
                }
            } else {
                PollResult::Pending
            }
        } else {
            PollResult::Failed(-5)
        }
    }
}

// Storage for active futures
lazy_static::lazy_static! {
    static ref NET_FUTURES: Mutex<HashMap<u64, NetFuture>> = Mutex::new(HashMap::new());
}

static NEXT_FUTURE_ID: AtomicU64 = AtomicU64::new(1);

/// State for async network promise
#[repr(C)]
struct NetPromiseState {
    header: StateMachineHeader,
    promise_type: NetPromiseType,
    future_id: u64,
}

/// Poll function for async network operations
unsafe extern "C" fn net_async_poll(state: *mut u8) -> i64 {
    let sm = &mut *(state as *mut NetPromiseState);

    if sm.header.async_state().is_finished() {
        return PollResult::Failed(-1).to_abi();
    }

    if let Ok(futures) = NET_FUTURES.lock() {
        if let Some(future) = futures.get(&sm.future_id) {
            let result = future.poll();
            match result {
                PollResult::Ready(v) => {
                    sm.header.set_completed();
                    return PollResult::Ready(v).to_abi();
                }
                PollResult::Failed(e) => {
                    sm.header.set_failed();
                    return PollResult::Failed(e).to_abi();
                }
                PollResult::Pending => {
                    return PollResult::Pending.to_abi();
                }
            }
        }
    }

    sm.header.set_failed();
    PollResult::Failed(-6).to_abi()
}

/// Connect to TCP server asynchronously
/// Returns a ZrtlPromise pointer. Poll until ready, result is stream handle.
#[no_mangle]
pub unsafe extern "C" fn net_tcp_connect_async(address: StringConstPtr) -> *const ZrtlPromise {
    let addr_str = match zrtl_string_to_str(address) {
        Some(s) => s.to_string(),
        None => {
            let state = Box::new(NetPromiseState {
                header: StateMachineHeader::new(),
                promise_type: NetPromiseType::TcpConnect,
                future_id: 0,
            });
            let state_ptr = Box::into_raw(state) as *mut u8;
            (*(state_ptr as *mut NetPromiseState)).header.set_failed();
            let promise = Box::new(ZrtlPromise::new(state_ptr, net_async_poll));
            return Box::into_raw(promise);
        }
    };

    let future = NetFuture::spawn_tcp_connect(addr_str);
    let future_id = NEXT_FUTURE_ID.fetch_add(1, Ordering::SeqCst);

    if let Ok(mut futures) = NET_FUTURES.lock() {
        futures.insert(future_id, future);
    }

    let state = Box::new(NetPromiseState {
        header: StateMachineHeader::new(),
        promise_type: NetPromiseType::TcpConnect,
        future_id,
    });
    let state_ptr = Box::into_raw(state) as *mut u8;

    let promise = Box::new(ZrtlPromise::new(state_ptr, net_async_poll));
    Box::into_raw(promise)
}

/// Accept TCP connection asynchronously
/// Returns a ZrtlPromise pointer. Poll until ready, result is stream handle.
#[no_mangle]
pub extern "C" fn net_tcp_accept_async(listener_handle: u64) -> *const ZrtlPromise {
    let future = NetFuture::spawn_tcp_accept(listener_handle);
    let future_id = NEXT_FUTURE_ID.fetch_add(1, Ordering::SeqCst);

    if let Ok(mut futures) = NET_FUTURES.lock() {
        futures.insert(future_id, future);
    }

    let state = Box::new(NetPromiseState {
        header: StateMachineHeader::new(),
        promise_type: NetPromiseType::TcpAccept,
        future_id,
    });
    let state_ptr = Box::into_raw(state) as *mut u8;

    let promise = Box::new(unsafe { ZrtlPromise::new(state_ptr, net_async_poll) });
    Box::into_raw(promise)
}

/// Read from TCP connection asynchronously
/// Returns a ZrtlPromise pointer. Poll until ready, result is bytes read count.
/// Use net_async_read_result to get the actual data.
#[no_mangle]
pub extern "C" fn net_tcp_read_async(handle: u64, max_bytes: u32) -> *const ZrtlPromise {
    let future = NetFuture::spawn_tcp_read(handle, max_bytes);
    let future_id = NEXT_FUTURE_ID.fetch_add(1, Ordering::SeqCst);

    if let Ok(mut futures) = NET_FUTURES.lock() {
        futures.insert(future_id, future);
    }

    let state = Box::new(NetPromiseState {
        header: StateMachineHeader::new(),
        promise_type: NetPromiseType::TcpRead,
        future_id,
    });
    let state_ptr = Box::into_raw(state) as *mut u8;

    let promise = Box::new(unsafe { ZrtlPromise::new(state_ptr, net_async_poll) });
    Box::into_raw(promise)
}

/// Get the data from a completed read promise
#[no_mangle]
pub extern "C" fn net_async_read_result(promise: *const ZrtlPromise) -> ArrayPtr {
    if promise.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let promise_ref = &*promise;
        let state = promise_ref.state_machine as *mut NetPromiseState;
        if state.is_null() {
            return std::ptr::null_mut();
        }

        if let Ok(futures) = NET_FUTURES.lock() {
            if let Some(future) = futures.get(&(*state).future_id) {
                if let Ok(fstate) = future.state.lock() {
                    if let Some(data) = &fstate.data {
                        let arr = array_new::<u8>(data.len());
                        if arr.is_null() {
                            return std::ptr::null_mut();
                        }
                        for &byte in data {
                            array_push(arr, byte);
                        }
                        return arr;
                    }
                }
            }
        }
    }

    std::ptr::null_mut()
}

/// Free a network promise after use
#[no_mangle]
pub extern "C" fn net_promise_free(promise: *mut ZrtlPromise) {
    if promise.is_null() {
        return;
    }

    unsafe {
        let promise_box = Box::from_raw(promise);
        if !promise_box.state_machine.is_null() {
            let state = Box::from_raw(promise_box.state_machine as *mut NetPromiseState);
            // Clean up the future
            if let Ok(mut futures) = NET_FUTURES.lock() {
                futures.remove(&state.future_id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zrtl::{string_free, string_new};

    #[test]
    fn test_tcp_listener() {
        // Just test that we can create and close a listener
        let addr = string_new("127.0.0.1:0"); // Port 0 = OS assigns port
        unsafe {
            let handle = net_tcp_listen(addr);
            // May fail in CI due to network restrictions
            if handle != 0 {
                net_tcp_listener_close(handle);
            }
            string_free(addr);
        }
    }

    #[test]
    fn test_udp_socket() {
        // Just test that we can create and close a UDP socket
        let addr = string_new("127.0.0.1:0");
        unsafe {
            let handle = net_udp_bind(addr);
            // May fail in CI due to network restrictions
            if handle != 0 {
                net_udp_close(handle);
            }
            string_free(addr);
        }
    }

    #[test]
    fn test_async_connect_invalid() {
        // Test async connect with invalid address
        let addr = zrtl::string_new("invalid:99999");
        unsafe {
            let promise = net_tcp_connect_async(addr);
            assert!(!promise.is_null());
            // Don't wait for it - just verify promise was created
            net_promise_free(promise as *mut ZrtlPromise);
            string_free(addr);
        }
    }
}
