//! The Lua 5.4 C API for zylua: `lua.h` and `lauxlib.h`, linked into
//! the executable, for C modules loaded with `require` and
//! `package.loadlib`.
//!
//! The API's functions are C (`c/lapi.c`, and the reference's
//! `lauxlib.c` unchanged), each a veneer over a core written in Rust
//! here. A core never raises: it returns, with an error left pending
//! when the operation raised one, and the veneer then raises. A raise
//! is a jump in C to the protect frame the call of the C function set
//! up, so it crosses only C frames. What Lua means by an operation
//! (indexing, calling, raising) comes from the frontend's library
//! through a [`bridge::Bridge`], compiled into the running program the
//! first time a native library opens; a program that loads none pays
//! for none of it.
//!
//! The frontend reaches this crate through the ZRTL symbols of
//! [`static_plugin`] and hands over what it owns through [`install`].

pub mod api;
pub mod bridge;
pub mod calls;
pub mod load;
pub mod state;
pub mod values;

use std::sync::OnceLock;

use state::{Global, State};
use values::Any;

pub use values::Tags;

/// What the frontend provides: the bridge's compiled entries, a way to
/// root a word for the collector, and its box tags.
pub struct Hooks {
    /// Compile the bridge's entries into the running program.
    pub resolve: fn() -> Result<bridge::Resolved, String>,
    /// Register `len` bytes at an address as a root of the collector.
    pub add_root: fn(*const u8, usize),
    pub tags: Tags,
}

static HOOKS: OnceLock<Hooks> = OnceLock::new();
static BRIDGE: OnceLock<bridge::Bridge> = OnceLock::new();
static READY: OnceLock<Result<(), String>> = OnceLock::new();

/// The main thread's State, as a word the collector reads as a root.
static MAIN: std::sync::atomic::AtomicPtr<State> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// Hand the API what the frontend owns. Cheap: nothing is compiled or
/// allocated until a native library first opens.
pub fn install(hooks: Hooks) {
    values::set_tags(hooks.tags);
    let _ = HOOKS.set(hooks);
}

/// The bridge; present once [`ready`] has succeeded.
pub(crate) fn bridge() -> &'static bridge::Bridge {
    BRIDGE
        .get()
        .expect("the C API bridge is compiled before C code runs")
}

/// Make the API usable: compile the bridge, check that the program
/// keeps its globals where C code reaches them, and make the main
/// State and the registry. Done once; the answer stands.
pub(crate) fn ready() -> Result<(), String> {
    READY.get_or_init(prepare).clone()
}

fn prepare() -> Result<(), String> {
    let hooks = HOOKS
        .get()
        .ok_or("the C API was not installed by this host")?;
    let resolved = (hooks.resolve)()?;
    let _ = BRIDGE.set(bridge::Bridge::from_resolved(&resolved)?);
    let b = bridge();
    if (b.shared)() == 0 {
        return Err(
            "C code cannot run in this program: its globals are not kept in the globals table"
                .to_string(),
        );
    }
    (hooks.add_root)(
        MAIN.as_ptr() as *const u8,
        std::mem::size_of::<*mut State>(),
    );
    (hooks.add_root)(
        load::RESULT.as_ptr() as *const u8,
        std::mem::size_of::<*mut State>(),
    );
    let global = values::zeroed(std::mem::size_of::<Global>()) as *mut Global;
    let main = State::create(global, std::ptr::null());
    // SAFETY: a fresh block of the Global's size.
    unsafe {
        global.write(Global {
            main,
            light: std::ptr::null(),
            panic: None,
        });
    }
    MAIN.store(main, std::sync::atomic::Ordering::Release);
    // SAFETY: the main State is rooted from here on.
    unsafe {
        (*global).light = (b.new_table)();
    }
    (b.init)();
    Ok(())
}

/// The main thread's State for a host that embeds the running program,
/// the API made ready first: the `lua_State` a C host would have from
/// `luaL_newstate`. The program must keep its globals in the globals
/// table (`zyntax_lua::open_host` opens one that does).
pub fn host_state() -> Result<state::L, String> {
    ready()?;
    // SAFETY: the main State is made once the API is ready, and lives on.
    Ok(unsafe { (*main_state()).l() })
}

/// The main thread's State.
pub(crate) fn main_state() -> *mut State {
    MAIN.load(std::sync::atomic::Ordering::Acquire)
}

/// The State of the Lua thread running now, made the first time C code
/// runs on it.
pub(crate) fn current_state() -> *mut State {
    let b = bridge();
    let co = (b.current)();
    if co.is_null() {
        return main_state();
    }
    thread_state(co)
}

/// The State of coroutine `co`, made on first use.
pub(crate) fn thread_state(co: Any) -> *mut State {
    let b = bridge();
    let p = (b.thread_state)(co);
    if p != 0 {
        return p as *mut State;
    }
    // SAFETY: the main State is made before any other.
    let global = unsafe { (*main_state()).global };
    let s = State::create(global, co);
    (b.set_thread_state)(co, values::box_pointer(s as *mut u8, values::tags().light));
    s
}

mod exports {
    include!(concat!(env!("OUT_DIR"), "/exports.rs"));
}

pub use exports::EXPORT_NAMES;

mod plugin {
    use crate::api::{c_ud_meta, c_ud_set_meta};
    use crate::load::{c_error, c_loaded, c_loadfunc, c_loadlib};

    zrtl::zrtl_plugin! {
        name: "lua_capi",
        symbols: [
            ("$LuaC$loadlib", c_loadlib),
            ("$LuaC$loadfunc", c_loadfunc),
            ("$LuaC$loaded", c_loaded),
            ("$LuaC$error", c_error),
            ("$LuaC$ud_meta", c_ud_meta),
            ("$LuaC$ud_set_meta", c_ud_set_meta),
        ]
    }
}

/// The plugin, for the host's batch of linked plugins. Reading the
/// table of every exported function keeps each of them, and the C
/// objects that define them, in the executable whatever the linker
/// strips.
pub fn static_plugin() -> zrtl::StaticPlugin {
    std::hint::black_box(&exports::EXPORTS);
    plugin::static_plugin()
}
