//! Handler segments for a host that switches between stacks on one
//! thread: a `with` scope a stack leaves open is in scope on that stack
//! alone, and comes back when the stack runs again.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::host_futures::{handler_stack_depth, task_handler_segment_count};
use zyntax_embed::{TieredConfig, TieredRuntime};

const SRC: &str = r#"
effect Counter {
    def bump()
    def total(): i64
}

handler Tally for Counter {
    var n: i64 = 0
    def bump() { self.n = self.n + 1 }
    def total(): i64 { return self.n }
}

@effect(Counter)
def bump_and_read(): i64 { bump() return total() }
"#;

fn runtime() -> TieredRuntime {
    let mut rt = TieredRuntime::new(TieredConfig::development()).expect("runtime should start");
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(SRC, "<handler_segments>")
        .expect("parse");
    rt.compile_typed_program(program).expect("compile");
    rt
}

#[test]
fn a_frame_left_open_on_one_stack_is_in_scope_there_alone() {
    let mut rt = runtime();
    let token = rt.get_effect_handler("Tally").expect("resolve Tally");
    let instance = rt.new_handler_instance(token).expect("mint");
    let (a, b) = (rt.new_handler_segment(), rt.new_handler_segment());
    assert_ne!(a, b);
    let depth = handler_stack_depth();

    // Stack A installs a handler and is switched away from with it open.
    let scope = rt.enter_handler_segment(a);
    let frame = rt.push_handler_instance(instance).expect("install");
    assert_eq!(rt.call::<i64>("bump_and_read", &[]).ok(), Some(1));
    rt.leave_handler_segment(a, scope);
    assert_eq!(handler_stack_depth(), depth, "A's frame is lifted off");

    // Stack B runs with nothing of A's in scope.
    let scope = rt.enter_handler_segment(b);
    let out = rt.call::<i64>("bump_and_read", &[]);
    assert!(out.is_err(), "B sees A's handler: {out:?}");
    rt.leave_handler_segment(b, scope);

    // A runs again and finds its handler and its state where it left them.
    let scope = rt.enter_handler_segment(a);
    assert_eq!(rt.call::<i64>("bump_and_read", &[]).ok(), Some(2));
    rt.pop_effect_handler(frame);
    rt.leave_handler_segment(a, scope);
    assert_eq!(handler_stack_depth(), depth);
    rt.forget_handler_segment(a);
    rt.forget_handler_segment(b);
}

#[test]
fn forgetting_a_segment_drops_the_frames_it_left_open() {
    let mut rt = runtime();
    let token = rt.get_effect_handler("Tally").expect("resolve Tally");
    let instance = rt.new_handler_instance(token).expect("mint");
    let id = rt.new_handler_segment();
    let before = task_handler_segment_count();

    let scope = rt.enter_handler_segment(id);
    rt.push_handler_instance(instance).expect("install");
    rt.leave_handler_segment(id, scope);
    assert_eq!(task_handler_segment_count(), before + 1);

    rt.forget_handler_segment(id);
    assert_eq!(task_handler_segment_count(), before);
    let scope = rt.enter_handler_segment(id);
    let out = rt.call::<i64>("bump_and_read", &[]);
    rt.leave_handler_segment(id, scope);
    assert!(
        out.is_err(),
        "a forgotten segment still has its frame: {out:?}"
    );
}
