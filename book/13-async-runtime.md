# Async Native Runtime

This chapter covers the Promise-based async native runtime in Zyntax. The async system compiles async functions to efficient state machines that can be executed with native performance.

## Overview

Zyntax provides first-class async support through:

- **Promise-based ABI**: Async functions return `*Promise<T>` pointers containing state machine and poll function references
- **State Machine Compilation**: Async functions are compiled to efficient state machines
- **Native Polling**: Poll functions use the C ABI for maximum performance
- **Rust Integration**: `ZyntaxPromise` implements the Rust `Future` trait for seamless integration

## How Async Compilation Works

When you write an async function:

```zig
async fn double(x: i32) i32 {
    return x * 2;
}
```

The compiler generates two functions:

1. **Entry Function** (`double`): Returns a `*Promise<i32>` containing:
   - A pointer to the allocated state machine
   - A pointer to the internal poll function

2. **Poll Function** (`__double_poll`): The internal state machine implementation:
   - Takes a state machine pointer as input
   - Returns poll result as `i64` (0 = Pending, non-zero = Ready(value))

### Promise Memory Layout

The Promise struct has a 16-byte layout on 64-bit systems:

```
+------------------+------------------+
| state_machine    | poll_fn          |
| *mut u8 (8 bytes)| fn ptr (8 bytes) |
+------------------+------------------+
offset 0           offset 8
```

### State Machine Memory Layout

The state machine struct is dynamically sized:

```
+----------+----------+----------+-----+
| state    | capture1 | capture2 | ... |
| u32      | T1       | T2       | ... |
+----------+----------+----------+-----+
offset 0   offset 4   offset 8   ...
```

- **state**: Current state index (0 = initial, higher = after await points)
- **captures**: Function parameters and locals that need to survive across await points

## Using Async Functions from Rust

### Basic Async Call

```rust
use zyntax_embed::{ZyntaxRuntime, LanguageGrammar, ZyntaxValue, ZyntaxPromise};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let grammar = LanguageGrammar::compile_zyn_file("grammars/zig.zyn")?;
    let mut runtime = ZyntaxRuntime::new()?;

    runtime.compile_with_grammar(&grammar, r#"
        async fn compute(x: i32) i32 {
            return x * 2;
        }
    "#)?;

    // Call async function - returns a Promise
    let promise: ZyntaxPromise = runtime.call_async("compute", &[ZyntaxValue::Int(21)])?;

    // Block until completion
    let result: i32 = promise.await_result()?;
    assert_eq!(result, 42);

    Ok(())
}
```

### Manual Polling

For non-blocking execution, poll the promise manually:

```rust
use zyntax_embed::{ZyntaxPromise, PromiseState};

let promise = runtime.call_async("long_running_task", &[])?;

loop {
    match promise.poll() {
        PromiseState::Pending => {
            // Do other work while waiting
            std::thread::yield_now();
        }
        PromiseState::Ready(value) => {
            println!("Task completed with: {:?}", value);
            break;
        }
        PromiseState::Failed(error) => {
            eprintln!("Task failed: {}", error);
            break;
        }
    }
}
```

### Poll with Limit

Limit the number of polls for timeout behavior:

```rust
let promise = runtime.call_async("compute", &[21.into()])?;

// Poll up to 100 times
match promise.poll_with_limit(100) {
    PromiseState::Ready(value) => {
        println!("Completed: {:?}", value);
    }
    PromiseState::Pending => {
        println!("Still pending after 100 polls");
    }
    PromiseState::Failed(e) => {
        println!("Failed: {}", e);
    }
}
```

### Await with Timeout

Use deadline-based waiting:

```rust
use std::time::Duration;

let promise = runtime.call_async("slow_task", &[])?;

match promise.await_with_timeout(Duration::from_secs(5)) {
    Ok(value) => println!("Got: {:?}", value),
    Err(e) => println!("Timeout or error: {}", e),
}
```

### Promise Chaining

Chain transformations using `.then()` and `.catch()`:

```rust
let promise = runtime.call_async("fetch_data", &[url.into()])?;

let processed = promise
    .then(|data| {
        // Transform successful result
        ZyntaxValue::String(format!("Processed: {:?}", data))
    })
    .catch(|error| {
        // Handle errors
        ZyntaxValue::String(format!("Error: {}", error))
    });

let result = processed.await_result()?;
```

### Async/Await Integration

`ZyntaxPromise` implements `std::future::Future`, enabling use with Rust's async/await:

```rust
async fn process_async(runtime: &ZyntaxRuntime) -> Result<i32, RuntimeError> {
    let promise = runtime.call_async("compute", &[10.into()])?;

    // Use .await directly
    let result: i32 = promise.await?;

    Ok(result)
}

// Use with any async runtime
#[tokio::main]
async fn main() {
    let runtime = ZyntaxRuntime::new().unwrap();
    // ... compile code ...

    let result = process_async(&runtime).await.unwrap();
    println!("Result: {}", result);
}
```

## Poll Convention

The poll function returns an `i64` with the following convention:

| Return Value | Meaning |
|--------------|---------|
| `0` | `Pending` - Not yet complete, poll again later |
| `> 0` | `Ready(value)` - Completed with positive value |
| `< 0` | `Ready(value)` - Completed with negative value |

For void async functions, `Ready` returns `1`.

## State Machine Internals

### Capture Analysis

The compiler analyzes which variables need to be "captured" (stored in the state machine) to survive across await points:

```zig
async fn example(x: i32, y: i32) i32 {
    const a = x + 1;      // Used after await, must be captured
    await some_future();   // <-- Await point
    return a + y;          // 'a' and 'y' needed here
}
```

The state machine for this function captures:
- `x` (parameter, needed to compute `a`)
- `y` (parameter, needed in return)
- `a` (local, computed before await, used after)

### Simple vs. Multi-State Functions

**Simple async functions** (no await points) use a streamlined wrapper:

```rust
// Compiled as single-state: just loads params, executes, returns Ready
async fn simple(x: i32) i32 {
    return x * 2;
}
```

**Multi-state async functions** (with await points) generate switch dispatch:

```rust
// Compiled with state machine dispatch
async fn complex(x: i32) i32 {
    const a = x + 1;
    await delay(100);  // State 0 -> State 1
    const b = a * 2;
    await delay(200);  // State 1 -> State 2
    return b;
}
```

## Writing Async-Aware Grammars

To support async in your language grammar, define the async keyword and function modifiers:

```zyn
// Function definition with optional async modifier
// async_modifier? gives Option<()>; .is_some() converts to bool
function_def = {
    async_modifier? ~ "fn" ~ name:identifier
    ~ "(" ~ params:param_list? ~ ")"
    ~ ret:return_type?
    ~ body:block
}
  -> TypedDeclaration::Function {
      name: intern(name),
      params: params.unwrap_or([]),
      return_type: ret,
      body: Some(body),
      is_async: async_modifier.is_some(),
  }

async_modifier = { "async" }
```

The `is_async: async_modifier.is_some()` field tells the compiler to apply async transformation when the `async` keyword is present.

## Performance Considerations

### Allocation

- **State machine**: Heap-allocated via `malloc` on each async call
- **Promise**: Heap-allocated (16 bytes) on each async call
- **Cleanup**: Currently manual; future versions will integrate with ARC

### Overhead

| Operation | Overhead |
|-----------|----------|
| Async call | ~1 malloc (state machine) + 1 malloc (Promise) |
| Poll | Function call + state load + potential state store |
| Await | Repeated polling until Ready |

For hot async paths, consider:
1. Reusing state machines when possible
2. Using sync functions for simple computations
3. Batching async operations

## TieredRuntime and Async

The `TieredRuntime` automatically optimizes async functions:

```rust
use zyntax_embed::TieredRuntime;

let mut runtime = TieredRuntime::production()?;

// Async functions start at Tier 0 (baseline)
runtime.load_module("zig", r#"
    async fn hot_async(x: i32) i32 {
        return x * 2;
    }
"#)?;

// After enough calls, automatically promoted to optimized tiers
for _ in 0..10000 {
    let promise = runtime.call_async("hot_async", &[i.into()])?;
    let _: i32 = promise.await_result()?;
}

// Check tier
let tier = runtime.get_function_tier("hot_async")?;
println!("Optimization tier: {:?}", tier);
```

## Real-World Examples

### Example 1: Async Loops with State

Async functions can contain while loops with mutable state. The state machine correctly captures and restores loop variables across poll boundaries:

```zig
async fn sum_range(n: i32) i32 {
    var total: i32 = 0;
    var i: i32 = 1;
    while (i <= n) {
        total = total + i;
        i = i + 1;
    }
    return total;
}
```

```rust
let promise = runtime.call_async("sum_range", &[ZyntaxValue::Int(100)])?;

// Poll until completion
while promise.is_pending() {
    promise.poll();
}

// sum_range(100) = 1+2+...+100 = 5050
assert_eq!(promise.state(), PromiseState::Ready(ZyntaxValue::Int(5050)));
```

### Example 2: Await Inside Loops

A powerful pattern is awaiting other async functions inside loops. Each iteration creates a new nested Promise:

```zig
async fn double(x: i32) i32 {
    return x * 2;
}

async fn sum_doubled(n: i32) i32 {
    var total: i32 = 0;
    var i: i32 = 1;
    while (i <= n) {
        const doubled = await double(i);  // Await in loop!
        total = total + doubled;
        i = i + 1;
    }
    return total;
}
```

```rust
let promise = runtime.call_async("sum_doubled", &[ZyntaxValue::Int(5)])?;

while promise.is_pending() {
    promise.poll();
}

// sum_doubled(5) = double(1) + double(2) + ... + double(5)
//                = 2 + 4 + 6 + 8 + 10 = 30
assert_eq!(promise.state(), PromiseState::Ready(ZyntaxValue::Int(30)));
```

### Example 3: Chained Async Calls

Async functions can await other async functions and perform additional computation:

```zig
async fn step1(x: i32) i32 {
    return x + 10;
}

async fn step2(x: i32) i32 {
    const result = await step1(x);
    return result * 2;
}

async fn step3(x: i32) i32 {
    const result = await step2(x);
    return result + 5;
}
```

```rust
let promise = runtime.call_async("step3", &[ZyntaxValue::Int(5)])?;

while promise.is_pending() {
    promise.poll();
}

// step3(5) = step2(5) + 5 = (step1(5) * 2) + 5 = ((5+10) * 2) + 5 = 35
assert_eq!(promise.state(), PromiseState::Ready(ZyntaxValue::Int(35)));
```

### Example 4: Long-Running Process with Await

Await a long-running async function and process the result:

```zig
// A long-running async process that sums 1 to n
async fn long_sum(n: i32) i32 {
    var total: i32 = 0;
    var i: i32 = 1;
    while (i <= n) {
        total = total + i;
        i = i + 1;
    }
    return total;
}

// Awaits the long-running process and adds a constant
async fn add_to_sum(n: i32) i32 {
    const sum = await long_sum(n);
    return sum + 100;
}
```

```rust
let promise = runtime.call_async("add_to_sum", &[ZyntaxValue::Int(50)])?;

while promise.is_pending() {
    promise.poll();
}

// add_to_sum(50) = long_sum(50) + 100 = 1275 + 100 = 1375
assert_eq!(promise.state(), PromiseState::Ready(ZyntaxValue::Int(1375)));
```

### Example 5: Multiple Parameters

Async functions handle multiple parameters correctly:

```zig
async fn sum_with_multiplier(start: i32, end: i32, multiplier: i32) i32 {
    var total: i32 = 0;
    var i: i32 = start;
    while (i <= end) {
        total = total + (i * multiplier);
        i = i + 1;
    }
    return total;
}
```

```rust
let promise = runtime.call_async("sum_with_multiplier", &[
    ZyntaxValue::Int(1),
    ZyntaxValue::Int(5),
    ZyntaxValue::Int(2),
])?;

while promise.is_pending() {
    promise.poll();
}

// sum_with_multiplier(1, 5, 2) = (1*2)+(2*2)+(3*2)+(4*2)+(5*2) = 30
assert_eq!(promise.state(), PromiseState::Ready(ZyntaxValue::Int(30)));
```

## Promise Combinators (Parallel Execution)

Zyntax provides JavaScript-style Promise combinators for running multiple async operations in parallel.

### PromiseAll - Wait for All

`PromiseAll` waits for all promises to complete, similar to JavaScript's `Promise.all()`:

```rust
use zyntax_embed::{ZyntaxRuntime, ZyntaxValue, PromiseAll};

// Define multiple async functions
runtime.load_module("zig", r#"
async fn compute(x: i32) i32 {
    return x * 2;
}

async fn sum_range(n: i32) i32 {
    var total: i32 = 0;
    var i: i32 = 1;
    while (i <= n) {
        total = total + i;
        i = i + 1;
    }
    return total;
}
"#)?;

// Create multiple promises
let promises = vec![
    runtime.call_async("compute", &[ZyntaxValue::Int(5)])?,     // 10
    runtime.call_async("compute", &[ZyntaxValue::Int(10)])?,    // 20
    runtime.call_async("sum_range", &[ZyntaxValue::Int(100)])?, // 5050
];

// Wait for all to complete
let mut all = PromiseAll::new(promises);
let results = all.await_all()?;

// results = [Int(10), Int(20), Int(5050)]
println!("All completed after {} polls", all.poll_count());
```

If any promise fails, `PromiseAll` returns the first error immediately (fast-fail).

### PromiseRace - First to Complete

`PromiseRace` resolves as soon as any promise completes, similar to `Promise.race()`:

```rust
use zyntax_embed::{ZyntaxRuntime, ZyntaxValue, PromiseRace};

runtime.load_module("zig", r#"
async fn quick(x: i32) i32 {
    return x * 2;
}

async fn slow(n: i32) i32 {
    var total: i32 = 0;
    var i: i32 = 1;
    while (i <= n) {
        total = total + i;
        i = i + 1;
    }
    return total;
}
"#)?;

let promises = vec![
    runtime.call_async("slow", &[ZyntaxValue::Int(1000)])?,  // Takes many polls
    runtime.call_async("quick", &[ZyntaxValue::Int(21)])?,   // Completes quickly
];

let mut race = PromiseRace::new(promises);
let (winner_index, value) = race.await_first()?;

// winner_index = index of first promise to complete
// value = result from that promise
println!("Promise {} won with {:?}", winner_index, value);
```

### PromiseAllSettled - Collect All Results

`PromiseAllSettled` waits for all promises regardless of success or failure:

```rust
use zyntax_embed::{ZyntaxRuntime, ZyntaxValue, PromiseAllSettled, SettledResult};

let promises = vec![
    runtime.call_async("compute", &[ZyntaxValue::Int(1)])?,
    runtime.call_async("compute", &[ZyntaxValue::Int(2)])?,
    runtime.call_async("compute", &[ZyntaxValue::Int(3)])?,
];

let mut settled = PromiseAllSettled::new(promises);
let results = settled.await_all();

for (i, result) in results.iter().enumerate() {
    match result {
        SettledResult::Fulfilled(value) => println!("Promise {} succeeded: {:?}", i, value),
        SettledResult::Rejected(error) => println!("Promise {} failed: {}", i, error),
    }
}
```

### Timeout Support

All combinators support timeout-based waiting:

```rust
use std::time::Duration;

// PromiseAll with timeout
let mut all = PromiseAll::new(promises);
match all.await_all_with_timeout(Duration::from_secs(5)) {
    Ok(results) => println!("All completed: {:?}", results),
    Err(e) => println!("Timeout or error: {}", e),
}

// PromiseRace with timeout
let mut race = PromiseRace::new(promises);
match race.await_first_with_timeout(Duration::from_secs(1)) {
    Ok((index, value)) => println!("Winner: {} = {:?}", index, value),
    Err(e) => println!("Timeout or error: {}", e),
}
```

### Manual Polling for Combinators

For non-blocking execution, poll the combinator manually:

```rust
use zyntax_embed::{PromiseAll, PromiseAllState};

let mut all = PromiseAll::new(promises);

loop {
    match all.poll() {
        PromiseAllState::Pending => {
            // Do other work while waiting
            process_other_events();
        }
        PromiseAllState::AllReady(values) => {
            println!("All {} promises completed!", values.len());
            break;
        }
        PromiseAllState::Failed(error) => {
            println!("A promise failed: {}", error);
            break;
        }
    }
}
```

## Async Computation Pipeline (Legacy Example)

```rust
use zyntax_embed::{ZyntaxRuntime, LanguageGrammar, ZyntaxValue};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let grammar = LanguageGrammar::compile_zyn_file("grammars/zig.zyn")?;
    let mut runtime = ZyntaxRuntime::new()?;

    runtime.compile_with_grammar(&grammar, r#"
        async fn step1(x: i32) i32 {
            return x + 10;
        }

        async fn step2(x: i32) i32 {
            return x * 2;
        }

        async fn pipeline(input: i32) i32 {
            const a = step1(input);
            const b = step2(a);
            return b;
        }
    "#)?;

    // Execute the async pipeline
    let promise = runtime.call_async("pipeline", &[ZyntaxValue::Int(5)])?;

    // Poll until complete
    let mut polls = 0;
    loop {
        polls += 1;
        match promise.poll() {
            PromiseState::Ready(value) => {
                println!("Pipeline result: {:?} (after {} polls)", value, polls);
                break;
            }
            PromiseState::Pending => continue,
            PromiseState::Failed(e) => {
                eprintln!("Pipeline failed: {}", e);
                break;
            }
        }
    }

    Ok(())
}
```

## Debugging Async Functions

### Logging

Enable trace logging to see async compilation details:

```bash
RUST_LOG=zyntax_compiler::async_support=trace cargo run
```

### Inspecting Generated Functions

The compiler generates predictable function names:

| Original Function | Generated Functions |
|-------------------|---------------------|
| `async fn foo(...)` | `foo` (entry), `__foo_poll` (internal) |
| `async fn bar(...)` | `bar` (entry), `__bar_poll` (internal) |

List functions to verify:

```rust
let functions = runtime.load_module("zig", async_source)?;
println!("Generated: {:?}", functions);
// Output: ["foo", "__foo_poll", "bar", "__bar_poll"]
```

### Common Issues

1. **Promise returns wrong value**: Check that the return type matches the expected size (i32 → i64 extension)

2. **Infinite polling**: Ensure your async function actually completes (has a return path)

3. **Capture errors**: If parameters aren't available after await, check that capture analysis includes them

## API Reference

### ZyntaxPromise

```rust
impl ZyntaxPromise {
    /// Poll once, returning current state
    pub fn poll(&self) -> PromiseState;

    /// Poll up to `limit` times
    pub fn poll_with_limit(&self, limit: usize) -> PromiseState;

    /// Block until completion
    pub fn await_result<T: FromZyntax>(&self) -> Result<T, RuntimeError>;

    /// Block with timeout
    pub fn await_with_timeout(&self, timeout: Duration) -> Result<ZyntaxValue, RuntimeError>;

    /// Chain a success handler
    pub fn then<F>(self, f: F) -> Self where F: Fn(ZyntaxValue) -> ZyntaxValue;

    /// Chain an error handler
    pub fn catch<F>(self, f: F) -> Self where F: Fn(String) -> ZyntaxValue;

    /// Get the number of polls so far
    pub fn poll_count(&self) -> usize;

    /// Check if completed
    pub fn is_completed(&self) -> bool;
}
```

### PromiseState

```rust
pub enum PromiseState {
    /// Not yet complete
    Pending,
    /// Completed successfully with value
    Ready(ZyntaxValue),
    /// Failed with error message
    Failed(String),
}
```

### Runtime Methods

```rust
impl ZyntaxRuntime {
    /// Call an async function, returning a Promise
    pub fn call_async(&self, name: &str, args: &[ZyntaxValue]) -> Result<ZyntaxPromise, RuntimeError>;
}

impl TieredRuntime {
    /// Call an async function with tiered optimization
    pub fn call_async(&self, name: &str, args: &[ZyntaxValue]) -> Result<ZyntaxPromise, RuntimeError>;
}
```

## Next Steps

- See [Embedding SDK](./12-embedding-sdk.md) for general runtime usage
- See [HIR Builder](./11-hir-builder.md) for custom backend integration
- See [Grammar Syntax](./04-grammar-syntax.md) for writing async-aware grammars
