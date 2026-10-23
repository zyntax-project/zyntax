enum State { Idle, Running, Done }
function step(s: State): State {
    let next = s;
    if (next === State.Idle) {
        next = State.Running;
    } else if (next === State.Running) {
        next = State.Done;
    }
    return next;
}
let s = State.Idle;
let rounds = 0;
while (s !== State.Done) { s = step(s); rounds++; }
console.log(rounds, s);
