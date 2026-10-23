enum Direction { Up, Down = 5, Left, Right = Left * 2 }
const enum Flags { None = 0, A = 1 << 0, B = 1 << 1, AB = A | B }
function turn(d: Direction): Direction {
    if (d === Direction.Up) return Direction.Down;
    if (d === Direction.Down) return Direction.Left;
    return Direction.Up;
}
let d = Direction.Up;
d = turn(d);
console.log(d, turn(d), Direction.Right, Flags.AB, Flags.A | Flags.B);
let n: number = Direction.Left + 1;
const k: Direction = n;
console.log(n, k === Direction.Right, typeof Direction.Up);
