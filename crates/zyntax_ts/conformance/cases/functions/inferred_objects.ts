type Point = { x: number; y: number };
function make(x: number) { return { y: 2, x: x }; }
function alias(p: Point) { return p; }
function score(x: number) {
    const p: Point = make(x);
    const q: Point = alias(p);
    q.x += 1;
    return p.x * 10 + q.y;
}
let sum = 0;
for (let i = 0; i < 300; i++) sum += score(i);
console.log(sum);
