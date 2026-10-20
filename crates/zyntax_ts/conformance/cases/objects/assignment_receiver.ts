type Cell = { x: number };
function plain(): number {
    let a: Cell = { x: 1 };
    const b: Cell = { x: 9 };
    const old: Cell = a;
    a.x = (a = b).x;
    return old.x * 10 + a.x;
}
function compound(): number {
    let a: Cell = { x: 1 };
    const b: Cell = { x: 9 };
    const old: Cell = a;
    a.x += (a = b).x;
    return old.x * 10 + a.x;
}
console.log(plain(), compound());
