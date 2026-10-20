type First = { readonly x: number };
type Second = { readonly x: number };
type Mutable = { x: number };
function choose(flag: boolean, a: First, b: Second) {
    if (flag) return a;
    return b;
}
function chooseMutable(flag: boolean, a: Mutable) {
    if (flag) return a;
    return { x: 8 };
}
const a: First = { x: 3 };
const b: Second = { x: 7 };
const c: Mutable = { x: 2 };
console.log(choose(false, a, b).x, chooseMutable(false, c).x);
