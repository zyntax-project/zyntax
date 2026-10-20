function total(x: number) {
    const a: number = middle(x);
    return a + middle(x + 1);
}
function middle(x: number) { return leaf(x) + leaf(x + 1); }
function leaf(x: number) { return x * 2; }
console.log(total(3));
