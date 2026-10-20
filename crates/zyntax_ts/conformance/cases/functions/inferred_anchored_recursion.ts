function odd(n: number) { return !even(n); }
function even(n: number): boolean {
    if (n == 0) return true;
    return odd(n - 1);
}
console.log(even(8), odd(8), odd(9));
