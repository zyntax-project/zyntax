function label(x: number) {
    const suffix: string = "!";
    if (x > 0) return "yes" + suffix;
    return "no" + suffix;
}
function positive(x: number) { return x > 0; }
function absolute(x: number) { if (x < 0) return -x; return x; }
console.log(label(1), label(-1), positive(1), positive(-1), absolute(-3));
