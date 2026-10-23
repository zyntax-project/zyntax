function describe(n: number): string {
    switch (n % 3) {
        case 0: { const label = "fizz"; return label; }
        case 1: { let count = n; count++; return "next " + count; }
        default: return "other";
    }
}
let out = "";
for (let i = 0; i < 4; i++) {
    out = out + describe(i) + ";";
}
console.log(out);
