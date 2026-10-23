function pick(k: number): number {
    let v: number;
    switch (k) {
        case 0: v = 10; break;
        default: v = 20;
    }
    return v;
}
console.log(pick(0), pick(3));
