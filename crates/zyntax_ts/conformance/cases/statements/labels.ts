let found = -1;
outer: for (let i = 0; i < 5; i++) {
    for (let j = 0; j < 5; j++) {
        if (i * j === 6) { found = i * 10 + j; break outer; }
        if (j > i) continue outer;
    }
}
let skipped = 0;
rows: for (let r = 0; r < 4; r++) {
    let c = 0;
    while (c < 4) {
        c++;
        if (c === 2) continue rows;
        skipped++;
    }
}
let total = 0;
block: {
    total += 1;
    if (total > 0) break block;
    total += 100;
}
console.log(found, skipped, total);
