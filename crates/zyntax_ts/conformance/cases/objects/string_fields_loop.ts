class Tag { label: string = "start"; weight: number = 0; }
function make(i: number): Tag { const t = new Tag(); t.weight = i; if (i % 2 === 0) t.label = "even"; return t; }
function run(n: number): number {
    let evens = 0;
    for (let i = 0; i < n; i++) {
        const t = make(i);
        const r = { name: t.label, score: t.weight };
        if (r.name === "even") evens++;
    }
    return evens;
}
console.log(run(200000));
