let x: { o: boolean; n: number } = { o: false, n: 1 };
if (x["o"] === false) { x["o"] = true; x[`n`] += 2; }
console.log(x.o, x["n"]);
