function make(): number { return 6; }
const base = make();
let hits = 0;
function bump(): number { hits += base; return hits; }
bump();
console.log(bump(), base);
