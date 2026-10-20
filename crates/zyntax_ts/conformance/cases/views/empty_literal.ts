interface Empty {}
function make(): Empty { return { x: 1, y: 2 }; }
const p: Empty = { x: 3, y: 4 }; const q = make();
console.log(p === p, q === q);
