enum E { A = 1, B = 2 }
let x: E = E.A;
while (x !== E.B) { x = E.B; }
console.log(x === E.A);
