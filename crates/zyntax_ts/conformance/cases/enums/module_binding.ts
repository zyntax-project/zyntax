enum Mode { Slow = 1, Fast = 2 }
let mode = Mode.Slow;
let budget: number = 0;
function speed(): void { mode = Mode.Fast; }
function spend(): void { budget += mode === Mode.Fast ? 10 : 1; }
spend(); speed(); spend(); spend();
console.log(mode, budget);
