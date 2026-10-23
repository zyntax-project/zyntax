let total: number;
function f(): number { return total; }
function decide(): boolean { return true; }
if (decide()) { total = 1; }
console.log(total, f());
