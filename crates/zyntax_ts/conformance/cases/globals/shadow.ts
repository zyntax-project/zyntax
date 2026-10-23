let value = 7;
function param(value: number): number { return value + 1; }
function local(): number { const value = 100; return value; }
function outer(): number { return value; }
function nested(): number { { let value = 1; value++; } return value; }
console.log(param(1), local(), outer(), nested());
