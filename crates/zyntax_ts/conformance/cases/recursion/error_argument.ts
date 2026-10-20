function identity(n: number): number { return n; }
function loop(n: number) { return identity(loop(n)); }
