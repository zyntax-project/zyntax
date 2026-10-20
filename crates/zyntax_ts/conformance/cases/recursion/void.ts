function discarded(n: number) { if (n > 0) discarded(n - 1); }
function partial(n: number) { if (n > 0) return partial(n - 1); }
discarded(5); partial(0); console.log(7);
