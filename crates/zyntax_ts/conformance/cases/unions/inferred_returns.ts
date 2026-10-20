function partial(b: boolean) { if (b) return 7; }
function mixed(b: boolean) { if (b) return undefined; return 2; }
function bare(b: boolean) { if (b) return; return 5; }
function absent(): undefined {}
function empty() { return; }
function noValue(): void { return undefined; }
console.log(partial(true), partial(false), mixed(true), mixed(false));
console.log(bare(true), bare(false), absent(), empty(), noValue());
