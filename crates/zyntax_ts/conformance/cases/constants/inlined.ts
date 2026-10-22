const N = 8;
const SCALE: number = N * 2 + 1;
const HALF = SCALE / 2;
const LIMIT = -N;
const DEBUG = false;
const READY = !DEBUG && N > 3;
const NAME = "ab" + "cd";
const TAU = Math.PI * 2;
function area(r: number): number { return TAU / 2 * r * r; }
function sum(xs: number[]): number { let t = 0; for (let i = 0; i < N; i++) { t += xs[i] * SCALE; } return t; }
function label(): string { return NAME + "!"; }
function half(): number { return HALF + LIMIT; }
function shadow(N: number): number { return N + 1; }
function ready(): boolean { return READY; }
const xs: number[] = [1, 2, 3, 4, 5, 6, 7, 8];
console.log(area(2), sum(xs), label(), half(), shadow(100), ready());
