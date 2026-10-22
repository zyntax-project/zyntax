let h = 0;
for (let i = 0; i < 10; i++) { h = (h * 31 + i) | 0; h = h >>> 1; h ^= i << 2; }
console.log(h, 7 & 3, 5 | 8, 6 ^ 3, ~5, ~~3.7, -1 >>> 0, -8 >> 1, 1 << 31, 1 << 32, 5 << 33);
console.log(2147483648 | 0, 4294967296 | 0, 4294967297 | 0, -2147483649 | 0, 1e10 | 0, NaN | 0, Infinity | 0, 1.9e19 | 0, -1.5e19 | 0, 3.9 | 0, -3.9 | 0);
let x = 12345;
x <<= 3; x >>= 1; x >>>= 2; x |= 1; x &= 0xFFFF; x ^= 0xAAAA;
console.log(x, 0xFFFFFFFF >>> 28, (0xDEADBEEF ^ 0xCAFEBABE) >>> 0, 123456789 * 987 | 0);
function xorshift(seed: number): number { let s = seed | 0; s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return s >>> 0; }
let r = 2463534242; for (let i = 0; i < 5; i++) { r = xorshift(r); }
console.log(r);
