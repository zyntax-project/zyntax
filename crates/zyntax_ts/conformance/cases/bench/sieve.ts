function sieve(n: number): number {
  const flags: boolean[] = [];
  for (let i = 0; i <= n; i++) { flags.push(true); }
  let count = 0;
  for (let i = 2; i <= n; i++) {
    if (flags[i]) { count++; for (let j = i * 2; j <= n; j += i) { flags[j] = false; } }
  }
  return count;
}
console.log(sieve(20000));
