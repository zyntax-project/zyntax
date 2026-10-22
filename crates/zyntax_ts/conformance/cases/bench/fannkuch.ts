function fannkuch(n: number): number {
  const perm: number[] = []; const perm1: number[] = []; const count: number[] = [];
  for (let i = 0; i < n; i++) { perm.push(0); perm1.push(i); count.push(0); }
  let maxFlips = 0, r = n;
  while (true) {
    while (r !== 1) { count[r - 1] = r; r--; }
    for (let i = 0; i < n; i++) { perm[i] = perm1[i]; }
    let flips = 0, k = perm[0];
    while (k !== 0) {
      let i = 0, j = k;
      while (i < j) { const t = perm[i]; perm[i] = perm[j]; perm[j] = t; i++; j--; }
      flips++; k = perm[0];
    }
    if (flips > maxFlips) { maxFlips = flips; }
    while (true) {
      if (r === n) { return maxFlips; }
      const p0 = perm1[0];
      for (let i = 0; i < r; i++) { perm1[i] = perm1[i + 1]; }
      perm1[r] = p0;
      count[r] = count[r] - 1;
      if (count[r] > 0) { break; }
      r++;
    }
  }
}
console.log(fannkuch(7));
