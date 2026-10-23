let running = true;
let rounds = 0;
const label = "rounds";
let mode = "fast";
function run(limit: number): void {
    while (running) {
        rounds++;
        if (rounds >= limit) running = false;
    }
}
function describe(): string { return label + ":" + mode; }
function slow(): void { mode = "slow"; }
run(7);
console.log(rounds, running, describe());
slow();
console.log(describe());
