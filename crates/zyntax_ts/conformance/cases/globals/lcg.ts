let seed = 42;
function next(): number {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed / 2147483648;
}
function id<T>(x: T): T { return x; }
class Sampler {
    total: number = 0;
    draw(): void { this.total += next(); }
}
const sampler = new Sampler();
for (let i = 0; i < 4; i++) sampler.draw();
console.log(sampler.total, id(seed));
