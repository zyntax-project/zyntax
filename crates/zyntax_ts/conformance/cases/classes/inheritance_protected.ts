class Counter {
  protected count = 0;
  private step = 1;
  bump(): number {
    this.count = this.count + this.step;
    return this.count;
  }
}

class Doubler extends Counter {
  constructor(private factor: number) {
    super();
  }
  bump(): number {
    this.count = this.count + this.factor;
    return this.count * 2;
  }
  peek(): number {
    return this.count;
  }
}

function run(c: Counter, times: number): number {
  let last = 0;
  for (let i = 0; i < times; i++) {
    last = c.bump();
  }
  return last;
}

const plain = new Counter();
const doubled = new Doubler(3);
console.log(run(plain, 3), run(doubled, 3), doubled.peek());
