class Counter {
  static count = 0;
  static label: string = "hits";
  static limit: number;
  id: number;
  constructor() {
    Counter.count = Counter.count + 1;
    this.id = Counter.count;
  }
  static reset(): void {
    Counter.count = 0;
  }
}
class Sub extends Counter {
  static extra = Counter.count + 10;
}
function make(n: number): number {
  for (let i = 0; i < n; i++) {
    new Counter();
  }
  return Counter.count;
}
console.log(Counter.count, make(3), Sub.extra, Sub.count, Counter.label);
Counter.reset();
Counter.limit = 5;
console.log(Counter.count, new Counter().id, Counter.limit);
