class Builder {
  count = 0;
  add(): number {
    return this.count;
  }
  step() {
    this.count = this.count + 1;
    return this;
  }
}
class Labeled extends Builder {
  tag = 5;
  mark() {
    this.tag = this.tag + 10;
    return this;
  }
}
const b = new Labeled();
const r = b.step().mark().step().mark();
console.log(r.count, r.tag, b.step().tag);
