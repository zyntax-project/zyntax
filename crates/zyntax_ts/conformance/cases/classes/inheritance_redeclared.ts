class Base {
  size = 1;
  mark = 0;
  area(): number {
    return this.size * 10 + this.mark;
  }
}
class Big extends Base {
  extra = 3;
  size = this.extra + 4;
}
class Bigger extends Big {
  mark = 2;
}
const items: Base[] = [new Base(), new Big(), new Bigger()];
let sum = 0;
for (const item of items) {
  sum = sum + item.area();
}
console.log(new Base().area(), new Big().area(), new Bigger().area(), sum);
