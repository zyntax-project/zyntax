class Temperature {
  private celsius = 0;
  get fahrenheit(): number {
    return this.celsius * 9 / 5 + 32;
  }
  set fahrenheit(value: number) {
    this.celsius = (value - 32) * 5 / 9;
  }
  get kelvin(): number {
    return this.celsius + 273;
  }
}
class Scaled extends Temperature {
  get kelvin(): number {
    return 1000;
  }
}
function read(t: Temperature): number {
  return t.kelvin;
}
const t = new Temperature();
t.fahrenheit = 212;
const s = new Scaled();
const assigned = (s.fahrenheit = 32);
console.log(t.fahrenheit, t.kelvin, read(t), read(s), assigned, s.fahrenheit);
