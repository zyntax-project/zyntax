class Shape {
  area(): number {
    return 0;
  }
}

class Square extends Shape {
  side = 2;
  constructor() {
    console.log(this.side);
  }
}
console.log(new Square().area());
