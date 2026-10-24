class Shape {
  area(): number {
    return 0;
  }
}

class Square extends Shape {
  side = 2;
  constructor() {
    this.side = 3;
    super();
  }
}
console.log(new Square().area());
