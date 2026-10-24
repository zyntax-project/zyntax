class Square extends Shape {
  side = 2;
}

class Shape {
  area(): number {
    return 0;
  }
}
console.log(new Square().area());
