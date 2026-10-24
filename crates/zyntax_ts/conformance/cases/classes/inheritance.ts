class Animal {
  legs: number;
  constructor(legs: number) {
    this.legs = legs;
  }
  speak(): number {
    return this.legs;
  }
  describe(): number {
    return this.speak() * 10;
  }
}

class Dog extends Animal {
  tail = 1;
  constructor() {
    super(4);
  }
  speak(): number {
    return this.legs + this.tail;
  }
}

class Puppy extends Dog {
  speak(): number {
    return super.speak() + 100;
  }
}

function total(a: Animal): number {
  return a.describe();
}

const a = new Animal(2);
const d = new Dog();
const p = new Puppy();
console.log(a.speak(), d.speak(), p.speak());
console.log(total(a), total(d), total(p));
console.log(d.legs, p.tail);
