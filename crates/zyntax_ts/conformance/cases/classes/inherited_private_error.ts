class Account {
  private balance = 10;
  read(): number {
    return this.balance;
  }
}

class Savings extends Account {
  steal(): number {
    return this.balance;
  }
}
console.log(new Savings().steal());
