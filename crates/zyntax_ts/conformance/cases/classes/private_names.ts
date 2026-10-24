class Account {
  #balance = 0;
  #limit: number;
  static #opened = 0;
  x = 1;
  constructor(limit: number) {
    this.#limit = limit;
    Account.#opened = Account.#opened + 1;
  }
  #clamp(value: number): number {
    return value > this.#limit ? this.#limit : value;
  }
  deposit(amount: number): number {
    this.#balance = this.#clamp(this.#balance + amount);
    this.#balance++;
    return this.#balance;
  }
  static opened(): number {
    return Account.#opened;
  }
}
const a = new Account(10);
const b = new Account(100);
console.log(a.deposit(5), a.deposit(20), b.deposit(3), Account.opened(), a.x);
