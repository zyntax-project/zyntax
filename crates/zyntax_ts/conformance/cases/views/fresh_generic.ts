interface X { x: number }
function identity<T>(p: T): T { return p; } identity<X>({ x: 1, y: 2 });
