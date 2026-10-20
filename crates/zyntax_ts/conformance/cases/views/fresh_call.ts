interface X { x: number }
function f(p: X): number { return p.x; } f({ x: 1, y: 2 });
