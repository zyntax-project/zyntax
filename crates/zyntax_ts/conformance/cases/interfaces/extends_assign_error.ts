interface Point { x: number; y: number }
interface Point3 extends Point { z: number }
const p: Point3 = { x: 1, y: 2 };
