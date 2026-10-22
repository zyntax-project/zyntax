import {identity,make,alias,Read} from './barrel';const p:Read={x:7};const q=identity(p);const xs=make(q.x);const ys=alias(xs);ys[1]=ys[1]+3;console.log(xs[1]+p.x);
