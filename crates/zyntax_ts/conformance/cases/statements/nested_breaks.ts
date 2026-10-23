let log = "";
for (let i = 0; i < 3; i++) {
    switch (i) {
        case 0:
            for (let j = 0; j < 5; j++) { if (j === 2) break; log = log + j; }
            break;
        case 1:
            let k = 0;
            while (true) { k++; if (k > 2) break; }
            log = log + "k" + k;
            continue;
        default:
            log = log + "d";
    }
    log = log + "|";
}
console.log(log);
