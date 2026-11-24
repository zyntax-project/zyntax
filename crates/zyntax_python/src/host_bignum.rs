//! The magnitude arithmetic of the arbitrary-precision int library: add,
//! subtract, multiply, divide, compare and shift over little-endian
//! 30-bit limbs, each held in a slot of a list of ints. Operands are
//! trimmed (no zero limb on top); results are written into slices the
//! caller made and zero filled, and the caller trims them.

const BITS: u32 = 30;
const BASE: i64 = 1 << BITS;
const MASK: i64 = BASE - 1;
/// Operands this long, both, multiply by Karatsuba.
const KARATSUBA: usize = 48;

/// `-1`, `0` or `1` as `a` is less than, equal to or more than `b`.
pub fn cmp(a: &[i64], b: &[i64]) -> i64 {
    if a.len() != b.len() {
        return if a.len() > b.len() { 1 } else { -1 };
    }
    for i in (0..a.len()).rev() {
        if a[i] != b[i] {
            return if a[i] > b[i] { 1 } else { -1 };
        }
    }
    0
}

/// `a + b` into `out`, which holds one limb more than the longer.
pub fn add(a: &[i64], b: &[i64], out: &mut [i64]) {
    let (a, b) = if a.len() < b.len() { (b, a) } else { (a, b) };
    let mut carry = 0;
    for i in 0..a.len() {
        let s = a[i] + carry + b.get(i).copied().unwrap_or(0);
        out[i] = s & MASK;
        carry = s >> BITS;
    }
    out[a.len()] = carry;
}

/// `a - b` into `out`, for `a >= b`.
pub fn sub(a: &[i64], b: &[i64], out: &mut [i64]) {
    let mut borrow = 0;
    for i in 0..a.len() {
        let mut d = a[i] - borrow - b.get(i).copied().unwrap_or(0);
        if d < 0 {
            d += BASE;
            borrow = 1;
        } else {
            borrow = 0;
        }
        out[i] = d;
    }
}

fn trimmed(mut v: Vec<i64>) -> Vec<i64> {
    while v.last() == Some(&0) {
        v.pop();
    }
    v
}

fn trim_slice(a: &[i64]) -> &[i64] {
    let mut n = a.len();
    while n > 0 && a[n - 1] == 0 {
        n -= 1;
    }
    &a[..n]
}

fn add_vec(a: &[i64], b: &[i64]) -> Vec<i64> {
    let mut out = vec![0; a.len().max(b.len()) + 1];
    add(a, b, &mut out);
    trimmed(out)
}

fn sub_vec(a: &[i64], b: &[i64]) -> Vec<i64> {
    let mut out = vec![0; a.len()];
    sub(a, b, &mut out);
    trimmed(out)
}

fn shifted(a: &[i64], k: usize) -> Vec<i64> {
    if a.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0; k];
    out.extend_from_slice(a);
    out
}

/// The product by rows into a zeroed `out`: one row of the longer operand for each limb of
/// the shorter, so a small multiplier makes few rows.
fn mul_school(a: &[i64], b: &[i64], out: &mut [i64]) {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let m = long.len();
    for (i, &si) in short.iter().enumerate() {
        let row = &mut out[i..i + m + 1];
        let mut carry = 0;
        for (slot, &x) in row.iter_mut().zip(long) {
            let t = *slot + si * x + carry;
            *slot = t & MASK;
            carry = t >> BITS;
        }
        row[m] = carry;
    }
}

fn mul_vec(a: &[i64], b: &[i64]) -> Vec<i64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0; a.len() + b.len()];
    mul(a, b, &mut out);
    trimmed(out)
}

/// `a * b` into `out`, which holds `a.len() + b.len()` limbs.
pub fn mul(a: &[i64], b: &[i64], out: &mut [i64]) {
    if a.is_empty() || b.is_empty() {
        out.fill(0);
        return;
    }
    if a.len() < KARATSUBA || b.len() < KARATSUBA {
        mul_school(a, b, out);
        return;
    }
    let k = a.len().min(b.len()) / 2;
    let (a0, a1) = (trim_slice(&a[..k]), &a[k..]);
    let (b0, b1) = (trim_slice(&b[..k]), &b[k..]);
    let z0 = mul_vec(a0, b0);
    let z2 = mul_vec(a1, b1);
    let z1 = sub_vec(
        &sub_vec(&mul_vec(&add_vec(a0, a1), &add_vec(b0, b1)), &z0),
        &z2,
    );
    let r = add_vec(&z0, &shifted(&z1, k));
    let r = add_vec(&r, &shifted(&z2, 2 * k));
    out.fill(0);
    out[..r.len()].copy_from_slice(&r);
}

/// `a / d` into `q`, which holds `a.len()` limbs, for `d` one limb;
/// the remainder.
pub fn divmod_small(a: &[i64], d: i64, q: &mut [i64]) -> i64 {
    let mut r = 0;
    for i in (0..a.len()).rev() {
        let cur = (r << BITS) | a[i];
        let qi = cur / d;
        q[i] = qi;
        r = cur - qi * d;
    }
    r
}

/// `a << n` into `out`, which holds `a.len() + n / 30 + 1` limbs.
pub fn shl(a: &[i64], n: usize, out: &mut [i64]) {
    let (limbs, bits) = (n / 30, (n % 30) as u32);
    let mut carry = 0;
    for (i, &x) in a.iter().enumerate() {
        let t = (x << bits) | carry;
        out[limbs + i] = t & MASK;
        carry = t >> BITS;
    }
    out[limbs + a.len()] = carry;
}

/// `a >> n` into `out`, which holds `a.len()` limbs.
pub fn shr(a: &[i64], n: usize, out: &mut [i64]) {
    let (limbs, bits) = (n / 30, (n % 30) as u32);
    for i in limbs..a.len() {
        let mut v = a[i] >> bits;
        if i + 1 < a.len() {
            v |= (a[i + 1] << (BITS - bits)) & MASK;
        }
        out[i - limbs] = v;
    }
}

fn bit_length(x: i64) -> u32 {
    64 - x.leading_zeros()
}

/// `a / b` into `q` and `a % b` into `r`; `b` is not empty. `q` holds
/// `a.len()` limbs, at least one, and `r` holds `b.len()`.
pub fn divmod(a: &[i64], b: &[i64], q: &mut [i64], r: &mut [i64]) {
    q.fill(0);
    r.fill(0);
    if cmp(a, b) < 0 {
        r[..a.len()].copy_from_slice(a);
        return;
    }
    let n = b.len();
    if n == 1 {
        let rem = divmod_small(a, b[0], q);
        r[0] = rem;
        return;
    }
    // Knuth's algorithm D on normalised copies.
    let s = BITS - bit_length(b[n - 1]);
    let mut v = vec![0; n + 1];
    shl(b, s as usize, &mut v);
    v.truncate(n);
    let mut u = vec![0; a.len() + 1];
    shl(a, s as usize, &mut u);
    let m = a.len() - n;
    let (vtop, vsec) = (v[n - 1], v[n - 2]);
    for j in (0..=m).rev() {
        let num = u[j + n] * BASE + u[j + n - 1];
        let mut qhat = num / vtop;
        let mut rhat = num - qhat * vtop;
        while qhat >= BASE || qhat * vsec > rhat * BASE + u[j + n - 2] {
            qhat -= 1;
            rhat += vtop;
            if rhat >= BASE {
                break;
            }
        }
        let (mut borrow, mut carry) = (0, 0);
        for i in 0..n {
            let p = qhat * v[i] + carry;
            carry = p >> BITS;
            let mut t = u[i + j] - (p & MASK) - borrow;
            if t < 0 {
                t += BASE;
                borrow = 1;
            } else {
                borrow = 0;
            }
            u[i + j] = t;
        }
        let t = u[j + n] - carry - borrow;
        u[j + n] = t;
        if t < 0 {
            qhat -= 1;
            let mut carry = 0;
            for i in 0..n {
                let t2 = u[i + j] + v[i] + carry;
                u[i + j] = t2 & MASK;
                carry = t2 >> BITS;
            }
            u[j + n] += carry;
        }
        q[j] = qhat;
    }
    shr(trim_slice(&u[..n]), s as usize, r);
}

/// `m = m * k + add` in place; the carry out of the top limb.
pub fn mul_small_add(m: &mut [i64], k: i64, add: i64) -> i64 {
    let mut carry = add;
    for x in m.iter_mut() {
        let t = *x * k + carry;
        *x = t & MASK;
        carry = t >> BITS;
    }
    carry
}

/// A list's slots as a slice.
///
/// # Safety
/// `data` must point at `len` readable ints, or `len` must be zero.
unsafe fn slots<'a>(data: *const i64, len: i64) -> &'a [i64] {
    if len <= 0 || data.is_null() {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(data, len as usize) }
    }
}

/// A list's slots as a mutable slice.
///
/// # Safety
/// `data` must point at `len` writable ints, or `len` must be zero.
unsafe fn slots_mut<'a>(data: *mut i64, len: i64) -> &'a mut [i64] {
    if len <= 0 || data.is_null() {
        &mut []
    } else {
        unsafe { std::slice::from_raw_parts_mut(data, len as usize) }
    }
}

pub(crate) extern "C" fn host_big_cmp(a: *const i64, an: i64, b: *const i64, bn: i64) -> i64 {
    unsafe { cmp(slots(a, an), slots(b, bn)) }
}

pub(crate) extern "C" fn host_big_add(
    a: *const i64,
    an: i64,
    b: *const i64,
    bn: i64,
    out: *mut i64,
    on: i64,
) {
    unsafe { add(slots(a, an), slots(b, bn), slots_mut(out, on)) }
}

pub(crate) extern "C" fn host_big_sub(
    a: *const i64,
    an: i64,
    b: *const i64,
    bn: i64,
    out: *mut i64,
    on: i64,
) {
    unsafe { sub(slots(a, an), slots(b, bn), slots_mut(out, on)) }
}

pub(crate) extern "C" fn host_big_mul(
    a: *const i64,
    an: i64,
    b: *const i64,
    bn: i64,
    out: *mut i64,
    on: i64,
) {
    unsafe { mul(slots(a, an), slots(b, bn), slots_mut(out, on)) }
}

pub(crate) extern "C" fn host_big_divmod_small(
    a: *const i64,
    an: i64,
    d: i64,
    q: *mut i64,
    qn: i64,
) -> i64 {
    unsafe { divmod_small(slots(a, an), d, slots_mut(q, qn)) }
}

pub(crate) extern "C" fn host_big_shl(a: *const i64, an: i64, n: i64, out: *mut i64, on: i64) {
    unsafe { shl(slots(a, an), n as usize, slots_mut(out, on)) }
}

pub(crate) extern "C" fn host_big_shr(a: *const i64, an: i64, n: i64, out: *mut i64, on: i64) {
    unsafe { shr(slots(a, an), n as usize, slots_mut(out, on)) }
}

pub(crate) extern "C" fn host_big_divmod(
    a: *const i64,
    an: i64,
    b: *const i64,
    bn: i64,
    q: *mut i64,
    qn: i64,
    r: *mut i64,
    rn: i64,
) {
    unsafe {
        divmod(
            slots(a, an),
            slots(b, bn),
            slots_mut(q, qn),
            slots_mut(r, rn),
        )
    }
}

pub(crate) extern "C" fn host_big_mul_small_add(m: *mut i64, n: i64, k: i64, add: i64) -> i64 {
    unsafe { mul_small_add(slots_mut(m, n), k, add) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A limb vector as a big integer, for the checks below.
    fn value(m: &[i64]) -> u128 {
        m.iter().rev().fold(0u128, |v, &x| (v << 30) | x as u128)
    }

    fn limbs(mut x: u128) -> Vec<i64> {
        let mut m = Vec::new();
        while x > 0 {
            m.push((x & MASK as u128) as i64);
            x >>= 30;
        }
        m
    }

    #[test]
    fn small_values_add_subtract_and_compare() {
        let (a, b) = (limbs(123_456_789_012_345_678), limbs(987_654_321_098_765));
        let mut s = vec![0; a.len().max(b.len()) + 1];
        add(&a, &b, &mut s);
        assert_eq!(value(&s), 123_456_789_012_345_678 + 987_654_321_098_765);
        let mut d = vec![0; a.len()];
        sub(&a, &b, &mut d);
        assert_eq!(value(&d), 123_456_789_012_345_678 - 987_654_321_098_765);
        assert_eq!(cmp(&a, &b), 1);
        assert_eq!(cmp(&b, &a), -1);
        assert_eq!(cmp(&a, &a), 0);
    }

    #[test]
    fn products_match_wide_arithmetic() {
        let (x, y) = (0x1234_5678_9abc_def0u128, 0xfedc_ba98_7654_3210u128);
        let (a, b) = (limbs(x), limbs(y));
        let mut p = vec![0; a.len() + b.len()];
        mul(&a, &b, &mut p);
        assert_eq!(value(&p), x * y);
    }

    #[test]
    fn karatsuba_agrees_with_the_schoolbook_product() {
        let a: Vec<i64> = (0..120).map(|i| (i * 7919 + 13) % BASE + 1).collect();
        let b: Vec<i64> = (0..90).map(|i| (i * 104729 + 7) % BASE + 1).collect();
        let mut fast = vec![0; a.len() + b.len()];
        mul(&a, &b, &mut fast);
        let mut slow = vec![0; a.len() + b.len()];
        mul_school(&a, &b, &mut slow);
        assert_eq!(fast, slow);
    }

    #[test]
    fn division_inverts_multiplication() {
        let a: Vec<i64> = (0..40).map(|i| (i * 31337 + 5) % BASE + 1).collect();
        let b: Vec<i64> = (0..13).map(|i| (i * 65537 + 3) % BASE + 1).collect();
        let mut q = vec![0; a.len()];
        let mut r = vec![0; b.len()];
        divmod(&a, &b, &mut q, &mut r);
        let q = trimmed(q);
        let r = trimmed(r);
        assert_eq!(cmp(&r, &b), -1);
        let back = add_vec(&mul_vec(&q, &b), &r);
        assert_eq!(back, a);
    }

    #[test]
    fn division_by_a_limb_and_by_a_larger_value() {
        let a = limbs(10u128.pow(30) + 7);
        let mut q = vec![0; a.len()];
        let r = divmod_small(&a, 1_000_000_000, &mut q);
        assert_eq!(value(&q), (10u128.pow(30) + 7) / 1_000_000_000);
        assert_eq!(r as u128, (10u128.pow(30) + 7) % 1_000_000_000);
        let small = limbs(5);
        let big = limbs(10u128.pow(20));
        let mut q = vec![0; 1];
        let mut r = vec![0; big.len()];
        divmod(&small, &big, &mut q, &mut r);
        assert_eq!(value(&q), 0);
        assert_eq!(value(&r), 5);
    }

    #[test]
    fn shifts_move_bits_across_limbs() {
        let a = limbs(0xdead_beef_cafe_f00d);
        let mut l = vec![0; a.len() + 100 / 30 + 1];
        shl(&a, 100, &mut l);
        assert_eq!(value(&l), 0xdead_beef_cafe_f00du128 << 100);
        let mut r = vec![0; a.len()];
        shr(&a, 17, &mut r);
        assert_eq!(value(&r), 0xdead_beef_cafe_f00du128 >> 17);
    }

    #[test]
    fn a_small_multiplier_with_a_carry() {
        let mut m = limbs(999_999_999_999);
        let carry = mul_small_add(&mut m, 36, 35);
        let mut v = value(&m);
        v += (carry as u128) << (30 * m.len());
        assert_eq!(v, 999_999_999_999 * 36 + 35);
    }
}
