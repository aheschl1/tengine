
pub fn add<T: std::ops::Add<Output = T>>(a: T, b: T) -> T {
    a + b
}

pub fn sub<T: std::ops::Sub<Output = T>>(a: T, b: T) -> T {
    a - b
}

pub fn mul<T: std::ops::Mul<Output = T>>(a: T, b: T) -> T {
    a * b
}