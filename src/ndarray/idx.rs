use crate::ndarray::Dim;

pub enum Idx<'a> {
    Coord(&'a [Dim]),
    At(usize),
    Item
}

impl<'a> From<&'a [Dim]> for Idx<'a>
{
    fn from(value: &'a [Dim]) -> Self {
        Idx::Coord(value)
    }
}

impl<'a> From<&'a Vec<Dim>> for Idx<'a> {
    fn from(value: &'a Vec<Dim>) -> Self {
        Idx::Coord(value.as_slice())
    }
}

impl From<Vec<Dim>> for Idx<'_> {
    fn from(value: Vec<Dim>) -> Self {
        Idx::Coord(Box::leak(value.into_boxed_slice()))
    }
}

impl<'a> From<Dim> for Idx<'a> {
    fn from(value: Dim) -> Self {
        Idx::At(value)
    }
}

impl<'a> From<i32> for Idx<'a> {
    fn from(value: i32) -> Self {
        Idx::At(value as usize)
    }
}


impl<'a> From<i64> for Idx<'a> {
    fn from(value: i64) -> Self {
        Idx::At(value as usize)
    }
}

impl<'a> From<()> for Idx<'a> {
    fn from(_: ()) -> Self {
        Idx::Item
    }
}