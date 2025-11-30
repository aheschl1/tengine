use crate::core::Dim;

pub enum Idx<'a> {
    Coord(&'a [Dim]),
    At(usize),
    Item
}

impl<'a> From<&Idx<'a>> for Idx<'a> {
    /// Clones an index reference into an owned index.
    fn from(value: &Idx<'a>) -> Self {
        match value {
            Idx::Coord(coords) => Idx::Coord(coords),
            Idx::At(i) => Idx::At(*i),
            Idx::Item => Idx::Item,
        }
    }
}

impl<'a> From<&'a [Dim]> for Idx<'a>
{
    /// Converts a slice of coordinates into a multi-dimensional index.
    fn from(value: &'a [Dim]) -> Self {
        Idx::Coord(value)
    }
}

impl<'a, const C: usize> From<&'a [Dim; C]> for Idx<'a>
{
    /// Converts a slice of coordinates into a multi-dimensional index.
    fn from(value: &'a [Dim; C]) -> Self {
        Idx::Coord(value)
    }
}

impl<'a> From<&'a Vec<Dim>> for Idx<'a> {
    /// Converts a vector of coordinates into a multi-dimensional index.
    fn from(value: &'a Vec<Dim>) -> Self {
        Idx::Coord(value.as_slice())
    }
}

impl From<Vec<Dim>> for Idx<'_> {
    /// Converts an owned vector of coordinates into a multi-dimensional index.
    fn from(value: Vec<Dim>) -> Self {
        Idx::Coord(Box::leak(value.into_boxed_slice()))
    }
}

impl<'a> From<Dim> for Idx<'a> {
    /// Treats a single dimension value as a 1-D index.
    fn from(value: Dim) -> Self {
        Idx::At(value)
    }
}

impl<'a> From<i32> for Idx<'a> {
    /// Converts an i32 into a 1-D index.
    fn from(value: i32) -> Self {
        Idx::At(value as usize)
    }
}


impl<'a> From<i64> for Idx<'a> {
    /// Converts an i64 into a 1-D index.
    fn from(value: i64) -> Self {
        Idx::At(value as usize)
    }
}

impl<'a> From<()> for Idx<'a> {
    /// Represents indexing the single item of a scalar tensor.
    fn from(_: ()) -> Self {
        Idx::Item
    }
}