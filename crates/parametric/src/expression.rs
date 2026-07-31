//! The authored expression tree and its evaluator.
//!
//! An expression is what the author *typed*, retained rather than collapsed to a number, so
//! that `height = width * 2` survives as a relationship instead of decaying into two numbers
//! that have to be kept in step by hand. Evaluating one yields a
//! [`Quantity`] — a value that carries its own dimension — and
//! the dimension check happens as a consequence of the arithmetic rather than as a separate
//! pass.
//!
//! ## Exact, and only exact
//!
//! The operators are `+ - * /` over exact rationals, and that is the whole language. There
//! is deliberately no `sin`/`sqrt`: they would make a result irrational, and the invariant
//! this crate exists to hold is that an authored value is exact so a persisted document is
//! float-free and a density re-target re-evaluates losslessly. A trigonometric relationship
//! between two entities is a **constraint** for the solver (which is floating-point and
//! unbothered), not an expression for the author to type.
//!
//! ## Where a length literal gets its density
//!
//! `3 blocks` is only a count of voxels once `d` is known, so evaluation takes a density —
//! the same rule as [`Measurement::to_voxels`](crate::units::Measurement::to_voxels). It is
//! supplied at *eval* time and never stored, which is precisely what lets one expression
//! re-evaluate correctly at a new density.

use std::collections::{BTreeMap, BTreeSet};

use crate::dimension::Dimension;
use crate::quantity::{Quantity, QuantityError};
use crate::units::{AngleMeasurement, ExactRational, Measurement};

/// The name of the built-in parameter carrying the document's voxels-per-block.
///
/// It resolves [`Dimension::DIMENSIONLESS`] — voxels over blocks is a ratio of two lengths —
/// so `3 blocks * voxel_density` types as a length by the ordinary multiplication rule and
/// needs no special case anywhere in the evaluator.
pub const VOXEL_DENSITY: &str = "voxel_density";

/// A leaf value in an expression: something the author wrote down directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Literal {
    /// A length, as its authored blocks+voxels expression.
    Length(Measurement),
    /// An angle, in exact degrees.
    Angle(AngleMeasurement),
    /// A pure number — a scale factor, a count.
    Number(ExactRational),
}

/// A binary operator. Addition and subtraction require matching dimensions; multiplication
/// and division combine them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operator {
    /// `+`
    Add,
    /// `-`
    Subtract,
    /// `*`
    Multiply,
    /// `/`
    Divide,
}

/// An authored expression tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expression {
    /// A value written down directly.
    Literal(Literal),
    /// A reference to a named parameter, or to a built-in such as [`VOXEL_DENSITY`].
    Symbol(String),
    /// Unary minus.
    Negate(Box<Expression>),
    /// Two operands combined.
    Binary {
        /// The left operand.
        left: Box<Expression>,
        /// What to do with them.
        operator: Operator,
        /// The right operand.
        right: Box<Expression>,
    },
}

impl Expression {
    /// A length literal.
    pub fn length(measurement: Measurement) -> Self {
        Expression::Literal(Literal::Length(measurement))
    }

    /// An angle literal.
    pub fn angle(angle: AngleMeasurement) -> Self {
        Expression::Literal(Literal::Angle(angle))
    }

    /// A pure-number literal.
    pub fn number(value: ExactRational) -> Self {
        Expression::Literal(Literal::Number(value))
    }

    /// A whole-number literal — the common case, so it does not have to be spelled out.
    pub fn whole(value: i64) -> Self {
        Self::number(ExactRational::from_integer(value as i128))
    }

    /// A reference to a named parameter.
    pub fn symbol(name: impl Into<String>) -> Self {
        Expression::Symbol(name.into())
    }

    /// `self + other`.
    pub fn plus(self, other: Expression) -> Self {
        self.binary(Operator::Add, other)
    }

    /// `self - other`.
    pub fn minus(self, other: Expression) -> Self {
        self.binary(Operator::Subtract, other)
    }

    /// `self * other`.
    pub fn times(self, other: Expression) -> Self {
        self.binary(Operator::Multiply, other)
    }

    /// `self / other`.
    pub fn divided_by(self, other: Expression) -> Self {
        self.binary(Operator::Divide, other)
    }

    fn binary(self, operator: Operator, other: Expression) -> Self {
        Expression::Binary {
            left: Box::new(self),
            operator,
            right: Box::new(other),
        }
    }

    /// Every symbol this expression mentions, deduplicated — what the dependency graph and
    /// the cycle check are built from.
    pub fn referenced_symbols(&self) -> BTreeSet<String> {
        let mut found = BTreeSet::new();
        self.collect_symbols(&mut found);
        found
    }

    fn collect_symbols(&self, into: &mut BTreeSet<String>) {
        match self {
            Expression::Literal(_) => {}
            Expression::Symbol(name) => {
                into.insert(name.clone());
            }
            Expression::Negate(inner) => inner.collect_symbols(into),
            Expression::Binary { left, right, .. } => {
                left.collect_symbols(into);
                right.collect_symbols(into);
            }
        }
    }
}

/// Why an expression could not be evaluated, or a parameter could not be defined.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluationError {
    /// A symbol that names no parameter and no built-in.
    UnknownSymbol {
        /// The name as written.
        name: String,
    },
    /// A parameter that depends on itself, directly or through others. Carries the cycle in
    /// the order it was walked, so the message can name the loop rather than just assert one.
    CircularReference {
        /// The names on the cycle, in walk order.
        cycle: Vec<String>,
    },
    /// A parameter defined over a name that is already a built-in. Shadowing
    /// [`VOXEL_DENSITY`] would let a document silently mean something else at another
    /// density, so it is refused rather than resolved by precedence.
    ShadowsBuiltIn {
        /// The built-in name the definition collided with.
        name: String,
    },
    /// The arithmetic itself failed — mismatched dimensions, a zero divisor, an overflow.
    Quantity(QuantityError),
}

impl From<QuantityError> for EvaluationError {
    fn from(error: QuantityError) -> Self {
        EvaluationError::Quantity(error)
    }
}

impl core::fmt::Display for EvaluationError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EvaluationError::UnknownSymbol { name } => {
                write!(formatter, "unknown parameter `{name}`")
            }
            EvaluationError::CircularReference { cycle } => {
                write!(formatter, "`{}` depends on itself", cycle.join("` → `"))
            }
            EvaluationError::ShadowsBuiltIn { name } => {
                write!(formatter, "`{name}` is built in and cannot be redefined")
            }
            EvaluationError::Quantity(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for EvaluationError {}

/// The named parameters of a document, and the evaluator over them.
///
/// Definitions may reference each other in any order — `gap` may be defined before the
/// `wall` it divides — because resolution happens at evaluation, walking the references.
/// What is *not* allowed is a cycle, which [`define`](Self::define) rejects at definition
/// time so the table can never hold one.
#[derive(Debug, Clone, Default)]
pub struct SymbolTable {
    parameters: BTreeMap<String, Expression>,
}

impl SymbolTable {
    /// An empty table. The built-ins are not stored — they are resolved during evaluation,
    /// so they cost nothing and cannot be deleted.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether a name is built in, and so neither definable nor deletable.
    pub fn is_built_in(name: &str) -> bool {
        name == VOXEL_DENSITY
    }

    /// Define or redefine a parameter.
    ///
    /// Refuses a name that shadows a built-in, and refuses a definition that would put the
    /// table into a cycle — checked against the table as it *would be*, so redefining an
    /// existing parameter is judged on its new expression rather than its old one.
    pub fn define(
        &mut self,
        name: impl Into<String>,
        expression: Expression,
    ) -> Result<(), EvaluationError> {
        let name = name.into();
        if Self::is_built_in(&name) {
            return Err(EvaluationError::ShadowsBuiltIn { name });
        }
        let displaced = self.parameters.insert(name.clone(), expression);
        match self.find_cycle_from(&name) {
            Some(cycle) => {
                // Put the table back exactly as it was: a rejected definition must leave no
                // trace, or the next call would be judged against a table the caller never
                // agreed to.
                match displaced {
                    Some(previous) => self.parameters.insert(name, previous),
                    None => self.parameters.remove(&name),
                };
                Err(EvaluationError::CircularReference { cycle })
            }
            None => Ok(()),
        }
    }

    /// Remove a parameter, returning whether it was there.
    pub fn remove(&mut self, name: &str) -> bool {
        self.parameters.remove(name).is_some()
    }

    /// The expression a parameter is defined as.
    pub fn get(&self, name: &str) -> Option<&Expression> {
        self.parameters.get(name)
    }

    /// Every defined parameter name, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.parameters.keys().map(String::as_str)
    }

    /// Evaluate an expression at the document density.
    ///
    /// `density` is voxels-per-block: it scales every block term and is the value
    /// [`VOXEL_DENSITY`] resolves to.
    pub fn evaluate(
        &self,
        expression: &Expression,
        density: u32,
    ) -> Result<Quantity, EvaluationError> {
        self.evaluate_within(expression, density, &mut Vec::new())
    }

    /// Evaluate a named parameter at the document density.
    pub fn evaluate_symbol(&self, name: &str, density: u32) -> Result<Quantity, EvaluationError> {
        self.resolve_symbol(name, density, &mut Vec::new())
    }

    fn evaluate_within(
        &self,
        expression: &Expression,
        density: u32,
        visiting: &mut Vec<String>,
    ) -> Result<Quantity, EvaluationError> {
        match expression {
            Expression::Literal(Literal::Length(measurement)) => {
                Ok(Quantity::from_measurement(*measurement, density))
            }
            Expression::Literal(Literal::Angle(angle)) => Ok(Quantity::from_angle(*angle)),
            Expression::Literal(Literal::Number(value)) => Ok(Quantity::dimensionless(*value)),
            Expression::Symbol(name) => self.resolve_symbol(name, density, visiting),
            Expression::Negate(inner) => {
                Ok(self.evaluate_within(inner, density, visiting)?.negated()?)
            }
            Expression::Binary {
                left,
                operator,
                right,
            } => {
                let left = self.evaluate_within(left, density, visiting)?;
                let right = self.evaluate_within(right, density, visiting)?;
                Ok(match operator {
                    Operator::Add => left.plus(right)?,
                    Operator::Subtract => left.minus(right)?,
                    Operator::Multiply => left.times(right),
                    Operator::Divide => left.divided_by(right)?,
                })
            }
        }
    }

    fn resolve_symbol(
        &self,
        name: &str,
        density: u32,
        visiting: &mut Vec<String>,
    ) -> Result<Quantity, EvaluationError> {
        if name == VOXEL_DENSITY {
            return Ok(Quantity::dimensionless(ExactRational::from_integer(
                density as i128,
            )));
        }
        let expression =
            self.parameters
                .get(name)
                .ok_or_else(|| EvaluationError::UnknownSymbol {
                    name: name.to_string(),
                })?;
        // `define` refuses cycles, so this cannot fire for a table built through the public
        // API. It is kept because the recursion would otherwise be unbounded, and an
        // unbounded recursion is a stack overflow rather than an error message.
        if visiting.iter().any(|seen| seen == name) {
            let mut cycle = visiting.clone();
            cycle.push(name.to_string());
            return Err(EvaluationError::CircularReference { cycle });
        }
        visiting.push(name.to_string());
        let value = self.evaluate_within(expression, density, visiting);
        visiting.pop();
        value
    }

    /// Walk the reference graph from `start`, returning the cycle through it if there is one.
    fn find_cycle_from(&self, start: &str) -> Option<Vec<String>> {
        let mut path = Vec::new();
        let mut on_path = BTreeSet::new();
        self.walk_for_cycle(start, &mut path, &mut on_path)
    }

    fn walk_for_cycle(
        &self,
        name: &str,
        path: &mut Vec<String>,
        on_path: &mut BTreeSet<String>,
    ) -> Option<Vec<String>> {
        if on_path.contains(name) {
            let mut cycle = path.clone();
            cycle.push(name.to_string());
            return Some(cycle);
        }
        let expression = self.parameters.get(name)?;
        path.push(name.to_string());
        on_path.insert(name.to_string());
        for referenced in expression.referenced_symbols() {
            if let Some(cycle) = self.walk_for_cycle(&referenced, path, on_path) {
                return Some(cycle);
            }
        }
        on_path.remove(name);
        path.pop();
        None
    }
}

/// The dimension an expression will produce, without computing its value.
///
/// Answers "would this fit in that field?" before anything is stored — the check the
/// parameters panel wants as you type, and the one a driving dimension applies before it
/// hands a value to the solver. Uses a density of 1 because a dimension never depends on
/// one: scaling a length by any number leaves it a length.
pub fn dimension_of(
    table: &SymbolTable,
    expression: &Expression,
) -> Result<Dimension, EvaluationError> {
    table.evaluate(expression, 1).map(|value| value.dimension)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DENSITY: u32 = 16;

    fn blocks(count: i64) -> Expression {
        Expression::length(Measurement::new(
            ExactRational::from_integer(count as i128),
            0,
        ))
    }

    fn voxels(count: i64) -> Expression {
        Expression::length(Measurement::from_voxels(count))
    }

    fn table_with(pairs: &[(&str, Expression)]) -> SymbolTable {
        let mut table = SymbolTable::new();
        for (name, expression) in pairs {
            table
                .define(*name, expression.clone())
                .expect("fixture defines no cycles");
        }
        table
    }

    #[test]
    fn a_parameter_can_be_defined_in_terms_of_another() {
        // The whole reason the panel is worth building: `height = width * 2` survives as a
        // relationship, so changing `width` moves `height`.
        let table = table_with(&[
            ("width", blocks(2)),
            (
                "height",
                Expression::symbol("width").times(Expression::whole(2)),
            ),
        ]);
        let height = table.evaluate_symbol("height", DENSITY).expect("resolves");
        assert_eq!(height.dimension, Dimension::LENGTH);
        assert_eq!(height.to_whole_voxels(), Ok(64));
    }

    #[test]
    fn voxel_density_is_built_in_and_dimensionless() {
        // Voxels-per-block is a ratio of two lengths, so multiplying by it leaves a length —
        // no special case in the evaluator, just the ordinary rule.
        let table = SymbolTable::new();
        let density = table
            .evaluate(&Expression::symbol(VOXEL_DENSITY), DENSITY)
            .expect("built in");
        assert_eq!(density.dimension, Dimension::DIMENSIONLESS);
        assert_eq!(density.value, ExactRational::from_integer(16));

        // Scaling a LENGTH by it is the useful case — one voxel times the density is one
        // block's worth, and it re-derives at whatever density the document is on. Scaling a
        // bare number by it stays a bare number, which is why the assertion below is on a
        // length literal and not on `Expression::whole(1)`.
        let one_block_in_voxels = voxels(1).times(Expression::symbol(VOXEL_DENSITY));
        let scaled = table
            .evaluate(&one_block_in_voxels, DENSITY)
            .expect("resolves");
        assert_eq!(scaled.dimension, Dimension::LENGTH);
        assert_eq!(scaled.to_whole_voxels(), Ok(16));
        assert_eq!(
            table
                .evaluate(&one_block_in_voxels, 32)
                .expect("resolves")
                .to_whole_voxels(),
            Ok(32),
            "it tracks the density rather than baking one in"
        );
    }

    #[test]
    fn density_does_not_promote_a_bare_number_to_a_length() {
        // The mistake this guards: `2 * voxel_density` reads like "two blocks" and is not —
        // both operands are dimensionless, so the result is a count, and the voxel door
        // refuses it rather than silently treating a number as a distance.
        let table = SymbolTable::new();
        let count = Expression::whole(2).times(Expression::symbol(VOXEL_DENSITY));
        let value = table.evaluate(&count, DENSITY).expect("resolves");
        assert_eq!(value.dimension, Dimension::DIMENSIONLESS);
        assert!(matches!(
            value.to_whole_voxels(),
            Err(QuantityError::MismatchedDimensions { .. })
        ));
    }

    #[test]
    fn the_same_expression_re_evaluates_at_a_new_density() {
        // Density is supplied at eval and never stored, which is exactly what makes a
        // density re-target lossless. "3.5 blocks" is 56 voxels at d16 and 112 at d32.
        let half_blocks = Expression::length(Measurement::new(
            ExactRational::new(7, 2).expect("non-zero denominator"),
            0,
        ));
        let table = SymbolTable::new();
        assert_eq!(
            table
                .evaluate(&half_blocks, 16)
                .expect("resolves")
                .to_whole_voxels(),
            Ok(56)
        );
        assert_eq!(
            table
                .evaluate(&half_blocks, 32)
                .expect("resolves")
                .to_whole_voxels(),
            Ok(112)
        );
    }

    #[test]
    fn adding_a_length_to_an_angle_is_refused() {
        let table = SymbolTable::new();
        let mixed = blocks(1).plus(Expression::angle(AngleMeasurement::from_degrees(45)));
        assert!(matches!(
            table.evaluate(&mixed, DENSITY),
            Err(EvaluationError::Quantity(
                QuantityError::MismatchedDimensions { .. }
            ))
        ));
    }

    #[test]
    fn a_direct_self_reference_is_refused_at_definition() {
        // Refused when DEFINED, not when evaluated, so the table can never hold a cycle and
        // no later read has to defend against one.
        let mut table = SymbolTable::new();
        let result = table.define(
            "wall",
            Expression::symbol("wall").times(Expression::whole(2)),
        );
        assert!(matches!(
            result,
            Err(EvaluationError::CircularReference { .. })
        ));
        assert!(
            table.get("wall").is_none(),
            "a refused define leaves no trace"
        );
    }

    #[test]
    fn an_indirect_cycle_is_refused_and_leaves_the_table_untouched() {
        let mut table = table_with(&[("a", Expression::symbol("b")), ("b", blocks(1))]);
        // b -> a would close a -> b -> a.
        let result = table.define("b", Expression::symbol("a"));
        assert!(matches!(
            result,
            Err(EvaluationError::CircularReference { .. })
        ));
        // The rejected definition must not have displaced the good one.
        assert_eq!(table.get("b"), Some(&blocks(1)));
        assert_eq!(
            table
                .evaluate_symbol("a", DENSITY)
                .expect("still resolves")
                .to_whole_voxels(),
            Ok(16)
        );
    }

    #[test]
    fn a_built_in_cannot_be_redefined() {
        // Shadowing would let a document mean something different at another density with
        // nothing on screen to say so.
        let mut table = SymbolTable::new();
        assert_eq!(
            table.define(VOXEL_DENSITY, Expression::whole(8)),
            Err(EvaluationError::ShadowsBuiltIn {
                name: VOXEL_DENSITY.to_string()
            })
        );
    }

    #[test]
    fn an_unknown_symbol_names_itself() {
        let table = SymbolTable::new();
        assert_eq!(
            table.evaluate(&Expression::symbol("nope"), DENSITY),
            Err(EvaluationError::UnknownSymbol {
                name: "nope".to_string()
            })
        );
    }

    #[test]
    fn a_ratio_of_parameters_scales_a_length() {
        // `wall / gap` is dimensionless, so it can multiply anything — the composition the
        // dimension algebra exists to permit.
        let table = table_with(&[("wall", voxels(32)), ("gap", voxels(8))]);
        let scaled = Expression::symbol("wall")
            .divided_by(Expression::symbol("gap"))
            .times(voxels(3));
        let value = table.evaluate(&scaled, DENSITY).expect("resolves");
        assert_eq!(value.dimension, Dimension::LENGTH);
        assert_eq!(value.to_whole_voxels(), Ok(12));
    }

    #[test]
    fn division_stays_exact_through_a_round_trip() {
        // A third of a wall, tripled, is the wall again — the property a float would lose
        // and the reason the language is rationals only.
        let table = table_with(&[("wall", voxels(10))]);
        let there_and_back = Expression::symbol("wall")
            .divided_by(Expression::whole(3))
            .times(Expression::whole(3));
        assert_eq!(
            table
                .evaluate(&there_and_back, DENSITY)
                .expect("resolves")
                .to_whole_voxels(),
            Ok(10)
        );
    }

    #[test]
    fn dimension_is_answerable_without_a_density() {
        let table = table_with(&[("wall", blocks(2))]);
        assert_eq!(
            dimension_of(&table, &Expression::symbol("wall")),
            Ok(Dimension::LENGTH)
        );
        assert_eq!(
            dimension_of(
                &table,
                &Expression::symbol("wall").divided_by(Expression::symbol("wall"))
            ),
            Ok(Dimension::DIMENSIONLESS)
        );
    }

    #[test]
    fn referenced_symbols_finds_every_mention_once() {
        let expression = Expression::symbol("a")
            .plus(Expression::symbol("b"))
            .times(Expression::symbol("a"));
        let found = expression.referenced_symbols();
        assert_eq!(found.len(), 2);
        assert!(found.contains("a") && found.contains("b"));
    }

    #[test]
    fn negation_keeps_the_dimension() {
        let table = SymbolTable::new();
        let value = table
            .evaluate(&Expression::Negate(Box::new(voxels(5))), DENSITY)
            .expect("resolves");
        assert_eq!(value.dimension, Dimension::LENGTH);
        assert_eq!(value.to_whole_voxels(), Ok(-5));
    }
}
