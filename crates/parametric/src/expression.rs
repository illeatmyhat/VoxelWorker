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
#[allow(clippy::use_self)]
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
    #[must_use]
    pub const fn length(measurement: Measurement) -> Self {
        Self::Literal(Literal::Length(measurement))
    }

    /// An angle literal.
    #[must_use]
    pub const fn angle(angle: AngleMeasurement) -> Self {
        Self::Literal(Literal::Angle(angle))
    }

    /// A pure-number literal.
    #[must_use]
    pub const fn number(value: ExactRational) -> Self {
        Self::Literal(Literal::Number(value))
    }

    /// A whole-number literal — the common case, so it does not have to be spelled out.
    #[must_use]
    pub fn whole(value: i64) -> Self {
        Self::number(ExactRational::from_integer(i128::from(value)))
    }

    /// The [`Measurement`] this expression IS, when it is one length literal and nothing else.
    ///
    /// **The authored blocks+voxels split survives only here.** Evaluating any expression yields
    /// a [`Quantity`] — a flat voxel count at one density — and a count cannot say whether the
    /// author wrote `3 blocks` or `48 voxels`. That distinction is what a density re-target reads,
    /// so a caller that must RETAIN what was typed asks this first and only falls back to the
    /// evaluated value when the answer is `None`.
    ///
    /// `None` for anything compound, for a symbol, and for an angle or a bare number — none of
    /// which is a length someone wrote down.
    #[must_use]
    pub const fn as_authored_length(&self) -> Option<Measurement> {
        match self {
            Self::Literal(Literal::Length(measurement)) => Some(*measurement),
            Self::Literal(Literal::Angle(_) | Literal::Number(_))
            | Self::Symbol(_)
            | Self::Negate(_)
            | Self::Binary { .. } => None,
        }
    }

    /// Whether the author wrote NO unit word anywhere: every leaf is a bare number.
    ///
    /// This is what a binding asks before applying its default unit, and it is a question about
    /// the TREE rather than about the evaluated dimension, because a dimensionless answer arises
    /// two ways — a bare count, and a ratio that cancelled (`3 blocks / 1 block`). The first named
    /// no unit and takes the field's; the second named two and is a ratio, not a length. A symbol
    /// is a unit the table will supply, so it counts as one named.
    #[must_use]
    pub fn names_no_unit(&self) -> bool {
        match self {
            Self::Literal(Literal::Number(_)) => true,
            Self::Literal(Literal::Length(_) | Literal::Angle(_)) | Self::Symbol(_) => false,
            Self::Negate(inner) => inner.names_no_unit(),
            Self::Binary { left, right, .. } => left.names_no_unit() && right.names_no_unit(),
        }
    }

    /// A reference to a named parameter.
    #[must_use]
    pub fn symbol(name: impl Into<String>) -> Self {
        Self::Symbol(name.into())
    }

    /// `self + other`.
    #[must_use]
    pub fn plus(self, other: Self) -> Self {
        self.binary(Operator::Add, other)
    }

    /// `self - other`.
    #[must_use]
    pub fn minus(self, other: Self) -> Self {
        self.binary(Operator::Subtract, other)
    }

    /// `self * other`.
    #[must_use]
    pub fn times(self, other: Self) -> Self {
        self.binary(Operator::Multiply, other)
    }

    /// `self / other`.
    #[must_use]
    pub fn divided_by(self, other: Self) -> Self {
        self.binary(Operator::Divide, other)
    }

    fn binary(self, operator: Operator, other: Self) -> Self {
        Self::Binary {
            left: Box::new(self),
            operator,
            right: Box::new(other),
        }
    }

    /// Every symbol this expression mentions, deduplicated — what the dependency graph and
    /// the cycle check are built from.
    #[must_use]
    pub fn referenced_symbols(&self) -> BTreeSet<String> {
        let mut found = BTreeSet::new();
        self.collect_symbols(&mut found);
        found
    }

    fn collect_symbols(&self, into: &mut BTreeSet<String>) {
        match self {
            Self::Literal(_) => {}
            Self::Symbol(name) => {
                into.insert(name.clone());
            }
            Self::Negate(inner) => inner.collect_symbols(into),
            Self::Binary { left, right, .. } => {
                left.collect_symbols(into);
                right.collect_symbols(into);
            }
        }
    }
}

/// Why a typed expression could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpressionParseError {
    /// Nothing but whitespace.
    Empty,
    /// A measurement literal inside the expression is malformed. Carries the literal grammar's
    /// own complaint rather than restating it, so `8/0 blocks` reads the same wherever it is
    /// typed.
    Measurement(crate::units::MeasurementParseError),
    /// A token that cannot begin an operand — an operator with nothing to its left, a stray
    /// bracket, a character the lexer could not read.
    UnexpectedToken {
        /// The token as written.
        text: String,
    },
    /// The input stopped where an operand was required: a trailing `+`, an open bracket.
    UnexpectedEnd,
    /// An opening bracket with no closing one.
    UnclosedParen,
    /// A complete expression followed by something that is not an operator.
    TrailingInput {
        /// The first token past the expression.
        text: String,
    },
}

impl From<crate::units::MeasurementParseError> for ExpressionParseError {
    fn from(error: crate::units::MeasurementParseError) -> Self {
        Self::Measurement(error)
    }
}

impl core::fmt::Display for ExpressionParseError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty => write!(formatter, "empty expression"),
            Self::Measurement(error) => write!(formatter, "{error}"),
            Self::UnexpectedToken { text } => {
                write!(formatter, "`{text}` cannot start a value here")
            }
            Self::UnexpectedEnd => {
                write!(formatter, "the expression ends where a value was expected")
            }
            Self::UnclosedParen => write!(formatter, "a `(` is never closed"),
            Self::TrailingInput { text } => {
                write!(formatter, "`{text}` is left over after the expression")
            }
        }
    }
}

impl std::error::Error for ExpressionParseError {}

/// Read an authored expression.
///
/// The grammar, over the token stream the units lexer produces:
///
/// ```text
/// expression  := term (('+' | '-') term)*
/// term        := factor (('*' | '/') factor)*
/// factor      := '-' factor | primary
/// primary     := measurement | number | symbol | '(' expression ')'
/// measurement := (number+ length_word)+
/// angle       := number+ degree_word
/// ```
///
/// **A measurement literal is a GREEDY MUNCH, and that is what makes the two grammars compose.**
/// `3 blocks 8 voxels` is ONE operand, not a product of four tokens, and `3 8/16 blocks` is one
/// operand too — the sixteenths idiom is a run of numbers closed by a unit. The munch is handed
/// straight to the literal grammar's own reader, so the duplicate-unit rule and the
/// sub-voxel rejection are not restated here. `3 blocks * 2` stops the munch at `*`, because `*`
/// is not a unit word.
///
/// A bare number is DIMENSIONLESS — a count or a scale factor. `2 * 3 blocks` is a length;
/// `2 * 3` is the number six; `3 blocks + 2` is a dimension error, raised at evaluation rather
/// than here, because this layer reads structure and the evaluator judges dimensions.
///
/// **An ANGLE literal is the same shape with a different unit.** `45 deg` is one operand, and so
/// is `22 1/2 degrees`. Which literal reader a munch goes to is decided by its first unit word;
/// mixing the two dimensions inside one munch is refused by the reader that got it, by name.
/// Whether an angle is a legal ANSWER is not this layer's question — `3 blocks + 45 deg` parses
/// and then fails to evaluate, the same way `3 blocks + 2` does.
///
/// # Errors
///
/// Returns the first structural complaint, or the literal grammar's own error for a malformed
/// measurement inside the expression.
pub fn parse(input: &str) -> Result<Expression, ExpressionParseError> {
    let tokens = crate::units::tokenise(input);
    if tokens.is_empty() {
        return Err(ExpressionParseError::Empty);
    }
    let mut reader = TokenReader {
        tokens: &tokens,
        at: 0,
    };
    let expression = reader.read_sum()?;
    reader.peek().map_or(Ok(expression), |token| {
        Err(ExpressionParseError::TrailingInput {
            text: describe(token),
        })
    })
}

/// How a token reads back in an error message.
fn describe(token: &crate::units::Token) -> String {
    use crate::units::Token;
    match token {
        Token::Number(text) | Token::Word(text) | Token::Unexpected(text) => text.clone(),
        Token::Operator(sign) => sign.to_string(),
        Token::OpenParen => "(".to_owned(),
        Token::CloseParen => ")".to_owned(),
    }
}

/// A cursor over the token slice. Recursive descent, one level per precedence tier.
struct TokenReader<'a> {
    tokens: &'a [crate::units::Token],
    at: usize,
}

impl TokenReader<'_> {
    fn peek(&self) -> Option<&crate::units::Token> {
        self.tokens.get(self.at)
    }

    const fn advance(&mut self) {
        self.at = self.at.saturating_add(1);
    }

    /// `+` and `-`, left-associative.
    fn read_sum(&mut self) -> Result<Expression, ExpressionParseError> {
        let mut left = self.read_product()?;
        while let Some(&crate::units::Token::Operator(sign @ ('+' | '-'))) = self.peek() {
            self.advance();
            let right = self.read_product()?;
            left = if sign == '+' {
                left.plus(right)
            } else {
                left.minus(right)
            };
        }
        Ok(left)
    }

    /// `*` and `/`, left-associative and binding tighter than `+`.
    fn read_product(&mut self) -> Result<Expression, ExpressionParseError> {
        let mut left = self.read_signed()?;
        while let Some(&crate::units::Token::Operator(sign @ ('*' | '/'))) = self.peek() {
            self.advance();
            let right = self.read_signed()?;
            left = if sign == '*' {
                left.times(right)
            } else {
                left.divided_by(right)
            };
        }
        Ok(left)
    }

    /// A leading `-` on a bracket or a symbol. A minus on a NUMBER never reaches here: the lexer
    /// folds a prefix minus into the number it precedes, which is what keeps `-3b` a signed
    /// literal in both grammars.
    fn read_signed(&mut self) -> Result<Expression, ExpressionParseError> {
        if self.peek() == Some(&crate::units::Token::Operator('-')) {
            self.advance();
            return Ok(Expression::Negate(Box::new(self.read_signed()?)));
        }
        self.read_operand()
    }

    fn read_operand(&mut self) -> Result<Expression, ExpressionParseError> {
        use crate::units::Token;
        let Some(token) = self.peek() else {
            return Err(ExpressionParseError::UnexpectedEnd);
        };
        match token {
            Token::OpenParen => {
                self.advance();
                let inner = self.read_sum()?;
                match self.peek() {
                    Some(Token::CloseParen) => {
                        self.advance();
                        Ok(inner)
                    }
                    _ => Err(ExpressionParseError::UnclosedParen),
                }
            }
            Token::Number(_) => self.read_number_or_measurement(),
            Token::Word(name) => {
                // A unit word with no number in front of it is not an operand. Reading it as a
                // parameter called `blocks` would turn a typo into an unknown-parameter error
                // that names the wrong problem.
                if crate::units::is_unit_word(name) {
                    return Err(ExpressionParseError::UnexpectedToken { text: name.clone() });
                }
                let symbol = Expression::Symbol(name.clone());
                self.advance();
                Ok(symbol)
            }
            other => Err(ExpressionParseError::UnexpectedToken {
                text: describe(other),
            }),
        }
    }

    /// The greedy munch: as many `number+ unit_word` groups as run on from here.
    ///
    /// Ends at the first token that is neither a number continuing a group nor a unit word
    /// closing one. If the run closed no group at all, the operand was a bare dimensionless
    /// number and the single leading number is taken instead.
    ///
    /// **The FIRST unit word picks the reader, and the reader judges the rest.** The munch itself
    /// is dimension-blind — it stops on any unit word — so `3 blocks 45 deg` is munched whole and
    /// handed to the length grammar, which refuses the degree by name. Filtering the munch by
    /// dimension instead would end the operand at `45` and report the leftover `deg` as trailing
    /// input, which describes the parser's state rather than the author's mistake.
    fn read_number_or_measurement(&mut self) -> Result<Expression, ExpressionParseError> {
        use crate::units::{Token, UnitDimension};
        let start = self.at;
        let mut end_of_last_group = None;
        let mut dimension = None;
        let mut cursor = self.at;
        loop {
            let numbers_started = cursor;
            while matches!(self.tokens.get(cursor), Some(Token::Number(_))) {
                cursor = cursor.saturating_add(1);
            }
            if cursor == numbers_started {
                break;
            }
            match self.tokens.get(cursor) {
                Some(Token::Word(word)) if crate::units::is_unit_word(word) => {
                    dimension = dimension.or_else(|| crate::units::unit_dimension(word));
                    cursor = cursor.saturating_add(1);
                    end_of_last_group = Some(cursor);
                }
                _ => break,
            }
        }
        if let Some(end) = end_of_last_group {
            let munched = self.tokens.get(start..end).unwrap_or(&[]);
            let literal = match dimension {
                Some(UnitDimension::Angle) => {
                    Expression::angle(crate::units::angle_from_tokens(munched)?)
                }
                _ => Expression::length(crate::units::measurement_from_tokens(munched)?),
            };
            self.at = end;
            return Ok(literal);
        }
        // No unit closed a group, so this is a bare count. Exactly one number belongs to the
        // operand — `3 4` is two operands with nothing between them, and the caller reports the
        // second as left over rather than silently summing them.
        let Some(Token::Number(text)) = self.tokens.get(start) else {
            return Err(ExpressionParseError::UnexpectedEnd);
        };
        let value = crate::units::rational_from_number_token(text)?;
        self.at = start.saturating_add(1);
        Ok(Expression::number(value))
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
        Self::Quantity(error)
    }
}

impl core::fmt::Display for EvaluationError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownSymbol { name } => {
                write!(formatter, "unknown parameter `{name}`")
            }
            Self::CircularReference { cycle } => {
                write!(formatter, "`{}` depends on itself", cycle.join("` → `"))
            }
            Self::ShadowsBuiltIn { name } => {
                write!(formatter, "`{name}` is built in and cannot be redefined")
            }
            Self::Quantity(error) => write!(formatter, "{error}"),
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
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether a name is built in, and so neither definable nor deletable.
    #[must_use]
    pub fn is_built_in(name: &str) -> bool {
        name == VOXEL_DENSITY
    }

    /// Define or redefine a parameter.
    ///
    /// Refuses a name that shadows a built-in, and refuses a definition that would put the
    /// table into a cycle — checked against the table as it *would be*, so redefining an
    /// existing parameter is judged on its new expression rather than its old one.
    ///
    /// # Errors
    ///
    /// Returns an error when `name` shadows a built-in or when the new definition introduces
    /// a circular reference.
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
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Expression> {
        self.parameters.get(name)
    }

    /// Every defined parameter name, sorted.
    #[must_use = "iterate over the table's parameter names"]
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.parameters.keys().map(String::as_str)
    }

    /// Evaluate an expression at the document density.
    ///
    /// `density` is voxels-per-block: it scales every block term and is the value
    /// [`VOXEL_DENSITY`] resolves to.
    ///
    /// # Errors
    ///
    /// Returns the first unknown-symbol, cycle, or quantity-arithmetic error encountered.
    pub fn evaluate(
        &self,
        expression: &Expression,
        density: u32,
    ) -> Result<Quantity, EvaluationError> {
        self.evaluate_within(expression, density, &mut Vec::new())
    }

    /// Evaluate a named parameter at the document density.
    ///
    /// # Errors
    ///
    /// Returns the first unknown-symbol, cycle, or quantity-arithmetic error encountered.
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
                i128::from(density),
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
///
/// # Errors
///
/// Returns the same evaluation error that would occur while evaluating the expression.
pub fn dimension_of(
    table: &SymbolTable,
    expression: &Expression,
) -> Result<Dimension, EvaluationError> {
    table.evaluate(expression, 1).map(|value| value.dimension)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::as_conversions,
        clippy::expect_used,
        clippy::panic,
        clippy::unwrap_used
    )]

    use super::*;

    const DENSITY: u32 = 16;

    fn blocks(count: i64) -> Expression {
        Expression::length(Measurement::new(
            ExactRational::from_integer(i128::from(count)),
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

    /// An angle is an operand of the same grammar, not a second parser bolted on. It munches the
    /// same way, takes the same number forms, and composes with the same operators.
    #[test]
    fn an_angle_literal_is_an_operand_like_any_other() {
        let degrees = |numerator: i128, denominator: i128| {
            AngleMeasurement::new(ExactRational::new(numerator, denominator).expect("valid"))
        };
        let cases: [(&str, Expression); 4] = [
            ("45 deg", Expression::angle(degrees(45, 1))),
            ("45deg", Expression::angle(degrees(45, 1))),
            ("22 1/2 degrees", Expression::angle(degrees(45, 2))),
            (
                "45 deg / 2",
                Expression::angle(degrees(45, 1)).divided_by(Expression::whole(2)),
            ),
        ];
        for (text, expected) in cases {
            assert_eq!(parse(text), Ok(expected), "`{text}`");
        }
    }

    /// A compound angle loses NOTHING, which is why there is no `as_authored_angle` beside
    /// `as_authored_length`. The evaluator is exact rationals and they are closed under all four
    /// operators, so `45 deg / 2` evaluates to exactly 45/2 degrees — there is no authored split
    /// to rescue, the way blocks-and-voxels has one.
    #[test]
    fn a_compound_angle_stays_exact() {
        let value = SymbolTable::new()
            .evaluate(&parse("45 deg / 2").expect("parses"), DENSITY)
            .expect("resolves");
        assert_eq!(value.dimension, Dimension::ANGLE);
        assert_eq!(
            value.value,
            ExactRational::new(45, 2).expect("a valid rational")
        );
    }

    /// Mixing the dimensions inside ONE munch is the literal grammar's complaint, by name, and
    /// not a structural one about a leftover token.
    #[test]
    fn a_degree_inside_a_length_literal_is_named() {
        assert_eq!(
            parse("3 blocks 45 deg"),
            Err(ExpressionParseError::Measurement(
                crate::units::MeasurementParseError::WrongDimension {
                    unit_text: "deg".to_owned(),
                    reading: "length",
                }
            ))
        );
    }

    /// Mixing them across an OPERATOR parses fine and fails where dimensions are judged. Same as
    /// `3 blocks + 2`: this layer reads structure, the evaluator reads dimensions.
    #[test]
    fn a_length_plus_an_angle_parses_and_then_refuses() {
        let mixed = parse("3 blocks + 45 deg").expect("structurally fine");
        assert!(matches!(
            SymbolTable::new().evaluate(&mixed, DENSITY),
            Err(EvaluationError::Quantity(
                QuantityError::MismatchedDimensions { .. }
            ))
        ));
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

    /// What a parsed expression is WORTH, at a density, so a test reads as the author would.
    fn worth(input: &str) -> Result<i64, String> {
        let expression = parse(input).map_err(|error| error.to_string())?;
        SymbolTable::new()
            .evaluate(&expression, DENSITY)
            .map_err(|error| error.to_string())?
            .to_whole_voxels()
            .map_err(|error| error.to_string())
    }

    /// A single measurement literal must mean, through the expression grammar, exactly what it
    /// has always meant through the literal one.
    ///
    /// This is the compatibility half of the lexer split. The field that reads these is about to
    /// stop calling `units::parse`, and every spelling it accepted must survive the change —
    /// including the two the grammars could plausibly disagree about: the sixteenths idiom
    /// (`3 8/16 blocks`, a run of numbers closed by one unit) and the signed offset (`-1b 4v`,
    /// where the minus belongs to the first term and not to the sum).
    #[test]
    fn every_literal_spelling_means_the_same_through_the_expression_grammar() {
        for spelling in [
            "3 blocks",
            "3b",
            "3.5 blocks",
            "8/16 blocks",
            "3 8/16 blocks",
            "56 voxels",
            "3 blocks 8 voxels",
            "3b 8v",
            "-3b",
            "-1b 4v",
            "-3.5 blocks",
            "-8/16 blocks",
            "-12 voxels",
            "-1 blocks 4 voxels",
        ] {
            let through_the_literal = crate::units::parse(spelling)
                .unwrap_or_else(|error| panic!("`{spelling}` parses as a literal: {error}"))
                .to_voxels(DENSITY)
                .expect("lands on whole voxels");
            assert_eq!(
                worth(spelling),
                Ok(through_the_literal),
                "`{spelling}` must mean the same to both grammars"
            );
        }
    }

    /// Arithmetic, precedence, and the two readings of a slash.
    ///
    /// The slash cases are the ones the lexer had to decide. `8/16 blocks` is half a block — a
    /// closed-up fraction is one operand. `8 / 16 blocks` is not a length at all, so it is not
    /// here; it is in the dimension test below. And `24 / 2 / 3` is four, not thirty-six: a
    /// closed-up fraction directly after a division sign would have re-associated it.
    #[test]
    fn arithmetic_binds_the_way_arithmetic_binds() {
        assert_eq!(worth("2 * 3 blocks"), Ok(96));
        assert_eq!(worth("3 blocks * 2"), Ok(96));
        assert_eq!(worth("1 block + 4 voxels"), Ok(20));
        assert_eq!(worth("1 block - 4 voxels"), Ok(12));
        assert_eq!(worth("2 blocks + 2 * 1 block"), Ok(64));
        assert_eq!(worth("(2 blocks + 2 blocks) * 2"), Ok(128));
        assert_eq!(worth("2 blocks / 2"), Ok(16));
        assert_eq!(worth("-(1 block)"), Ok(-16));
        assert_eq!(worth("8/16 blocks"), Ok(8));
        assert_eq!(worth("24 voxels / 2 / 3"), Ok(4));
        // Whitespace is the whole difference between a division and a fraction, so these two
        // spellings mean different things on purpose. Pinned because it is the surprise.
        assert_eq!(worth("24 voxels / 2/3"), Ok(36));
        assert_eq!(worth("3 blocks * voxel_density / voxel_density"), Ok(48));
    }

    /// The two things this grammar refuses on purpose, and the two ways it can be malformed.
    ///
    /// A SYMBOL parses and then fails to resolve. That is not a stub: the table is empty, and an
    /// empty table's honest answer to `width` is that it knows no such parameter. The same code
    /// path serves a real table the day one exists.
    #[test]
    fn a_symbol_and_a_dimension_mismatch_are_refused_where_they_belong() {
        assert!(parse("2 * width + 3 blocks").is_ok(), "a symbol PARSES");
        assert_eq!(
            worth("2 * width + 3 blocks"),
            Err("unknown parameter `width`".to_owned())
        );
        // A dimensionless result is not a length, and asking it for voxels is the mismatch —
        // caught by the same rule, one layer along.
        let density = SymbolTable::new()
            .evaluate(&parse("voxel_density").expect("parses"), DENSITY)
            .expect("resolves");
        assert_eq!(density.dimension, Dimension::DIMENSIONLESS);
        assert!(density.to_whole_voxels().is_err());
        // Adding a count to a length is a DIMENSION error, and it is the evaluator's to raise —
        // the parser reads structure and judges nothing about what the structure means.
        assert!(parse("3 blocks + 2").is_ok());
        assert!(worth("3 blocks + 2").is_err());

        assert_eq!(parse(""), Err(ExpressionParseError::Empty));
        assert_eq!(parse("   "), Err(ExpressionParseError::Empty));
        assert_eq!(
            parse("3 blocks +"),
            Err(ExpressionParseError::UnexpectedEnd)
        );
        assert_eq!(parse("(3 blocks"), Err(ExpressionParseError::UnclosedParen));
        assert_eq!(
            parse("3 blocks 4 blocks"),
            Err(ExpressionParseError::Measurement(
                crate::units::MeasurementParseError::DuplicateUnit {
                    unit_text: "blocks".to_owned()
                }
            )),
            "the munch takes both groups, and the literal grammar rejects the duplicate"
        );
        assert_eq!(
            parse("3 4"),
            Err(ExpressionParseError::TrailingInput {
                text: "4".to_owned()
            }),
            "two bare numbers with nothing between them are not one operand"
        );
        assert_eq!(
            parse("blocks"),
            Err(ExpressionParseError::UnexpectedToken {
                text: "blocks".to_owned()
            }),
            "a unit word alone is a malformed literal, not a parameter called `blocks`"
        );
        assert_eq!(
            parse("3 blocks @ 2"),
            Err(ExpressionParseError::TrailingInput {
                text: "@".to_owned()
            }),
            "a character the lexer cannot read is named, never skipped"
        );
    }
}
