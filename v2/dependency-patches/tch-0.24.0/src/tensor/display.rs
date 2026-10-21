/// Pretty printing of tensors
/// This implementation should be in line with the PyTorch version.
/// https://github.com/pytorch/pytorch/blob/7b419e8513a024e172eae767e24ec1b849976b13/torch/_tensor_str.py
use crate::{Kind, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BasicKind {
    Float,
    Int,
    Bool,
    Complex,
}

impl BasicKind {
    fn for_tensor(t: &Tensor) -> BasicKind {
        match t.f_kind() {
            Err(_) => BasicKind::Complex,
            Ok(kind) => match kind {
                Kind::Int | Kind::Int8 | Kind::Uint8 | Kind::Int16 | Kind::Int64 => BasicKind::Int,
                Kind::BFloat16
                | Kind::QInt8
                | Kind::QUInt8
                | Kind::QInt32
                | Kind::Half
                | Kind::Float
                | Kind::Double
                | Kind::Float8e5m2
                | Kind::Float8e4m3fn
                | Kind::Float8e5m2fnuz
                | Kind::Float8e4m3fnuz => BasicKind::Float,
                Kind::Bool => BasicKind::Bool,
                Kind::ComplexHalf | Kind::ComplexFloat | Kind::ComplexDouble => BasicKind::Complex,
            },
        }
    }

    fn _is_floating_point(&self) -> bool {
        match self {
            BasicKind::Float => true,
            BasicKind::Bool | BasicKind::Int | BasicKind::Complex => false,
        }
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.defined() {
            match self.f_kind() {
                Err(err) => write!(f, "Tensor[{:?}, {:?}]", self.size(), err),
                Ok(kind) => {
                    let (is_int, is_float) = match kind {
                        Kind::Int | Kind::Int8 | Kind::Uint8 | Kind::Int16 | Kind::Int64 => {
                            (true, false)
                        }
                        Kind::BFloat16
                        | Kind::QInt8
                        | Kind::QUInt8
                        | Kind::QInt32
                        | Kind::Half
                        | Kind::Float
                        | Kind::Double
                        | Kind::Float8e5m2
                        | Kind::Float8e4m3fn
                        | Kind::Float8e5m2fnuz
                        | Kind::Float8e4m3fnuz => (false, true),
                        Kind::Bool
                        | Kind::ComplexHalf
                        | Kind::ComplexFloat
                        | Kind::ComplexDouble => (false, false),
                    };
                    match (self.size().as_slice(), is_int, is_float) {
                        ([], true, false) => write!(f, "[{}]", i64::try_from(self).unwrap()),
                        ([s], true, false) if *s < 10 => {
                            write!(f, "{:?}", Vec::<i64>::try_from(self).unwrap())
                        }
                        ([], false, true) => write!(f, "[{}]", f64::try_from(self).unwrap()),
                        ([s], false, true) if *s < 10 => {
                            write!(f, "{:?}", Vec::<f64>::try_from(self).unwrap())
                        }
                        _ => write!(f, "Tensor[{:?}, {:?}]", self.size(), kind),
                    }
                }
            }
        } else {
            write!(f, "Tensor[Undefined]")
        }
    }
}

/// Options for Tensor pretty printing
pub struct PrinterOptions {
    precision: usize,
    threshold: usize,
    edge_items: usize,
    line_width: usize,
    sci_mode: Option<bool>,
}

lazy_static! {
    static ref PRINT_OPTS: std::sync::Mutex<PrinterOptions> =
        std::sync::Mutex::new(Default::default());
}

pub fn set_print_options(options: PrinterOptions) {
    *PRINT_OPTS.lock().unwrap() = options
}

pub fn set_print_options_default() {
    *PRINT_OPTS.lock().unwrap() = Default::default()
}

pub fn set_print_options_short() {
    *PRINT_OPTS.lock().unwrap() = PrinterOptions {
        precision: 2,
        threshold: 1000,
        edge_items: 2,
        line_width: 80,
        sci_mode: None,
    }
}

pub fn set_print_options_full() {
    *PRINT_OPTS.lock().unwrap() = PrinterOptions {
        precision: 4,
        threshold: usize::MAX,
        edge_items: 3,
        line_width: 80,
        sci_mode: None,
    }
}

impl Default for PrinterOptions {
    fn default() -> Self {
        Self { precision: 4, threshold: 1000, edge_items: 3, line_width: 80, sci_mode: None }
    }
}

trait TensorFormatter {
    type Elem;

    fn fmt<T: std::fmt::Write>(&self, v: Self::Elem, max_w: usize, f: &mut T) -> std::fmt::Result;

    fn value(tensor: &Tensor) -> Self::Elem;

    fn values(tensor: &Tensor) -> Vec<Self::Elem>;

    fn max_width(&self, to_display: &Tensor) -> usize {
        let mut max_width = 1;
        for v in Self::values(to_display) {
            let mut fmt_size = FmtSize::new();
            let _res = self.fmt(v, 1, &mut fmt_size);
            max_width = usize::max(max_width, fmt_size.final_size())
        }
        max_width
    }

    fn write_newline_indent(i: usize, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f)?;
        for _ in 0..i {
            write!(f, " ")?
        }
        Ok(())
    }

    fn fmt_tensor(
        &self,
        t: &Tensor,
        indent: usize,
        max_w: usize,
        summarize: bool,
        po: &PrinterOptions,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        let size = t.size();
        let edge_items = po.edge_items as i64;
        write!(f, "[")?;
        match size.as_slice() {
            [] => self.fmt(Self::value(t), max_w, f)?,
            [v] if summarize && *v > 2 * edge_items => {
                for v in Self::values(&t.slice(0, None, Some(edge_items), 1)).into_iter() {
                    self.fmt(v, max_w, f)?;
                    write!(f, ", ")?;
                }
                write!(f, "...")?;
                for v in Self::values(&t.slice(0, Some(-edge_items), None, 1)).into_iter() {
                    write!(f, ", ")?;
                    self.fmt(v, max_w, f)?
                }
            }
            [_] => {
                let elements_per_line = usize::max(1, po.line_width / (max_w + 2));
                for (i, v) in Self::values(t).into_iter().enumerate() {
                    if i > 0 {
                        if i % elements_per_line == 0 {
                            write!(f, ",")?;
                            Self::write_newline_indent(indent, f)?
                        } else {
                            write!(f, ", ")?;
                        }
                    }
                    self.fmt(v, max_w, f)?
                }
            }
            _ => {
                if summarize && size[0] > 2 * edge_items {
                    for i in 0..edge_items {
                        self.fmt_tensor(&t.get(i), indent + 1, max_w, summarize, po, f)?;
                        write!(f, ",")?;
                        Self::write_newline_indent(indent, f)?
                    }
                    write!(f, "...")?;
                    Self::write_newline_indent(indent, f)?;
                    for i in size[0] - edge_items..size[0] {
                        self.fmt_tensor(&t.get(i), indent + 1, max_w, summarize, po, f)?;
                        if i + 1 != size[0] {
                            write!(f, ",")?;
                            Self::write_newline_indent(indent, f)?
                        }
                    }
                } else {
                    for i in 0..size[0] {
                        self.fmt_tensor(&t.get(i), indent + 1, max_w, summarize, po, f)?;
                        if i + 1 != size[0] {
                            write!(f, ",")?;
                            Self::write_newline_indent(indent, f)?
                        }
                    }
                }
            }
        }
        write!(f, "]")?;
        Ok(())
    }
}

struct FloatFormatter {
    int_mode: bool,
    sci_mode: bool,
    precision: usize,
}

struct FmtSize {
    current_size: usize,
}

impl FmtSize {
    fn new() -> Self {
        Self { current_size: 0 }
    }

    fn final_size(self) -> usize {
        self.current_size
    }
}

impl std::fmt::Write for FmtSize {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.current_size += s.len();
        Ok(())
    }
}

impl FloatFormatter {
    fn new(t: &Tensor, po: &PrinterOptions) -> Self {
        let mut int_mode = true;
        let mut sci_mode = false;

        let _guard = crate::no_grad_guard();
        let t = t.to_device(crate::Device::Cpu);

        // Rather than containing all values, this should only include
        // values that end up being displayed according to [threshold].
        let nonzero_finite_vals = {
            let t = t.reshape([-1]);
            t.masked_select(&t.isfinite().logical_and(&t.ne(0.)))
        };

        let values = Vec::<f64>::try_from(&nonzero_finite_vals).unwrap();
        if nonzero_finite_vals.numel() > 0 {
            let nonzero_finite_abs = nonzero_finite_vals.abs();
            let nonzero_finite_min = nonzero_finite_abs.min().double_value(&[]);
            let nonzero_finite_max = nonzero_finite_abs.max().double_value(&[]);

            for &value in values.iter() {
                if value.ceil() != value {
                    int_mode = false;
                    break;
                }
            }

            sci_mode = nonzero_finite_max / nonzero_finite_min > 1000.
                || nonzero_finite_max > 1e8
                || nonzero_finite_min < 1e-4
        }

        match po.sci_mode {
            None => {}
            Some(v) => sci_mode = v,
        }
        Self { int_mode, sci_mode, precision: po.precision }
    }
}

impl TensorFormatter for FloatFormatter {
    type Elem = f64;

    fn fmt<T: std::fmt::Write>(&self, v: Self::Elem, max_w: usize, f: &mut T) -> std::fmt::Result {
        if self.sci_mode {
            write!(f, "{v:width$.prec$e}", v = v, width = max_w, prec = self.precision)
        } else if self.int_mode {
            if v.is_finite() {
                write!(f, "{v:width$.0}.", v = v, width = max_w - 1)
            } else {
                write!(f, "{v:max_w$.0}")
            }
        } else {
            write!(f, "{v:width$.prec$}", v = v, width = max_w, prec = self.precision)
        }
    }

    fn value(tensor: &Tensor) -> Self::Elem {
        tensor.double_value(&[])
    }

    fn values(tensor: &Tensor) -> Vec<Self::Elem> {
        Vec::<Self::Elem>::try_from(tensor.reshape(-1)).unwrap()
    }
}

struct IntFormatter;

impl TensorFormatter for IntFormatter {
    type Elem = i64;

    fn fmt<T: std::fmt::Write>(&self, v: Self::Elem, max_w: usize, f: &mut T) -> std::fmt::Result {
        write!(f, "{v:max_w$}")
    }

    fn value(tensor: &Tensor) -> Self::Elem {
        tensor.int64_value(&[])
    }

    fn values(tensor: &Tensor) -> Vec<Self::Elem> {
        Vec::<Self::Elem>::try_from(tensor.reshape(-1)).unwrap()
    }
}

struct BoolFormatter;

impl TensorFormatter for BoolFormatter {
    type Elem = bool;

    fn fmt<T: std::fmt::Write>(&self, v: Self::Elem, max_w: usize, f: &mut T) -> std::fmt::Result {
        let v = if v { "true" } else { "false" };
        write!(f, "{v:max_w$}")
    }

    fn value(tensor: &Tensor) -> Self::Elem {
        tensor.int64_value(&[]) != 0
    }

    fn values(tensor: &Tensor) -> Vec<Self::Elem> {
        Vec::<Self::Elem>::try_from(tensor.reshape(-1)).unwrap()
    }
}

fn get_summarized_data(t: &Tensor, edge_items: i64) -> Tensor {
    let size = t.size();
    if size.is_empty() {
        t.shallow_clone()
    } else if size.len() == 1 {
        if size[0] > 2 * edge_items {
            Tensor::cat(
                &[t.slice(0, None, Some(edge_items), 1), t.slice(0, Some(-edge_items), None, 1)],
                0,
            )
        } else {
            t.shallow_clone()
        }
    } else if size[0] > 2 * edge_items {
        let mut vs: Vec<_> =
            (0..edge_items).map(|i| get_summarized_data(&t.get(i), edge_items)).collect();
        for i in (size[0] - edge_items)..size[0] {
            vs.push(get_summarized_data(&t.get(i), edge_items))
        }
        Tensor::stack(&vs, 0)
    } else {
        let vs: Vec<_> = (0..size[0]).map(|i| get_summarized_data(&t.get(i), edge_items)).collect();
        Tensor::stack(&vs, 0)
    }
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.defined() {
            let po = PRINT_OPTS.lock().unwrap();
            let summarize = self.numel() > po.threshold;
            let basic_kind = BasicKind::for_tensor(self);
            let to_display = if summarize {
                get_summarized_data(self, po.edge_items as i64)
            } else {
                self.shallow_clone()
            };
            match basic_kind {
                BasicKind::Int => {
                    let tf = IntFormatter;
                    let max_w = tf.max_width(&to_display);
                    tf.fmt_tensor(self, 1, max_w, summarize, &po, f)?;
                    writeln!(f)?;
                }
                BasicKind::Float => {
                    let tf = FloatFormatter::new(&to_display, &po);
                    let max_w = tf.max_width(&to_display);
                    tf.fmt_tensor(self, 1, max_w, summarize, &po, f)?;
                    writeln!(f)?;
                }
                BasicKind::Bool => {
                    let tf = BoolFormatter;
                    let max_w = tf.max_width(&to_display);
                    tf.fmt_tensor(self, 1, max_w, summarize, &po, f)?;
                    writeln!(f)?;
                }
                BasicKind::Complex => {}
            };
            let kind = match self.f_kind() {
                Ok(kind) => format!("{kind:?}"),
                Err(err) => format!("{err:?}"),
            };
            write!(f, "Tensor[{:?}, {}]", self.size(), kind)
        } else {
            write!(f, "Tensor[Undefined]")
        }
    }
}
