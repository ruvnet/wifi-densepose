//! Variable stores.
use super::Init;
use crate::tensor::Tensor;
use crate::wrappers::stream::ReadSeekAdapter;
use crate::{Device, Kind, TchError};
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::HashMap;
use std::io::{Read, Seek};
use std::ops::Div;
use std::sync::{Arc, Mutex, MutexGuard};

/// The separator is used to separate path elements in the tensor names.
const SEP: char = '.';

#[derive(Debug)]
pub struct Var {
    pub tensor: Tensor,
    pub group: usize,
}

// When the variable store is frozen, the trainable_variables vector
// still contains the same tensors however these tensors are set not
// to require gradients.
#[derive(Debug)]
pub struct Variables {
    pub named_variables: HashMap<String, Tensor>,
    pub trainable_variables: Vec<Var>,
}

/// A VarStore is used to store variables used by one or multiple layers.
/// It specifies a single device where all variables are stored.
#[derive(Debug)]
pub struct VarStore {
    pub variables_: Arc<Mutex<Variables>>,
    device: Device,
    kind: Kind,
}

/// A variable store with an associated path for variables naming.
#[derive(Debug, Clone)]
pub struct Path<'a> {
    path: Vec<String>,
    group: usize,
    var_store: &'a VarStore,
}

/// An Entry holds an entry corresponding to a given name in Path.
#[derive(Debug)]
pub struct Entry<'a> {
    name: &'a str,
    variables: MutexGuard<'a, Variables>,
    // This field holds the mutex lock
    path: &'a Path<'a>,
}

impl VarStore {
    /// Creates a new var-store located on the specified device.
    pub fn new(device: Device) -> VarStore {
        let variables =
            Variables { named_variables: HashMap::new(), trainable_variables: Vec::new() };
        VarStore { variables_: Arc::new(Mutex::new(variables)), device, kind: Kind::Float }
    }

    pub fn merge(var_stores: Vec<(VarStore, Option<&str>)>) -> Result<VarStore, TchError> {
        let mut new_var_store = VarStore::new(Device::Cpu);

        if var_stores.is_empty() {
            Ok(new_var_store)
        } else {
            let mut new_variables =
                Variables { named_variables: HashMap::new(), trainable_variables: Vec::new() };
            let device = var_stores[0].0.device();

            for (var_store, prefix) in var_stores {
                if var_store.device() != device {
                    return Err(TchError::Torch(format!(
                        "All VarStores must be on the same device, got {:?} and {:?}",
                        device,
                        var_store.device()
                    )));
                }
                for (var_name, var) in var_store.variables() {
                    let new_var_name = format!("{}{}", prefix.unwrap_or(""), var_name);
                    match new_variables.named_variables.entry(new_var_name) {
                        Occupied(v) => {
                            return Err(TchError::Torch(format!(
                                "Duplicate variable name found: {}. Provide a unique prefix to allow merge operation",
                                v.key(),
                            )));
                        }
                        Vacant(v) => {
                            v.insert(var);
                        }
                    }
                }
                for trainable_var in
                    var_store.variables_.lock().unwrap().trainable_variables.drain(..)
                {
                    new_variables.trainable_variables.push(trainable_var);
                }
            }
            new_var_store.variables_ = Arc::new(Mutex::new(new_variables));
            new_var_store.device = device;

            Ok(new_var_store)
        }
    }

    /// Gets the device for this var-store.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Gets the default kind of new variables
    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// Returns the number of tensors currently stored on this var-store.
    pub fn len(&self) -> usize {
        let variables = self.variables_.lock().unwrap();
        variables.named_variables.len()
    }

    /// Returns true if no tensors are currently stored on this var-store.
    pub fn is_empty(&self) -> bool {
        let variables = self.variables_.lock().unwrap();
        variables.named_variables.is_empty()
    }

    /// Returns all the trainable variables for this var-store.
    pub fn trainable_variables(&self) -> Vec<Tensor> {
        let variables = self.variables_.lock().unwrap();
        variables.trainable_variables.iter().map(|v| v.tensor.shallow_clone()).collect()
    }

    /// Returns all variables along with their names.
    pub fn variables(&self) -> HashMap<String, Tensor> {
        let variables = self.variables_.lock().unwrap();
        variables
            .named_variables
            .iter()
            .map(|(name, v)| (name.clone(), v.shallow_clone()))
            .collect()
    }

    /// Gets the root path for this variable store.
    ///
    /// Variables are named and organized using paths. This function returns
    /// the top level path for the var store and can be combined with '/'
    /// to create sub-paths.
    pub fn root(&self) -> Path<'_> {
        Path { path: vec![], group: 0, var_store: self }
    }

    /// Saves the var-store variable values to a file.
    ///
    /// Weight values for all the tensors currently stored in the
    /// var-store are saved in the given file.
    ///
    /// If the given path ends with the suffix `.safetensors`, the file will
    /// be saved in safetensors format. Otherwise, libtorch C++ module format
    /// will be used. Note that saving in pickle format (`.pt` extension) is
    /// not supported by the C++ API of Torch.
    pub fn save<T: AsRef<std::path::Path>>(&self, path: T) -> Result<(), TchError> {
        let variables = self.variables_.lock().unwrap();
        let named_tensors = variables.named_variables.iter().collect::<Vec<_>>();
        match path.as_ref().extension().and_then(|x| x.to_str()) {
            Some("safetensors") => Tensor::write_safetensors(named_tensors.as_slice(), path),
            Some(_) | None => Tensor::save_multi(named_tensors.as_slice(), path),
        }
    }

    /// Saves the var-store variable values to a stream.
    ///
    /// Weight values for all the tensors currently stored in the
    /// var-store gets saved in the given stream.
    pub fn save_to_stream<W: std::io::Write>(&self, stream: W) -> Result<(), TchError> {
        let variables = self.variables_.lock().unwrap();
        let named_tensors = variables.named_variables.iter().collect::<Vec<_>>();
        Tensor::save_multi_to_stream(named_tensors.as_slice(), stream)
    }

    fn named_tensors<T: AsRef<std::path::Path>>(
        &self,
        path: T,
    ) -> Result<HashMap<String, Tensor>, TchError> {
        let named_tensors = match path.as_ref().extension().and_then(|x| x.to_str()) {
            Some("bin") | Some("pt") => Tensor::loadz_multi_with_device(&path, self.device),
            Some("safetensors") => Tensor::read_safetensors(path),
            Some(_) | None => Tensor::load_multi_with_device(&path, self.device),
        };
        Ok(named_tensors?.into_iter().collect())
    }

    /// Copies the data from source tensor to destination
    ///
    /// Updates the precision of the destination to match the source
    fn copy_data_with_precision_update(src: &Tensor, dst: &mut Tensor) -> Result<(), TchError> {
        dst.set_data(&dst.to_kind(src.kind()));
        dst.f_copy_(src)
    }

    fn load_internal<T: AsRef<std::path::Path>>(&mut self, path: T) -> Result<(), TchError> {
        let named_tensors = self.named_tensors(&path)?;
        let mut variables = self.variables_.lock().unwrap();
        for (name, var) in variables.named_variables.iter_mut() {
            match named_tensors.get(name) {
                Some(src) => crate::no_grad(|| {
                    Self::copy_data_with_precision_update(src, var)
                        .map_err(|e| e.path_context(name))
                })?,
                None => {
                    return Err(TchError::TensorNameNotFound(
                        name.to_string(),
                        path.as_ref().to_string_lossy().into_owned(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Loads the var-store variable values from a file.
    ///
    /// Weight values for all the tensors currently stored in the
    /// var-store are loaded from the given file. Note that the set of
    /// variables stored in the var-store is not changed, only the values
    /// for these tensors are modified.
    ///
    /// The format of the file is deduced from the file extension:
    /// - `.safetensors`: The file is assumed to be in safetensors format.
    /// - `.bin` or `.pt`: The file is assumed to be in pickle format.
    /// - Otherwise, the file is assumed to be in libtorch C++ module format.
    pub fn load<T: AsRef<std::path::Path>>(&mut self, path: T) -> Result<(), TchError> {
        if self.device != Device::Mps {
            self.load_internal(path)
        } else {
            // Current workaround to allow loading in MPS device.
            // On new libtorch releases check if direct loading becomes possible and revert
            // See (https://github.com/LaurentMazare/tch-rs/issues/609#issuecomment-1427071598).
            self.set_device(Device::Cpu);
            let or_error = self.load_internal(path);
            // Be cautious not to early exit so as to ensure that the device is set back to Mps
            // even on errors.
            self.set_device(Device::Mps);
            or_error
        }
    }

    /// Loads the var-store variable values from a stream.
    ///
    /// Weight values for all the tensors currently stored in the
    /// var-store gets loaded from the given stream. Note that the set of
    /// variables stored in the var-store is not changed, only the values
    /// for these tensors are modified.
    pub fn load_from_stream<S: Read + Seek>(&mut self, stream: S) -> Result<(), TchError> {
        let adapter = ReadSeekAdapter::new(stream);
        let named_tensors = Tensor::load_multi_from_stream_with_device(adapter, self.device)?;
        let named_tensors: HashMap<_, _> = named_tensors.into_iter().collect();
        let mut variables = self.variables_.lock().unwrap();
        for (name, var) in variables.named_variables.iter_mut() {
            match named_tensors.get(name) {
                Some(src) => crate::no_grad(|| {
                    Self::copy_data_with_precision_update(src, var)
                        .map_err(|e| e.path_context(name))
                })?,
                None => {
                    return Err(TchError::TensorNameNotFound(
                        name.to_string(),
                        "source stream".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Loads the var-store variable values from a file if it exists.
    ///
    /// Weight values for the tensors currently stored in the var-store and the given file get
    /// loaded from the given file. If a variable in the var store is not present in the given file,
    /// it is skipped and its values are not updated. This method should be used if pre-trained
    /// weight for only parts of the model are available.
    /// Note that the set of variables stored in the var-store is not changed, only the values
    /// for these tensors are modified.
    ///
    /// Returns a String Vector containing the names of missing variables.
    pub fn load_partial<T: AsRef<std::path::Path>>(
        &mut self,
        path: T,
    ) -> Result<Vec<String>, TchError> {
        let named_tensors = self.named_tensors(&path)?;
        let mut variables = self.variables_.lock().unwrap();
        let mut missing_variables = Vec::new();
        for (name, var) in variables.named_variables.iter_mut() {
            match named_tensors.get(name) {
                Some(src) => crate::no_grad(|| {
                    Self::copy_data_with_precision_update(src, var)
                        .map_err(|e| e.path_context(name))
                })?,
                None => {
                    missing_variables.push(name.to_owned());
                }
            }
        }
        Ok(missing_variables)
    }

    /// Freezes a var store.
    ///
    /// Gradients for the variables in this store are not tracked
    /// anymore.
    pub fn freeze(&mut self) {
        let variables = self.variables_.lock().unwrap();
        for variable in variables.trainable_variables.iter() {
            let _v = variable.tensor.set_requires_grad(false);
        }
    }

    /// Unfreezes a var store.
    ///
    /// Gradients for the variables in this store are tracked again.
    pub fn unfreeze(&mut self) {
        let variables = self.variables_.lock().unwrap();
        for variable in variables.trainable_variables.iter() {
            let _v = variable.tensor.set_requires_grad(true);
        }
    }

    /// Casts all variables in a var store to the target kind and sets the default kind
    /// for new variables.
    ///
    /// For floating-point conversion, methods `half`, `bfloat16`, `float` and `double`
    /// should be preferred as they ensure only float-like variables will be converted
    /// to the target type.
    pub fn set_kind(&mut self, kind: Kind) {
        self.root().set_kind(kind);
        self.kind = kind;
    }

    /// Casts all float-like variable of a var store to half-precision (Half kind).
    pub fn half(&mut self) {
        self.root().half();
    }

    /// Casts all float-like variable of a var store to bfloat16-precision (BFloat16 kind).
    pub fn bfloat16(&mut self) {
        self.root().bfloat16();
    }

    /// Casts all float-like variable of a var store to single-precision (Float kind).
    pub fn float(&mut self) {
        self.root().float();
    }

    /// Casts all float-like variable of a var store to single-precision (Double kind).
    pub fn double(&mut self) {
        self.root().double();
    }

    /// Migrates a var store and all its tensor to a target device.
    pub fn set_device(&mut self, device: Device) {
        let mut variables = self.variables_.lock().unwrap();
        for (_, variable) in variables.named_variables.iter_mut() {
            variable.set_data(&variable.to_device(device));
        }
        self.device = device
    }

    /// Copies variable values from a source var store to this var store.
    ///
    /// All the variables in this var store have to exist with the same
    /// name in the source var store, otherwise an error is returned.
    pub fn copy(&mut self, src: &VarStore) -> Result<(), TchError> {
        let mut variables = self.variables_.lock().unwrap();
        let src_variables = src.variables_.lock().unwrap();
        let device = self.device;
        for name in variables.named_variables.keys() {
            if !src_variables.named_variables.contains_key(name) {
                return Err(TchError::TensorNameNotFound(
                    name.to_string(),
                    "src var-store".to_string(),
                ));
            }
        }
        for (name, var) in variables.named_variables.iter_mut() {
            let src_var = src_variables.named_variables.get(name).unwrap();
            crate::no_grad(|| var.f_copy_(&src_var.to_device(device)))?;
        }
        Ok(())
    }
}

impl<'a> Path<'a> {
    /// Get the components of the path.
    pub fn components(&self) -> impl Iterator<Item = &str> {
        self.path.iter().map(String::as_str)
    }

    /// Gets a sub-path of the given path.
    pub fn sub<T: std::string::ToString>(&self, s: T) -> Path<'a> {
        let s = s.to_string();
        if s.chars().any(|x| x == SEP) {
            panic!("sub name cannot contain {SEP} {s}");
        }
        let mut path = self.path.clone();
        path.push(s);
        Path { path, group: self.group, var_store: self.var_store }
    }

    pub fn set_group(&self, group: usize) -> Path<'a> {
        Path { path: self.path.clone(), group, var_store: self.var_store }
    }

    /// Gets the device where the var-store variables are stored.
    pub fn device(&self) -> Device {
        self.var_store.device
    }

    /// Gets the default kind of new variables
    pub fn kind(&self) -> Kind {
        self.var_store.kind
    }

    pub fn path(&self, name: &str) -> String {
        if name.chars().any(|x| x == SEP) {
            panic!("variable name cannot contain {SEP} {name}");
        }
        if self.path.is_empty() {
            name.to_string()
        } else {
            format!("{}{}{}", self.path.join(&SEP.to_string()), SEP, name)
        }
    }

    /// Casts all variables in a var store sub-path to the target kind .
    ///
    /// Only the variable in the path sub-tree are cast to the target kind:
    /// other var store variables are unaffected. For floating-point conversion, methods
    /// `half`, `bfloat16`, `float` and `double` should be preferred as they ensure only
    /// float-like variables will be converted to the target type.
    pub fn set_kind(&mut self, kind: Kind) {
        let path_root = self.path.join(SEP.to_string().as_str());
        let mut variables = self.var_store.variables_.lock().unwrap();
        for (variable_name, variable) in variables.named_variables.iter_mut() {
            if variable_name.starts_with(&path_root) {
                variable.set_data(&variable.to_kind(kind));
            }
        }
    }

    /// Casts all float-like variables in a var store sub-path to the target kind .
    ///
    /// Only the float-like variable in the path sub-tree are cast to the target kind:
    /// other var store variables are unaffected
    fn set_float_kind(&mut self, kind: Kind) {
        let path_root = self.path.join(SEP.to_string().as_str());
        let mut variables = self.var_store.variables_.lock().unwrap();
        for (variable_name, variable) in variables.named_variables.iter_mut() {
            if variable_name.starts_with(&path_root) & variable.is_floating_point() {
                variable.set_data(&variable.to_kind(kind));
            }
        }
    }

    /// Casts all float-like variables in a var store sub-path to half-precision (Half kind).
    ///
    /// Only the variable in the path sub-tree are cast to half-precision:
    /// other var store variables are unaffected
    pub fn half(&mut self) {
        self.set_float_kind(Kind::Half);
    }

    /// Casts all float-like variables in a var store sub-path to bfloat16-precision (BFloat16 kind).
    ///
    /// Only the variable in the path sub-tree are cast to bfloat16-precision:
    /// other var store variables are unaffected
    pub fn bfloat16(&mut self) {
        self.set_float_kind(Kind::BFloat16);
    }

    /// Casts all float-like variables in a var store sub-path to single-precision (Float kind).
    ///
    /// Only the variable in the path sub-tree are cast to single-precision:
    /// other var store variables are unaffected
    pub fn float(&mut self) {
        self.set_float_kind(Kind::Float);
    }

    /// Casts all float-like variables in a var store sub-path to double-precision (Double kind).
    ///
    /// Only the variable in the path sub-tree are cast to double-precision:
    /// other var store variables are unaffected
    pub fn double(&mut self) {
        self.set_float_kind(Kind::Double);
    }

    pub fn add(&self, name: &str, tensor: Tensor, trainable: bool) -> Tensor {
        let path = self.path(name);
        let mut variables = self.var_store.variables_.lock().unwrap();
        let path = if variables.named_variables.contains_key(&path) {
            format!("{}__{}", path, variables.named_variables.len())
        } else {
            path
        };
        let tensor = if trainable { tensor.set_requires_grad(true) } else { tensor };
        if trainable {
            let var = Var { tensor: tensor.shallow_clone(), group: self.group };
            variables.trainable_variables.push(var);
        };
        variables.named_variables.insert(path, tensor.shallow_clone());
        tensor
    }

    fn get_or_add_with_lock(
        &self,
        name: &str,
        tensor: Tensor,
        trainable: bool,
        mut variables: MutexGuard<Variables>,
    ) -> Tensor {
        let path = self.path(name);
        if let Some(var) = variables.named_variables.get(&path) {
            return var.shallow_clone();
        }

        let tensor = if trainable { tensor.set_requires_grad(true) } else { tensor };
        if trainable {
            let var = Var { tensor: tensor.shallow_clone(), group: self.group };
            variables.trainable_variables.push(var);
        }
        variables.named_variables.insert(path, tensor.shallow_clone());
        tensor
    }

    /// Creates a new variable initialized with zeros.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable will not be trainable so
    /// gradients will not be tracked.
    /// The variable uses a float tensor initialized with zeros.
    pub fn f_zeros_no_train(&self, name: &str, dims: &[i64]) -> Result<Tensor, TchError> {
        let z = Tensor::f_zeros(dims, (Kind::Float, self.device()))?;
        Ok(self.add(name, z, false))
    }

    /// Creates a new variable initialized with ones.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable will not be trainable so
    /// gradients will not be tracked.
    /// The variable uses a float tensor initialized with ones.
    pub fn f_ones_no_train(&self, name: &str, dims: &[i64]) -> Result<Tensor, TchError> {
        let o = Tensor::f_ones(dims, (Kind::Float, self.device()))?;
        Ok(self.add(name, o, false))
    }

    /// Creates a new variable.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized as per the
    /// related argument.
    pub fn f_var(&self, name: &str, dims: &[i64], init: Init) -> Result<Tensor, TchError> {
        let v = super::f_init(init, dims, self.device(), self.kind())?;
        Ok(self.add(name, v, true))
    }

    /// Creates a new variable initialized with zeros.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized with zeros.
    pub fn f_zeros(&self, name: &str, dims: &[i64]) -> Result<Tensor, TchError> {
        self.f_var(name, dims, Init::Const(0.))
    }

    /// Creates a new variable initialized with ones.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized with ones.
    pub fn f_ones(&self, name: &str, dims: &[i64]) -> Result<Tensor, TchError> {
        self.f_var(name, dims, Init::Const(1.))
    }

    /// Creates a new variable initialized randomly with normal distribution.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized randomly using a
    /// standard normal distribution.
    pub fn f_randn_standard(&self, name: &str, dims: &[i64]) -> Result<Tensor, TchError> {
        let init = Init::Randn { mean: 0., stdev: 1. };
        self.f_var(name, dims, init)
    }

    /// Creates a new variable initialized randomly with normal distribution.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized randomly using a
    /// normal distribution with the specified mean and standard deviation.
    pub fn f_randn(
        &self,
        name: &str,
        dims: &[i64],
        mean: f64,
        stdev: f64,
    ) -> Result<Tensor, TchError> {
        self.f_var(name, dims, Init::Randn { mean, stdev })
    }

    /// Creates a new variable initialized randomly with uniform distribution.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized randomly using a
    /// uniform distribution between the specified bounds.
    pub fn f_uniform(
        &self,
        name: &str,
        dims: &[i64],
        lo: f64,
        up: f64,
    ) -> Result<Tensor, TchError> {
        self.f_var(name, dims, Init::Uniform { lo, up })
    }

    /// Creates a new variable initialized randomly with kaiming uniform.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized randomly using a
    /// uniform distribution which bounds follow Kaiming initialization.
    pub fn f_kaiming_uniform(&self, name: &str, dims: &[i64]) -> Result<Tensor, TchError> {
        self.f_var(name, dims, super::init::DEFAULT_KAIMING_UNIFORM)
    }

    /// Creates a new variable initialized randomly with kaiming normal.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized randomly using a
    /// normal distribution which stdev follow Kaiming initialization.
    pub fn f_kaiming_normal(&self, name: &str, dims: &[i64]) -> Result<Tensor, TchError> {
        self.f_var(name, dims, super::init::DEFAULT_KAIMING_NORMAL)
    }

    /// Creates a new variable initialized randomly with an orthogonal matrix
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized randomly with an orthogonal
    /// matrix as described in *Exact solutions to the nonlinear dynamics
    /// of learning in deep linear neural networks* - Saxe, A. et. al. (2013).
    /// The input tensor must have at least 2 dimensions, and for tensors
    /// with more than 2 dimensions the trailing dimensions are flattened.
    pub fn f_orthogonal(&self, name: &str, dims: &[i64], gain: f64) -> Result<Tensor, TchError> {
        self.f_var(name, dims, Init::Orthogonal { gain })
    }

    /// Creates a new variable initialized by copying an existing tensor.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized by copying some
    /// given tensor.
    pub fn f_var_copy(&self, name: &str, t: &Tensor) -> Result<Tensor, TchError> {
        let mut v = self.f_zeros(name, &t.size())?;
        crate::no_grad(|| v.f_copy_(t))?;
        Ok(v)
    }

    /// Creates a new variable initialized with zeros.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable will not be trainable so
    /// gradients will not be tracked.
    /// The variable uses a float tensor initialized with zeros.
    pub fn zeros_no_train(&self, name: &str, dims: &[i64]) -> Tensor {
        self.f_zeros_no_train(name, dims).unwrap()
    }

    /// Creates a new variable initialized with ones.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable will not be trainable so
    /// gradients will not be tracked.
    /// The variable uses a float tensor initialized with ones.
    pub fn ones_no_train(&self, name: &str, dims: &[i64]) -> Tensor {
        self.f_ones_no_train(name, dims).unwrap()
    }

    /// Creates a new variable.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized as per the
    /// related argument.
    pub fn var(&self, name: &str, dims: &[i64], init: Init) -> Tensor {
        self.f_var(name, dims, init).unwrap()
    }

    /// Creates a new variable initialized with zeros.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized with zeros.
    pub fn zeros(&self, name: &str, dims: &[i64]) -> Tensor {
        self.f_zeros(name, dims).unwrap()
    }

    /// Creates a new variable initialized with ones.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized with ones.
    pub fn ones(&self, name: &str, dims: &[i64]) -> Tensor {
        self.f_ones(name, dims).unwrap()
    }

    /// Creates a new variable initialized randomly with normal distribution.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized randomly using a
    /// standard normal distribution.
    pub fn randn_standard(&self, name: &str, dims: &[i64]) -> Tensor {
        self.f_randn_standard(name, dims).unwrap()
    }

    /// Creates a new variable initialized randomly with normal distribution.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized randomly using a
    /// normal distribution with the specified mean and standard deviation.
    pub fn randn(&self, name: &str, dims: &[i64], mean: f64, stdev: f64) -> Tensor {
        self.f_randn(name, dims, mean, stdev).unwrap()
    }

    /// Creates a new variable initialized randomly with uniform distribution.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized randomly using a
    /// uniform distribution between the specified bounds.
    pub fn uniform(&self, name: &str, dims: &[i64], lo: f64, up: f64) -> Tensor {
        self.f_uniform(name, dims, lo, up).unwrap()
    }

    /// Creates a new variable initialized randomly with kaiming uniform.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized randomly using a
    /// uniform distribution which bounds follow Kaiming initialization.
    pub fn kaiming_uniform(&self, name: &str, dims: &[i64]) -> Tensor {
        self.f_kaiming_uniform(name, dims).unwrap()
    }

    /// Creates a new variable initialized randomly with kaiming normal.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized randomly using a
    /// normal distribution which stdev follow Kaiming initialization.
    pub fn kaiming_normal(&self, name: &str, dims: &[i64]) -> Tensor {
        self.f_kaiming_normal(name, dims).unwrap()
    }

    /// Creates a new variable initialized randomly with an orthogonal matrix
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized randomly with an orthogonal
    /// matrix as described in *Exact solutions to the nonlinear dynamics
    /// of learning in deep linear neural networks* - Saxe, A. et. al. (2013).
    /// The input tensor must have at least 2 dimensions, and for tensors
    /// with more than 2 dimensions the trailing dimensions are flattened.
    pub fn orthogonal(&self, name: &str, dims: &[i64], gain: f64) -> Tensor {
        self.f_orthogonal(name, dims, gain).unwrap()
    }

    /// Creates a new variable initialized by copying an existing tensor.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized by copying some
    /// given tensor.
    pub fn var_copy(&self, name: &str, t: &Tensor) -> Tensor {
        self.f_var_copy(name, t).unwrap()
    }

    /// Gets the tensor corresponding to a given name if present.
    pub fn get(&self, name: &str) -> Option<Tensor> {
        let path = self.path(name);
        let variables = self.var_store.variables_.lock().unwrap();
        variables.named_variables.get(&path).map(|v| v.shallow_clone())
    }

    /// Gets the entry corresponding to a given name for in-place manipulation.
    pub fn entry<'b>(&'b self, name: &'b str) -> Entry<'b> {
        let variables = self.var_store.variables_.lock().unwrap();
        Entry { name, variables, path: self }
    }
}

impl Entry<'_> {
    /// Returns the existing entry if, otherwise create a new variable.
    ///
    /// If this entry name matches the name of a variables stored in the
    /// var store, the corresponding tensor is returned. Otherwise a new
    /// variable is added to the var-store with the entry name and is
    /// initialized according to the init parameter.
    pub fn or_var(self, dims: &[i64], init: Init) -> Tensor {
        let v = super::init(init, dims, self.path.device());
        self.path.get_or_add_with_lock(self.name, v, true, self.variables)
    }

    /// Returns the existing entry if, otherwise create a new variable.
    pub fn or_var_copy(self, tensor: &Tensor) -> Tensor {
        let mut v = self.or_zeros(&tensor.size());
        crate::no_grad(|| v.copy_(tensor));
        v
    }

    /// Returns the existing entry if, otherwise create a new variable.
    pub fn or_kaiming_uniform(self, dims: &[i64]) -> Tensor {
        self.or_var(dims, super::init::DEFAULT_KAIMING_NORMAL)
    }

    /// Returns the existing entry if, otherwise create a new variable.
    pub fn or_kaiming_normal(self, dims: &[i64]) -> Tensor {
        self.or_var(dims, super::init::DEFAULT_KAIMING_NORMAL)
    }

    /// Returns the existing entry if, otherwise create a new variable.
    pub fn or_orthogonal(self, dims: &[i64], gain: f64) -> Tensor {
        self.or_var(dims, Init::Orthogonal { gain })
    }

    /// Returns the existing entry if, otherwise create a new variable.
    pub fn or_ones(self, dims: &[i64]) -> Tensor {
        self.or_var(dims, Init::Const(1.))
    }

    /// Returns the existing entry if, otherwise create a new variable.
    pub fn or_ones_no_train(self, dims: &[i64]) -> Tensor {
        let o = Tensor::ones(dims, (Kind::Float, self.path.device()));
        self.path.get_or_add_with_lock(self.name, o, true, self.variables)
    }

    /// Returns the existing entry if, otherwise create a new variable.
    pub fn or_randn(self, dims: &[i64], mean: f64, stdev: f64) -> Tensor {
        self.or_var(dims, Init::Randn { mean, stdev })
    }

    /// Returns the existing entry if, otherwise create a new variable.
    pub fn or_randn_standard(self, dims: &[i64]) -> Tensor {
        let init = Init::Randn { mean: 0., stdev: 1. };
        self.or_var(dims, init)
    }

    /// Returns the existing entry if, otherwise create a new variable.
    pub fn or_uniform(self, dims: &[i64], lo: f64, up: f64) -> Tensor {
        self.or_var(dims, Init::Uniform { lo, up })
    }

    /// Returns the existing entry if, otherwise create a new variable.
    pub fn or_zeros(self, dims: &[i64]) -> Tensor {
        self.or_var(dims, Init::Const(0.))
    }

    /// Returns the existing entry if, otherwise create a new variable.
    pub fn or_zeros_no_train(self, dims: &[i64]) -> Tensor {
        let z = Tensor::zeros(dims, (Kind::Float, self.path.device()));
        self.path.get_or_add_with_lock(self.name, z, true, self.variables)
    }
}

impl<'a, T> Div<T> for &mut Path<'a>
where
    T: std::string::ToString,
{
    type Output = Path<'a>;

    fn div(self, rhs: T) -> Self::Output {
        self.sub(rhs.to_string())
    }
}

impl<'a, T> Div<T> for &Path<'a>
where
    T: std::string::ToString,
{
    type Output = Path<'a>;

    fn div(self, rhs: T) -> Self::Output {
        self.sub(rhs.to_string())
    }
}

impl<'a, T> Div<T> for Path<'a>
where
    T: std::string::ToString,
{
    type Output = Path<'a>;

    fn div(self, rhs: T) -> Self::Output {
        self.sub(rhs.to_string())
    }
}
