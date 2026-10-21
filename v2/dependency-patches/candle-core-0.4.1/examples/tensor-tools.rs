use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{Device, Result};
use clap::{Parser, Subcommand, ValueEnum};
use rayon::prelude::*;

#[derive(ValueEnum, Debug, Clone)]
enum QuantizationMode {
    /// The default quantization includes all 2d tensors, except the output tensor which always
    /// uses Q6_K.
    Llama,
}

impl QuantizationMode {
    fn quantize(&self, name: &str, tensor: QTensor, dtype: GgmlDType) -> Result<QTensor> {
        match self {
            Self::Llama => {
                // Same behavior as the llama.cpp quantization.
                let should_quantize = name.ends_with(".weight") && tensor.rank() == 2;
                if should_quantize {
                    let tensor = tensor.dequantize(&Device::Cpu)?;
                    if name == "output.weight" {
                        QTensor::quantize(&tensor, GgmlDType::Q6K)
                    } else {
                        QTensor::quantize(&tensor, dtype)
                    }
                } else {
                    Ok(tensor)
                }
            }
        }
    }
}

#[derive(ValueEnum, Debug, Clone)]
enum Quantization {
    #[value(name = "q4_0")]
    Q4_0,
    #[value(name = "q4_1")]
    Q4_1,
    #[value(name = "q5_0")]
    Q5_0,
    #[value(name = "q5_1")]
    Q5_1,
    #[value(name = "q8_0")]
    Q8_0,
    #[value(name = "q8_1")]
    Q8_1,
    Q2k,
    Q3k,
    Q4k,
    Q5k,
    Q6k,
    Q8k,
    F16,
    F32,
}

impl Quantization {
    fn dtype(&self) -> GgmlDType {
        match self {
            Quantization::Q4_0 => GgmlDType::Q4_0,
            Quantization::Q4_1 => GgmlDType::Q4_1,
            Quantization::Q5_0 => GgmlDType::Q5_0,
            Quantization::Q5_1 => GgmlDType::Q5_1,
            Quantization::Q8_0 => GgmlDType::Q8_0,
            Quantization::Q8_1 => GgmlDType::Q8_1,
            Quantization::Q2k => GgmlDType::Q2K,
            Quantization::Q3k => GgmlDType::Q3K,
            Quantization::Q4k => GgmlDType::Q4K,
            Quantization::Q5k => GgmlDType::Q5K,
            Quantization::Q6k => GgmlDType::Q6K,
            Quantization::Q8k => GgmlDType::Q8K,
            Quantization::F16 => GgmlDType::F16,
            Quantization::F32 => GgmlDType::F32,
        }
    }
}

#[derive(ValueEnum, Debug, Clone)]
enum Format {
    Safetensors,
    Npz,
    Ggml,
    Gguf,
    Pth,
    Pickle,
}

impl Format {
    fn infer<P: AsRef<std::path::Path>>(p: P) -> Option<Self> {
        p.as_ref()
            .extension()
            .and_then(|e| e.to_str())
            .and_then(|e| match e {
                // We don't infer any format for .bin as it can be used for ggml/gguf or pytorch.
                "safetensors" | "safetensor" => Some(Self::Safetensors),
                "npz" => Some(Self::Npz),
                "pth" | "pt" => Some(Self::Pth),
                "ggml" => Some(Self::Ggml),
                "gguf" => Some(Self::Gguf),
                _ => None,
            })
    }
}

#[derive(Subcommand, Debug, Clone)]
enum Command {
    Ls {
        files: Vec<std::path::PathBuf>,

        /// The file format to use, if unspecified infer from the file extension.
        #[arg(long, value_enum)]
        format: Option<Format>,

        /// Enable verbose mode.
        #[arg(short, long)]
        verbose: bool,
    },

    Quantize {
        /// The input file(s), in safetensors format.
        in_file: Vec<std::path::PathBuf>,

        /// The output file, in gguf format.
        #[arg(long)]
        out_file: std::path::PathBuf,

        /// The quantization schema to apply.
        #[arg(long, value_enum)]
        quantization: Quantization,

        /// Which tensor to quantize.
        #[arg(long, value_enum, default_value_t = QuantizationMode::Llama)]
        mode: QuantizationMode,
    },

    Dequantize {
        /// The input file, in gguf format.
        in_file: std::path::PathBuf,

        /// The output file, in safetensors format.
        #[arg(long)]
        out_file: std::path::PathBuf,
    },
}

#[derive(Parser, Debug, Clone)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

fn run_ls(
    file: &std::path::PathBuf,
    format: Option<Format>,
    verbose: bool,
    device: &Device,
) -> Result<()> {
    let format = match format {
        Some(format) => format,
        None => match Format::infer(file) {
            Some(format) => format,
            None => {
                println!(
                    "{file:?}: cannot infer format from file extension, use the --format flag"
                );
                return Ok(());
            }
        },
    };
    match format {
        Format::Npz => {
            let tensors = candle_core::npy::NpzTensors::new(file)?;
            let mut names = tensors.names();
            names.sort();
            for name in names {
                let shape_dtype = match tensors.get_shape_and_dtype(name) {
                    Ok((shape, dtype)) => format!("[{shape:?}; {dtype:?}]"),
                    Err(err) => err.to_string(),
                };
                println!("{name}: {shape_dtype}")
            }
        }
        Format::Safetensors => {
            let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(file)? };
            let mut tensors = tensors.tensors();
            tensors.sort_by(|a, b| a.0.cmp(&b.0));
            for (name, view) in tensors.iter() {
                let dtype = view.dtype();
                let dtype = match candle_core::DType::try_from(dtype) {
                    Ok(dtype) => format!("{dtype:?}"),
                    Err(_) => format!("{dtype:?}"),
                };
                let shape = view.shape();
                println!("{name}: [{shape:?}; {dtype}]")
            }
        }
        Format::Pth => {
            let mut tensors = candle_core::pickle::read_pth_tensor_info(file, verbose, None)?;
            tensors.sort_by(|a, b| a.name.cmp(&b.name));
            for tensor_info in tensors.iter() {
                println!(
                    "{}: [{:?}; {:?}]",
                    tensor_info.name,
                    tensor_info.layout.shape(),
                    tensor_info.dtype,
                );
                if verbose {
                    println!("    {:?}", tensor_info);
                }
            }
        }
        Format::Pickle => {
            let file = std::fs::File::open(file)?;
            let mut reader = std::io::BufReader::new(file);
            let mut stack = candle_core::pickle::Stack::empty();
            stack.read_loop(&mut reader)?;
            for (i, obj) in stack.stack().iter().enumerate() {
                println!("{i} {obj:?}");
            }
        }
        Format::Ggml => {
            let mut file = std::fs::File::open(file)?;
            let content = candle_core::quantized::ggml_file::Content::read(&mut file, device)?;
            let mut tensors = content.tensors.into_iter().collect::<Vec<_>>();
            tensors.sort_by(|a, b| a.0.cmp(&b.0));
            for (name, qtensor) in tensors.iter() {
                println!("{name}: [{:?}; {:?}]", qtensor.shape(), qtensor.dtype());
            }
        }
        Format::Gguf => {
            let mut file = std::fs::File::open(file)?;
            let content = gguf_file::Content::read(&mut file)?;
            if verbose {
                let mut metadata = content.metadata.into_iter().collect::<Vec<_>>();
                metadata.sort_by(|a, b| a.0.cmp(&b.0));
                println!("metadata entries ({})", metadata.len());
                for (key, value) in metadata.iter() {
                    println!("  {key}: {value:?}");
                }
            }
            let mut tensors = content.tensor_infos.into_iter().collect::<Vec<_>>();
            tensors.sort_by(|a, b| a.0.cmp(&b.0));
            for (name, info) in tensors.iter() {
                println!("{name}: [{:?}; {:?}]", info.shape, info.ggml_dtype);
            }
        }
    }
    Ok(())
}

fn run_quantize_safetensors(
    in_files: &[std::path::PathBuf],
    out_file: std::path::PathBuf,
    q: Quantization,
) -> Result<()> {
    let mut out_file = std::fs::File::create(out_file)?;
    let mut tensors = std::collections::HashMap::new();
    for in_file in in_files.iter() {
        let in_tensors = candle_core::safetensors::load(in_file, &Device::Cpu)?;
        tensors.extend(in_tensors)
    }
    println!("tensors: {}", tensors.len());

    let dtype = q.dtype();
    let block_size = dtype.block_size();

    let qtensors = tensors
        .into_par_iter()
        .map(|(name, tensor)| {
            let should_quantize = tensor.rank() == 2 && tensor.dim(1)? % block_size == 0;
            println!("  quantizing {name} {tensor:?} {should_quantize}");
            let tensor = if should_quantize {
                QTensor::quantize(&tensor, dtype)?
            } else {
                QTensor::quantize(&tensor, GgmlDType::F32)?
            };
            Ok((name, tensor))
        })
        .collect::<Result<Vec<_>>>()?;
    let qtensors = qtensors
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();
    gguf_file::write(&mut out_file, &[], &qtensors)?;
    Ok(())
}

fn run_dequantize(
    in_file: std::path::PathBuf,
    out_file: std::path::PathBuf,
    device: &Device,
) -> Result<()> {
    let mut in_file = std::fs::File::open(in_file)?;
    let content = gguf_file::Content::read(&mut in_file)?;
    let mut tensors = std::collections::HashMap::new();
    for (tensor_name, _) in content.tensor_infos.iter() {
        let tensor = content.tensor(&mut in_file, tensor_name, device)?;
        let tensor = tensor.dequantize(device)?;
        tensors.insert(tensor_name.to_string(), tensor);
    }
    candle_core::safetensors::save(&tensors, out_file)?;
    Ok(())
}

fn run_quantize(
    in_files: &[std::path::PathBuf],
    out_file: std::path::PathBuf,
    q: Quantization,
    qmode: QuantizationMode,
    device: &Device,
) -> Result<()> {
    if in_files.is_empty() {
        candle_core::bail!("no specified input files")
    }
    if let Some(extension) = out_file.extension() {
        if extension == "safetensors" {
            candle_core::bail!("the generated file cannot use the safetensors extension")
        }
    }
    if let Some(extension) = in_files[0].extension() {
        if extension == "safetensors" {
            return run_quantize_safetensors(in_files, out_file, q);
        }
    }

    if in_files.len() != 1 {
        candle_core::bail!("only a single in-file can be used when quantizing gguf files")
    }

    // Open the out file early so as to fail directly on missing directories etc.
    let mut out_file = std::fs::File::create(out_file)?;
    let mut in_ = std::fs::File::open(&in_files[0])?;
    let content = gguf_file::Content::read(&mut in_)?;
    println!("tensors: {}", content.tensor_infos.len());

    let dtype = q.dtype();
    let qtensors = content
        .tensor_infos
        .par_iter()
        .map(|(name, _)| {
            println!("  quantizing {name}");
            let mut in_file = std::fs::File::open(&in_files[0])?;
            let tensor = content.tensor(&mut in_file, name, device)?;
            let tensor = qmode.quantize(name, tensor, dtype)?;
            Ok((name, tensor))
        })
        .collect::<Result<Vec<_>>>()?;
    let qtensors = qtensors
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();

    let metadata = content
        .metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();
    gguf_file::write(&mut out_file, metadata.as_slice(), &qtensors)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = Device::Cpu;
    match args.command {
        Command::Ls {
            files,
            format,
            verbose,
        } => {
            let multiple_files = files.len() > 1;
            for file in files.iter() {
                if multiple_files {
                    println!("--- {file:?} ---");
                }
                run_ls(file, format.clone(), verbose, &device)?
            }
        }
        Command::Quantize {
            in_file,
            out_file,
            quantization,
            mode,
        } => run_quantize(&in_file, out_file, quantization, mode, &device)?,
        Command::Dequantize { in_file, out_file } => run_dequantize(in_file, out_file, &device)?,
    }
    Ok(())
}
