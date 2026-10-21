use anyhow::{bail, ensure, Result};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use tch::{
    nn,
    nn::{FuncT, ModuleT},
    Tensor,
};

#[derive(Debug)]
struct Block {
    block_type: String,
    parameters: BTreeMap<String, String>,
}

impl Block {
    fn get(&self, key: &str) -> Result<&str> {
        match self.parameters.get(key) {
            None => bail!("cannot find {} in {}", key, self.block_type),
            Some(value) => Ok(value),
        }
    }
}

#[derive(Debug)]
pub struct Darknet {
    blocks: Vec<Block>,
    parameters: BTreeMap<String, String>,
}

impl Darknet {
    fn get(&self, key: &str) -> Result<&str> {
        match self.parameters.get(key) {
            None => bail!("cannot find {} in net parameters", key),
            Some(value) => Ok(value),
        }
    }
}

struct Accumulator {
    block_type: Option<String>,
    parameters: BTreeMap<String, String>,
    net: Darknet,
}

impl Accumulator {
    fn new() -> Accumulator {
        Accumulator {
            block_type: None,
            parameters: BTreeMap::new(),
            net: Darknet { blocks: vec![], parameters: BTreeMap::new() },
        }
    }

    fn finish_block(&mut self) {
        match &self.block_type {
            None => (),
            Some(block_type) => {
                if block_type == "net" {
                    self.net.parameters = self.parameters.clone();
                } else {
                    let block = Block {
                        block_type: block_type.to_string(),
                        parameters: self.parameters.clone(),
                    };
                    self.net.blocks.push(block);
                }
                self.parameters.clear();
            }
        }
        self.block_type = None;
    }
}

pub fn parse_config<T: AsRef<Path>>(path: T) -> Result<Darknet> {
    let file = File::open(path.as_ref())?;
    let mut acc = Accumulator::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.trim();
        if line.starts_with('[') {
            ensure!(line.ends_with(']'), "line does not end with ']' {}", line);
            let line = &line[1..line.len() - 1];
            acc.finish_block();
            acc.block_type = Some(line.to_string());
        } else {
            let key_value: Vec<&str> = line.splitn(2, '=').collect();
            ensure!(key_value.len() == 2, "missing equal {}", line);
            let prev = acc
                .parameters
                .insert(key_value[0].trim().to_owned(), key_value[1].trim().to_owned());
            ensure!(prev.is_none(), "multiple value for key {}", line);
        }
    }
    acc.finish_block();
    Ok(acc.net)
}

enum Bl {
    Layer(Box<dyn ModuleT>),
    Route(Vec<usize>),
    Shortcut(usize),
    Yolo(i64, Vec<(i64, i64)>),
}

fn conv(vs: nn::Path, index: usize, p: i64, b: &Block) -> Result<(i64, Bl)> {
    let activation = b.get("activation")?;
    let filters = b.get("filters")?.parse::<i64>()?;
    let pad = b.get("pad")?.parse::<i64>()?;
    let size = b.get("size")?.parse::<i64>()?;
    let stride = b.get("stride")?.parse::<i64>()?;
    let pad = if pad != 0 { (size - 1) / 2 } else { 0 };
    let (bn, bias) = match b.parameters.get("batch_normalize") {
        Some(p) if p.parse::<i64>()? != 0 => {
            let vs = &vs / format!("batch_norm_{index}");
            let bn = nn::batch_norm2d(vs, filters, Default::default());
            (Some(bn), false)
        }
        Some(_) | None => (None, true),
    };
    let conv_cfg = nn::ConvConfig { stride, padding: pad, bias, ..Default::default() };
    let vs = &vs / format!("conv_{index}");
    let conv = nn::conv2d(vs, p, filters, size, conv_cfg);
    let leaky = match activation {
        "leaky" => true,
        "linear" => false,
        otherwise => bail!("unsupported activation {}", otherwise),
    };
    let func = nn::func_t(move |xs, train| {
        let xs = xs.apply(&conv);
        let xs = match &bn {
            Some(bn) => xs.apply_t(bn, train),
            None => xs,
        };
        if leaky {
            xs.maximum(&(&xs * 0.1))
        } else {
            xs
        }
    });
    Ok((filters, Bl::Layer(Box::new(func))))
}

fn upsample(prev_channels: i64) -> Result<(i64, Bl)> {
    let layer = nn::func_t(|xs, _is_training| {
        let (_n, _c, h, w) = xs.size4().unwrap();
        xs.upsample_nearest2d([2 * h, 2 * w], 2.0, 2.0)
    });
    Ok((prev_channels, Bl::Layer(Box::new(layer))))
}

fn int_list_of_string(s: &str) -> Result<Vec<i64>> {
    let res: Result<Vec<_>, _> = s.split(',').map(|xs| xs.trim().parse::<i64>()).collect();
    Ok(res?)
}

fn usize_of_index(index: usize, i: i64) -> usize {
    if i >= 0 {
        i as usize
    } else {
        (index as i64 + i) as usize
    }
}

fn route(index: usize, p: &[(i64, Bl)], block: &Block) -> Result<(i64, Bl)> {
    let layers = int_list_of_string(block.get("layers")?)?;
    let layers: Vec<usize> = layers.into_iter().map(|l| usize_of_index(index, l)).collect();
    let channels = layers.iter().map(|&l| p[l].0).sum();
    Ok((channels, Bl::Route(layers)))
}

fn shortcut(index: usize, p: i64, block: &Block) -> Result<(i64, Bl)> {
    let from = block.get("from")?.parse::<i64>()?;
    Ok((p, Bl::Shortcut(usize_of_index(index, from))))
}

fn yolo(p: i64, block: &Block) -> Result<(i64, Bl)> {
    let classes = block.get("classes")?.parse::<i64>()?;
    let flat = int_list_of_string(block.get("anchors")?)?;
    ensure!(flat.len() % 2 == 0, "even number of anchors");
    let anchors: Vec<_> = (0..(flat.len() / 2)).map(|i| (flat[2 * i], flat[2 * i + 1])).collect();
    let mask = int_list_of_string(block.get("mask")?)?;
    let anchors = mask.into_iter().map(|i| anchors[i as usize]).collect();
    Ok((p, Bl::Yolo(classes, anchors)))
}

// Apply f to a slice of tensor xs and replace xs values with f output.
fn slice_apply_and_set<F>(xs: &mut Tensor, start: i64, len: i64, f: F)
where
    F: FnOnce(&Tensor) -> Tensor,
{
    let mut slice = xs.narrow(2, start, len);
    let src = f(&slice);
    slice.copy_(&src)
}

fn detect(xs: &Tensor, image_height: i64, classes: i64, anchors: &[(i64, i64)]) -> Tensor {
    let (bsize, _channels, height, _width) = xs.size4().unwrap();
    let stride = image_height / height;
    let grid_size = image_height / stride;
    let bbox_attrs = 5 + classes;
    let nanchors = anchors.len() as i64;
    let mut xs = xs
        .view((bsize, bbox_attrs * nanchors, grid_size * grid_size))
        .transpose(1, 2)
        .contiguous()
        .view((bsize, grid_size * grid_size * nanchors, bbox_attrs));
    let grid = Tensor::arange(grid_size, tch::kind::FLOAT_CPU);
    let a = grid.repeat([grid_size, 1]);
    let b = a.tr().contiguous();
    let x_offset = a.view((-1, 1));
    let y_offset = b.view((-1, 1));
    let xy_offset =
        Tensor::cat(&[x_offset, y_offset], 1).repeat([1, nanchors]).view((-1, 2)).unsqueeze(0);
    let anchors: Vec<f32> = anchors
        .iter()
        .flat_map(|&(x, y)| vec![x as f32 / stride as f32, y as f32 / stride as f32].into_iter())
        .collect();
    let anchors =
        Tensor::from_slice(&anchors).view((-1, 2)).repeat([grid_size * grid_size, 1]).unsqueeze(0);
    slice_apply_and_set(&mut xs, 0, 2, |xs| xs.sigmoid() + xy_offset);
    slice_apply_and_set(&mut xs, 4, 1 + classes, Tensor::sigmoid);
    slice_apply_and_set(&mut xs, 2, 2, |xs| xs.exp() * anchors);
    slice_apply_and_set(&mut xs, 0, 4, |xs| xs * stride);
    xs
}

impl Darknet {
    pub fn height(&self) -> Result<i64> {
        let image_height = self.get("height")?.parse::<i64>()?;
        Ok(image_height)
    }

    pub fn width(&self) -> Result<i64> {
        let image_width = self.get("width")?.parse::<i64>()?;
        Ok(image_width)
    }

    pub fn build_model(&self, vs: &nn::Path) -> Result<FuncT<'_>> {
        let mut blocks: Vec<(i64, Bl)> = vec![];
        let mut prev_channels: i64 = 3;
        for (index, block) in self.blocks.iter().enumerate() {
            let channels_and_bl = match block.block_type.as_str() {
                "convolutional" => conv(vs / index, index, prev_channels, block)?,
                "upsample" => upsample(prev_channels)?,
                "shortcut" => shortcut(index, prev_channels, block)?,
                "route" => route(index, &blocks, block)?,
                "yolo" => yolo(prev_channels, block)?,
                otherwise => bail!("unsupported block type {}", otherwise),
            };
            prev_channels = channels_and_bl.0;
            blocks.push(channels_and_bl);
        }
        let image_height = self.height()?;
        let func = nn::func_t(move |xs, train| {
            let mut prev_ys: Vec<Tensor> = vec![];
            let mut detections: Vec<Tensor> = vec![];
            for (_, b) in blocks.iter() {
                let ys = match b {
                    Bl::Layer(l) => {
                        let xs = prev_ys.last().unwrap_or(xs);
                        l.forward_t(xs, train)
                    }
                    Bl::Route(layers) => {
                        let layers: Vec<_> = layers.iter().map(|&i| &prev_ys[i]).collect();
                        Tensor::cat(&layers, 1)
                    }
                    Bl::Shortcut(from) => prev_ys.last().unwrap() + prev_ys.get(*from).unwrap(),
                    Bl::Yolo(classes, anchors) => {
                        let xs = prev_ys.last().unwrap_or(xs);
                        detections.push(detect(xs, image_height, *classes, anchors));
                        Tensor::default()
                    }
                };
                prev_ys.push(ys);
            }
            Tensor::cat(&detections, 1)
        });
        Ok(func)
    }
}
