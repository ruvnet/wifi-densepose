//! Monocular depth estimation via MiDaS ONNX + backprojection to 3D points.
#![allow(dead_code)]

use crate::depth_anything::{DepthAnythingBackend, DepthAnythingConfig};
use crate::pointcloud::{ColorPoint, PointCloud};
use anyhow::Result;
use std::sync::{Mutex, OnceLock};

/// Default camera intrinsics (approximate for HD webcam)
#[derive(Clone, Debug)]
pub struct CameraIntrinsics {
    pub fx: f32, // focal length x (pixels)
    pub fy: f32, // focal length y (pixels)
    pub cx: f32, // principal point x
    pub cy: f32, // principal point y
    pub width: u32,
    pub height: u32,
}

impl Default for CameraIntrinsics {
    fn default() -> Self {
        Self {
            fx: 525.0,
            fy: 525.0, // typical webcam focal length
            cx: 320.0,
            cy: 240.0, // center of 640x480
            width: 640,
            height: 480,
        }
    }
}

impl CameraIntrinsics {
    pub fn for_dimensions(width: u32, height: u32) -> Self {
        let default = Self::default();
        Self {
            fx: default.fx * width as f32 / default.width as f32,
            fy: default.fy * height as f32 / default.height as f32,
            cx: width as f32 / 2.0,
            cy: height as f32 / 2.0,
            width,
            height,
        }
    }

    pub fn from_matrix_or_default(matrix: &[f32; 9], width: u32, height: u32) -> Self {
        let values = (matrix[0], matrix[4], matrix[2], matrix[5]);
        if [values.0, values.1, values.2, values.3]
            .iter()
            .all(|value| value.is_finite())
            && values.0 > 0.0
            && values.1 > 0.0
        {
            Self {
                fx: values.0,
                fy: values.1,
                cx: values.2,
                cy: values.3,
                width,
                height,
            }
        } else {
            Self::for_dimensions(width, height)
        }
    }
}

#[derive(Clone, Debug)]
pub struct DepthEstimate {
    pub depth: Vec<f32>,
    pub confidence: Option<Vec<f32>>,
    pub width: u32,
    pub height: u32,
    pub intrinsics: CameraIntrinsics,
    pub extrinsics: Option<[f32; 12]>,
    pub is_metric: bool,
    pub backend: &'static str,
    pub min_confidence: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BackendSelection {
    Auto,
    DepthAnything,
    Midas,
    Heuristic,
}

struct DepthEstimator {
    selection: BackendSelection,
    depth_anything: Option<DepthAnythingBackend>,
}

static DEFAULT_ESTIMATOR: OnceLock<std::result::Result<Mutex<DepthEstimator>, String>> =
    OnceLock::new();

/// Backproject a depth map to 3D points using camera intrinsics.
///
/// depth_map: row-major [height x width] in meters
/// rgb: optional row-major [height x width x 3] color
pub fn backproject_depth(
    depth_map: &[f32],
    intrinsics: &CameraIntrinsics,
    rgb: Option<&[u8]>,
    downsample: u32,
) -> PointCloud {
    backproject_depth_with_confidence(depth_map, intrinsics, rgb, None, 0.0, downsample)
}

fn backproject_depth_with_confidence(
    depth_map: &[f32],
    intrinsics: &CameraIntrinsics,
    rgb: Option<&[u8]>,
    confidence: Option<&[f32]>,
    min_confidence: f32,
    downsample: u32,
) -> PointCloud {
    let mut cloud = PointCloud::new("camera_depth");
    let w = intrinsics.width;
    let h = intrinsics.height;
    let step = downsample.max(1);

    for y in (0..h).step_by(step as usize) {
        for x in (0..w).step_by(step as usize) {
            let idx = (y * w + x) as usize;
            if idx >= depth_map.len() {
                continue;
            }
            let z = depth_map[idx];

            let point_confidence = confidence
                .and_then(|values| values.get(idx).copied())
                .unwrap_or(1.0);

            // Skip invalid depths
            if z <= 0.01
                || z > 10.0
                || !z.is_finite()
                || !point_confidence.is_finite()
                || point_confidence < min_confidence
            {
                continue;
            }

            // Backproject: (u, v, z) → (X, Y, Z)
            let px = (x as f32 - intrinsics.cx) * z / intrinsics.fx;
            let py = (y as f32 - intrinsics.cy) * z / intrinsics.fy;

            let (r, g, b) = if let Some(rgb_data) = rgb {
                let ri = idx * 3;
                if ri + 2 < rgb_data.len() {
                    (rgb_data[ri], rgb_data[ri + 1], rgb_data[ri + 2])
                } else {
                    (128, 128, 128)
                }
            } else {
                // Color by depth (blue=near, red=far)
                let t = ((z - 0.5) / 4.0).clamp(0.0, 1.0);
                (
                    (t * 255.0) as u8,
                    ((1.0 - t) * 128.0) as u8,
                    ((1.0 - t) * 255.0) as u8,
                )
            };

            cloud.points.push(ColorPoint {
                x: px,
                y: py,
                z,
                r,
                g,
                b,
                intensity: point_confidence,
            });
        }
    }
    cloud
}

/// Backproject a typed estimate, preserving model dimensions, calibrated camera
/// intrinsics, confidence, and RGB alignment.
pub fn backproject_estimate(
    estimate: &DepthEstimate,
    source_rgb: Option<(&[u8], u32, u32)>,
    downsample: u32,
) -> PointCloud {
    let resized_rgb = source_rgb.and_then(|(rgb, width, height)| {
        resize_rgb_nearest(rgb, width, height, estimate.width, estimate.height).ok()
    });
    backproject_depth_with_confidence(
        &estimate.depth,
        &estimate.intrinsics,
        resized_rgb.as_deref(),
        estimate.confidence.as_deref(),
        estimate.min_confidence,
        downsample,
    )
}

impl DepthEstimator {
    fn from_env() -> Result<Self> {
        let selection = match std::env::var("RUVIEW_DEPTH_BACKEND")
            .unwrap_or_else(|_| "auto".into())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "auto" => BackendSelection::Auto,
            "depth-anything" | "depth_anything" | "da" => BackendSelection::DepthAnything,
            "midas" => BackendSelection::Midas,
            "heuristic" | "demo" => BackendSelection::Heuristic,
            value => anyhow::bail!(
                "unknown RUVIEW_DEPTH_BACKEND '{value}'; expected auto, depth-anything, midas, or heuristic"
            ),
        };

        let configured = std::env::var_os("RUVIEW_DA_LIBRARY").is_some()
            || std::env::var_os("RUVIEW_DA_MODEL").is_some();
        let depth_anything = if selection == BackendSelection::DepthAnything
            || (selection == BackendSelection::Auto && configured)
        {
            match DepthAnythingConfig::from_env()
                .and_then(|config| DepthAnythingBackend::load(&config))
            {
                Ok(backend) => Some(backend),
                Err(error) if selection == BackendSelection::Auto => {
                    eprintln!(
                        "Depth Anything auto-configuration unavailable ({error:#}); falling back to MiDaS/heuristic"
                    );
                    None
                }
                Err(error) => return Err(error),
            }
        } else {
            None
        };

        Ok(Self {
            selection,
            depth_anything,
        })
    }

    fn estimate(&mut self, rgb: &[u8], width: u32, height: u32) -> Result<DepthEstimate> {
        if matches!(
            self.selection,
            BackendSelection::DepthAnything | BackendSelection::Auto
        ) {
            if let Some(backend) = self.depth_anything.as_mut() {
                match backend.infer_rgb(rgb, width, height) {
                    Ok(estimate) => return Ok(estimate),
                    Err(error) if self.selection == BackendSelection::Auto => {
                        eprintln!(
                            "Depth Anything inference unavailable ({error:#}); falling back to MiDaS/heuristic"
                        );
                    }
                    Err(error) => return Err(error),
                }
            } else if self.selection == BackendSelection::DepthAnything {
                anyhow::bail!("Depth Anything backend was selected but is not initialized");
            }
        }

        if matches!(
            self.selection,
            BackendSelection::Midas | BackendSelection::Auto
        ) {
            match estimate_depth_midas_server(rgb, width, height) {
                Ok(depth) => {
                    return Ok(DepthEstimate {
                        depth,
                        confidence: None,
                        width,
                        height,
                        intrinsics: CameraIntrinsics::for_dimensions(width, height),
                        extrinsics: None,
                        is_metric: false,
                        backend: "midas",
                        min_confidence: 0.0,
                    })
                }
                Err(error) if self.selection == BackendSelection::Auto => {
                    eprintln!("MiDaS service unavailable ({error:#}); using heuristic depth");
                }
                Err(error) => return Err(error),
            }
        }

        let depth = estimate_depth_heuristic(rgb, width, height)?;
        Ok(DepthEstimate {
            depth,
            confidence: None,
            width,
            height,
            intrinsics: CameraIntrinsics::for_dimensions(width, height),
            extrinsics: None,
            is_metric: false,
            backend: "heuristic",
            min_confidence: 0.0,
        })
    }
}

/// Run the configured depth backend. The estimator is initialized once so a
/// native model context is reused across frames.
pub fn estimate_depth_frame(image_data: &[u8], width: u32, height: u32) -> Result<DepthEstimate> {
    let estimator = DEFAULT_ESTIMATOR.get_or_init(|| {
        DepthEstimator::from_env()
            .map(Mutex::new)
            .map_err(|error| format!("{error:#}"))
    });
    let estimator = estimator
        .as_ref()
        .map_err(|error| anyhow::anyhow!(error.clone()))?;
    estimator
        .lock()
        .map_err(|_| anyhow::anyhow!("depth estimator lock poisoned"))?
        .estimate(image_data, width, height)
}

/// Compatibility helper for training code that only consumes the dense map.
pub fn estimate_depth(image_data: &[u8], width: u32, height: u32) -> Result<Vec<f32>> {
    Ok(estimate_depth_frame(image_data, width, height)?.depth)
}

fn estimate_depth_heuristic(image_data: &[u8], width: u32, height: u32) -> Result<Vec<f32>> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| anyhow::anyhow!("RGB dimensions overflow"))?;
    if width == 0 || height == 0 || image_data.len() < expected {
        anyhow::bail!("RGB buffer is smaller than {width}x{height} RGB8");
    }
    let w = width as usize;
    let h = height as usize;
    let mut lum = vec![0.0f32; w * h];
    for (i, lum_i) in lum.iter_mut().enumerate() {
        let ri = i * 3;
        if ri + 2 < image_data.len() {
            *lum_i = (0.299 * image_data[ri] as f32
                + 0.587 * image_data[ri + 1] as f32
                + 0.114 * image_data[ri + 2] as f32)
                / 255.0;
        }
    }
    let mut edges = vec![0.0f32; w * h];
    if w >= 3 && h >= 3 {
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let gx = -lum[(y - 1) * w + x - 1] + lum[(y - 1) * w + x + 1]
                    - 2.0 * lum[y * w + x - 1]
                    + 2.0 * lum[y * w + x + 1]
                    - lum[(y + 1) * w + x - 1]
                    + lum[(y + 1) * w + x + 1];
                let gy = -lum[(y - 1) * w + x - 1]
                    - 2.0 * lum[(y - 1) * w + x]
                    - lum[(y - 1) * w + x + 1]
                    + lum[(y + 1) * w + x - 1]
                    + 2.0 * lum[(y + 1) * w + x]
                    + lum[(y + 1) * w + x + 1];
                edges[y * w + x] = (gx * gx + gy * gy).sqrt().min(1.0);
            }
        }
    }
    let mut depth_map = vec![3.0f32; w * h];
    for i in 0..w * h {
        let base = 1.0 + (1.0 - lum[i]) * 3.5;
        let edge_boost = edges[i] * 1.5;
        depth_map[i] = (base - edge_boost).max(0.3);
    }
    Ok(depth_map)
}

fn resize_rgb_nearest(
    rgb: &[u8],
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
) -> Result<Vec<u8>> {
    let expected = (source_width as usize)
        .checked_mul(source_height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| anyhow::anyhow!("RGB dimensions overflow"))?;
    if source_width == 0
        || source_height == 0
        || target_width == 0
        || target_height == 0
        || rgb.len() < expected
    {
        anyhow::bail!("invalid RGB resize input");
    }
    let target_len = (target_width as usize)
        .checked_mul(target_height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| anyhow::anyhow!("target RGB dimensions overflow"))?;
    let mut output = vec![0u8; target_len];
    for y in 0..target_height {
        let source_y = ((y as u64 * source_height as u64) / target_height as u64) as u32;
        for x in 0..target_width {
            let source_x = ((x as u64 * source_width as u64) / target_width as u64) as u32;
            let source = ((source_y * source_width + source_x) * 3) as usize;
            let target = ((y * target_width + x) * 3) as usize;
            output[target..target + 3].copy_from_slice(&rgb[source..source + 3]);
        }
    }
    Ok(output)
}

/// Call MiDaS depth server running on GPU (127.0.0.1:9885).
fn estimate_depth_midas_server(rgb: &[u8], width: u32, height: u32) -> Result<Vec<f32>> {
    let expected = (width * height * 3) as usize;
    if rgb.len() < expected {
        anyhow::bail!("rgb too small");
    }

    // Send RGB as JSON array to depth server
    let rgb_list: Vec<u8> = rgb[..expected].to_vec();
    let body = serde_json::json!({
        "width": width,
        "height": height,
        "rgb": rgb_list,
    });
    let body_bytes = serde_json::to_vec(&body)?;

    let client = std::net::TcpStream::connect_timeout(
        &"127.0.0.1:9885".parse()?,
        std::time::Duration::from_millis(500),
    )?;
    client.set_read_timeout(Some(std::time::Duration::from_secs(5)))?;
    client.set_write_timeout(Some(std::time::Duration::from_secs(2)))?;

    use std::io::{Read, Write};
    let mut stream = client;
    let req = format!(
        "POST /depth HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body_bytes.len()
    );
    stream.write_all(req.as_bytes())?;
    stream.write_all(&body_bytes)?;

    // Read response
    let mut resp = Vec::new();
    stream.read_to_end(&mut resp)?;

    // Skip HTTP headers
    let body_start = resp
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|p| p + 4)
        .unwrap_or(0);
    let depth_bytes = &resp[body_start..];

    let n = (width * height) as usize;
    if depth_bytes.len() < n * 4 {
        anyhow::bail!("depth response too small");
    }

    let depth: Vec<f32> = depth_bytes[..n * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    Ok(depth)
}

/// Capture depth cloud from camera (placeholder — real impl uses nokhwa or v4l2).
pub async fn capture_depth_cloud(_frames: usize) -> Result<PointCloud> {
    eprintln!("Camera capture not available (no camera on this machine).");
    eprintln!("Use --demo for synthetic data, or run on a machine with a camera.");
    Ok(demo_depth_cloud())
}

/// Generate a demo depth point cloud (synthetic room scene).
pub fn demo_depth_cloud() -> PointCloud {
    let _cloud = PointCloud::new("demo_camera_depth");
    let intrinsics = CameraIntrinsics::default();

    // Simulate a depth map: room with walls at 3m, floor, and a person at 2m
    let w = 160; // downsampled
    let h = 120;
    let mut depth = vec![3.0f32; w * h];

    // Floor plane (bottom third)
    for y in (h * 2 / 3)..h {
        for x in 0..w {
            depth[y * w + x] = 1.0 + (y - h * 2 / 3) as f32 * 0.05;
        }
    }

    // Person silhouette (center, depth=2m)
    for y in (h / 4)..(h * 3 / 4) {
        for x in (w * 2 / 5)..(w * 3 / 5) {
            let dy = (y as f32 - h as f32 / 2.0).abs() / (h as f32 / 4.0);
            let dx = (x as f32 - w as f32 / 2.0).abs() / (w as f32 / 5.0);
            if dx * dx + dy * dy < 1.0 {
                depth[y * w + x] = 2.0 + (dx * dx + dy * dy) * 0.3;
            }
        }
    }

    let scaled_intrinsics = CameraIntrinsics {
        fx: intrinsics.fx * w as f32 / intrinsics.width as f32,
        fy: intrinsics.fy * h as f32 / intrinsics.height as f32,
        cx: w as f32 / 2.0,
        cy: h as f32 / 2.0,
        width: w as u32,
        height: h as u32,
    };

    backproject_depth(&depth, &scaled_intrinsics, None, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backproject_2x2_depth_yields_four_points() {
        // 2x2 image, depth=1m everywhere; trivial intrinsics.
        let intr = CameraIntrinsics {
            fx: 1.0,
            fy: 1.0,
            cx: 0.5,
            cy: 0.5,
            width: 2,
            height: 2,
        };
        let depth = vec![1.0f32; 4];
        let cloud = backproject_depth(&depth, &intr, None, 1);
        assert_eq!(cloud.points.len(), 4, "2x2 depth → 4 backprojected points");
        // Every point should be at z=1.0.
        for p in &cloud.points {
            assert!((p.z - 1.0).abs() < 1e-6, "z should be 1.0, got {}", p.z);
        }
        // With cx=0.5, cy=0.5 the four pixel centers backproject symmetrically
        // about the optical axis: x in {-0.5, 0.5}, y in {-0.5, 0.5}.
        let mut xs: Vec<f32> = cloud.points.iter().map(|p| p.x).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((xs[0] + 0.5).abs() < 1e-6);
        assert!((xs.last().unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn backproject_rejects_invalid_depth() {
        let intr = CameraIntrinsics {
            fx: 1.0,
            fy: 1.0,
            cx: 0.5,
            cy: 0.5,
            width: 2,
            height: 2,
        };
        // All pixels NaN → no points.
        let depth = vec![f32::NAN; 4];
        let cloud = backproject_depth(&depth, &intr, None, 1);
        assert_eq!(cloud.points.len(), 0);
    }

    #[test]
    fn confidence_gate_filters_and_propagates_intensity() {
        let estimate = DepthEstimate {
            depth: vec![1.0, 1.0, 1.0, 1.0],
            confidence: Some(vec![0.2, 0.9, 1.1, 2.0]),
            width: 2,
            height: 2,
            intrinsics: CameraIntrinsics {
                fx: 1.0,
                fy: 1.0,
                cx: 0.5,
                cy: 0.5,
                width: 2,
                height: 2,
            },
            extrinsics: Some([0.0; 12]),
            is_metric: true,
            backend: "test",
            min_confidence: 1.0,
        };
        let cloud = backproject_estimate(&estimate, None, 1);
        assert_eq!(cloud.points.len(), 2);
        assert_eq!(cloud.points[0].intensity, 1.1);
        assert_eq!(cloud.points[1].intensity, 2.0);
    }

    #[test]
    fn backprojection_tolerates_short_depth_buffer() {
        let intr = CameraIntrinsics {
            fx: 1.0,
            fy: 1.0,
            cx: 0.5,
            cy: 0.5,
            width: 2,
            height: 2,
        };
        let cloud = backproject_depth(&[1.0], &intr, None, 1);
        assert_eq!(cloud.points.len(), 1);
    }

    #[test]
    fn nearest_rgb_resize_preserves_corner_colors() {
        let source = [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
        let resized = resize_rgb_nearest(&source, 2, 2, 4, 4).unwrap();
        assert_eq!(&resized[0..3], &[255, 0, 0]);
        assert_eq!(&resized[(15 * 3)..(16 * 3)], &[255, 255, 255]);
    }

    #[test]
    fn heuristic_handles_one_pixel_image() {
        let depth = estimate_depth_heuristic(&[127, 127, 127], 1, 1).unwrap();
        assert_eq!(depth.len(), 1);
        assert!(depth[0].is_finite());
    }

    #[test]
    fn model_intrinsics_fall_back_when_invalid() {
        let intr = CameraIntrinsics::from_matrix_or_default(&[0.0; 9], 320, 240);
        assert!(intr.fx > 0.0 && intr.fy > 0.0);
        assert_eq!((intr.width, intr.height), (320, 240));
    }
}
