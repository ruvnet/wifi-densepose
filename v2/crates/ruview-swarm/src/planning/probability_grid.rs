//! Bayesian probability grid for victim localization.

use crate::types::GridCell;

/// 2-D grid tracking posterior victim probability per cell.
pub struct ProbabilityGrid {
    pub cells: Vec<Vec<GridCell>>,
    pub cell_size_m: f64,
    pub width: u32,
    pub height: u32,
}

impl ProbabilityGrid {
    pub fn new(width: u32, height: u32, cell_size_m: f64) -> Self {
        let cells = (0..height)
            .map(|y| {
                (0..width)
                    .map(|x| GridCell {
                        x_idx: x,
                        y_idx: y,
                        victim_probability: 0.5, // uninformative prior
                        pheromone: 0.0,
                        last_scanned_ms: 0,
                    })
                    .collect()
            })
            .collect();
        Self { cells, cell_size_m, width, height }
    }

    /// Bayesian update: P(victim | detection) or P(victim | no detection).
    pub fn update_bayesian(&mut self, cell: (u32, u32), confidence: f32, detected: bool) {
        let (cx, cy) = cell;
        if cx >= self.width || cy >= self.height {
            return;
        }
        let c = &mut self.cells[cy as usize][cx as usize];
        let prior = c.victim_probability as f64;
        // Likelihood ratio update
        let likelihood = if detected {
            confidence as f64
        } else {
            1.0 - confidence as f64
        };
        let denom = likelihood * prior + (1.0 - likelihood) * (1.0 - prior);
        c.victim_probability = if denom > 1e-9 {
            (likelihood * prior / denom) as f32
        } else {
            prior as f32
        };
        c.pheromone = (c.pheromone + 0.1).min(1.0);
    }

    /// Returns the cell (x, y) with highest expected value: P * (1 - scanned_weight).
    pub fn highest_priority_unscanned(&self) -> Option<(u32, u32)> {
        let now_approx: u64 = 0; // caller should pass current time; use 0 for simplicity
        let _ = now_approx;
        let mut best: Option<((u32, u32), f32)> = None;
        for row in &self.cells {
            for cell in row {
                let scanned_weight = if cell.last_scanned_ms > 0 { cell.pheromone } else { 0.0 };
                let score = cell.victim_probability * (1.0 - scanned_weight);
                if best.as_ref().is_none_or(|(_, bs)| score > *bs) {
                    best = Some(((cell.x_idx, cell.y_idx), score));
                }
            }
        }
        best.map(|(pos, _)| pos)
    }

    /// Mark a cell as scanned. Returns true if this is the first scan of this cell.
    pub fn mark_scanned(&mut self, cell: (u32, u32)) -> bool {
        let (cx, cy) = cell;
        if cx >= self.width || cy >= self.height {
            return false;
        }
        let c = &mut self.cells[cy as usize][cx as usize];
        if c.last_scanned_ms == 0 {
            c.last_scanned_ms = 1; // mark as visited
            true
        } else {
            false
        }
    }

    /// Fraction of cells that have been scanned at least once.
    pub fn coverage_pct(&self) -> f64 {
        let total: usize = self.cells.iter().flatten().count();
        let scanned: usize = self.cells.iter().flatten().filter(|c| c.last_scanned_ms > 0).count();
        if total == 0 { 1.0 } else { scanned as f64 / total as f64 }
    }

    /// Return the next cell for systematic boustrophedon sweep (row-by-row, unscanned first).
    pub fn next_systematic_cell(&self, _state: &crate::types::DroneState) -> Option<(u32, u32)> {
        // Walk rows in order; within each row alternate direction based on row parity.
        for yi in 0..self.height {
            let x_iter: Box<dyn Iterator<Item = u32>> = if yi % 2 == 0 {
                Box::new(0..self.width)
            } else {
                Box::new((0..self.width).rev())
            };
            for xi in x_iter {
                if self.cells[yi as usize][xi as usize].last_scanned_ms == 0 {
                    return Some((xi, yi));
                }
            }
        }
        None
    }

    /// Merge another grid's probabilities using weighted average.
    pub fn apply_gossip_update(&mut self, remote: &ProbabilityGrid) {
        let h = self.height.min(remote.height) as usize;
        let w = self.width.min(remote.width) as usize;
        for y in 0..h {
            for x in 0..w {
                let local = &mut self.cells[y][x];
                let r = remote.cells[y][x].victim_probability;
                local.victim_probability = (local.victim_probability + r) / 2.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayesian_update_increases_probability() {
        let mut grid = ProbabilityGrid::new(10, 10, 2.0);
        grid.update_bayesian((5, 5), 0.9, true);
        assert!(grid.cells[5][5].victim_probability > 0.5);
    }

    #[test]
    fn test_bayesian_update_decreases_probability() {
        let mut grid = ProbabilityGrid::new(10, 10, 2.0);
        grid.update_bayesian((5, 5), 0.9, false);
        assert!(grid.cells[5][5].victim_probability < 0.5);
    }

    #[test]
    fn test_highest_priority_returns_cell() {
        let mut grid = ProbabilityGrid::new(5, 5, 2.0);
        // Boost one cell
        grid.cells[2][3].victim_probability = 0.99;
        grid.cells[2][3].pheromone = 0.0;
        let best = grid.highest_priority_unscanned();
        assert!(best.is_some());
        assert_eq!(best.unwrap(), (3, 2));
    }
}
