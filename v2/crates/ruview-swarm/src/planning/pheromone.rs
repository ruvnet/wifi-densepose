//! Stigmergic pheromone evaporation for coverage tracking.

use crate::types::GridCell;

/// Evaporate pheromones across all cells.
/// `rate`: fraction decayed per tick (e.g. 0.01 = 1% per tick).
pub fn evaporate(cells: &mut [Vec<GridCell>], rate: f32) {
    for row in cells.iter_mut() {
        for cell in row.iter_mut() {
            cell.pheromone = (cell.pheromone * (1.0 - rate)).max(0.0);
        }
    }
}

/// Deposit pheromone at a cell (clamp to 1.0).
pub fn deposit(cells: &mut [Vec<GridCell>], x: u32, y: u32, amount: f32) {
    if let Some(row) = cells.get_mut(y as usize) {
        if let Some(cell) = row.get_mut(x as usize) {
            cell.pheromone = (cell.pheromone + amount).min(1.0);
        }
    }
}
