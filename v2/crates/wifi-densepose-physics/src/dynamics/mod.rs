//! Optional bounded Rapier articulated-dynamics audit.
//!
//! All bodies, constraints, and snapshots are process-owned. The auditor
//! scores a candidate but never mutates or selects the public pose.

use rapier3d::prelude::{
    ColliderBuilder, PhysicsWorld, Pose, RigidBodyBuilder, RigidBodyHandle, Rotation,
    SphericalJointBuilder, Vector,
};
use wifi_densepose_core::FloorPlane;

use crate::skeleton::OBSERVED_EDGES;

const SEGMENT_RADIUS_M: f32 = 0.035;
const MAX_SUBSTEPS: u8 = 8;

#[derive(Clone, Debug)]
struct Segment {
    handle: RigidBodyHandle,
    target_position: Pose,
    endpoints: (usize, usize),
    midpoint: Vector,
    rotation: Rotation,
}

#[derive(Clone, Debug)]
struct JointAnchor {
    first: usize,
    second: usize,
    local_first: Vector,
    local_second: Vector,
}

/// Scalar dynamics assessment safe for metrics and provenance.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DynamicsAssessment {
    pub stable: bool,
    pub segment_count: u8,
    pub joint_count: u8,
    pub contact_count: u16,
    pub substeps: u8,
    pub max_tracking_error_m: f32,
    pub max_joint_anchor_error_m: f32,
    pub max_floor_penetration_m: f32,
    pub max_control_effort: f32,
}

/// Process-owned humanoid assembled from capsule segments and spherical joints.
pub struct DynamicsAuditor {
    world: PhysicsWorld,
    segments: Vec<Segment>,
    anchors: Vec<JointAnchor>,
    floor: FloorPlane,
}

impl DynamicsAuditor {
    /// Build a bounded articulated model from one candidate pose.
    #[must_use]
    pub fn new(joints: &[[f32; 3]; 17], floor: FloorPlane) -> Self {
        let mut world = PhysicsWorld::new();
        world.gravity = Vector::new(0.0, 0.0, -9.81);
        let floor_normal = Vector::from_array(floor.normal);
        let floor_point = floor_normal * -floor.offset_m;
        let floor_rotation = Rotation::from_rotation_arc(Vector::Z, floor_normal);
        world.insert(
            RigidBodyBuilder::fixed().pose(Pose::from_parts(floor_point, floor_rotation)),
            ColliderBuilder::cuboid(10.0, 10.0, 0.02).friction(0.8),
        );

        let mut segments = Vec::with_capacity(OBSERVED_EDGES.len());
        for endpoints in OBSERVED_EDGES {
            let a = Vector::from_array(joints[endpoints.0]);
            let b = Vector::from_array(joints[endpoints.1]);
            let delta = b - a;
            let length = delta.length().max(SEGMENT_RADIUS_M * 2.0);
            let midpoint = (a + b) * 0.5;
            let direction = delta.try_normalize().unwrap_or(Vector::Y);
            let rotation = Rotation::from_rotation_arc(Vector::Y, direction);
            let position = Pose::from_parts(midpoint, rotation);
            let (handle, _) = world.insert(
                RigidBodyBuilder::dynamic()
                    .pose(position)
                    .linear_damping(0.8)
                    .angular_damping(0.8)
                    .can_sleep(false),
                ColliderBuilder::capsule_y(
                    (length * 0.5 - SEGMENT_RADIUS_M).max(0.001),
                    SEGMENT_RADIUS_M,
                )
                .density(450.0)
                .friction(0.7),
            );
            segments.push(Segment {
                handle,
                target_position: position,
                endpoints,
                midpoint,
                rotation,
            });
        }

        let mut anchors = Vec::new();
        for first in 0..segments.len() {
            for second in (first + 1)..segments.len() {
                let Some(shared) =
                    shared_endpoint(segments[first].endpoints, segments[second].endpoints)
                else {
                    continue;
                };
                let point = Vector::from_array(joints[shared]);
                let local_first =
                    segments[first].rotation.inverse() * (point - segments[first].midpoint);
                let local_second =
                    segments[second].rotation.inverse() * (point - segments[second].midpoint);
                world.insert_impulse_joint(
                    segments[first].handle,
                    segments[second].handle,
                    SphericalJointBuilder::new()
                        .local_anchor1(local_first)
                        .local_anchor2(local_second)
                        .contacts_enabled(false),
                );
                anchors.push(JointAnchor {
                    first,
                    second,
                    local_first,
                    local_second,
                });
            }
        }
        Self {
            world,
            segments,
            anchors,
            floor,
        }
    }

    /// Run bounded PD-controlled audit steps and return scalar residuals.
    #[must_use]
    pub fn audit(
        &mut self,
        joints: &[[f32; 3]; 17],
        dt_seconds: f32,
        requested_substeps: u8,
    ) -> DynamicsAssessment {
        let substeps = requested_substeps.clamp(1, MAX_SUBSTEPS);
        self.world.integration_parameters.dt =
            dt_seconds.clamp(1.0 / 240.0, 0.05) / f32::from(substeps);
        let mut max_control_effort = 0.0f32;
        for segment in &mut self.segments {
            let a = Vector::from_array(joints[segment.endpoints.0]);
            let b = Vector::from_array(joints[segment.endpoints.1]);
            let midpoint = (a + b) * 0.5;
            let direction = (b - a).try_normalize().unwrap_or(Vector::Y);
            let rotation = Rotation::from_rotation_arc(Vector::Y, direction);
            segment.target_position = Pose::from_parts(midpoint, rotation);
        }
        for _ in 0..substeps {
            for segment in &self.segments {
                let Some(body) = self.world.bodies.get_mut(segment.handle) else {
                    continue;
                };
                let position_error = segment.target_position.translation - body.translation();
                let rotation_error =
                    (segment.target_position.rotation * body.rotation().inverse()).to_scaled_axis();
                let force = position_error * 120.0 - body.linvel() * 18.0;
                let torque = rotation_error * 35.0 - body.angvel() * 8.0;
                max_control_effort = max_control_effort.max(force.length()).max(torque.length());
                body.add_force(force, true);
                body.add_torque(torque, true);
            }
            self.world.step();
        }

        let max_tracking_error_m = self
            .segments
            .iter()
            .filter_map(|segment| {
                self.world
                    .bodies
                    .get(segment.handle)
                    .map(|body| (segment, body))
            })
            .map(|(segment, body)| {
                (segment.target_position.translation - body.translation()).length()
            })
            .fold(0.0, f32::max);
        let max_joint_anchor_error_m = self
            .anchors
            .iter()
            .filter_map(|anchor| {
                let first = self.world.bodies.get(self.segments[anchor.first].handle)?;
                let second = self.world.bodies.get(self.segments[anchor.second].handle)?;
                let point_first = first.position() * anchor.local_first;
                let point_second = second.position() * anchor.local_second;
                Some((point_first - point_second).length())
            })
            .fold(0.0, f32::max);
        let max_floor_penetration_m = self
            .segments
            .iter()
            .filter_map(|segment| self.world.bodies.get(segment.handle))
            .map(|body| (-self.floor.signed_distance(body.translation().to_array())).max(0.0))
            .fold(0.0, f32::max);
        let contact_count = self
            .world
            .narrow_phase
            .contact_pairs()
            .filter(|pair| pair.has_any_active_contact())
            .count()
            .try_into()
            .unwrap_or(u16::MAX);
        let stable = max_tracking_error_m.is_finite()
            && max_joint_anchor_error_m.is_finite()
            && max_floor_penetration_m.is_finite()
            && max_control_effort.is_finite();
        DynamicsAssessment {
            stable,
            segment_count: self.segments.len().try_into().unwrap_or(u8::MAX),
            joint_count: self.anchors.len().try_into().unwrap_or(u8::MAX),
            contact_count,
            substeps,
            max_tracking_error_m,
            max_joint_anchor_error_m,
            max_floor_penetration_m,
            max_control_effort,
        }
    }
}

fn shared_endpoint(first: (usize, usize), second: (usize, usize)) -> Option<usize> {
    [first.0, first.1]
        .into_iter()
        .find(|endpoint| *endpoint == second.0 || *endpoint == second.1)
}
