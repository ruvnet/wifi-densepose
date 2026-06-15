# Guardian Visual Language

Guardian is the operator-facing HUD and performance-instrument layer for RuView.

It combines RF telemetry, scene state, audio matrix routing, macro controls, and morphing scenes into a portable tactical instrument interface.

## Design Principles

- Instrument first, decoration second.
- Dense but readable.
- Black, white, red as primary system colors.
- CRT/phosphor only where persistence or signal memory matters.
- Swiss grid discipline for layout.
- Every panel must represent live state, stored state, routing, or operator action.
- Avoid decorative cyberpunk language unless it maps to a real signal or control.

## Product Layers

- RuView: sensing, CSI, pose, vitals, training, OTA, data plumbing.
- Guardian HUD: live operator interface and scene control.
- RF Forge Jr: hardware/RF architecture and timing spine.
- ALPACA Guardian: RF-to-audio scene engine and performance layer.

## UI Objects

- Screen: a visible state surface.
- Scene: a stored configuration of macros, routing, audio state, and RF response.
- Morph: continuous interpolation between two scenes.
- Macro: a controllable performance dimension.
- Matrix: routing truth.
- Ribbon: continuous frequency/control surface.
- Phosphor trace: persistence, memory, signal afterimage.

## Color

- Black: background, machine body, signal void.
- White/cream: paper, labels, measurement surfaces.
- Red: alert, selected state, action path, transition.
- Green/cyan phosphor: live signal persistence only.

## Rule

If a visual element does not correspond to signal, state, control, routing, or operator action, remove it.
