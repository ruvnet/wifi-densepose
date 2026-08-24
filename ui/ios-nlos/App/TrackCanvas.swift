import Foundation
import RuViewNLOSCore
import SwiftUI

enum TrackCanvasMode: String, CaseIterable, Identifiable {
    case targets
    case pointCloud

    var id: String { rawValue }
}

struct TrackCanvas: View {
    let tracks: [NLOSTrack]
    let mode: TrackCanvasMode

    var body: some View {
        Canvas { context, size in
            var background = Path()
            background.addRect(CGRect(origin: .zero, size: size))
            context.fill(
                background,
                with: .color(Color(red: 0.014, green: 0.029, blue: 0.043))
            )
            drawGrid(context: &context, size: size)
            if mode == .pointCloud {
                drawPointCloud(context: &context, size: size)
            } else {
                drawRadar(context: &context, size: size)

                let radiusMeters = max(
                    5,
                    min(100, tracks.flatMap { [abs($0.positionM.x), abs($0.positionM.z)] }.max() ?? 5)
                )

                for track in tracks {
                    draw(
                        track: track,
                        context: &context,
                        size: size,
                        radiusMeters: radiusMeters
                    )
                }
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .overlay {
            RoundedRectangle(cornerRadius: 16)
                .stroke(Color.cyan.opacity(0.24), lineWidth: 1)
        }
        .shadow(color: Color.cyan.opacity(0.08), radius: 16)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(
            mode == .pointCloud
                ? "Projected LiDAR reconstruction cloud with \(tracks.count) gated hidden target hypotheses"
                : "Plan view with \(tracks.count) gated hidden target hypotheses"
        )
    }

    private func drawPointCloud(context: inout GraphicsContext, size: CGSize) {
        let cyan = Color(red: 0.129, green: 0.831, blue: 0.906)
        let green = Color(red: 0.345, green: 0.949, blue: 0.545)
        let orange = Color(red: 1.000, green: 0.714, blue: 0.361)

        for row in 0..<10 {
            for column in 0..<16 {
                let x = size.width * (0.08 + CGFloat(column) / 17)
                let floorY = size.height * (0.58 + CGFloat(row) * 0.032)
                let wallY = size.height * (0.12 + CGFloat(row) * 0.039)
                let wallX = x + CGFloat(row - 5) * 1.8
                let floorOpacity = 0.28 + Double(row) * 0.025
                let wallColor = row > 7 ? orange : cyan
                context.fill(
                    Path(ellipseIn: CGRect(x: x - 1.1, y: floorY - 1.1, width: 2.2, height: 2.2)),
                    with: .color(cyan.opacity(floorOpacity))
                )
                context.fill(
                    Path(ellipseIn: CGRect(x: wallX - 1.15, y: wallY - 1.15, width: 2.3, height: 2.3)),
                    with: .color(wallColor.opacity(0.52))
                )
            }
        }

        let goldenAngle = Double.pi * (3 - sqrt(5.0))
        for (trackIndex, track) in tracks.prefix(16).enumerated() {
            let centerX = size.width / 2
                + CGFloat(max(-6, min(6, track.positionM.x))) * size.width * 0.055
                - CGFloat(max(0, min(8, track.positionM.z))) * size.width * 0.012
            let centerY = size.height * 0.63
                - CGFloat(max(0, min(4, track.positionM.y))) * size.height * 0.10
                + CGFloat(max(0, min(8, track.positionM.z))) * size.height * 0.018
            let radiusX = max(8, min(34, CGFloat(sqrt(track.covarianceDiagonalM2.x)) * 28))
            let radiusY = max(8, min(38, CGFloat(sqrt(track.covarianceDiagonalM2.y)) * 30))
            let color = track.state == .degraded ? orange : green

            for sample in 0..<72 {
                let normalizedY = 1 - 2 * ((Double(sample) + 0.5) / 72)
                let radial = sqrt(max(0, 1 - normalizedY * normalizedY))
                let theta = Double(sample) * goldenAngle + Double(trackIndex) * 0.73
                let shell = 0.5 + Double((sample * 37 + trackIndex * 17) % 47) / 94
                let point = CGPoint(
                    x: centerX + CGFloat(cos(theta) * radial * shell) * radiusX,
                    y: centerY + CGFloat(normalizedY * shell) * radiusY
                )
                let diameter: CGFloat = sample.isMultiple(of: 5) ? 3.2 : 2.1
                context.fill(
                    Path(ellipseIn: CGRect(
                        x: point.x - diameter / 2,
                        y: point.y - diameter / 2,
                        width: diameter,
                        height: diameter
                    )),
                    with: .color(color.opacity(0.82))
                )
            }

            var lockLine = Path()
            lockLine.move(to: CGPoint(x: centerX, y: centerY - radiusY - 10))
            lockLine.addLine(to: CGPoint(x: centerX, y: centerY + radiusY + 10))
            context.stroke(lockLine, with: .color(color.opacity(0.55)), lineWidth: 1)
            context.stroke(
                Path(ellipseIn: CGRect(
                    x: centerX - radiusX - 5,
                    y: centerY - radiusY - 5,
                    width: (radiusX + 5) * 2,
                    height: (radiusY + 5) * 2
                )),
                with: .color(color.opacity(0.86)),
                style: StrokeStyle(lineWidth: 1.2, dash: [4, 3])
            )
            context.fill(
                Path(ellipseIn: CGRect(x: centerX - 3, y: centerY - 3, width: 6, height: 6)),
                with: .color(color)
            )
            context.draw(
                Text("\(String(track.trackId.prefix(12)).uppercased())  \(Int((track.confidence * 100).rounded()))%")
                    .font(.caption2.bold().monospaced())
                    .foregroundColor(color),
                at: CGPoint(
                    x: min(size.width - 62, centerX + radiusX + 34),
                    y: max(44, centerY - radiusY - 8)
                )
            )
        }

        context.draw(
            Text("LIDAR POINT CLOUD / RECONSTRUCTION")
                .font(.caption2.bold().monospaced())
                .foregroundColor(cyan),
            at: CGPoint(x: size.width / 2, y: 18)
        )
        context.draw(
            Text("NOT RAW IPHONE LIDAR")
                .font(.caption2.bold().monospaced())
                .foregroundColor(orange.opacity(0.82)),
            at: CGPoint(x: size.width / 2, y: 34)
        )
        context.draw(
            Text("\(tracks.count * 72) GATED TARGET RETURNS")
                .font(.caption2.bold().monospaced())
                .foregroundColor(Color.white.opacity(0.64)),
            at: CGPoint(x: size.width / 2, y: size.height - 16)
        )
    }

    private func draw(
        track: NLOSTrack,
        context: inout GraphicsContext,
        size: CGSize,
        radiusMeters: Double
    ) {
        let point = CGPoint(
            x: size.width / 2 + CGFloat(track.positionM.x / radiusMeters) * size.width * 0.43,
            y: size.height / 2 - CGFloat(track.positionM.z / radiusMeters) * size.height * 0.43
        )
        let uncertainty = min(
            36,
            max(9, CGFloat(sqrt(max(track.covarianceDiagonalM2.x, track.covarianceDiagonalM2.z))) * 14)
        )
        let color: Color = track.state == .degraded
            ? Color(red: 1.000, green: 0.612, blue: 0.231)
            : Color(red: 0.129, green: 0.831, blue: 0.906)
        let uncertaintyRect = CGRect(
            x: point.x - uncertainty,
            y: point.y - uncertainty,
            width: uncertainty * 2,
            height: uncertainty * 2
        )

        context.stroke(
            Path(ellipseIn: uncertaintyRect),
            with: .color(color.opacity(0.5)),
            style: StrokeStyle(lineWidth: 1, dash: [4, 3])
        )
        context.fill(
            Path(ellipseIn: CGRect(x: point.x - 11, y: point.y - 11, width: 22, height: 22)),
            with: .color(color.opacity(0.12))
        )
        context.fill(
            Path(ellipseIn: CGRect(x: point.x - 4, y: point.y - 4, width: 8, height: 8)),
            with: .color(color)
        )

        let labelPoint = CGPoint(
            x: min(max(point.x, 42), size.width - 42),
            y: min(point.y + uncertainty + 14, size.height - 12)
        )
        context.draw(
            Text(String(track.trackId.prefix(12)).uppercased())
                .font(.caption2.bold().monospaced())
                .foregroundColor(.white),
            at: labelPoint
        )
    }

    private func drawGrid(context: inout GraphicsContext, size: CGSize) {
        var minorGrid = Path()
        let spacing: CGFloat = 24
        var x: CGFloat = 0
        while x <= size.width {
            minorGrid.move(to: CGPoint(x: x, y: 0))
            minorGrid.addLine(to: CGPoint(x: x, y: size.height))
            x += spacing
        }
        var y: CGFloat = 0
        while y <= size.height {
            minorGrid.move(to: CGPoint(x: 0, y: y))
            minorGrid.addLine(to: CGPoint(x: size.width, y: y))
            y += spacing
        }
        context.stroke(minorGrid, with: .color(Color.cyan.opacity(0.065)), lineWidth: 0.5)

        var axes = Path()
        axes.move(to: CGPoint(x: size.width / 2, y: 0))
        axes.addLine(to: CGPoint(x: size.width / 2, y: size.height))
        axes.move(to: CGPoint(x: 0, y: size.height / 2))
        axes.addLine(to: CGPoint(x: size.width, y: size.height / 2))
        context.stroke(axes, with: .color(Color.cyan.opacity(0.34)), lineWidth: 1)
    }

    private func drawRadar(context: inout GraphicsContext, size: CGSize) {
        let center = CGPoint(x: size.width / 2, y: size.height / 2)
        let maximumDiameter = min(size.width, size.height) * 0.86
        for fraction in [0.25, 0.5, 0.75, 1.0] as [CGFloat] {
            let diameter = maximumDiameter * fraction
            context.stroke(
                Path(ellipseIn: CGRect(
                    x: center.x - diameter / 2,
                    y: center.y - diameter / 2,
                    width: diameter,
                    height: diameter
                )),
                with: .color(Color.cyan.opacity(fraction == 1 ? 0.2 : 0.12)),
                lineWidth: 0.8
            )
        }

        var sweep = Path()
        sweep.move(to: center)
        sweep.addLine(to: CGPoint(
            x: center.x + maximumDiameter * 0.31,
            y: center.y - maximumDiameter * 0.31
        ))
        context.stroke(sweep, with: .color(Color.green.opacity(0.32)), lineWidth: 1)

        context.fill(
            Path(ellipseIn: CGRect(x: center.x - 3, y: center.y - 3, width: 6, height: 6)),
            with: .color(Color.green.opacity(0.8))
        )

        context.draw(
            Text("RELAY ORIGIN")
                .font(.caption2.bold().monospaced())
                .foregroundColor(Color.white.opacity(0.5)),
            at: CGPoint(x: center.x, y: center.y + 17)
        )
    }
}
