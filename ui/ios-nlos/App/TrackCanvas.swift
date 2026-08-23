import Foundation
import RuViewNLOSCore
import SwiftUI

struct TrackCanvas: View {
    let tracks: [NLOSTrack]

    var body: some View {
        Canvas { context, size in
            var background = Path()
            background.addRect(CGRect(origin: .zero, size: size))
            context.fill(
                background,
                with: .color(Color(red: 0.014, green: 0.029, blue: 0.043))
            )
            drawGrid(context: &context, size: size)
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
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .overlay {
            RoundedRectangle(cornerRadius: 16)
                .stroke(Color.cyan.opacity(0.24), lineWidth: 1)
        }
        .shadow(color: Color.cyan.opacity(0.08), radius: 16)
        .accessibilityHidden(true)
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
