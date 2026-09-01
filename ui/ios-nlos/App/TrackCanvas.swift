import Foundation
import RuViewNLOSCore
import SwiftUI

struct TrackCanvas: View {
    let tracks: [NLOSTrack]

    var body: some View {
        Canvas { context, size in
            drawGrid(context: &context, size: size)
            let radiusMeters = max(
                5,
                min(100, tracks.flatMap { [abs($0.positionM.x), abs($0.positionM.z)] }.max() ?? 5)
            )

            for track in tracks {
                let point = CGPoint(
                    x: size.width / 2 + CGFloat(track.positionM.x / radiusMeters) * size.width * 0.45,
                    y: size.height / 2 - CGFloat(track.positionM.z / radiusMeters) * size.height * 0.45
                )
                let uncertainty = min(
                    34,
                    max(8, CGFloat(sqrt(max(track.covarianceDiagonalM2.x, track.covarianceDiagonalM2.z))) * 14)
                )
                let color: Color = track.state == .degraded ? .orange : .cyan
                let uncertaintyRect = CGRect(
                    x: point.x - uncertainty,
                    y: point.y - uncertainty,
                    width: uncertainty * 2,
                    height: uncertainty * 2
                )
                context.stroke(
                    Path(ellipseIn: uncertaintyRect),
                    with: .color(color.opacity(0.45)),
                    lineWidth: 1
                )
                context.fill(
                    Path(ellipseIn: CGRect(x: point.x - 5, y: point.y - 5, width: 10, height: 10)),
                    with: .color(color)
                )
                context.draw(
                    Text(String(track.trackId.prefix(12)))
                        .font(.caption2.monospaced())
                        .foregroundColor(.primary),
                    at: CGPoint(x: point.x, y: point.y + uncertainty + 10)
                )
            }
        }
        .background(Color(uiColor: .secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .overlay {
            RoundedRectangle(cornerRadius: 16)
                .stroke(Color.secondary.opacity(0.25), lineWidth: 1)
        }
        .accessibilityHidden(true)
    }

    private func drawGrid(context: inout GraphicsContext, size: CGSize) {
        var path = Path()
        for fraction in stride(
            from: CGFloat(0.1),
            through: CGFloat(0.9),
            by: CGFloat(0.1)
        ) {
            let x = size.width * fraction
            let y = size.height * fraction
            path.move(to: CGPoint(x: x, y: 0))
            path.addLine(to: CGPoint(x: x, y: size.height))
            path.move(to: CGPoint(x: 0, y: y))
            path.addLine(to: CGPoint(x: size.width, y: y))
        }
        context.stroke(path, with: .color(.secondary.opacity(0.12)), lineWidth: 0.5)

        var axes = Path()
        axes.move(to: CGPoint(x: size.width / 2, y: 0))
        axes.addLine(to: CGPoint(x: size.width / 2, y: size.height))
        axes.move(to: CGPoint(x: 0, y: size.height / 2))
        axes.addLine(to: CGPoint(x: size.width, y: size.height / 2))
        context.stroke(axes, with: .color(.secondary.opacity(0.5)), lineWidth: 1)
    }
}
