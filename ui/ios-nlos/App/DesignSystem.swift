import SwiftUI

enum RuViewTypography {
    enum Weight {
        case regular
        case medium
        case bold
    }

    static func outfit(_ size: CGFloat, weight: Weight = .regular) -> Font {
        let name: String
        switch weight {
        case .regular:
            name = "Outfit-Regular"
        case .medium:
            name = "Outfit-Medium"
        case .bold:
            name = "Outfit-Bold"
        }
        return .custom(name, size: size)
    }

    static func mono(_ size: CGFloat, medium: Bool = true) -> Font {
        .custom(medium ? "JetBrainsMono-Medium" : "JetBrainsMono-Regular", size: size)
    }
}
