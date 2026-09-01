// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "RuViewNLOS",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        .library(name: "RuViewNLOSCore", targets: ["RuViewNLOSCore"]),
        .library(name: "RuViewNLOSApple", targets: ["RuViewNLOSApple"]),
    ],
    targets: [
        .target(name: "RuViewNLOSCore"),
        .target(
            name: "RuViewNLOSApple",
            dependencies: ["RuViewNLOSCore"]
        ),
        .testTarget(
            name: "RuViewNLOSCoreTests",
            dependencies: ["RuViewNLOSCore"]
        ),
        .testTarget(
            name: "RuViewNLOSAppleTests",
            dependencies: ["RuViewNLOSApple"]
        ),
    ]
)
