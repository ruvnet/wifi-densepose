Pod::Spec.new do |s|
  s.name           = 'RuViewLidar'
  s.version        = '0.1.0'
  s.summary        = 'Bounded ARKit and RoomPlan capture for RuView commissioning'
  s.description    = 'Captures confidence-filtered visible-scene geometry without persisting raw camera or depth frames.'
  s.license        = { :type => 'MIT' }
  s.author         = { 'RuView' => 'opensource@ruv.net' }
  s.homepage       = 'https://github.com/ruvnet/RuView'
  s.platforms      = { :ios => '15.1' }
  s.swift_version  = '5.9'
  s.source         = { :git => 'https://github.com/ruvnet/RuView.git' }
  s.static_framework = true

  s.dependency 'ExpoModulesCore'
  s.frameworks = 'ARKit', 'AVFoundation', 'RoomPlan'
  s.source_files = '**/*.{h,m,mm,swift}'
end
