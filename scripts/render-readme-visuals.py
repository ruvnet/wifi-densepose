#!/usr/bin/env python3
"""Generate the RuView README visual system. Deterministic, no network."""
from pathlib import Path
from html import escape
import math,json
from readme_visuals_xml import parse_svg
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'assets/readme';OUT.mkdir(parents=True,exist_ok=True)
C='#78cfff';BLUE='#438fff';INK='#071124';WHITE='#f3f7ff';MUTED='#b0c2db'
LIME='#b8f36c';AMBER='#ffb45e'
def text(x,y,value,size=16,color=MUTED,anchor='start',extra=''):
 weight='font-weight="650" letter-spacing="-0.6"' if size>=25 else 'font-weight="500"'
 return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" {weight} {extra}>{escape(value)}</text>'
def path(d,color=C,width=1.5,extra=''):
 return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" {extra}/>'
def circle(x,y,r,color=C,extra=''):
 fill = '' if 'fill=' in extra else 'fill="none"'
 return f'<circle cx="{x}" cy="{y}" r="{r}" stroke="{color}" {fill} {extra}/>'

def icon(kind,x,y,scale=1):
 s=''
 if kind in ['signal','start']:
  for r in [13,25,38]:s+=path(f'M{-r*.8} {-r*.4}Q0 {-r*1.4} {r*.8} {-r*.4}',extra=f'class="breathe" style="animation-delay:-{r/20}s"')
  s+=circle(0,20,4,extra='fill="#72eee0"')+path('M0 12V-1')
 elif kind in ['presence','pose']:
  s+=circle(0,-28,8,LIME)+path('M0 -20V12M-24 -6L0 -14L24 -6M0 12L-18 38M0 12L18 38',LIME,width=2)
  for px,py in [(0,-14),(0,12),(-24,-6),(24,-6)]:s+=circle(px,py,2.3,AMBER,extra=f'fill="{AMBER}" class="spark"')
  if kind=='pose':
   for px,py in [(-18,38),(18,38)]:s+=circle(px,py,3,AMBER,extra='fill="#041117" class="breathe"')
  else:s+=f'<ellipse cx="0" cy="14" rx="40" ry="13" fill="none" stroke="{BLUE}" stroke-dasharray="3 8" class="flow"/>'
 elif kind in ['vitals','models']:
  s+=path('M-42 6H-27L-18 -10L-9 25L3 -25L14 10L24 -3L31 6H43',LIME,width=2,extra='class="trace"')+path('M-42 31H43','#214451',1)
 elif kind in ['edge','hardware']:
  s+='<rect x="-25" y="-25" width="50" height="50" rx="7" fill="#0a2630" stroke="#72eee0"/>'
  s+='<rect x="-13" y="-13" width="26" height="26" rx="3" fill="none" stroke="#70b6ff"/>'
  for v in [-16,0,16]:s+=path(f'M{v} -36V-25M{v} 25V36M-36 {v}H-25M25 {v}H36')
  s+=circle(0,0,4,extra='fill="#72eee0" class="breathe"')
 elif kind in ['home','automation']:
  s+=path('M-37 -3L0 -34L37 -3M-28 -10V32H28V-10M-9 32V8H9V32',width=2)
  s+=circle(0,-9,5,BLUE,extra='class="breathe"')
 elif kind in ['agent','agents']:
  for px,py in [(0,-33),(-35,23),(35,23)]:s+=path(f'M0 0L{px} {py}',BLUE,1,extra='class="flow"')+circle(px,py,7,extra='class="breathe"')
  s+=circle(0,0,15)+path('M-6 0L-1 5L8 -6')
 elif kind in ['seed','evidence']:
  s+=path('M0 -38L31 -20V18L0 38L-31 18V-20ZM-31 -20L0 -1L31 -20M0 -1V38',width=2)
  s+=path('M-15 1L-5 11L17 -12',BLUE,2,extra='class="trace"')
 else:
  for j in range(3):s+=path(f'M-38 {j*14-20}L0 {j*14-38}L38 {j*14-20}L0 {j*14-2}Z',extra='class="breathe"')
 base='<ellipse cx="0" cy="48" rx="43" ry="12" fill="#133a45" opacity=".5"/>'
 base+=path('M-43 48L0 27L43 48L0 69Z','#285c68',1)
 return f'<g transform="translate({x} {y}) scale({scale})">{base}<g class="float">{s}</g></g>'

def spatial(x,y,scale=1):
 """Projected geometry gives depth without scripts or CSS 3D dependencies."""
 s='<path d="M-82 36L0 -6L82 36L0 78Z" fill="#0a2733" stroke="#286171"/>'
 s+='<path d="M-82 36V-52L0 -94L82 -52V36M-82 -52L0 -10L82 -52M0 -10V78" fill="none" stroke="#366d80"/>'
 s+='<g class="scan"><path d="M-81 32L0 -10L81 32L0 74Z" fill="#5fe3d8" fill-opacity=".12" stroke="#72eee0" stroke-opacity=".7"/></g>'
 for px,py in [(-40,6),(0,-15),(40,6),(-20,30),(20,30),(0,49)]:
  s+=circle(px,py,2,BLUE,extra='fill="#70b6ff" class="spark"')
 s+='<g class="orbit"><ellipse cx="0" cy="4" rx="102" ry="30" fill="none" stroke="#70b6ff" stroke-opacity=".4" stroke-dasharray="2 8"/><circle cx="102" cy="4" r="4" fill="#91f6df"/></g>'
 s+='<ellipse cx="0" cy="36" rx="85" ry="28" fill="none" stroke="#70b6ff" class="wave"/>'
 return f'<g transform="translate({x} {y}) scale({scale})">{s}</g>'
def _frame(w,h,title,desc,body):
 return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc><defs><linearGradient id="wash" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#0b303a"/><stop offset=".55" stop-color="#06171e"/><stop offset="1" stop-color="#030c13"/></linearGradient><linearGradient id="accent"><stop stop-color="#91f6df"/><stop offset="1" stop-color="#26bad7"/></linearGradient><radialGradient id="halo"><stop stop-color="#35d4db" stop-opacity=".17"/><stop offset="1" stop-color="#35d4db" stop-opacity="0"/></radialGradient></defs><style>text{{font-family:Arial,Helvetica,sans-serif}}.mono{{font-family:ui-monospace,Consolas,monospace;letter-spacing:2px}}.flow{{stroke-dasharray:5 12;animation:flow 12s linear infinite}}.breathe{{animation:breathe 6s ease-in-out infinite}}.trace{{stroke-dasharray:400;stroke-dashoffset:0;animation:trace 10s ease-in-out infinite}}.float{{animation:float 9s ease-in-out infinite}}@keyframes flow{{to{{stroke-dashoffset:-136}}}}@keyframes breathe{{50%{{opacity:.4}}}}@keyframes trace{{0%,100%{{stroke-dashoffset:0}}50%{{stroke-dashoffset:90}}}}@keyframes float{{50%{{transform:translateY(-6px)}}}}@media(prefers-reduced-motion:reduce){{*{{animation:none!important}}}}</style><rect x=".5" y=".5" width="{w-1}" height="{h-1}" rx="18" fill="url(#wash)" stroke="#245360"/>{body}</svg>'''
MOTION='''<style>
text{font-family:"Avenir Next","Segoe UI",Helvetica,Arial,sans-serif}
.mono{font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace;letter-spacing:1.6px}
.float{animation:float 5s ease-in-out infinite}
.flow{stroke-dasharray:5 12;animation:flow 5s linear infinite}
.breathe{animation:breathe 4s ease-in-out infinite}
.trace{stroke-dasharray:400;animation:trace 6s ease-in-out infinite}
.scan{animation:scan 7s ease-in-out infinite}
.orbit{transform-box:fill-box;transform-origin:center;animation:orbit 18s linear infinite}
.spark{animation:spark 3.5s ease-in-out infinite}
.wave{transform-box:fill-box;transform-origin:center;animation:wave 4s ease-out infinite}
@keyframes float{0%,100%{transform:translate(0,0)}50%{transform:translate(3px,-10px)}}
@keyframes scan{0%,100%{transform:translateY(0)}50%{transform:translateY(-76px)}}
@keyframes orbit{to{transform:rotate(360deg)}}
@keyframes spark{50%{opacity:.25}}
@keyframes flow{to{stroke-dashoffset:-136}}
@keyframes breathe{50%{opacity:.4}}
@keyframes trace{0%,100%{stroke-dashoffset:0}50%{stroke-dashoffset:90}}
@keyframes wave{0%{transform:scale(.5);opacity:.7}100%{transform:scale(1.3);opacity:0}}
@media(prefers-reduced-motion:reduce){*{animation:none!important}}
</style>'''

def frame(w,h,title,desc,body):
 value=_frame(w,h,title,desc,body)
 value=value[:value.index('<style>')]+MOTION+value[value.index('</style>')+8:]
 palette={'#0b303a':'#102749','#06171e':'#071429','#030c13':'#03091a',
          '#245360':'#2d4e77','#91f6df':'#9ee7ff','#26bad7':'#348bff',
          '#35d4db':'#348bff','#72eee0':C,'#70b6ff':BLUE,'#041117':INK,
          '#133a45':'#142c50','#285c68':'#2e527d','#0a2733':'#0b1b35',
          '#286171':'#2e5b8e','#366d80':'#4473a5','#5fe3d8':'#438fff',
          '#0a2630':'#0c2242','#12343d':'#162f50','#4b9ba4':'#5490c5',
          '#071b23':'#091b35','#285360':'#315685','#214451':'#233f63','#3c8897':'#437cb2'}
 for old,new in palette.items():value=value.replace(old,new)
 return value

manifest=[]
def save(name,w,h,title,desc,body,target):
 name+='-spatial-v3'
 s=frame(w,h,title,desc,body);raw=(s+'\n').encode('utf-8');parse_svg(raw);(OUT/(name+'.svg')).write_bytes(raw);manifest.append({'file':'assets/readme/'+name+'.svg','title':title,'target':target,'width':w,'height':h})
# Main title panel: a radio field passes through an illustrated room.
s=text(48,44,'RUVIEW / COGNITUM ECOSYSTEM',13,C,extra='class="mono"')+text(48,121,'RuView',76,WHITE)+text(48,184,'Spaces become signals.',38,WHITE)+text(48,234,'Signals become understanding.',32,C)+text(50,279,'Explore RF sensing, edge inference and local automation.',18)
s+='<rect x="48" y="326" width="270" height="52" rx="12" fill="url(#accent)"/>'+text(72,359,'Explore the guide',20,INK)+path('M272 352H292M285 345L292 352L285 359',INK,2)
s+=text(49,419,'RF SENSING  /  EDGE MODELS  /  LOCAL ACTION',11,MUTED,extra='class="mono"')
s+='<circle cx="950" cy="235" r="235" fill="url(#halo)"/>'
for i in range(9):s+=path(f'M{750+i*40} 282L{610+i*80} 443','#12343d',1)
for y in [306,336,372,418]:s+=path(f'M680 {y}H1168','#12343d',1)
s+='<g class="float">'+path('M803 157L1008 106L1137 177L930 236ZM803 157V314L930 391L1137 324V177M930 236V391M1008 106V268L803 314M1008 268L1137 324','#4b9ba4',1.5)+'</g>'
s+='<g transform="translate(0 80)"><g class="scan"><path d="M803 157L1008 106L1137 177L930 236Z" fill="#72eee0" fill-opacity=".1" stroke="#72eee0" stroke-opacity=".6"/></g></g>'
s+='<g transform="translate(974 282)"><g class="orbit"><ellipse rx="135" ry="36" fill="none" stroke="#70b6ff" stroke-dasharray="3 11" opacity=".4"/><circle cx="135" cy="0" r="4" fill="#91f6df"/></g></g>'
for i in range(3):
 s+=f'<g transform="translate(974 312)"><ellipse rx="140" ry="44" fill="none" stroke="{BLUE}" class="wave" style="animation-delay:-{i*1.3}s"/></g>'
for i in range(24):
 s+=circle(849+(i%6)*44,198+(i//6)*35,1.7,BLUE,extra=f'fill="#70b6ff" class="spark" style="animation-delay:-{i%7/2}s"')
s+=icon('presence',976,265,1.35)+icon('edge',780,349,.7)
for i in range(4):s+=path(f'M{794+i*12} {333-i*9}Q{830+i*25} {245-i*24} {896+i*29} {207-i*3}',C,1.4,extra=f'class="flow" style="animation-delay:-{i}s" opacity="{.8-i*.12}"')
for x,y in [(803,157),(1008,106),(1137,177),(930,236),(930,391),(803,314),(1137,324)]:s+=circle(x,y,3,extra='fill="#72eee0" class="breathe"')
s+=text(1148,419,'CONCEPTUAL RF FIELD',10,MUTED,'end',extra='class="mono"')
save('hero',1200,450,'RuView: spaces become signals','Animated conceptual RF field and room geometry. Explore the RuView user guide. This illustration is not a sensor measurement.',s,'docs/user-guide.md')
# Visual navigation icons, each receives a real outer link in Markdown.
nav=[('start','Start here','docs/user-guide.md'),('hardware','Hardware','firmware/esp32-csi-node/README.md'),('models','Models','https://huggingface.co/ruvnet/wifi-densepose-pretrained'),('agents','AI toolkit','harness/ruview/README.md'),('home','Smart home','docs/integrations/home-assistant.md'),('seed','Cognitum','https://cognitum.one/seed')]
for k,label,target in nav:save('nav-'+k,180,116,label,'Open '+label+'. Animated navigation icon.',icon(k,90,45,.65)+text(90,98,label,16,WHITE,'middle'),target)
# Capability cards. No unverified performance claims.
cards=[('presence','01','Presence & motion','Explore changes in room occupancy.','Inspect motion features from CSI.','SENSING GUIDE','docs/user-guide.md'),('vitals','02','Breathing & vital signals','Inspect phase-derived signal trends.','Validate estimates on your hardware.','SIGNAL PIPELINE','v2/crates/wifi-densepose-signal/README.md'),('pose','03','Pose research','Train and compare body keypoints.','Live accuracy differs from benchmarks.','READ THE EVIDENCE','docs/benchmarks/pose-estimation-cog.md'),('edge','04','Edge sensor mesh','Provision, capture and calibrate.','Explore ESP32 firmware and runbooks.','HARDWARE GUIDE','firmware/esp32-csi-node/README.md'),('automation','05','Local automation','Connect rooms to your smart home.','Home Assistant, MQTT and Matter.','INTEGRATION GUIDE','docs/integrations/home-assistant.md'),('agents','06','A guided AI operator','Source-cited setup and verification.','CLI, MCP and agent workflows.','TOOLKIT GUIDE','harness/ruview/README.md')]
for k,num,title,l1,l2,cta,target in cards:
 s=text(30,39,'RUVIEW / '+num,11,C,extra='class="mono"')+text(30,94,title,29,WHITE)+text(30,136,l1,16)+text(30,164,l2,16)
 s+='<circle cx="465" cy="170" r="110" fill="url(#halo)"/>'+spatial(480,161,.53)+icon(k,474,155,.68)+path('M30 204H550','#245360',1)+path('M30 204H550',C,1,extra='class="flow" opacity=".5"')
 s+=text(30,245,cta,12,C,extra='class="mono"')+path('M510 239H540M532 231L540 239L532 247',C,1.5)
 save('feature-'+k,580,276,title,l1+' '+l2+' Conceptual animation; follow the linked source for capability limits.',s,target)
# Overview uses three clearly labelled stages rather than fabricated telemetry.
s=text(36,44,'SIGNAL TO CONTEXT',12,C,extra='class="mono"')+text(36,89,'A sensing loop you can inspect.',34,WHITE)
for i,(label,sub,kind,tail) in enumerate([('Capture RF','CSI from supported hardware','signal','Provision and calibrate'),('Extract context','Signal processing and models','models','Compare against real data'),('Act locally','Events, APIs and integrations','automation','Apply rules and review outputs')]):
 x=36+i*396;s+=f'<rect x="{x}" y="120" width="336" height="220" rx="14" fill="#071b23" stroke="#285360"/>'
 s+=text(x+24,153,f'0{i+1}',12,C)+text(x+24,188,label,26,WHITE)+text(x+24,216,sub,15)+spatial(x+269,278,.35)+icon(kind,x+269,270,.5)+text(x+24,311,tail,13)
 if i<2:s+=path(f'M{x+344} 229H{x+384}M{x+376} 221L{x+384} 229L{x+376} 237',C,2,extra='class="flow"')
s+=text(36,373,'CONCEPTUAL PIPELINE / HARDWARE AND ENVIRONMENT VALIDATION MATTER',11,MUTED,extra='class="mono"')
save('overview',1200,396,'RuView sensing overview','Conceptual sequence: capture RF, extract context, act locally. Links to architecture and user documentation.',s,'docs/user-guide.md')
# Chapter headings act as visual landmarks, all remain links.
headers=[('features','Explore the capabilities','SENSING / MODELS / AUTOMATION','presence','docs/user-guide.md'),('edge','Intelligence at the edge','SENSORS / FIRMWARE / MODULES','edge','firmware/esp32-csi-node/README.md'),('learning','From recordings to models','CAPTURE / TRAIN / EVALUATE','models','docs/user-guide.md'),('evidence','Evidence before accuracy claims','BENCHMARKS / REPRODUCERS / LIMITS','evidence','docs/benchmarks/pose-estimation-cog.md'),('docs','Choose your next step','GUIDES / HARDWARE / INTEGRATIONS','agents','docs/user-guide.md')]
for k,title,sub,kind,target in headers:
 s=icon(kind,76,75,.75)+text(140,63,title,34,WHITE)+text(142,99,sub,12,C,extra='class="mono"')
 s+=spatial(1012,84,.55)
 s+=path('M1140 70H1160M1152 62L1160 70L1152 78',C,2)
 save('header-'+k,1200,148,title,'Animated section heading. '+sub,s,target)
# Brand destination; original ruOS promotion remains in the README.
s=text(44,42,'COGNITUM ONE',12,C,extra='class="mono"')+text(44,99,'Ambient Intelligence.',42,WHITE)+text(44,147,'At the edge of the physical world.',25,C)+text(44,190,'Explore the platform behind the ecosystem.',16)
s+='<rect x="44" y="224" width="256" height="46" rx="11" fill="url(#accent)"/>'+text(65,254,'Explore Cognitum',18,INK)+path('M263 247H284M277 240L284 247L277 254',INK,2)
s+='<circle cx="947" cy="151" r="173" fill="url(#halo)"/>'+spatial(945,173,1.55)+icon('seed',945,145,1.65)
for j in range(3):s+=f'<ellipse cx="945" cy="150" rx="{140+j*20}" ry="{30+j*13}" transform="rotate(-25 945 150)" fill="none" stroke="#3c8897" stroke-dasharray="5 18" class="flow"/>'
save('cognitum',1200,304,'Explore Cognitum One','Ambient Intelligence at the edge of the physical world. Open cognitum.one.',s,'https://cognitum.one')
(OUT/'manifest.json').write_bytes((json.dumps({'schema_version':1,'style':'RuView navy, electric blue, lime and amber','motion':'Decorative CSS; prefers-reduced-motion supported; no script or remote dependencies','assets':manifest},indent=2)+'\n').encode('utf-8'))
print(f'Generated {len(manifest)} SVGs, {sum((ROOT/x["file"]).stat().st_size for x in manifest):,} bytes')
