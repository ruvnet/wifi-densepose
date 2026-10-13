import {trainExperiment,loadArtifact,rank} from './ranker.mjs';
try {
  const args=process.argv.slice(2), opts={};
  if(args.length%2 || args.length>8) throw new Error('Expected --train DIR or --artifact DIR, --context ID, optional --k 1..3');
  for(let i=0;i<args.length;i+=2) {
    if(!['--train','--artifact','--context','--k'].includes(args[i])||opts[args[i]]!==undefined||!args[i+1]||args[i+1].length>4096) throw new Error('Invalid arguments');
    opts[args[i]]=args[i+1];
  }
  if(Boolean(opts['--train'])===Boolean(opts['--artifact'])) throw new Error('Select exactly one train or artifact mode');
  const model=opts['--train']?(await trainExperiment(opts['--train'])).model:loadArtifact(opts['--artifact']);
  const result=rank(model,opts['--context']??'context_0_8',opts['--k']===undefined?3:Number(opts['--k']));
  process.stdout.write(JSON.stringify(result)+'\n');
} catch(error) {process.stderr.write(String(error.message)+'\n');process.exitCode=1;}
