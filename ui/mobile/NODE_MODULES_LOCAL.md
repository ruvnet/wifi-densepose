# Local node_modules note

On the Steam Deck, node_modules is kept off /home to avoid inode exhaustion.

If dependencies appear missing after reboot, remount or recreate the SD-card-backed node_modules location before running:

npm install
npm test
npm run lint
./node_modules/.bin/tsc --noEmit
