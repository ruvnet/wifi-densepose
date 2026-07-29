import test from 'node:test';
import assert from 'node:assert/strict';
import { booleanFlag, run, validSkillName } from '../bin/cli.js';

test('skill lookup rejects traversal and only accepts bounded slugs', async () => {
  assert.equal(validSkillName('secure-plugin'), true);
  assert.equal(validSkillName('../README'), false);
  assert.equal(validSkillName('..\\README'), false);
  assert.equal(validSkillName('a'.repeat(65)), false);

  const original = console.error;
  console.error = () => {};
  try {
    assert.equal(await run(['skill', '../../../README']), 2);
    assert.equal(await run(['skill', '..\\..\\README']), 2);
  } finally {
    console.error = original;
  }
});

test('security-relevant boolean flags accept explicit true/false without silent downgrade', () => {
  assert.equal(booleanFlag(true, '--strict'), true);
  assert.equal(booleanFlag('true', '--strict'), true);
  assert.equal(booleanFlag('false', '--strict'), false);
  assert.throws(() => booleanFlag('yes', '--strict'), /bare flag, true, or false/);
});
