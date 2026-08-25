#!/usr/bin/env node
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const constants = require('../tui/lib/protocol/constants');
const framing = require('../tui/lib/protocol/framing');

const modelPath = process.argv[2] || '../gemma-4-12B-it';
const quantFlag = process.argv[3] || '--q4';
const binPath = path.resolve(__dirname, '../zig-out/bin/infer');
const child = spawn(binPath, ['--model', modelPath, '--serve', '--gpu', quantFlag], {
  stdio: ['pipe', 'pipe', 'inherit'],
});

let rxBuffer = Buffer.alloc(0);
let currentTurn = 1;

child.stdout.on('data', (chunk) => {
  rxBuffer = Buffer.concat([rxBuffer, chunk]);
  while (true) {
    const frame = framing.parsedFrame(rxBuffer);
    if (!frame) break;
    rxBuffer = rxBuffer.subarray(16 + frame.header.payloadLen);

    if (frame.header.opcode === constants.OP_STREAM_THOUGHT) {
      const text = frame.payload.subarray(24).toString('utf-8').replace(/\u2581/g, ' ');
      process.stdout.write(`\x1b[33m${text}\x1b[0m`);
    } else if (frame.header.opcode === constants.OP_STREAM_CONTENT) {
      const text = frame.payload.subarray(24).toString('utf-8').replace(/\u2581/g, ' ');
      process.stdout.write(`\x1b[32m${text}\x1b[0m`);
    } else if (frame.header.opcode === constants.OP_TURN_COMPLETE) {
      console.log(`\n\n\x1b[36m[Turn ${currentTurn} Complete]\x1b[0m\n`);
      if (currentTurn === 1) {
        currentTurn = 2;
        sendTurn2();
      } else {
        const shutdownHdr = framing.headerBuffer(constants.OP_SHUTDOWN, 999, 0);
        child.stdin.write(shutdownHdr);
      }
    }
  }
});

function makeConfigFrame(maxTokens = 128, temp = 0.0) {
  const payload = Buffer.alloc(20);
  payload.writeUInt32LE(512, 0);
  payload.writeFloatLE(temp, 4);
  payload.writeFloatLE(0.95, 8);
  payload.writeFloatLE(0.0, 12);
  payload.writeUInt32LE(maxTokens, 16);
  const hdr = framing.headerBuffer(constants.OP_SET_CONFIG, 1, payload.length);
  return Buffer.concat([hdr, payload]);
}

const kernelPath = path.resolve(__dirname, '../tui/PROMPT_KERNEL.md');
const kernel = fs.existsSync(kernelPath) ? fs.readFileSync(kernelPath, 'utf-8').trim() : '';

function sendTurn1() {
  console.log('\x1b[34m--- SENDING TURN 1: "How are you doing today?" ---\x1b[0m');
  const prompt = `<|turn>system\n<|think|>\n${kernel}<turn|>\n<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n`;
  child.stdin.write(makeConfigFrame(512, 0.0));
  child.stdin.write(framing.streamInputFrame(prompt, 1));
}

function sendTurn2() {
  console.log('\x1b[34m--- SENDING TURN 2: "What tools do you see that you have available?" ---\x1b[0m');
  const prompt = `<|turn>user\nIn your context, what tools do you see that you have available?<turn|>\n<|turn>model\n`;
  child.stdin.write(makeConfigFrame(512, 0.0));
  child.stdin.write(framing.streamInputFrame(prompt, 2));
}

sendTurn1();
