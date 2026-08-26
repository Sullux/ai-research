import { spawn } from 'child_process';

const MAGIC = 0x53554C58;
const OP_STREAM_INPUT = 0x0001;
const OP_SET_CONFIG = 0x0004;
const OP_SHUTDOWN = 0x000F;
const OP_STREAM_STATUS = 0x0100;
const OP_STREAM_CONTENT = 0x0101;
const OP_STREAM_THOUGHT = 0x0102;
const OP_TURN_COMPLETE = 0x0103;

function makeHeader(opcode, msgId, payloadLen) {
  const buf = Buffer.alloc(16);
  buf.writeUInt32LE(MAGIC, 0);
  buf.writeUInt16LE(1, 4);
  buf.writeUInt16LE(msgId, 6);
  buf.writeUInt16LE(opcode, 8);
  buf.writeUInt16LE(0, 10);
  buf.writeUInt32LE(payloadLen, 12);
  return buf;
}

function makeStreamInputFrame(text, msgId = 1) {
  const textBytes = Buffer.from(text, 'utf-8');
  const payload = Buffer.alloc(8 + textBytes.length);
  payload.writeUInt8(0x00, 0);
  textBytes.copy(payload, 8);
  const hdr = makeHeader(OP_STREAM_INPUT, msgId, payload.length);
  return Buffer.concat([hdr, payload]);
}

const proc = spawn('./zig-out/bin/infer', ['--model', '../gemma-4-12B-it-qat-q4_0-unquantized', '--serve', '--gpu', '--q4'], {
  stdio: ['pipe', 'pipe', 'inherit'],
});

let rx = Buffer.alloc(0);
let promptStart = 0;
let firstTokenTime = 0;
let tokenCount = 0;

proc.stdout.on('data', (chunk) => {
  rx = Buffer.concat([rx, chunk]);
  while (rx.length >= 16) {
    const magic = rx.readUInt32LE(0);
    if (magic !== MAGIC) break;
    const opcode = rx.readUInt16LE(8);
    const payloadLen = rx.readUInt32LE(12);
    if (rx.length < 16 + payloadLen) break;

    const payload = rx.subarray(16, 16 + payloadLen);
    rx = rx.subarray(16 + payloadLen);

    if (opcode === OP_STREAM_STATUS) {
      const status = payload.readUInt8(0);
      const tokSec = payload.readFloatLE(4);
      const curTok = payload.readUInt32LE(16);
      const totalTok = payload.readUInt32LE(20);
      console.log(`[STATUS] code=${status} speed=${tokSec.toFixed(1)} tok/s, progress=${curTok}/${totalTok}`);
    } else if (opcode === OP_STREAM_THOUGHT) {
      if (firstTokenTime === 0) {
        firstTokenTime = Date.now();
        console.log(`[FIRST TOKEN] TTFT: ${firstTokenTime - promptStart} ms`);
      }
      tokenCount++;
      const t = payload.subarray(24).toString('utf-8');
      process.stdout.write(`\x1b[33m${t}\x1b[0m`);
    } else if (opcode === OP_STREAM_CONTENT) {
      if (firstTokenTime === 0) {
        firstTokenTime = Date.now();
        console.log(`[FIRST TOKEN] TTFT: ${firstTokenTime - promptStart} ms`);
      }
      tokenCount++;
      const t = payload.subarray(24).toString('utf-8');
      process.stdout.write(`\x1b[32m${t}\x1b[0m`);
    } else if (opcode === OP_TURN_COMPLETE) {
      const totalTime = Date.now() - firstTokenTime;
      console.log(`\n[COMPLETE] ${tokenCount} decode tokens in ${totalTime} ms (${(tokenCount / (totalTime / 1000)).toFixed(2)} tok/s)`);
      proc.stdin.write(makeHeader(OP_SHUTDOWN, 99, 0));
    }
  }
});

// Wait 4s for model to load, then send prompt
setTimeout(() => {
  console.log('[SENDING PROMPT]');
  promptStart = Date.now();
  const prompt = '<|turn>user\nHow are you today?<turn|>\n<|turn>model\n<|channel>thought\n';
  proc.stdin.write(makeStreamInputFrame(prompt));
}, 6000);
