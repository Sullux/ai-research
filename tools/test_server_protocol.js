import { spawn } from 'child_process';

const MAGIC = 0x53554C58;
const VERSION = 1;

const OP_STREAM_INPUT = 0x0001;
const OP_ABORT = 0x0002;
const OP_MEM_QUERY = 0x0003;
const OP_SET_CONFIG = 0x0004;
const OP_PING = 0x000E;
const OP_SHUTDOWN = 0x000F;

const OP_STREAM_CONTENT = 0x0101;
const OP_STREAM_THOUGHT = 0x0102;
const OP_TURN_COMPLETE = 0x0103;
const OP_MEM_RESPONSE = 0x0105;
const OP_STATUS = 0x0106;
const OP_PONG = 0x010E;
const OP_ERROR = 0x01FF;

function makeHeader(opcode, msgId, payloadLen) {
  const buf = Buffer.alloc(16);
  buf.writeUInt32LE(MAGIC, 0);
  buf.writeUInt16LE(VERSION, 4);
  buf.writeUInt16LE(msgId, 6);
  buf.writeUInt16LE(opcode, 8);
  buf.writeUInt16LE(0, 10);
  buf.writeUInt32LE(payloadLen, 12);
  return buf;
}

function makeStreamInputFrame(text, msgId = 1) {
  const textBytes = Buffer.from(text, 'utf-8');
  const payload = Buffer.alloc(8 + textBytes.length);
  payload.writeUInt8(0x00, 0); // mode = text
  payload.writeUInt8(0x00, 1);
  payload.writeUInt16LE(0, 2);
  payload.writeUInt32LE(0, 4);
  textBytes.copy(payload, 8);
  const hdr = makeHeader(OP_STREAM_INPUT, msgId, payload.length);
  return Buffer.concat([hdr, payload]);
}

function makePingFrame(msgId = 42) {
  return makeHeader(OP_PING, msgId, 0);
}

function makeShutdownFrame() {
  return makeHeader(OP_SHUTDOWN, 0, 0);
}

function makeSetConfigFrame(thinkingBudget = 256, temp = 1.0, topP = 0.95, qThresh = 0.001) {
  const payload = Buffer.alloc(16);
  payload.writeUInt32LE(thinkingBudget, 0);
  payload.writeFloatLE(temp, 4);
  payload.writeFloatLE(topP, 8);
  payload.writeFloatLE(qThresh, 12);
  const hdr = makeHeader(OP_SET_CONFIG, 0, 16);
  return Buffer.concat([hdr, payload]);
}

function makeMemQueryFrame(queryText, msgId = 100) {
  const queryBytes = Buffer.from(queryText, 'utf-8');
  const payload = Buffer.alloc(24 + queryBytes.length);
  payload.writeUInt8(0x00, 0); // mode = keywords
  payload.writeUInt8(5, 1); // top_k = 5
  payload.writeUInt16LE(0, 2); // cursor
  payload.writeBigUInt64LE(0n, 4); // timestamp
  payload.writeFloatLE(1.0, 12);
  payload.writeFloatLE(0.5, 16);
  payload.writeFloatLE(1.0, 20);
  queryBytes.copy(payload, 24);
  const hdr = makeHeader(OP_MEM_QUERY, msgId, payload.length);
  return Buffer.concat([hdr, payload]);
}

async function runProtocolTest() {
  const modelDir = process.argv[2] || '../gemma-4-E2B';
  const extraArgs = process.argv.slice(3);
  const args = ['--model', modelDir, '--serve', ...extraArgs];
  console.log(`Spawning ./zig-out/bin/infer ${args.join(' ')} ...`);
  const proc = spawn('./zig-out/bin/infer', args, {
    stdio: ['pipe', 'pipe', 'inherit'],
  });

  let rxBuffer = Buffer.alloc(0);
  const receivedFrames = [];

  proc.stdout.on('data', (chunk) => {
    rxBuffer = Buffer.concat([rxBuffer, chunk]);
    while (rxBuffer.length >= 16) {
      const magic = rxBuffer.readUInt32LE(0);
      if (magic !== MAGIC) {
        console.error('Invalid magic received:', magic.toString(16));
        proc.kill();
        process.exit(1);
      }
      const version = rxBuffer.readUInt16LE(4);
      const msgId = rxBuffer.readUInt16LE(6);
      const opcode = rxBuffer.readUInt16LE(8);
      const payloadLen = rxBuffer.readUInt32LE(12);

      if (rxBuffer.length < 16 + payloadLen) break; // wait for full payload

      const payload = rxBuffer.subarray(16, 16 + payloadLen);
      rxBuffer = rxBuffer.subarray(16 + payloadLen);

      receivedFrames.push({ opcode, msgId, payload });
      onFrameReceived({ opcode, msgId, payload });
    }
  });

  let pongReceived = false;
  let turnCompleteReceived = false;
  let memResponseReceived = false;
  let generatedTokens = [];

  function onFrameReceived(frame) {
    switch (frame.opcode) {
      case OP_PONG:
        console.log(`✓ OP_PONG received (msgId: ${frame.msgId})`);
        pongReceived = true;
        break;
      case OP_STREAM_CONTENT:
      case OP_STREAM_THOUGHT:
        const tokId = frame.payload.readUInt32LE(0);
        const tokType = frame.payload.readUInt8(20);
        const textLen = frame.payload.readUInt16LE(22);
        const text = frame.payload.toString('utf-8', 24, 24 + textLen);
        generatedTokens.push(text);
        process.stdout.write(text);
        break;
      case OP_TURN_COMPLETE:
        const totalTok = frame.payload.readUInt32LE(0);
        const elapsedMs = frame.payload.readUInt32LE(4);
        const tokSec = frame.payload.readFloatLE(8);
        const reason = frame.payload.readUInt8(12);
        console.log(`\n✓ OP_TURN_COMPLETE: ${totalTok} tokens in ${elapsedMs}ms (${tokSec.toFixed(1)} tok/s, reason: ${reason})`);
        turnCompleteReceived = true;
        break;
      case OP_MEM_RESPONSE:
        const count = frame.payload.readUInt8(0);
        const status = frame.payload.readUInt8(1);
        console.log(`✓ OP_MEM_RESPONSE: count=${count}, status=${status}`);
        memResponseReceived = true;
        break;
      case OP_ERROR:
        console.error(`✗ OP_ERROR:`, frame.payload.toString('utf-8'));
        break;
    }
  }

  // 1. Send PING
  console.log('Sending OP_PING...');
  proc.stdin.write(makePingFrame(101));
  await new Promise((r) => setTimeout(r, 200));

  // 2. Send SET_CONFIG
  console.log('Sending OP_SET_CONFIG...');
  proc.stdin.write(makeSetConfigFrame(128, 1.0, 0.95, 0.001));

  // 3. Send STREAM_INPUT
  console.log('\nSending OP_STREAM_INPUT ("What is 2 + 2?")...');
  proc.stdin.write(makeStreamInputFrame('What is 2 + 2?', 1));

  // Wait for turn completion
  const start = Date.now();
  while (!turnCompleteReceived && Date.now() - start < 15000) {
    await new Promise((r) => setTimeout(r, 100));
  }

  // 4. Send MEM_QUERY
  console.log('\nSending OP_MEM_QUERY ("arithmetic addition")...');
  proc.stdin.write(makeMemQueryFrame('arithmetic addition', 200));
  await new Promise((r) => setTimeout(r, 300));

  // 5. Send SHUTDOWN
  console.log('Sending OP_SHUTDOWN...');
  proc.stdin.write(makeShutdownFrame());

  await new Promise((r) => proc.on('close', r));
  console.log('✓ Engine shutdown cleanly with code:', proc.exitCode);

  if (!pongReceived) throw new Error('OP_PONG not received');
  if (!turnCompleteReceived) throw new Error('OP_TURN_COMPLETE not received');
  if (!memResponseReceived) throw new Error('OP_MEM_RESPONSE not received');

  console.log('\n🎉 ALL PROTOCOL VERIFICATION TESTS PASSED!');
}

runProtocolTest().catch((err) => {
  console.error('Test failed:', err);
  process.exit(1);
});
