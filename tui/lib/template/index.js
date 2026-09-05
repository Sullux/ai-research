const formatTurn1 = (systemPrompt, userText) =>
  `<|turn>system\n<|think|>\n${systemPrompt}\n<turn|>\n<|turn>user\n${userText}\n<turn|>\n<|turn>model\n`

const formatUserTurn = (userText) =>
  `<|turn>user\n${userText}\n<turn|>\n<|turn>model\n`

const formatUserDecisionTurn = (userText, waitingTasks = []) => {
  const waitingLines = waitingTasks
    .map((w) => `- Step ${w.id}: ${w.waitingForUser?.brief || w.brief}`)
    .join('\n')

  return [
    `<|turn>user\n${userText}\n<turn|>\n<|turn>model\n<|channel>thought\n`,
    `User message received: "${userText}"`,
    '',
    'Tasks currently awaiting user intervention:',
    waitingLines,
    '',
    'Decision:',
    "1. If user's message fulfills an awaiting task, resume that task or call `done()`.",
    '2. If user provided a new unrelated instruction, prioritize answering/planning it.',
    '3. If user cancelled a task, mark it complete/aborted.',
    'Next action:',
  ].join('\n') + '\n'
}

const formatToolResponse = (toolName, result) =>
  `<|tool_response>response:${toolName}${JSON.stringify(result)}<tool_response|>\n<|turn>model\n<|channel>thought\n`

const formatStepTick = (planId, planBrief, stepId, stepBrief) => [
  '<|turn>model',
  '<|channel>thought',
  `Focus: Plan ${planId || ''} - ${planBrief || ''}`,
  `Active Step ${stepId}: ${stepBrief}`,
  'Status: In progress.',
  'Next action:',
].join('\n') + '\n'

const formatTimerWake = (stepId, reason) => [
  '<|turn>model',
  '<|channel>thought',
  `Timer expired for Step ${stepId || ''} (${reason || 'timer'}). Checking state for updates.`,
  'Next action:',
].join('\n') + '\n'

const formatResumeAfterInterrupt = (stepId, brief) => [
  '<|turn>model',
  '<|channel>thought',
  `Interruption handled. Automatically resuming Step ${stepId}: ${brief}.`,
  'Previous context remains active in episodic memory.',
  'Next action:',
].join('\n') + '\n'

const formatNotificationInterrupt = (notItem) => {
  const isTruncated = Boolean(notItem.extra?.isTruncated || notItem.preview?.endsWith('...'))
  const lines = [
    '<|turn>model',
    '<|channel>thought',
    `[Interrupt Event: ${notItem.id} | Source: ${notItem.source}]`,
    `Payload: ${notItem.preview}`,
  ]
  if (isTruncated) {
    lines.push(`Input event ${notItem.id} is truncated.`)
    lines.push(`Required action: You must use read with path "${notItem.source}" and offset 0 to inspect the full content, snooze to defer, or ack to dismiss before proceeding.`)
  } else {
    lines.push('Evaluate interrupt: execute immediate action, snooze to defer, or ack when addressed.')
  }
  lines.push('Next action:')
  return lines.join('\n') + '\n'
}

const formatContinuationNudge = () => [
  '<|turn>model',
  '',
].join('\n')

const formatTruncatedTurn = (userText, eventId, relPath) => [
  `<|turn>user\n${userText}\n<turn|>\n<|turn>model\n<|channel>thought\n`,
  `Input event ${eventId} is truncated.`,
  `Required action: You must use read with path "${relPath}" and offset 0 to inspect the full content, snooze to defer, or ack to dismiss before delivering a final answer.`,
  'Next action:\n<|tool_call>',
].join('\n')

module.exports = {
  formatTurn1,
  formatUserTurn,
  formatUserDecisionTurn,
  formatTruncatedTurn,
  formatToolResponse,
  formatStepTick,
  formatTimerWake,
  formatResumeAfterInterrupt,
  formatNotificationInterrupt,
  formatContinuationNudge,
}
