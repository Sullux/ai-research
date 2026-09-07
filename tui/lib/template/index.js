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
    "1. If user's message fulfills an awaiting task, resume that task or call tool `done`.",
    '2. If user provided a new unrelated instruction, prioritize answering/planning it.',
    '3. If user cancelled a task, mark it complete/aborted.',
    'Next action:',
  ].join('\n') + '\n'
}

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

const formatBacklogResumeNudge = (notItem) => [
  '<|turn>model',
  '<|channel>thought',
  `Resuming previous context [Event: ${notItem.id} | Source: ${notItem.source}].`,
  `- If already satisfied or incorporated into previous answers: dismiss by calling tool \`ack\` with id "${notItem.id}".`,
  `- If pending work remains: address or continue it now, then call tool \`ack\` with id "${notItem.id}".`,
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
    lines.push(`Required action: You must call tool \`read\` with path "${notItem.source}" and offset 0 to inspect the full content, tool \`snooze\` to defer, or tool \`ack\` to dismiss before proceeding.`)
  } else {
    lines.push(`Evaluate interrupt: execute immediate action, call tool \`snooze\` with id "${notItem.id}" to defer, or call tool \`ack\` with id "${notItem.id}" when addressed.`)
  }
  lines.push('Next action:')
  return lines.join('\n') + '\n'
}

const formatTruncatedTurn = (userText, eventId, relPath) => [
  `<|turn>user\n${userText}\n<turn|>\n<|turn>model\n<|channel>thought\nNotice: Event ${eventId} payload is truncated.\nRequired action: Call tool \`read\` with path "${relPath}" and offset 0 to inspect before answering.\nNext action:\n`,
].join('\n')

const formatServicingCompletionNudge = (notItem) => [
  '<|turn>model',
  '<|channel>thought',
  `Notice: Notification "${notItem.id}" remains active.`,
  `- Call tool \`ack\` with id "${notItem.id}" to dismiss, or tool \`snooze\` with id "${notItem.id}" to defer.`,
  'Next action:',
].join('\n') + '\n'

module.exports = {
  formatTurn1,
  formatUserTurn,
  formatUserDecisionTurn,
  formatTruncatedTurn,
  formatStepTick,
  formatTimerWake,
  formatResumeAfterInterrupt,
  formatBacklogResumeNudge,
  formatNotificationInterrupt,
  formatServicingCompletionNudge,
}
