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
    "1. If user's message fulfills an awaiting task, resume that task or complete it using the done tool.",
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
  `- If already satisfied or incorporated into previous answers: dismiss notification "${notItem.id}" using the ack tool.`,
  `- If pending work remains: address or continue it now, then dismiss notification "${notItem.id}" using the ack tool.`,
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
    lines.push(`Required action: Inspect full content using the read tool with path "${notItem.source}" and offset 0, defer using the snooze tool, or dismiss using the ack tool.`)
  } else {
    lines.push(`Evaluate interrupt: execute immediate action, defer using the snooze tool, or dismiss notification "${notItem.id}" using the ack tool.`)
  }
  lines.push('Next action:')
  return lines.join('\n') + '\n'
}

const formatTruncatedTurn = (userText, eventId, relPath) => [
  `<|turn>user\n${userText}\n<turn|>\n<|turn>model\n<|channel>thought\nNotice: Event ${eventId} payload is truncated.\nRequired action: Inspect full content using the read tool with path "${relPath}" and offset 0 before answering.\nNext action:\n`,
].join('\n')

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
}
