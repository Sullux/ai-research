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

const formatBacklogResumeNudge = (notItem) => {
  const text = notItem.extra?.payload || notItem.preview || ''
  if (notItem.source?.startsWith('msg/user/')) {
    return `<|turn>user\n[Resume: ${notItem.id} | Source: ${notItem.source}] ${text}\nPlease continue addressing this task.\n<turn|>\n<|turn>model\n`
  }
  return `<|turn>user\n[Resume Event: ${notItem.id} | Source: ${notItem.source}] ${text}\nPlease continue addressing this item.\n<turn|>\n<|turn>model\n`
}

const formatNotificationInterrupt = (notItem) => {
  const text = notItem.extra?.payload || notItem.preview || ''
  if (notItem.source?.startsWith('msg/user/')) {
    return `<|turn>user\n${text}\n<turn|>\n<|turn>model\n`
  }
  return `<|turn>user\n[Event: ${notItem.id} | Source: ${notItem.source}]\n${text}\n<turn|>\n<|turn>model\n`
}

const formatTruncatedTurn = (userText) =>
  `<|turn>user\n${userText}\n<turn|>\n<|turn>model\n`

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
