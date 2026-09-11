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
  `- If already satisfied or incorporated into previous answers: continue active work.`,
  `- If pending work remains: address or continue it now.`,
  'If addressing user, exit thought channel with <channel|> and speak directly to the user.',
  'Next action:',
].join('\n') + '\n'

const formatNotificationInterrupt = (notItem) => {
  const lines = [
    '<|turn>model',
    '<|channel>thought',
    `[Interrupt Event: ${notItem.id} | Source: ${notItem.source}]`,
    `Payload: ${notItem.preview}`,
    'Evaluate interrupt in reasoning thoughts.',
    'If addressing or acknowledging the user, exit thought channel with <channel|> and deliver your response directly to the user.',
    'Next action:',
  ]
  return lines.join('\n') + '\n'
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
