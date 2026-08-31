const { refs } = require('./state')

const getPlanScroll = () => 0

const getPlanNodes = () => {
  if (!refs.orchestrator) return []
  const plans = refs.orchestrator.plans
  if (plans.length === 0) {
    return [
      {
        type: 'text',
        text: '  (No active plans. Model idle.)',
        fg: '#565f89',
        italic: true,
      },
    ]
  }

  const nodes = []
  for (const p of plans) {
    const isPlanDone = p.status === 'DONE'
    const planIcon = isPlanDone ? '✅ ' : '📋 '
    nodes.push({
      type: 'text',
      text: `${planIcon}Plan ${p.id}: ${p.brief}\n`,
      bold: true,
      fg: isPlanDone ? '#9ece6a' : '#7aa2f7',
      margin: { top: 1, bottom: 0 },
    })

    for (const s of p.steps) {
      let icon = '⬜ '
      let fg = '#9aa5ce'
      let tag = ''

      if (s.status === 'DONE') {
        icon = '✅ '
        fg = '#73daca'
      } else if (s.status === 'IN_PROGRESS') {
        icon = '▶ '
        fg = '#e0af68'
        tag = ' [ACTIVE]'
      } else if (s.status === 'WAITING_FOR_USER') {
        icon = '⏳ '
        fg = '#f7768e'
        tag = ' [WAITING]'
      } else if (s.status === 'DEFERRED') {
        icon = '⏱️ '
        fg = '#bb9af7'
        tag = ` [${s.deferReason || 'timer'}]`
      }

      nodes.push({
        type: 'text',
        text: `   ${icon}${s.id}: ${s.brief}${tag}\n`,
        fg,
      })
    }
  }

  return nodes
}

module.exports = {
  getPlanScroll,
  getPlanNodes,
}
