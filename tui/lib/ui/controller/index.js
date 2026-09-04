const { init, refs } = require('./state')
const {
  getLayoutTier,
  isClipped,
  getClippedBanner,
  getStatusText,
  getShortcutsText,
} = require('./layout')
const { getConversationNodes, getConversationScroll } = require('./conversation')
const { getStreamNodes, getStreamScroll } = require('./stream')
const { getPlanNodes, getPlanScroll } = require('./plan')
const { onSubmitInput } = require('./input')
const { onGlobalKey } = require('./keyboard')

module.exports = {
  init,
  refs,
  getLayoutTier,
  isClipped,
  getClippedBanner,
  getStatusText,
  getShortcutsText,
  getConversationNodes,
  getConversationScroll,
  getStreamNodes,
  getStreamScroll,
  getPlanNodes,
  getPlanScroll,
  onSubmitInput,
  onGlobalKey,
}
