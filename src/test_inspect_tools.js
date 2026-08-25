import fs from 'node:fs'
import { TOOLS } from '../tui/lib/tools/index.js'

// Let's inspect the tools defined in tui/lib/tools/index.js
console.log('Tools count:', Object.keys(TOOLS).length)
for (const [name, def] of Object.entries(TOOLS)) {
  console.log('Tool:', name, JSON.stringify(def))
}
