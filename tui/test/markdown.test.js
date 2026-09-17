const { describe, it } = require('node:test')
const assert = require('node:assert')
const { MarkdownControl, mapBlocks } = require('../lib/ui/markdown')
const { parse } = require('@sullux/markdown-compiler')

describe('MarkdownControl', () => {
  it('maps headings, paragraphs, lists, and code blocks into styled spans', () => {
    const md = [
      '# Heading 1',
      '## Heading 2',
      'Regular paragraph with **bold**, *italic*, and `code`.',
      '- Bullet A',
      '- Bullet B',
      '1. Step 1',
      '2. Step 2',
      '```js',
      'const a = 10',
      '```',
      '> Quoted insight',
    ].join('\n\n')

    const ast = parse(md)
    const spans = mapBlocks(ast.blocks, {
      fg: '#ffffff',
      h1: { bold: true, underline: true },
      code: { bg: '#1f2335' },
    })

    assert.ok(spans.length > 0)
    const boldSpan = spans.find((s) => s.text === 'bold')
    assert.ok(boldSpan)
    assert.strictEqual(boldSpan.bold, true)

    const italicSpan = spans.find((s) => s.text === 'italic')
    assert.ok(italicSpan)
    assert.strictEqual(italicSpan.italic, true)

    const codeSpan = spans.find((s) => s.text === ' code ')
    assert.ok(codeSpan)
    assert.strictEqual(codeSpan.bg, '#1f2335')

    const h1Span = spans.find((s) => s.text === 'Heading 1')
    assert.ok(h1Span)
    assert.strictEqual(h1Span.bold, true)
    assert.strictEqual(h1Span.underline, true)
  })

  it('measures, layouts, and renders through MarkdownControl contract', () => {
    const node = {
      type: 'markdown',
      text: '# Test Header\n\nHere is a list:\n- Item 1\n- Item 2',
      fg: '#ffffff',
    }
    const bounds = { x: 0, y: 0, width: 80, height: 20 }
    const measured = MarkdownControl.onMeasure(node, bounds)
    assert.ok(measured.width > 0)
    assert.ok(measured.height > 0)

    MarkdownControl.onLayout(node, bounds)
    const grid = Array.from({ length: 20 }, () =>
      Array.from({ length: 80 }, () => ({ char: ' ', style: '' })),
    )
    MarkdownControl.onRender(node, grid)

    const firstLine = grid[0].map((c) => c.char).join('')
    assert.ok(firstLine.includes('Test Header'))
  })
})
