#!/usr/bin/env node
/**
 * 把仓库 study/notes/ 下的学习笔记同步成 Hexo 文章，落到博客源目录。
 *
 * 单一事实来源是 git 仓库（study 分支、study/notes/*.md），博客只做发布面。
 * 不要在博客后台/本地改这些文章——下次同步会被覆盖。
 *
 *   node study/hexo/sync.mjs --dest <hexo-blog>/source/_posts/agentic-rl
 *   node study/hexo/sync.mjs --dest ./study/.hexo-out --dry-run   # 只看产物
 *
 * 产物：<dest>/YYYY-MM-DD-<slug>.md，frontmatter 带 permalink → /agentic-rl/<slug>/
 */
import { execFileSync } from 'node:child_process'
import { mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, basename, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const NOTES_DIR = resolve(HERE, '..', 'notes')
const REPO_ROOT = resolve(HERE, '..', '..')

const COLUMN = 'Agentic-RL'      // 分类名，决定 /categories/Agentic-RL/
const URL_ROOT = 'agentic-rl'    // permalink 前缀
const AUTHOR = 'Ciao'
const DEFAULT_TAGS = ['Agentic RL']

// ---------- 参数 ----------
const argv = process.argv.slice(2)
const argOf = (k) => {
  const i = argv.indexOf(k)
  return i >= 0 ? argv[i + 1] : undefined
}
const dest = argOf('--dest')
const dryRun = argv.includes('--dry-run')
if (!dest) {
  console.error('用法: node study/hexo/sync.mjs --dest <hexo-blog>/source/_posts/agentic-rl [--dry-run]')
  process.exit(1)
}
const DEST = resolve(process.cwd(), dest)

// ---------- 工具 ----------
const gitDate = (file) => {
  try {
    return execFileSync('git', ['log', '-1', '--format=%ad', '--date=format:%Y-%m-%d %H:%M:%S', '--', file], {
      cwd: REPO_ROOT,
      encoding: 'utf8',
    }).trim()
  } catch {
    return ''
  }
}

const splitFrontmatter = (raw) => {
  const m = /^---\n([\s\S]*?)\n---\n?/.exec(raw)
  return m ? { fm: m[1], body: raw.slice(m[0].length) } : { fm: '', body: raw }
}

/** 正文首个 H1 作为标题；没有就回退到文件名 */
const pickTitle = (body, fallback) => {
  const m = /^#\s+(.+)$/m.exec(body)
  return m ? m[1].trim() : fallback
}

const pickDescription = (body) => {
  const line = body
    .split('\n')
    .map((l) => l.trim())
    .find((l) => l && !l.startsWith('#') && !l.startsWith('>') && !l.startsWith('|') && !l.startsWith('-'))
  if (!line) return ''
  const plain = line.replace(/\[([^\]]+)\]\([^)]*\)/g, '$1').replace(/[*`]/g, '')
  return plain.length > 110 ? plain.slice(0, 110) + '…' : plain
}

/** VitePress 的相对 md 链接 → 博客里的专栏链接 */
const rewriteLinks = (body) =>
  body
    .replace(/\[([^\]]+)\]\(\.?\/([0-9a-z-]+)\.md\)/gi, (_, text, slug) => `[${text}](/${URL_ROOT}/${slug}/)`)
    .replace(/\[([^\]]+)\]\((?:\.\.\/)+(experiments[^)]*)\)/g, (_, text, p) =>
      `[${text}](https://github.com/datawhalechina/agentic-rl-lab/blob/main/${p.replace(/\.md$/, '')})`,
    )

/** 去掉正文里的首个 H1（标题已进 frontmatter），避免页内重复 */
const stripFirstH1 = (body) => body.replace(/^#\s+.+\n+/, '')

// ---------- 主流程 ----------
const files = readdirSync(NOTES_DIR)
  .filter((f) => f.endsWith('.md'))
  .sort()

if (!files.length) {
  console.error(`没找到笔记: ${NOTES_DIR}`)
  process.exit(1)
}

if (!dryRun) {
  rmSync(DEST, { recursive: true, force: true })
  mkdirSync(DEST, { recursive: true })
}

const manifest = []
for (const file of files) {
  const abs = join(NOTES_DIR, file)
  const raw = readFileSync(abs, 'utf8')
  const { body: body0 } = splitFrontmatter(raw)

  const stem = basename(file, '.md')
  const isIndex = stem === 'index'
  const slug = isIndex ? 'roadmap' : stem

  const title = isIndex
    ? 'Agentic RL Lab · 学习路线与进度'
    : pickTitle(body0, stem)
  const description = pickDescription(body0)
  const date = gitDate(abs) || new Date().toISOString().slice(0, 19).replace('T', ' ')

  const tags = [...DEFAULT_TAGS, isIndex ? 'Roadmap' : 'Notes']

  const body = rewriteLinks(stripFirstH1(body0)).trim() + '\n'

  const fm = [
    '---',
    `title: ${title}`,
    `date: ${date}`,
    `author: ${AUTHOR}`,
    `description: ${description}`,
    'categories:',
    ` - ${COLUMN}`,
    'tags:',
    ...tags.map((t) => ` - ${t}`),
    `permalink: ${URL_ROOT}/${slug}/`,
    '---',
    '',
  ].join('\n')

  const outName = `${date.slice(0, 10)}-${slug}.md`
  const outPath = join(DEST, outName)
  if (!dryRun) writeFileSync(outPath, fm + '\n' + body, 'utf8')
  manifest.push({ file, outName, permalink: `/${URL_ROOT}/${slug}/`, title })
}

console.log(`${dryRun ? '[dry-run] ' : ''}同步 ${manifest.length} 篇 → ${DEST}`)
for (const m of manifest) console.log(`  ${m.file} → ${m.outName}  (${m.permalink})  ${m.title}`)
if (!dryRun) {
  console.log('\n下一步：')
  console.log('  cd <hexo-blog> && npm run build && npx hexo server   # 本地预览')
  console.log('  cd <hexo-blog> && npm run deploy                    # 发布到 biochao.cc')
}
