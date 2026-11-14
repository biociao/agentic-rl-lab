import { defineConfig } from 'vitepress'

// 站点根目录 = study/，笔记正文放在 study/notes/
// 部署：GitHub Pages 项目站点 → https://biociao.github.io/agentic-rl-lab/
// 想换独立域名（如 arl.biochao.cc）见 study/README.md 的「换域名」一节
const BASE = '/agentic-rl-lab/'
const SITE = 'https://biociao.github.io/agentic-rl-lab'
const REPO = 'biociao/agentic-rl-lab'
const UPSTREAM_SITE = 'https://datawhalechina.github.io/agentic-rl-lab'

export default defineConfig({
  lang: 'zh-CN',
  title: 'Agentic RL 学习笔记',
  description: '跟着 Datawhale Agentic RL Lab 逐章精读、复现与记录',

  // 项目站点在子路径下，必须是 '/<repo>/'；换成自定义域时改成 '/'
  base: BASE,
  cleanUrls: true,
  lastUpdated: true,
  sitemap: { hostname: SITE },

  // README.md 是给仓库看的构建说明，不进站点
  srcExclude: ['README.md', 'node_modules/**'],

  markdown: {
    math: true,
    lineNumbers: true,
  },

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: `${BASE}favicon.svg` }],
    ['meta', { name: 'theme-color', content: '#2c3e50' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:site_name', content: 'Agentic RL 学习笔记' }],
  ],

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '学习路线', link: '/notes/' },
      { text: '第 00 章', link: '/notes/00-getting-started' },
      {
        text: '上游',
        items: [
          { text: '原仓库', link: `https://github.com/datawhalechina/agentic-rl-lab` },
          { text: '在线手册', link: UPSTREAM_SITE },
          { text: '我的 fork', link: `https://github.com/${REPO}` },
        ],
      },
    ],

    sidebar: {
      '/notes/': [
        {
          text: '从这里开始',
          items: [{ text: '学习路线与进度', link: '/notes/' }],
        },
        {
          text: '实验（按阅读顺序）',
          collapsed: false,
          items: [
            { text: '00 环境配置与基础概念', link: '/notes/00-getting-started' },
            { text: '01 Loss Functions', link: `${UPSTREAM_SITE}/experiments/loss-functions/` },
            { text: '02 GRPO', link: `${UPSTREAM_SITE}/experiments/grpo/` },
            { text: '03 Vision GRPO', link: `${UPSTREAM_SITE}/experiments/vision-grpo/` },
            { text: '04 PyTrio-Jev', link: `${UPSTREAM_SITE}/experiments/pytrio-jev/` },
            { text: '05 General OPD', link: `${UPSTREAM_SITE}/experiments/general-opd/` },
            { text: '06 Medical OPD', link: `${UPSTREAM_SITE}/experiments/medical-opd/` },
            { text: '07 DAPO', link: `${UPSTREAM_SITE}/experiments/dapo/` },
            { text: '08 GSPO', link: `${UPSTREAM_SITE}/experiments/gspo/` },
            { text: '09 OPSD', link: `${UPSTREAM_SITE}/experiments/opsd/` },
            { text: '10 AgentOPSD', link: `${UPSTREAM_SITE}/experiments/agentopsd/` },
            { text: '11 Search-R1', link: `${UPSTREAM_SITE}/experiments/search-r1/` },
            { text: '12 ReTool', link: `${UPSTREAM_SITE}/experiments/retool/` },
            { text: '13 ALFWorld', link: `${UPSTREAM_SITE}/experiments/alfworld/` },
            { text: '14 TEMPO', link: `${UPSTREAM_SITE}/experiments/tempo/` },
            { text: '15 Spec-o3', link: `${UPSTREAM_SITE}/experiments/spec-o3/` },
            { text: '16 Tournament-GRPO', link: `${UPSTREAM_SITE}/experiments/tournament-grpo/` },
          ],
        },
      ],
    },

    outline: { level: [2, 3], label: '本页目录' },
    docFooter: { prev: '上一篇', next: '下一篇' },
    lastUpdated: { text: '最后更新于', formatOptions: { dateStyle: 'short', timeStyle: 'short' } },
    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '目录',
    darkModeSwitchLabel: '外观',
    langMenuLabel: '语言',
    externalLinkIcon: true,

    search: {
      provider: 'local',
      options: { translations: { button: { buttonText: '搜索笔记', buttonAriaLabel: '搜索' } } },
    },

    editLink: {
      pattern: `https://github.com/${REPO}/edit/study/study/notes/:path`,
      text: '在 GitHub 上编辑此页',
    },

    socialLinks: [{ icon: 'github', link: `https://github.com/${REPO}` }],

    footer: {
      message: '个人学习笔记 · 内容以 CC BY-NC-SA 4.0 共享',
      copyright: '『我不生产知识，我只是知识的搬运工』',
    },
  },
})
