# 同步到 Hexo 博客（biochao.cc）· 专栏 Agentic-RL

把 `study/notes/*.md` 发布成个人博客上的一个**专栏**，笔记仍然只维护在本仓库。

## 设计原则

**单一事实来源 = 这个 git 仓库（`study` 分支）**，博客只是发布面。方向永远是：

```
study/notes/*.md  ──(sync.mjs)──►  hexo-blog/source/_posts/agentic-rl/*.md  ──(hexo deploy)──►  biochao.cc
```

因此：
- ❌ 不要在博客目录里改这些文章——下次同步整目录会被删掉重建；
- ✅ 所有修改都回到 `study/notes/` 提交，再跑一次同步 + 部署；
- 原始信息（谁写的、为什么这么写）留在 git 历史里，博客只承载成品。

## 用法

```bash
# 1. 同步（在工作区仓库根目录执行）
node study/hexo/sync.mjs --dest /Volumes/repo/ciao/hexo-blog/source/_posts/agentic-rl

# 只看产物、不落盘
node study/hexo/sync.mjs --dest ./study/.hexo-out --dry-run

# 2. 本地预览
cd /Volumes/repo/ciao/hexo-blog && npx hexo server
# → http://localhost:4000/categories/Agentic-RL/

# 3. 发布
cd /Volumes/repo/ciao/hexo-blog && npx hexo clean && npx hexo generate && npx hexo deploy
```

`hexo deploy` 走 `hexo-deployer-git`，推到 `hexo@biochao.cc:/var/www/repo/hexo.git` 的 `main` 分支，
由服务器侧的 hook 检出到站点目录——和现有 23 篇文章是同一条发布路径。

## frontmatter 映射

| 笔记 | Hexo frontmatter | 说明 |
| --- | --- | --- |
| 文件正文首个 `# H1` | `title` | 生成后从正文删掉，避免页内重复 |
| git 提交时间 | `date` | `git log -1 --date=format:...`，无提交时用当前时间 |
| — | `author: Ciao` | 与博客现有文章一致 |
| 正文首个普通段落（截 110 字） | `description` | 用于首页摘要与 SEO |
| — | `categories: [Agentic-RL]` | 专栏分类，自动生成 `/categories/Agentic-RL/` |
| — | `tags: [Agentic RL, Notes]` | 可再加章节专属标签 |
| 文件名 | `permalink: agentic-rl/<slug>` | 文章 URL 稳定为 `/agentic-rl/<slug>/`，不随日期变 |

笔记侧的 `](./00-getting-started.md)` 会被改写成 `(/agentic-rl/00-getting-started/)`，
指向仓库 `experiments/...` 的链接改写成 GitHub 上的对应路径——博客里不会出现死链。

## 博客端一次性配置

`_config.next.yml` 的 menu 里加一项，让专栏在导航栏可见：

```yaml
menu:
  home: / || fa fa-home
  agentic-rl: /categories/Agentic-RL/ || fa fa-flask    # ← 新增
  categories: /categories/ || fa fa-th
  tags: /tags/ || fa fa-tags
  archives: /archives/ || fa fa-archive
  about: /about/ || fa fa-user
```

改完 `npx hexo generate` 会重新生成全部页面的导航。

## 注意事项

- **渲染器是 `hexo-renderer-marked`**：标准 Markdown + GFM 表格可用；VitePress 专属语法
  （`::: tip` 容器、`layout: home` 首页）不会被渲染，所以 `study/index.md` 这个首页**不参与同步**，
  专栏入口直接用分类页。
- **图片**：笔记若有本地图片，放在博客的 `source/images/agentic-rl/` 下并在笔记里用同一路径引用；
  博客装了 `hexo-image-link`，相对路径会被重写，跨仓库引用容易断，建议统一走仓库 `study/` 内的图并同步过去。
- **Hexo 的 `layout: article`** 在现有 7 篇文章里出现过但没有对应模板文件，属无害的历史遗留；
  新生成的文章不带 `layout` 字段，走默认 post 模板。
- **同步是破坏性的**：`sync.mjs` 会先清空目标目录，所以目标目录必须是独占的
  `source/_posts/agentic-rl/`，不要指向 `source/_posts/` 根。

## 与 VitePress 站点的关系

仓库里同时存在 `study/.vitepress/`（GitHub Pages 版）。两者共用同一批 Markdown，互不干扰：

- 想只维护博客 → 删掉 `.vitepress/` 与 workflow，本目录保留；
- 想两个都发 → 两套产物各走各的流水线，笔记改一次、同步两次。
