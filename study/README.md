# study/ · 学习笔记站点

这里是 `study` 分支上的个人学习笔记，同时是一个独立的 [VitePress](https://vitepress.dev/) 站点，
发布在 **<https://biociao.github.io/agentic-rl-lab/>**——GitHub Pages 的免费域名，与个人博客
（`biochao.cc`，跑在阿里云 ECS 上）是两个完全独立的主机，互不影响，也不需要额外买域名。

> 为什么不用 `blog.biochao.cc` 那套？那是旧账号 `scelta.github.io` 仓库里留下的 CNAME，
> 该域名早已 NXDOMAIN，属废弃配置。而 `biociao.github.io` 下**没有**用户站点仓库，
> 所以项目站点不会被重定向到任何自定义域，用默认路径最省事。

## 目录结构

```
study/
├── index.md                 # 站点首页
├── notes/                   # 笔记正文
│   ├── index.md             #   学习路线与进度表
│   └── 00-getting-started.md
├── .vitepress/config.mts    # 站点配置（base、导航、侧边栏、搜索）
├── public/                  # 原样拷贝到产物根目录
│   └── favicon.svg
└── package.json             # 构建脚本与依赖
```

笔记是普通 Markdown：在 `notes/` 下新建文件，把链接加进 `notes/index.md` 的进度表和
`.vitepress/config.mts` 的 `sidebar` 即可。

## 本地预览

```bash
cd study
npm install            # 首次
npm run docs:dev       # http://localhost:5173/agentic-rl-lab/
npm run docs:build     # 产物在 .vitepress/dist
npm run docs:preview
```

> 本机 `~/.npm` 属主是 root，直接 `npm install` 会 EACCES。把缓存指到仓库内即可：
> `npm_config_cache="$PWD/../.npm-cache" npm install`。

## 发布

推送到 `study` 分支即触发 `.github/workflows/study-site.yml`：构建 → 上传 artifact → 部署 Pages。
站点地址：`https://biociao.github.io/agentic-rl-lab/`

首次启用（只需一次）：

1. **Pages**：仓库 Settings → Pages → Source 选 **GitHub Actions**。
2. **Actions**：fork 默认不跑 workflow。Settings → Actions → General → 允许本仓库运行 workflow
   （或在 Actions 页点 “I understand my workflows, go ahead and enable them”）。
3. 之后每次 `git push fork study` 自动上线。

## 换成独立域名（可选）

想要 `arl.biochao.cc` 这种自己的域名，三步：

1. **DNS**：在 `biochao.cc` 的解析（阿里云 hichina）加
   `CNAME  arl  →  biociao.github.io`（注意不要带路径和后缀点）。
2. **配置**：`.vitepress/config.mts` 里 `BASE` 改回 `'/'`、`SITE` 改为 `https://arl.biochao.cc`；
   并新建 `public/CNAME`，内容一行 `arl.biochao.cc`。
3. **Pages 设置**：Custom domain 填 `arl.biochao.cc`，证书签发后勾 **Enforce HTTPS**。

⚠️ `public/CNAME` 一旦存在，GitHub Pages 就只认那个自定义域，`biociao.github.io/agentic-rl-lab/`
会 301 跳走——所以不想用自定义域时**不要**留这个文件。

## 兜底：不用 Actions 手动发布

```bash
cd study
npm run docs:build
npx gh-pages -d .vitepress/dist -b gh-pages \
    -r https://github.com/biociao/agentic-rl-lab.git
```

然后 Settings → Pages → Source 选 **Deploy from a branch** → `gh-pages` / `root`。
（`base` 仍是 `/agentic-rl-lab/`，不用改。）

## 与上游的关系

- 本分支基于 `upstream/main`（`datawhalechina/agentic-rl-lab`）的副本；
- `upstream` remote 的 push URL 已设为 `no_push`，防止误推；
- 收上游更新：`git fetch upstream && git rebase upstream/main`。笔记只在 `study/` 目录，冲突面很小。
