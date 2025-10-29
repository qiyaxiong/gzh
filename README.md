## 公众号自动化创作工作流（Tavily 联网版）

### 功能
- Tavily 联网检索热点与用户关注点
- LLM 生成 3~5 个选题方向
- 手动选择主题
- 一键生成公众号图文（引言、3 个小标题、结尾），每段含配图占位符

### 运行前准备
1) 安装依赖：
```bash
pip install -r requirements.txt
```

2) 配置环境变量（推荐使用 .env 或 shell 导出）：
```bash
export OPENROUTER_API_KEY="<你的 OpenRouter API Key>"
export TAVILY_API_KEY="<你的 Tavily API Key>"
# 可选，覆盖默认模型
export OPENROUTER_MODEL="google/gemini-2.5-pro"
```

3) 启动应用：
```bash
streamlit run streamlit_app.py
```

### 说明
- LLM 通过 OpenAI Python SDK 指向 OpenRouter (`base_url=https://openrouter.ai/api/v1`)。
- 默认模型为 `google/gemini-2.5-pro`，可在侧边栏或通过 `OPENROUTER_MODEL` 修改。
- 文章每段前都会包含以下占位符，可后续替换为真实图片 URL 或 AI 生成图：
  - `{{引言配图URL}}`
  - `{{小标题一配图URL}}`
  - `{{小标题二配图URL}}`
  - `{{小标题三配图URL}}`
  - `{{结尾配图URL}}`

### 安全
- 请勿在代码中硬编码任何密钥。使用环境变量或 Streamlit Secrets 管理。
