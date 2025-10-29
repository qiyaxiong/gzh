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

# 实时语音对话系统

基于阿里云通义千问Omni模型的实时语音对话Web应用，支持语音输入、实时转录和语音回复。

## 功能特性

- 🎤 **实时语音对话**: 支持实时语音输入和输出
- 🗣️ **多音色选择**: 支持多种AI音色（Ethan、Emma、Lily、Oliver）
- 📝 **实时转录**: 显示用户输入和AI回复的文本转录
- 🔄 **自动重连**: 网络异常时自动重连机制
- 💬 **对话历史**: 保存完整的对话记录
- 📱 **响应式设计**: 支持桌面和移动设备

## 技术架构

- **后端**: FastAPI + WebSocket
- **前端**: 原生HTML/CSS/JavaScript
- **AI模型**: 阿里云通义千问Omni Realtime API
- **音频处理**: Web Audio API + PCM16编码

## 快速开始

### 1. 环境准备

```bash
# 安装Python依赖
pip install -r requirements.txt
```

### 2. 配置API密钥

设置阿里云DashScope API密钥：

```bash
export DASHSCOPE_API_KEY="your_api_key_here"
```

### 3. 启动服务器

```bash
python server.py
```

服务器将在 `http://localhost:9000` 启动。

### 4. 访问Web界面

打开浏览器访问 `http://localhost:9000` 即可开始使用。

## 使用说明

### 基本操作

1. **选择音色**: 在页面顶部选择喜欢的AI音色
2. **开始对话**: 点击"开始对话"按钮
3. **语音输入**: 允许麦克风权限后开始说话
4. **查看转录**: 可以显示/隐藏文本转录
5. **清空对话**: 点击"清空对话"按钮重置

### 音色选择

- **Ethan**: 男声，适合正式对话
- **Emma**: 女声，声音温和
- **Lily**: 女声，声音清晰
- **Oliver**: 男声，声音沉稳

### 注意事项

- 需要现代浏览器支持（Chrome、Firefox、Safari、Edge）
- 首次使用需要允许麦克风权限
- 建议在安静环境中使用以获得最佳效果
- 网络连接稳定时体验更佳

## 项目结构

```
├── server.py                 # FastAPI服务器
├── omni_realtime_client.py   # 通义千问Omni客户端
├── static/
│   └── index.html            # Web界面
├── requirements.txt          # Python依赖
└── README.md                # 说明文档
```

## 核心组件

### server.py
- FastAPI Web服务器
- WebSocket连接管理
- 音频数据处理
- 自动重连机制

### omni_realtime_client.py
- 通义千问Omni API客户端
- 音频流处理
- 事件回调处理
- 连接状态管理

### static/index.html
- 响应式Web界面
- 音频录制和播放
- 实时消息处理
- 用户交互控制

## 开发说明

### 音频格式要求
- 采样率: 16kHz
- 声道: 单声道
- 编码: PCM16
- 格式: 原始音频数据

### WebSocket消息格式
- 音频数据: `[0, ...audio_bytes]`
- 视频数据: `[1, ...video_bytes]`
- 文本消息: 直接发送字符串

### 错误处理
- 自动重连机制
- 心跳保活
- 异常恢复

## 故障排除

### 常见问题

1. **无法连接服务器**
   - 检查服务器是否启动
   - 确认端口9000未被占用
   - 检查防火墙设置

2. **麦克风无法使用**
   - 检查浏览器权限设置
   - 确认麦克风设备正常
   - 尝试刷新页面重新授权

3. **音频播放异常**
   - 检查浏览器音频支持
   - 确认音量设置正常
   - 尝试更换浏览器

4. **API调用失败**
   - 检查DASHSCOPE_API_KEY环境变量
   - 确认API密钥有效
   - 检查网络连接

### 调试模式

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 联系方式

如有问题，请通过GitHub Issues联系。
