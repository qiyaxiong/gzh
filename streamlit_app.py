import os
from typing import List, Dict, Any

import streamlit as st
from openai import OpenAI
import requests

try:
    # tavily-python SDK
    from tavily import TavilyClient
except Exception:  # pragma: no cover - optional import error surfaced in UI
    TavilyClient = None  # type: ignore


def get_llm_client(api_key: str) -> OpenAI:
    """Instantiate an OpenRouter-backed OpenAI client with the provided api_key."""
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def get_tavily_client(api_key: str) -> Any:
    """Instantiate Tavily client with the provided api_key."""
    if TavilyClient is None:
        raise RuntimeError(
            "tavily-python not installed. Please add 'tavily-python' to requirements and reinstall."
        )
    if not api_key:
        raise RuntimeError("Missing TAVILY_API_KEY.")
    return TavilyClient(api_key=api_key)


def call_llm_generate_topics(client: OpenAI, search_text: str, model: str, max_topics: int = 7) -> List[str]:
    """Use LLM to propose 5-7 WeChat article topics based on search snippets, no explanations."""
    system_prompt = (
        "你是一名资深公众号选题策划师。基于最新热点与用户关注点，"
        "只输出主题本身，不要任何解释。"
    )
    user_prompt = (
        "根据以下搜索结果，生成 5~7 个公众号选题。\n"
        "严格要求：每行仅包含一个主题，不要编号和标点修饰，不要附加说明。\n\n"
        + search_text
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        top_p=0.95,
    )
    content = (resp.choices[0].message.content or "").strip()
    lines = [line.strip(" -•\t").strip() for line in content.splitlines()]
    topics = [line for line in lines if line]
    # Keep up to max_topics
    return topics[:max_topics]


def call_llm_generate_article(client: OpenAI, topic: str, model: str, search_text: str = "", reference_content: str = "") -> str:
    """Generate a full WeChat-style article in Markdown with image placeholders for each section."""
    system_prompt = (
        "你是一名资深公众号写手兼视觉策划师。请使用生动、有趣、信息密度高的中文撰写。"
    )
    user_prompt = (
        f"根据主题《{topic}》生成一篇公众号文章，结构包含：\n"
        "- 引言\n- 小标题1\n- 小标题2\n- 小标题3\n- 结尾\n\n"
        "要求：\n"
        "- 全文 800~1200 字\n"
        "- 每段上方添加配图占位符（用花括号保留，不要替换为真实链接）：\n"
        "  {{引言配图URL}}\n  {{小标题一配图URL}}\n  {{小标题二配图URL}}\n  {{小标题三配图URL}}\n  {{结尾配图URL}}\n"
        "- 可适度引用搜索发现（如有提供）并在文中自然表述，不要罗列链接\n"
        "- 输出 Markdown，可直接用于公众号排版（包含合适的小标题与列表/加粗等）\n\n"
    )
    
    if reference_content:
        user_prompt += (
            "以下是参考公众号文章（用于学习风格、结构、语言特点，不要直接抄袭）：\n"
            "---参考文章开始---\n"
            + reference_content[:3000]  # Limit reference to avoid token overflow
            + "\n---参考文章结束---\n\n"
        )
    
    if search_text:
        user_prompt += "以下是可参考的搜索要点：\n" + search_text + "\n\n"
    
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        top_p=0.95,
    )
    return (resp.choices[0].message.content or "").strip()


def format_search_results(results: List[Dict[str, Any]], limit: int = 10) -> str:
    """Concatenate top-K Tavily results' titles and snippets to a single text."""
    chunks: List[str] = []
    for r in results[:limit]:
        title = r.get("title") or ""
        snippet = r.get("content") or r.get("snippet") or ""
        source = r.get("url") or ""
        piece = f"【{title}】\n{snippet}\n来源: {source}"
        chunks.append(piece)
    return "\n\n".join(chunks)


def _is_probably_image_url(url: str, strict_head_check: bool = False) -> bool:
    if not url or not url.startswith("http"):
        return False
    # Quick reject for common non-image hosts/paths
    lower_url = url.lower()
    if "github.com" in lower_url and not any(seg in lower_url for seg in ("/raw/", ".png", ".jpg", ".jpeg", ".webp", ".gif")):
        return False
    if any(lower_url.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif")):
        return True
    if not strict_head_check:
        return False
    try:
        resp = requests.head(url, timeout=5, allow_redirects=True)
        ctype = (resp.headers.get("Content-Type") or "").lower()
        return ctype.startswith("image/")
    except Exception:
        return False


def collect_image_urls_from_tavily(
    tavily_response: Dict[str, Any], max_urls: int = 5, strict_head_check: bool = False
) -> tuple[List[str], List[str]]:
    """Extract image URLs and descriptions from Tavily top-level 'images' field.

    Returns:
        (urls, descriptions): both lists of equal length
    """
    urls: List[str] = []
    descriptions: List[str] = []

    def add_url(u: str, desc: str = "") -> None:
        if not u:
            return
        if u not in urls and _is_probably_image_url(u, strict_head_check=strict_head_check):
            urls.append(u)
            descriptions.append(desc)

    # Tavily returns top-level 'images' field when include_images=True
    images = tavily_response.get("images") or []
    if isinstance(images, list):
        for it in images:
            if isinstance(it, str):
                add_url(it, "")
            elif isinstance(it, dict):
                url = str(it.get("url") or "")
                desc = str(it.get("description") or "")
                add_url(url, desc)
            if len(urls) >= max_urls:
                break

    return urls[:max_urls], descriptions[:max_urls]


def replace_placeholders_with_images(article_md: str, image_urls: List[str], render_markdown_images: bool = True) -> str:
    placeholders = [
        "{{引言配图URL}}",
        "{{小标题一配图URL}}",
        "{{小标题二配图URL}}",
        "{{小标题三配图URL}}",
        "{{结尾配图URL}}",
    ]
    replaced = article_md
    for idx, ph in enumerate(placeholders):
        url = image_urls[idx] if idx < len(image_urls) else None
        if url:
            replaced = replaced.replace(ph, f"![配图]({url})" if render_markdown_images else url)
    return replaced


def main() -> None:
    st.set_page_config(page_title="公众号选题 + 图文生成（联网版）", page_icon="📰", layout="centered")
    st.title("公众号选题 + 图文生成（联网版）")

    with st.sidebar:
        st.markdown("**配置**")
        # Safe secret access helper to avoid StreamlitSecretNotFoundError when secrets.toml is missing
        def get_secret_value(key: str) -> str:
            try:
                # st.secrets may raise if no secrets file exists
                return str(st.secrets.get(key))  # type: ignore[arg-type]
            except Exception:
                return ""

        default_model = get_secret_value("OPENROUTER_MODEL") or os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-pro")
        model = st.text_input("LLM 模型 (OpenRouter)", value=default_model, help="例如：google/gemini-2.5-pro 或 openai/gpt-4o-mini 等")
        
        st.markdown("**联网搜索**")
        enable_tavily_search = st.checkbox("启用 Tavily 联网搜索", value=True, help="关闭后可手动输入主题")
        
        if enable_tavily_search:
            max_results = st.slider("Tavily 结果数", min_value=5, max_value=20, value=10, step=1)
            query_text = st.text_area(
                "热点检索查询语句",
                value="当前热点话题、行业趋势、用户关注问题",
                help="会用于 Tavily 联网搜索，可自行改写以贴近你的行业/受众",
            )
        
        st.markdown("**密钥（可直接填写）**")
        input_openrouter = st.text_input("OpenRouter API Key", type="password", value="")
        if enable_tavily_search:
            input_tavily = st.text_input("Tavily API Key", type="password", value="")
        
        st.markdown("**参考文件**")
        reference_file = st.file_uploader(
            "上传公众号参考文件（可选）",
            type=["txt", "md"],
            help="上传已有公众号文章作为风格/结构参考，生成时会传给 LLM",
        )
        
        st.markdown("**图片**")
        if enable_tavily_search:
            auto_fill_images = st.checkbox("自动填充 Tavily 图片URL 到占位符", value=True)
            render_md_images = st.checkbox("以 Markdown 图片形式渲染 (![]())", value=True)
            strict_head_check = st.checkbox("严格校验图片URL (HTTP HEAD 校验 Content-Type)", value=True)
            strong_filter_display = st.checkbox("仅展示图片链接（强过滤）", value=True)
            show_image_links_section = st.checkbox("文章下方附上图片链接列表", value=True)
            preview_images_before_article = st.checkbox("在正文前展示图片预览", value=True)
        else:
            auto_fill_images = False
            render_md_images = True
            strict_head_check = False
            strong_filter_display = False
            show_image_links_section = False
            preview_images_before_article = False

    # Read reference file if uploaded
    reference_content = ""
    if reference_file is not None:
        try:
            reference_content = reference_file.read().decode("utf-8")
            st.session_state["reference_content"] = reference_content
        except Exception as e:
            st.warning(f"参考文件读取失败：{e}")
    else:
        reference_content = st.session_state.get("reference_content", "")

    # Resolve credentials with precedence: sidebar input > secrets > env
    openrouter_key = (input_openrouter or get_secret_value("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")).strip()
    
    if enable_tavily_search:
        tavily_key = (input_tavily or get_secret_value("TAVILY_API_KEY") or os.environ.get("TAVILY_API_KEY", "")).strip()
    else:
        tavily_key = ""

    # Initialize clients
    try:
        llm_client = get_llm_client(openrouter_key)
    except Exception as e:
        st.error(f"LLM 客户端初始化失败：{e}")
        return

    tavily_client = None
    if enable_tavily_search:
        try:
            tavily_client = get_tavily_client(tavily_key)
        except Exception as e:
            st.warning(f"Tavily 客户端不可用：{e}")
            tavily_client = None

    # Step 1: Generate topics from search or manual input
    if enable_tavily_search and st.button("生成选题"):
        if tavily_client is None:
            st.error("无法联网搜索，请正确安装并配置 tavily-python 与 TAVILY_API_KEY。")
        else:
            with st.spinner("正在联网搜索并分析……"):
                try:
                    search = tavily_client.search(
                        query=query_text,
                        search_depth="advanced",
                        max_results=max_results,
                        include_answer=False,
                        include_raw_content=True,
                        include_images=True,
                        include_image_descriptions=True,
                    )
                    # SDK returns a dict with 'results'
                    results = search.get("results", []) if isinstance(search, dict) else search
                    search_text = format_search_results(results, limit=max_results)
                    st.session_state["last_search_text"] = search_text
                    try:
                        urls, descs = collect_image_urls_from_tavily(search, max_urls=5, strict_head_check=strict_head_check)
                        st.session_state["last_image_urls"] = urls
                        st.session_state["last_image_descriptions"] = descs
                    except Exception:
                        st.session_state["last_image_urls"] = []
                        st.session_state["last_image_descriptions"] = []
                    topics = call_llm_generate_topics(llm_client, search_text, model, max_topics=7)
                    if not topics:
                        st.warning("未能生成主题，请调整检索或稍后再试。")
                    else:
                        st.session_state["themes"] = topics
                        st.success("已生成选题。")
                except Exception as e:
                    st.error(f"搜索或生成主题失败：{e}")
    
    # Manual topic input when Tavily is disabled
    if not enable_tavily_search:
        st.markdown("### 手动输入主题")
        manual_topics_input = st.text_area(
            "请输入主题列表（每行一个）",
            height=200,
            help="每行一个主题，例如：\n如何提升工作效率\nAI在教育中的应用\n健康饮食的5个技巧",
        )
        if st.button("使用手动主题"):
            manual_topics = [line.strip() for line in manual_topics_input.split("\n") if line.strip()]
            if manual_topics:
                st.session_state["themes"] = manual_topics
                st.session_state["last_image_urls"] = []
                st.session_state["last_image_descriptions"] = []
                st.success(f"已添加 {len(manual_topics)} 个主题。")
            else:
                st.warning("请至少输入一个主题。")

    # Regenerate topics without re-searching (uses last search text)
    if enable_tavily_search and st.button("重新生成选题"):
        last_search_text = st.session_state.get("last_search_text", "")
        if not last_search_text:
            st.error("暂无可用的搜索结果，请先点击『生成选题』进行联网检索。")
        else:
            with st.spinner("正在基于上次搜索结果重新生成……"):
                try:
                    topics = call_llm_generate_topics(llm_client, last_search_text, model, max_topics=7)
                    if not topics:
                        st.warning("未能生成主题，请稍后重试或重新检索。")
                    else:
                        st.session_state["themes"] = topics
                        st.success("已重新生成选题。")
                except Exception as e:
                    st.error(f"重新生成失败：{e}")

    # Topic selection UI
    themes: List[str] = st.session_state.get("themes", [])
    if themes:
        selected = st.radio("请选择一个主题：", themes, key="selected_theme")

        if st.button("生成公众号文章"):
            with st.spinner("正在生成文章……"):
                try:
                    # Optionally reuse last search text to enrich the article
                    last_search_text = st.session_state.get("last_search_text", "")
                    ref_content = st.session_state.get("reference_content", "")
                    article_md = call_llm_generate_article(
                        llm_client,
                        topic=selected,
                        model=model,
                        search_text=last_search_text,
                        reference_content=ref_content,
                    )
                    # Ensure placeholders present at least once each; if not, preprend them
                    placeholders = [
                        "{{引言配图URL}}",
                        "{{小标题一配图URL}}",
                        "{{小标题二配图URL}}",
                        "{{小标题三配图URL}}",
                        "{{结尾配图URL}}",
                    ]
                    missing = [p for p in placeholders if p not in article_md]
                    if missing:
                        prefix = "\n".join(missing) + "\n\n"
                        article_md = prefix + article_md

                    # Optionally auto-fill placeholders with image URLs from last search
                    st.session_state["auto_fill_images"] = auto_fill_images
                    image_urls = st.session_state.get("last_image_urls", [])
                    if auto_fill_images and image_urls:
                        article_md = replace_placeholders_with_images(article_md, image_urls, render_markdown_images=render_md_images)
                    elif auto_fill_images and not image_urls:
                        # If requested but没有抓到图片，则用纯链接填充（不渲染为图片）
                        article_md = replace_placeholders_with_images(article_md, [" ".join([])], render_markdown_images=False)

                    # Optional: preview images above article to ensure visibility even if Markdown fails
                    if preview_images_before_article and image_urls:
                        try:
                            image_descs = st.session_state.get("last_image_descriptions", [])
                            captions = image_descs if image_descs and len(image_descs) == len(image_urls) else [f"图{i+1}" for i in range(len(image_urls))]
                            st.image(image_urls, caption=captions, use_container_width=True)
                        except Exception:
                            pass

                    st.markdown(article_md)
                    
                    # Add copyable markdown code block
                    st.markdown("---")
                    st.markdown("### 📋 复制文章 Markdown")
                    with st.expander("点击展开/复制完整 Markdown 文本（纯文本，不渲染）", expanded=False):
                        st.text_area(
                            "可直接全选复制：",
                            value=article_md,
                            height=300,
                            label_visibility="collapsed",
                        )
                        st.download_button("下载为 article.md", data=article_md, file_name="article.md", mime="text/markdown")

                    # Optional: show collected image URLs as a list for manual copy
                    if show_image_links_section:
                        urls = image_urls
                        if strong_filter_display:
                            urls = [u for u in urls if _is_probably_image_url(u, strict_head_check=True)]
                        if urls:
                            st.markdown("**本次检索图片链接（可手动粘贴）：**")
                            for i, u in enumerate(urls, 1):
                                st.markdown(f"{i}. {u}")
                        else:
                            st.info("未从本次检索中提取到图片链接。")
                except Exception as e:
                    st.error(f"生成文章失败：{e}")


if __name__ == "__main__":
    main()


