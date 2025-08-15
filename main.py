import os
import requests
import time
from bs4 import BeautifulSoup
from feedgen.feed import FeedGenerator
from dotenv import load_dotenv
import google.generativeai as genai
import logging
from datetime import datetime
import pytz

# --- 配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# 配置Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.error("错误：GEMINI_API_KEY未在.env文件中设置。")
    exit()
genai.configure(api_key=GEMINI_API_KEY)

# 常量定义
HN_API_BASE_URL = "https://hacker-news.firebaseio.com/v0"
MAX_STORIES_TO_FETCH = 30
MAX_FEED_ENTRIES = 50
MAX_PROCESSED_IDS = 5000  # 新增：限制已处理ID文件的最大行数
RSS_FILE_PATH = 'hacker_news_summary_zh.xml'
PROCESSED_IDS_FILE = 'processed_ids.txt'

# --- 函数定义 ---

def get_processed_ids():
    """读取已处理的文章ID。"""
    if not os.path.exists(PROCESSED_IDS_FILE):
        return set()
    with open(PROCESSED_IDS_FILE, 'r') as f:
        return set(line.strip() for line in f)

def add_processed_id(story_id):
    """将新处理的文章ID追加到文件。"""
    with open(PROCESSED_IDS_FILE, 'a') as f:
        f.write(f"{story_id}\n")

def prune_processed_ids():
    """修剪已处理ID文件，防止其无限增长。"""
    try:
        if not os.path.exists(PROCESSED_IDS_FILE):
            return
        
        with open(PROCESSED_IDS_FILE, 'r') as f:
            lines = f.readlines()
        
        if len(lines) > MAX_PROCESSED_IDS:
            logging.info(f"已处理ID文件超过 {MAX_PROCESSED_IDS} 行，开始修剪...")
            lines_to_keep = lines[-MAX_PROCESSED_IDS:]
            with open(PROCESSED_IDS_FILE, 'w') as f:
                f.writelines(lines_to_keep)
            logging.info("修剪完成。")
    except Exception as e:
        logging.error(f"修剪ID文件时出错: {e}")


def get_top_story_ids(processed_ids):
    """获取Hacker News热门文章ID，并过滤掉已处理的。"""
    try:
        top_stories_url = f"{HN_API_BASE_URL}/topstories.json"
        response = requests.get(top_stories_url)
        response.raise_for_status()
        
        all_ids = [str(story_id) for story_id in response.json()]
        new_story_ids = [story_id for story_id in all_ids if story_id not in processed_ids]
        
        return new_story_ids[:MAX_STORIES_TO_FETCH]
    except requests.RequestException as e:
        logging.error(f"获取Hacker News热门文章ID失败: {e}")
        return []

def get_story_details(story_id):
    """获取文章详情。"""
    try:
        story_url = f"{HN_API_BASE_URL}/item/{story_id}.json"
        response = requests.get(story_url)
        response.raise_for_status()
        story_data = response.json()
        if story_data and 'url' in story_data:
            return {"id": story_id, "title": story_data.get("title", "无标题"), "url": story_data["url"]}
    except requests.RequestException as e:
        logging.error(f"获取文章详情失败 (ID: {story_id}): {e}")
    return None

def scrape_article_content(url):
    """抓取文章内容。"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)...'}
        response = requests.get(url, headers=headers, timeout=15, stream=True)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' not in content_type:
            logging.warning(f"跳过非HTML内容 (Content-Type: {content_type}, URL: {url})")
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = "\n".join([p.get_text() for p in paragraphs if len(p.get_text(strip=True)) > 100])
        if not content and soup.body:
            content = soup.body.get_text(separator='\n', strip=True)
        return content
    except Exception as e:
        logging.error(f"抓取或解析文章内容时出错 (URL: {url}): {e}")
    return None

def summarize_with_gemini(content):
    """使用Gemini进行总结。"""
    if not content or len(content.strip()) < 200:
        return "内容过短，无法生成有意义的摘要。"
    
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = f"请将以下英文文章内容总结成一段流畅的中文摘要，专注于核心观点和主要信息，字数在150字左右。不要添加任何个人评论或解释，直接给出摘要.\n\n文章内容:\n---\n{content[:8000]}\n---\n"
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"调用Gemini API失败: {e}")
        return "调用AI总结服务失败。"

def update_rss_feed(new_stories):
    """更新或创建RSS文件。"""
    fg = FeedGenerator()
    fg.title('Hacker News 中文摘要')
    fg.link(href='https://news.ycombinator.com/', rel='alternate')
    fg.description('由Gemini AI每日更新的Hacker News热门文章中文摘要')
    fg.language('zh-CN')

    existing_entries = []
    if os.path.exists(RSS_FILE_PATH):
        try:
            fg.load_extension('dc')
            fg.rss_file(RSS_FILE_PATH, load=True)
            existing_entries = fg.entry()
        except Exception as e:
            logging.warning(f"无法加载现有的RSS文件，将创建一个新的: {e}")

    for story in new_stories:
        fe = fg.add_entry(order='prepend')
        fe.title(story['title'])
        fe.link(href=story['url'])
        fe.description(story['summary'])
        fe.guid(story['url'], permalink=True)
        fe.pubDate(datetime.now(pytz.utc))

    if len(fg.entry()) > MAX_FEED_ENTRIES:
        fg.entry(fg.entry()[MAX_FEED_ENTRIES:], replace=True)

    fg.rss_file(RSS_FILE_PATH, pretty=True)
    logging.info(f"RSS源已成功更新并保存到: {RSS_FILE_PATH}")

# --- 主程序 ---
def main():
    """主执行函数"""
    logging.info("开始执行更新任务...")
    
    processed_ids = get_processed_ids()
    logging.info(f"已加载 {len(processed_ids)} 个已处理ID。")
    
    story_ids = get_top_story_ids(processed_ids)
    if not story_ids:
        logging.info("没有需要处理的新文章。")
        prune_processed_ids() # 即使没有新文章，也检查一下是否需要修剪
        logging.info("任务完成。")
        return

    logging.info(f"发现 {len(story_ids)} 篇新文章，开始处理...")
    newly_summarized_stories = []
    for i, story_id in enumerate(story_ids):
        logging.info(f"正在处理文章 {i+1}/{len(story_ids)} (ID: {story_id})...")
        details = get_story_details(story_id)
        
        if details:
            content = scrape_article_content(details['url'])
            if content:
                summary = summarize_with_gemini(content)
                details['summary'] = summary
                newly_summarized_stories.append(details)
                add_processed_id(details['id'])
                logging.info(f"  > 文章 {story_id} 处理成功。")
            else:
                logging.warning(f"  > 无法抓取文章内容，跳过。")
        
        if i < len(story_ids) - 1:
            time.sleep(5)

    if newly_summarized_stories:
        update_rss_feed(newly_summarized_stories)
    else:
        logging.info("本次运行没有成功处理任何新文章。")
    
    prune_processed_ids() # 在任务结束时修剪ID文件
    logging.info("所有任务已完成。")

if __name__ == "__main__":
    main()