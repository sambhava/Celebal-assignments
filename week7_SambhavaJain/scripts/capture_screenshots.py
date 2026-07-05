"""Drive the real Streamlit RAG app and capture screenshots for the README."""
import os, time
from playwright.sync_api import sync_playwright

URL = "http://localhost:8601"
OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "screenshots")
os.makedirs(OUT, exist_ok=True)
SAMPLE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sample_docs", "rag_overview.txt"))


def shot(page, name):
    path = os.path.join(OUT, name)
    page.screenshot(path=path, full_page=True)
    print("saved", os.path.relpath(path))


with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.goto(URL, wait_until="networkidle")
    page.wait_for_timeout(3000)

    # 1) landing page
    shot(page, "01_home.png")

    # upload the sample document
    page.set_input_files("input[type=file]", SAMPLE)
    page.wait_for_timeout(2000)

    # click "Process documents"
    page.get_by_role("button", name="Process documents").click()
    # wait for the success message
    page.wait_for_selector("text=Indexed", timeout=120000)
    page.wait_for_timeout(1500)
    shot(page, "02_indexed.png")

    # ask a question
    box = page.get_by_placeholder("What is the main idea of the document?")
    box.click()
    box.fill("What are the three stages of RAG and what does reranking do?")
    box.press("Enter")  # commit value so Streamlit reruns and enables the Ask button
    page.wait_for_timeout(1500)
    page.get_by_role("button", name="Ask", exact=True).click()

    # wait for an answer to render (the Sources expander shows "chunks used)")
    page.wait_for_selector("text=chunks used)", timeout=120000)
    page.wait_for_timeout(2000)
    shot(page, "03_answer.png")

    # expand the sources panel
    try:
        page.get_by_text("chunks used)", exact=False).first.click()
        page.wait_for_timeout(1500)
        shot(page, "04_sources.png")
    except Exception as e:
        print("sources expand skipped:", e)

    browser.close()
print("DONE")
