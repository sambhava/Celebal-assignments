"""Drive the redesigned NotebookLM-style app and capture screenshots."""
import os
from playwright.sync_api import sync_playwright

URL = "http://localhost:8608"
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
    shot(page, "01_empty_state.png")

    # add a source
    page.set_input_files("input[type=file]", SAMPLE)
    page.wait_for_timeout(2000)
    page.get_by_role("button", name="Add & summarize").click()

    # wait for the auto Overview card (its eyebrow) to appear
    page.wait_for_selector("text=Ask a follow-up", timeout=180000)
    page.wait_for_timeout(2000)
    shot(page, "02_overview.png")

    # ask a follow-up via the chat input
    chat = page.get_by_placeholder("Ask anything about your document…")
    chat.click()
    chat.fill("What does reranking do and why is it useful?")
    chat.press("Enter")
    page.wait_for_selector("text=passages)", timeout=180000)
    page.wait_for_timeout(2500)
    shot(page, "03_chat.png")

    # toggle dark mode and capture the same populated view
    page.get_by_role("button", name="🌙").click()
    page.wait_for_timeout(1800)
    shot(page, "04_dark_mode.png")

    browser.close()
print("DONE")
