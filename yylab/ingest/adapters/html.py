from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from ..records import Doc
from .base import Adapter

DROP_TAGS = ("script", "style", "noscript", "template", "svg", "iframe", "canvas", "form")
BLOCK_TAGS = (
    "p", "div", "section", "article", "header", "footer", "li", "tr", "h1", "h2", "h3",
    "h4", "h5", "h6", "blockquote", "pre", "table", "ul", "ol", "br",
)
WS_RE = re.compile(r"[ \t]+")
NL_RE = re.compile(r"\n{3,}")


class HtmlAdapter(Adapter):
    name = "html"
    extensions = {".html", ".htm", ".xhtml"}
    default_domain = "web"

    def extract(self, path: Path) -> Iterator[Doc]:
        from bs4 import BeautifulSoup

        raw = self._read_text(path)
        if not raw.strip():
            return
        soup = BeautifulSoup(raw, "lxml")
        for tag in soup(DROP_TAGS):
            tag.decompose()
        for tag in soup(BLOCK_TAGS):
            tag.insert_after("\n")
        title = soup.title.get_text(strip=True) if soup.title else ""
        main = soup.find(["article", "main"]) or soup.body or soup
        text = main.get_text()
        lines = [WS_RE.sub(" ", ln).strip() for ln in text.split("\n")]
        cleaned = NL_RE.sub("\n\n", "\n".join(ln for ln in lines if ln)).strip()
        if title:
            cleaned = f"{title}\n\n{cleaned}" if cleaned else title
        if cleaned:
            yield Doc(text=cleaned, source=str(path), domain=self.default_domain)
