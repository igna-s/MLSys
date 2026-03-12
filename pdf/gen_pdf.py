"""Generate two-column PDF writeup using markdown_pdf (PyMuPDF-based)."""
import os, re, base64

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from markdown_pdf import MarkdownPdf, Section

md = open('writeup.md', 'r', encoding='utf-8').read()

# Strip YAML frontmatter
if md.startswith('---'):
    end = md.find('---', 3)
    md = md[end+3:].strip()

# Embed images as base64 data URIs so they appear in the PDF
def embed_images(text):
    def replace_img(match):
        alt = match.group(1)
        src = match.group(2)
        if os.path.exists(src):
            with open(src, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = src.rsplit(".", 1)[-1]
            mime = {"png": "image/png", "jpg": "image/jpeg"}.get(ext, "image/png")
            return f'![{alt}](data:{mime};base64,{b64})'
        return match.group(0)
    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_img, text)

md = embed_images(md)

# Add CSS for two-column compact layout
css = """<style>
body {
    font-size: 8pt;
    line-height: 1.2;
    column-count: 2;
    column-gap: 12px;
    font-family: serif;
    text-align: justify;
}
h1 {
    font-size: 10pt;
    font-weight: bold;
    margin: 5pt 0 2pt 0;
    column-span: all;
    border-bottom: 0.4pt solid #888;
    padding-bottom: 1pt;
}
h2 { font-size: 8.5pt; font-weight: bold; margin: 3pt 0 1pt 0; }
p { margin: 1.5pt 0; }
ul, ol { margin: 1pt 0 1pt 10pt; padding-left: 6pt; }
li { margin: 0.5pt 0; }
blockquote {
    margin: 2pt 2pt; padding: 1pt 4pt;
    background: #f4f4f4; border-left: 1.5pt solid #999;
    font-size: 7.5pt;
}
code { font-size: 7pt; background: #f0f0f0; }
table { font-size: 7pt; border-collapse: collapse; margin: 2pt auto; width: 100%; }
th, td { border: 0.4pt solid #aaa; padding: 1pt 2pt; text-align: center; }
th { background: #e8e8e8; font-weight: bold; }
img {
    max-width: 90%;
    height: auto;
    display: block;
    margin: 3pt auto;
    column-span: all;
}
strong { font-weight: bold; }
</style>

"""

# Title header
title = """<div style="column-span: all; text-align: center; margin-bottom: 4pt; border-bottom: 1pt solid #333; padding-bottom: 3pt;">
<h1 style="font-size: 12pt; border: none; margin: 0 0 2pt 0;">MLSys 2026 Contest — Track A Submission</h1>
<p style="font-size: 8.5pt; font-style: italic; margin: 0;">Schwindt, Ignacio A. — April 2026</p>
</div>

"""

full_md = css + title + md

pdf = MarkdownPdf(toc_level=0)
pdf.meta['title'] = 'MLSys 2026 Contest — Track A Submission'
pdf.meta['author'] = 'Schwindt, Ignacio A.'
pdf.add_section(Section(full_md, toc=False))
pdf.save('writeup.pdf')

import fitz
doc = fitz.open('writeup.pdf')
n = len(doc)
doc.close()
print(f"Generated writeup.pdf — {n} pages")
if n != 2:
    print(f"WARNING: Expected 2 pages, got {n}")
