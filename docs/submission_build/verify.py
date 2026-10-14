"""Verify the generated docx preserved WRITEUP.md content (minus the intended edits)."""
import re, sys, zipfile
import xml.etree.ElementTree as ET

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

def docx_text(path):
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    paras = []
    for p in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
        t = "".join(n.text or "" for n in p.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"))
        paras.append(t)
    return paras

def norm(s):
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[*`_]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

md = open("C:/Users/DELL/threatwatch/docs/WRITEUP.md", encoding="utf-8").read()
paras = docx_text("C:/Users/DELL/threatwatch/ThreatWatch_Embedded_Systems_Writeup.docx")
doc_all = norm(" \u2022 ".join(paras))
doc_paras = [norm(p) for p in paras if norm(p)]

errors = []

# 1) every md content unit (paragraph/bullet/numitem/table cell) that is NOT intentionally
#    dropped/replaced must appear in the docx
dropped_markers = [
    "_Formatted for the provided template",
    "**Author:**",
    "Autonomous safety wearables must detect distress",  # original abstract replaced by trimmed one
    "Download link used: _<paste the drive link",        # replaced by UCI/GitHub links
    "improving to +0.131 over SVM purely",               # original abstract tail
]
# ascii diagrams intentionally replaced by figures 1 and 2 (5.1, 5.3); wiring diagram must REMAIN
in_fence = False
fence_buf, fence_after = [], None
units, last_h2 = [], None
for line in md.splitlines():
    if line.startswith("```"):
        in_fence = not in_fence
        if not in_fence and last_h2 and (last_h2.startswith("5.1") or last_h2.startswith("5.3")):
            units.append(("DROPPED_FENCE_5_1_5_3", "\n".join(fence_buf)))
        continue
    if in_fence:
        fence_buf.append(line)
        continue
    if line.startswith("### "):
        last_h2 = line[4:].strip()
    if line.startswith(("| ", "|--")) or line.startswith("#"):
        continue
    if line.startswith("* "):
        units.append(("BULLET", line[2:]))
    elif re.match(r"^\d{1,2}[a-z]?\. ", line):
        units.append(("NUM", line))
    elif line.strip() and not re.match(r"^-+\s*$", line):
        units.append(("P", line))

for kind, u in units:
    n = norm(u)
    if not n:
        continue
    if any(d in u for d in dropped_markers):
        continue
    if kind == "DROPPED_FENCE_5_1_5_3":
        if norm(u)[:80] in doc_all:
            errors.append("ASCII diagram of 5.1/5.3 should have been replaced by a figure but is still present")
        continue
    # wrapped md paragraphs: unit text may span multiple md lines -> rejoin handling done roughly:
    # check the first 70 normalized chars appear
    probe = n[:70]
    if probe and probe not in doc_all:
        errors.append(f"MISSING: {kind}: {probe}")

# 2) wiring diagram (6.2) must be present
if "3V3" not in doc_all or "0x57" not in doc_all or "GPIO21" not in doc_all:
    errors.append("Wiring ASCII diagram missing from docx")

# 3) survey table: exactly rows [1]..[8]; old rows 8,9,11,12 not in table order
for tag in ["[1] Wagh", "[2] Oliver", "[3] Almadhor", "[4] Schmidt", "[5] Yang", "[6] Ferdinando", "[7] Goodwin", "[8] Thummalakunta"]:
    if norm(tag) not in doc_all:
        errors.append(f"Survey row missing: {tag}")

# 4) references: 12 entries, correct new order
order = ["Wagh", "Oliver", "Almadhor", "Schmidt", "Yang", "Ferdinando", "Goodwin",
         "Thummalakunta", "Rodr\u00edguez-Rodr\u00edguez", "Prasanth", "Ghadi", "Mazhar"]
pos = -1
for i, name in enumerate(order, 1):
    m = re.search(r"\[" + str(i) + r"\]\s+" + re.escape(name), doc_all)
    if not m:
        errors.append(f"Reference [{i}] should be {name}")
    else:
        if m.start() < pos:
            errors.append(f"Reference [{i}] ({name}) out of order")
        pos = m.start()

# 5) figures + captions + placeholders
for cap in ["Figure 1:", "Figure 2:", "Figure 3:", "Figure 4:", "Figure 5:", "Figure 6:"]:
    if cap not in doc_all:
        errors.append(f"Caption missing: {cap}")
n_pending = doc_all.count("MEASUREMENT NOT YET RUN")
if n_pending != 3:
    errors.append(f"Expected 3 PENDING markers (10.3, 10.4, 10.5 row), found {n_pending}")

# 6) key measured numbers must survive verbatim
for num in ["0.617", "0.709", "0.486", "0.375", "+0.287", "99.7 %", "1 123 215", "284 796",
            "49 347", "4 354", "1 333", "0.5835", "13 411"]:
    if num not in doc_all:
        errors.append(f"Measured number missing: {num}")

# 7) abstract word count in doc: find abstract paragraph
abs_para = next((p for p in doc_paras if p.startswith("Autonomous safety wearables")), "")
wc = len(abs_para.split())
if not (230 <= wc <= 255):
    errors.append(f"Abstract word count {wc} outside 230-255")
if not any(p.startswith("Keywords:") for p in doc_paras):
    errors.append("Keywords paragraph missing")
kw = next(p for p in doc_paras if p.startswith("Keywords:"))
n_kw = len([k for k in kw.split("Keywords:")[1].split(",") if k.strip()])
if not (6 <= n_kw <= 10):
    errors.append(f"Keywords count {n_kw} outside 6-10")

# 8) dataset links present
if "archive.ics.uci.edu/dataset/465" not in doc_all:
    errors.append("UCI link missing")
if "github.com/WJMatthew/WESAD" not in doc_all:
    errors.append("GitHub link missing")

# 9) in-text citations resolve: [n] cited with n<=12; gap-3 [11] kept
if "to a server ([11])" not in doc_all:
    errors.append("Gap-3 [11] citation missing")
cited = set(int(x) for x in re.findall(r"\[(\d{1,2})\]", doc_all))
if not cited <= set(range(1, 13)):
    errors.append(f"Citation out of range: {sorted(cited)}")

# 10) title page fields
for t in ["Pratik Ambastha (25BCE0337)", "Adarsh Kumar Singh (25BCE0401)", "Arunya Srivastava (25BCE0344)",
          "Course: Embedded Systems", "Instructor: Prof. Vivek Rajpoot"]:
    if norm(t) not in doc_all:
        errors.append(f"Title page field missing: {t}")

print("paras:", len(paras))
if errors:
    print("FAILURES:", len(errors))
    for e in errors[:40]:
        print(" -", e)
    sys.exit(1)
print("ALL CONTENT CHECKS PASSED")
