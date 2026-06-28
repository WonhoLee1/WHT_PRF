import os

path = r"C:\Users\GOODMAN\.gemini\antigravity\brain\d27d4e34-8ab9-4a61-bbc5-db108318fbaf\walkthrough.md"

with open(path, "rb") as f:
    data = f.read()

# We need to find where the UTF-16LE data begins.
# Let's decode it. The first part is UTF-8 (from the artifact tool), 
# and the appended parts might be UTF-16LE with BOM or without BOM.
# A simple way: strip \x00 bytes where they shouldn't be, but this breaks Korean in UTF-16.
# Let's split by the known end of the original file, or just read the original file and re-append.
# Actually, I have the appended files: `dev_log\radioss_walkthrough_append.md` and `dev_log\radioss_result_analysis.md`.
# So I can just read the original `walkthrough.md` before the mess, or recreate the whole thing!

# The original walkthrough was 5874 bytes.
original_data = data[:5874].decode("utf-8", errors="ignore")

with open(r"dev_log\radioss_walkthrough_append.md", "r", encoding="utf-8") as f:
    append1 = f.read()

with open(r"dev_log\radioss_result_analysis.md", "r", encoding="utf-8") as f:
    append2 = f.read()

fixed_content = original_data + "\n" + append1 + "\n" + append2

with open(path, "w", encoding="utf-8") as f:
    f.write(fixed_content)

print("Fixed walkthrough.md")
