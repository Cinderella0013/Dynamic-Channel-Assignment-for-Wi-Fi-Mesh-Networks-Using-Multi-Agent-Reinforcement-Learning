"""Render a .docx to PDF: python docs/paper/build/render.py IN.docx OUT.pdf [--word]

Default: LibreOffice headless (about 15 s for the thesis and the paper together), for quick previews while
editing. --word: Microsoft Word through render.ps1 (about 1-2 min per document). Use --word for the final
build: LibreOffice paginates Thai text differently (thesis 48 vs 47 pages, paper 6 vs 5), and the TOC page
numbers written by `build_thesis.py --pages` must match what readers see in Word."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SOFFICE = r"C:\Program Files\LibreOffice\program\soffice.exe"
PS1 = Path(__file__).with_name("render.ps1")


def render(src, out, word=False):
    src, out = Path(src).resolve(), Path(out).resolve()
    if word:
        subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(PS1), "-In", str(src),
                        "-Out", str(out)], check=True)
        return out
    with tempfile.TemporaryDirectory() as tmp:
        # own profile dir, so a LibreOffice window the user has open does not block the headless run
        profile = Path(tmp, "profile").as_uri()
        subprocess.run([SOFFICE, "--headless", "--norestore", f"-env:UserInstallation={profile}",
                        "--convert-to", "pdf", "--outdir", tmp, str(src)], check=True, capture_output=True)
        shutil.move(str(Path(tmp, src.stem + ".pdf")), out)
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--word"]
    print(render(args[0], args[1], word="--word" in sys.argv))
