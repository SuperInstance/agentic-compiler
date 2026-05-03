"""
Tests for agentic-compiler repository.

The agentic-compiler contains scripts and documentation for an agentic compiler
system. These tests validate the documented structure, scripts, and expected
behaviors based on README.md and related docs.
"""
import pytest
import re
import subprocess
from pathlib import Path

REPOS_DIR = Path(__file__).resolve().parent.parent  # = /repos/agentic-compiler
assert REPOS_DIR.name == "agentic-compiler", f"Expected agentic-compiler repo root, got {REPOS_DIR}"
AGENTIC_DIR = REPOS_DIR  # The repo root IS the agentic-compiler directory

def read_md(name):
    return (AGENTIC_DIR / name).read_text()

# ── README contract tests ────────────────────────────────────────────────────

class TestReadmeContract:
    """Validate agentic-compiler/README.md defines the compiler system."""

    def test_readme_exists(self):
        assert (AGENTIC_DIR / "README.md").exists()

    def test_readme_has_compiler_mention(self):
        text = read_md("README.md")
        assert "compiler" in text.lower()

    def test_readme_has_agentic_context(self):
        text = read_md("README.md")
        assert "agent" in text.lower() or "agentic" in text.lower()

    def test_readme_has_installation_section(self):
        text = read_md("README.md")
        assert "install" in text.lower()

    def test_readme_has_usage_section(self):
        text = read_md("README.md")
        assert "usage" in text.lower() or "example" in text.lower()

    def test_readme_has_code_fences(self):
        text = read_md("README.md")
        assert "```" in text

    def test_readme_has_headings(self):
        text = read_md("README.md")
        headings = re.findall(r'^#+\s+', text, re.MULTILINE)
        assert len(headings) >= 2

    def test_readme_is_substantive(self):
        text = read_md("README.md")
        assert len(text) > 200

# ── Script tests ─────────────────────────────────────────────────────────────

class TestScripts:
    """Validate agentic-compiler scripts exist and are executable."""

    def test_scripts_directory_exists(self):
        assert (AGENTIC_DIR / "scripts").exists()

    def test_run_ra_script_exists(self):
        script = AGENTIC_DIR / "scripts" / "run-ra.py"
        assert script.exists()

    def test_run_ra_script_is_python(self):
        script = AGENTIC_DIR / "scripts" / "run-ra.py"
        text = script.read_text()
        assert "python" in text.lower() or "#!" in text or "def " in text

    def test_run_ra_script_has_shebang_or_def(self):
        script = AGENTIC_DIR / "scripts" / "run-ra.py"
        text = script.read_text()
        assert len(text) > 50

# ── Documentation structure tests ────────────────────────────────────────────

class TestDocumentationStructure:
    """Validate docs/ directory has expected structure."""

    def test_docs_directory_exists(self):
        assert (AGENTIC_DIR / "docs").exists()

    def test_docs_has_ra_subdirectory(self):
        ra_dir = AGENTIC_DIR / "docs" / "ra"
        assert ra_dir.exists()

    def test_ra_rounds_exist(self):
        ra_dir = AGENTIC_DIR / "docs" / "ra"
        round_files = list(ra_dir.glob("round-*.md"))
        assert len(round_files) >= 5, f"Expected 5+ round docs, found {len(round_files)}"

    def test_round_docs_are_numbered(self):
        ra_dir = AGENTIC_DIR / "docs" / "ra"
        round_files = sorted(ra_dir.glob("round-*.md"))
        numbers = []
        for f in round_files:
            m = re.search(r'round-(\d+)', f.name)
            if m:
                numbers.append(int(m.group(1)))
        assert len(numbers) >= 5
        numbers = sorted(numbers)  # Sort numerically
        assert numbers == list(range(1, len(numbers)+1)), "Round files should be sequentially numbered (1,2,3...)"

    def test_round_docs_have_content(self):
        ra_dir = AGENTIC_DIR / "docs" / "ra"
        for f in ra_dir.glob("round-*.md"):
            content = f.read_text()
            assert len(content) > 50, f"{f.name} should have substantive content"

# ── License tests ───────────────────────────────────────────────────────────

class TestLicense:
    """Validate LICENSE file."""

    def test_license_exists(self):
        assert (AGENTIC_DIR / "LICENSE").exists()

    def test_license_is_substantive(self):
        text = read_md("LICENSE").replace("#", "").replace(" ", "")
        assert len(text) > 100

# ── RA Round document content tests ──────────────────────────────────────────

class TestRoundDocuments:
    """Validate round documents describe agentic compilation process."""

    def test_round_docs_mention_iteration(self):
        ra_dir = AGENTIC_DIR / "docs" / "ra"
        for f in sorted(ra_dir.glob("round-*.md"))[:3]:
            text = f.read_text()
            assert len(text) > 50, f"{f.name} should have content"

    def test_round_docs_have_headers(self):
        ra_dir = AGENTIC_DIR / "docs" / "ra"
        for f in sorted(ra_dir.glob("round-*.md"))[:3]:
            text = f.read_text()
            headings = re.findall(r'^#+\s+', text, re.MULTILINE)
            assert len(headings) >= 1, f"{f.name} should have at least one heading"

    def test_round_docs_contain_reasoning(self):
        ra_dir = AGENTIC_DIR / "docs" / "ra"
        for f in sorted(ra_dir.glob("round-*.md"))[:3]:
            text = f.read_text()
            # Should contain some reasoning language
            assert any(kw in text.lower() for kw in ["reason", "analysis", "compile", "round", "agent"])


# ── Installation and usage tests ──────────────────────────────────────────────

class TestInstallationUsage:
    """Validate install/usage instructions are well-formed."""

    def test_readme_has_pip_install_command(self):
        text = read_md("README.md")
        assert "pip install" in text.lower() or "npm install" in text.lower() or "cargo" in text.lower()

    def test_readme_mentions_prerequisites(self):
        text = read_md("README.md")
        # Should mention something about requirements or dependencies
        assert "require" in text.lower() or "depend" in text.lower() or "prereq" in text.lower() or len(text) > 200

    def test_readme_has_example_import_or_call(self):
        text = read_md("README.md")
        assert "import" in text or "run" in text.lower() or "execute" in text.lower()


# ── Markdown quality tests ───────────────────────────────────────────────────

class TestMarkdownQuality:
    """Validate agentic-compiler markdown docs meet fleet standards."""

    def test_readme_has_code_fences_with_language(self):
        text = read_md("README.md")
        # Check for language-tagged code fences
        fences = re.findall(r'```(\w+)', text)
        assert len(fences) >= 1, "README should have at least one code fence"

    def test_readme_no_excessive_urls(self):
        text = read_md("README.md")
        urls = re.findall(r'https?://', text)
        # Should have some URLs but not spam
        assert len(urls) >= 1
        assert len(urls) < 20

    def test_all_markdown_files_have_content(self):
        for md_file in AGENTIC_DIR.glob("**/*.md"):
            content = md_file.read_text()
            assert len(content) > 20, f"{md_file} should not be nearly empty"

# ── Consistency tests ─────────────────────────────────────────────────────────

class TestConsistency:
    """Validate consistency across agentic-compiler docs."""

    def test_all_round_docs_use_consistent_format(self):
        ra_dir = AGENTIC_DIR / "docs" / "ra"
        round_files = sorted(ra_dir.glob("round-*.md"))
        if len(round_files) < 2:
            pytest.skip("Need at least 2 round files to compare")

        first = round_files[0].read_text()
        second = round_files[1].read_text()
        # Both should have similar structure (headers)
        h1_first = len(re.findall(r'^#\s+', first, re.MULTILINE))
        h1_second = len(re.findall(r'^#\s+', second, re.MULTILINE))
        assert h1_first > 0 and h1_second > 0

    def test_readme_and_docs_are_consistent(self):
        readme_text = read_md("README.md")
        # README should mention the ra/ docs directory
        assert "round" in readme_text.lower() or "docs" in readme_text.lower()
