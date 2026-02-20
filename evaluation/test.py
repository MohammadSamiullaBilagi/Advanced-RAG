import json
from pathlib import Path
from pydantic import BaseModel, Field


# **Step by step:**
# 1. `__file__` - Special variable containing the path of the current Python file
# 2. `Path(__file__)` - Converts string path to a `pathlib.Path` object
# 3. `.parent` - Gets the parent directory of the current file
# 4. `/ "tests.jsonl"` - Appends the filename (path division operator)
# 5. `str(...)` - Converts the Path object back to a string

# **Example:**
# ```
# If your script is at: /home/user/project/src/main.py
# Then TEST_FILE will be: /home/user/project/src/tests.jsonl

# TEST_FILE = "tests.jsonl"  # Only works if run from specific directory


TEST_FILE = str(Path(__file__).parent / "tests.jsonl")


class TestQuestion(BaseModel):
    """A test question with expected keywords and reference answer."""

    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category (e.g., direct_fact, spanning, temporal)")


def load_tests() -> list[TestQuestion]:
    """Load test questions from JSONL file."""
    tests = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            tests.append(TestQuestion(**data))
    return tests
