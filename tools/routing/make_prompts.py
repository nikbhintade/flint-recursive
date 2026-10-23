"""Build the T4 routing workload: 100 coding prompts in four kinds, written to data/routing/prompts.jsonl.

  humaneval  25  complete a function (openai/openai_humaneval)
  issue      25  explain the root cause of a GitHub issue and outline a fix (SWE-bench Verified)
  agent      25  one agent turn: issue -> earlier tool call that read a file -> file excerpt -> next action, with tools
  file       25  explain / review / test / refactor a real source file (llama.cpp and Strata sources)

Usage: python3 tools/routing/make_prompts.py
"""
import json
import pathlib
import random
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "routing" / "prompts.jsonl"
LLAMA = ROOT / "llama.cpp"
STRATA_TOOLS = ROOT.parent / "ai-content" / "strata" / "Strata" / "tools"
N = 25


def hf_rows(dataset, config, split, offset, length):
    url = (f"https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}"
           f"&split={split}&offset={offset}&length={length}")
    with urllib.request.urlopen(url, timeout=60) as r:
        return [x["row"] for x in json.load(r)["rows"]]


AGENT_SYSTEM = ("You are a software engineering agent working in a checked-out repository. Use the tools to inspect "
                "and change files; make the smallest correct change that fixes the issue.")
TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command in the repository root and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "str_replace", "description": "Replace one exact occurrence of old_str with new_str in a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old_str": {"type": "string"}, "new_str": {"type": "string"}},
            "required": ["path", "old_str", "new_str"]}}},
]


def preimage(patch):
    """The first file a patch touches, and the original lines around its hunks (context + removed lines)."""
    path, lines = None, []
    for line in patch.splitlines():
        if line.startswith("--- a/"):
            if path:
                break
            path = line[6:]
        elif line.startswith("@@"):
            lines.append("...")
        elif path and line[:1] in (" ", "-") and not line.startswith("---"):
            lines.append(line[1:])
    return path, "\n".join(lines)


FILE_TASKS = [
    "Explain what this code does, then point out one bug or inefficiency and show a fix.",
    "Review this code as a senior engineer: list the three most important problems, most severe first.",
    "Write unit tests for the most important function in this file. Use the testing style the language usually uses.",
    "Refactor the most complicated function in this file to be easier to read, without changing behaviour.",
    "Add input validation and clear error messages to the public functions in this file.",
]
FILE_GLOBS = [
    (LLAMA / "src", "*.cpp"), (LLAMA / "ggml" / "src" / "ggml-cuda", "*.cu"), (LLAMA / "gguf-py" / "gguf", "*.py"),
    (LLAMA / "tools" / "server" / "public_legacy", "*.js"), (LLAMA / "cmake", "*.cmake"), (LLAMA / "scripts", "*.sh"),
    (STRATA_TOOLS, "*.py"),
]
LANG = {".cpp": "cpp", ".cu": "cuda", ".py": "python", ".js": "javascript", ".cmake": "cmake", ".sh": "bash"}


def main():
    rng = random.Random(4)
    out = []

    for i, row in enumerate(hf_rows("openai/openai_humaneval", "openai_humaneval", "test", 0, 100)[::4][:N]):
        out.append({"id": f"humaneval-{i:02d}", "kind": "humaneval", "messages": [{"role": "user", "content":
                    "Complete this Python function. Reply with the full function.\n\n```python\n"
                    + row["prompt"] + "```"}]})

    swe = hf_rows("princeton-nlp/SWE-bench_Verified", "default", "test", 0, 100)
    rng.shuffle(swe)
    for i, row in enumerate(swe[:N]):
        out.append({"id": f"issue-{i:02d}", "kind": "issue", "messages": [{"role": "user", "content":
                    f"Repository: {row['repo']}\n\n{row['problem_statement'][:6000]}\n\n"
                    "Explain the most likely root cause and outline the fix."}]})

    for i, row in enumerate(swe[N:2 * N]):
        path, excerpt = preimage(row["patch"])
        out.append({"id": f"agent-{i:02d}", "kind": "agent", "tools": TOOLS, "messages": [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": f"Fix this issue in {row['repo']}:\n\n{row['problem_statement'][:5000]}"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function", "function": {
                "name": "bash", "arguments": json.dumps({"command": f"sed -n '1,400p' {path}"})}}]},
            {"role": "tool", "tool_call_id": "call_0", "content": excerpt[:12000]},
        ]})

    files = [p for d, g in FILE_GLOBS for p in sorted(d.glob(g)) if 3_000 < p.stat().st_size]
    rng.shuffle(files)
    by_lang = {}
    for p in files:
        by_lang.setdefault(p.suffix, []).append(p)
    picked = []
    while len(picked) < N and any(by_lang.values()):  # round-robin over languages
        for suffix in list(by_lang):
            if by_lang[suffix] and len(picked) < N:
                picked.append(by_lang[suffix].pop())
    for i, p in enumerate(picked):
        text = p.read_text(errors="replace")[: rng.choice([5_000, 10_000, 20_000, 30_000])]
        out.append({"id": f"file-{i:02d}", "kind": "file", "source": str(p.relative_to(ROOT.parent)),
                    "messages": [{"role": "user", "content":
                                  f"File `{p.name}`:\n\n```{LANG[p.suffix]}\n{text}\n```\n\n{FILE_TASKS[i % 5]}"}]})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    kinds = {}
    for row in out:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    print(f"{len(out)} prompts -> {OUT}  {kinds}")


if __name__ == "__main__":
    main()
