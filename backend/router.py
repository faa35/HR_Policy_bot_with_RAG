"""
router.py — Decides where to retrieve from: the user's uploaded PDFs, the
main HR policy knowledge base, or both.

This is a small, separate LLM call (cheap model, short prompt) made BEFORE
retrieval. It's deliberately its own module/function rather than logic
buried inside retriever.py or main.py, so the routing decision is easy to
read, test, and swap out independently of the retrieval/generation step.

Usage:
    decision = route(question, has_uploaded_docs=True, uploaded_filenames=[...])
    # decision is one of: "NEW", "OLD", "BOTH"
"""
from llama_index.llms.openai import OpenAI

import config

ROUTER_PROMPT = """You are a routing component inside an HR policy chatbot. \
There are two possible knowledge sources for answering the employee's question:

1. UPLOADED — document(s) the employee personally uploaded just now.
   Filename(s): {filenames}
   Content preview: "{content_preview}"
2. POLICY — the company's existing official HR policy documents, covering \
these organizations only: {known_organizations}

Decide which source(s) are needed to answer the question below.

Important: a question does NOT need to explicitly say "my upload" or "this \
pdf" to be about the uploaded document. If the question names a specific \
person, organization, or topic that appears in the UPLOADED content preview \
above but is NOT one of the known POLICY organizations listed, that is a \
strong signal to use NEW or BOTH — treat a named organization mismatch as \
just as strong a signal as an explicit reference to "the uploaded document."

Respond with EXACTLY ONE WORD, no punctuation, no explanation:
- NEW   -> only the uploaded documents are relevant
- OLD   -> only the official policy documents are relevant
- BOTH  -> the question needs both, or it's unclear which one applies

Question: {question}
Answer with one word (NEW, OLD, or BOTH):"""


def route(
    question: str,
    has_uploaded_docs: bool,
    uploaded_filenames: list[str],
    content_preview: str = "",
    known_organizations: str = "",
) -> str:
    """Return "NEW", "OLD", or "BOTH".

    If the user has no active uploaded docs, this short-circuits to "OLD"
    without spending an LLM call — there's nothing to route to.

    content_preview: a short snippet of the ACTUAL uploaded text (see
    user_index.content_preview()). This matters a lot — a filename alone
    (e.g. "Human-Resources-Policy.pdf") gives no signal about which
    organization or topic it covers, so without real content the router
    will systematically under-route to the uploaded doc whenever the
    question names something the filename doesn't.

    known_organizations: comma-separated list of orgs the main policy
    library actually covers (config.ORGANIZATIONS_STR). Without this, the
    router has no way to tell "the question names an org that's clearly
    NOT in the main library" apart from "the question is just phrased
    generically" — both look the same without this contrast.
    """
    if not has_uploaded_docs:
        return "OLD"

    config.require_api_key()
    llm = OpenAI(model=config.LLM_MODEL, api_key=config.OPENAI_API_KEY, temperature=0.0)

    prompt = ROUTER_PROMPT.format(
        filenames=", ".join(uploaded_filenames) or "uploaded documents",
        content_preview=content_preview.strip() or "(no preview available)",
        known_organizations=known_organizations or config.ORGANIZATIONS_STR,
        question=question,
    )
    raw = str(llm.complete(prompt)).strip().upper()

    if "BOTH" in raw:
        return "BOTH"
    if "NEW" in raw:
        return "NEW"
    if "OLD" in raw:
        return "OLD"
    # Model returned something unexpected — fail safe to BOTH so we don't
    # silently drop a source the answer might have needed.
    return "BOTH"