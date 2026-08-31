"""Local dummy knowledge base for testing the knowledge-retrieval tool.

Nothing in the reviewed project sources (code, flow doc, architecture
diagrams, pricing doc) contained real GoodScore knowledge-base content —
no S3 bucket, no S3 Vectors index, no AgentCore Gateway config. This file
is a stand-in for that, used ONLY when KNOWLEDGE_BASE_MODE=local_dummy
(see config.py, knowledge_gateway.py) — it exists purely so the
knowledge-retrieval tool path can be exercised end-to-end on a developer
machine. It is NOT a semantic/vector search (that's what S3 Vectors would
provide in production) — it's a small, explicit keyword-overlap ranking
over a handful of realistic GoodScore/credit-education FAQ entries.

Replace this module's role entirely with a real AgentCore Gateway call
(see knowledge_gateway.py) once Gateway URL, auth, and the real S3
Vectors-backed index are available — do not extend this file into a
production search implementation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Dummy content — plausible GoodScore FAQ / credit-education entries only.
# Written to sound like real GoodScore support content (matching the tone
# and topics referenced in prompt.py's SECTION 3 business rules — credit
# score, EMI, disputes, refresh cadence) but INVENTED for local testing,
# not sourced from any real GoodScore document.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnowledgeDocument:
    id: str
    title: str
    content: str
    tags: tuple[str, ...]


_DOCUMENTS: tuple[KnowledgeDocument, ...] = (
    KnowledgeDocument(
        id="kb-001",
        title="What is a credit score?",
        content=(
            "A credit score is a three-digit number, usually between 300 and 900, "
            "that summarises how reliably you have repaid borrowed money in the "
            "past. Lenders use it to decide whether to approve a loan or credit "
            "card, and at what interest rate. In India it is calculated by credit "
            "bureaus such as CIBIL, Equifax, and Experian using the repayment data "
            "reported by your banks and lenders."
        ),
        tags=("credit score", "basics", "cibil", "equifax"),
    ),
    KnowledgeDocument(
        id="kb-002",
        title="What counts as a good credit score?",
        content=(
            "Generally, a score of 750 or above is considered good and improves "
            "your chances of loan approval at better interest rates. 650-749 is "
            "fair, and below 650 may lead to rejections or higher interest rates. "
            "Different bureaus can show slightly different scores for the same "
            "person because they may not all have identical reporting data."
        ),
        tags=("credit score", "good score", "range"),
    ),
    KnowledgeDocument(
        id="kb-003",
        title="What factors affect your credit score?",
        content=(
            "Five main factors affect your credit score: payment history (paying "
            "EMIs and bills on time is the biggest factor), credit utilization "
            "(how much of your available credit limit you're using — lower is "
            "better), length of credit history, credit mix (a healthy balance of "
            "secured and unsecured credit), and the number of recent hard "
            "inquiries (too many loan/card applications in a short time can lower "
            "your score)."
        ),
        tags=("credit score", "factors", "utilization", "payment history"),
    ),
    KnowledgeDocument(
        id="kb-004",
        title="What is credit utilization and what ratio is ideal?",
        content=(
            "Credit utilization is the percentage of your total available credit "
            "limit that you are currently using across all your credit cards. For "
            "example, if your total limit is Rs. 1,00,000 and your outstanding "
            "balance is Rs. 30,000, your utilization is 30%. Keeping utilization "
            "below 30% is generally considered good for your credit score; going "
            "above 50% regularly can start to pull your score down."
        ),
        tags=("credit utilization", "credit card", "ratio"),
    ),
    KnowledgeDocument(
        id="kb-005",
        title="Does checking my own credit score lower it?",
        content=(
            "No. Checking your own credit score — like you do in the GoodScore "
            "app — is called a soft inquiry and has no impact on your score. Only "
            "a hard inquiry, which happens when a lender checks your report "
            "because you formally applied for a loan or credit card, can cause a "
            "small, temporary dip."
        ),
        tags=("soft inquiry", "hard inquiry", "myth", "credit check"),
    ),
    KnowledgeDocument(
        id="kb-006",
        title="What is a hard inquiry vs a soft inquiry?",
        content=(
            "A soft inquiry happens when you check your own score or when a "
            "company does a background check without it being tied to a credit "
            "application — it is invisible to lenders and doesn't affect your "
            "score. A hard inquiry happens when you apply for a loan, credit "
            "card, or similar credit product and the lender pulls your report to "
            "decide whether to approve you — this is visible to other lenders and "
            "can cause a small, temporary drop in your score if there are several "
            "in a short period."
        ),
        tags=("hard inquiry", "soft inquiry", "credit report"),
    ),
    KnowledgeDocument(
        id="kb-007",
        title="How often does my credit score refresh?",
        content=(
            "Credit bureaus typically update your score every 30 to 45 days as "
            "lenders report new repayment data to them. In the GoodScore app, "
            "you can request an on-demand score refresh once your last update is "
            "at least 30 days old — refreshing more often than that usually won't "
            "show a different number because the bureau's underlying data hasn't "
            "changed yet."
        ),
        tags=("score refresh", "update frequency", "bureau"),
    ),
    KnowledgeDocument(
        id="kb-008",
        title="What is an EMI and how does missing one affect my score?",
        content=(
            "EMI stands for Equated Monthly Instalment — a fixed monthly payment "
            "you make to repay a loan or credit card balance over time, covering "
            "both principal and interest. Missing an EMI payment is reported to "
            "the credit bureaus by your lender, usually after a short grace "
            "period, and can meaningfully lower your credit score because payment "
            "history is the single biggest factor bureaus look at. Paying even a "
            "few days late repeatedly can have a similar effect to a full miss."
        ),
        tags=("emi", "missed payment", "credit score impact"),
    ),
    KnowledgeDocument(
        id="kb-009",
        title="How do I dispute an error on my credit report?",
        content=(
            "If you spot an account, balance, or payment status on your credit "
            "report that looks incorrect, you can raise a dispute directly with "
            "the credit bureau (CIBIL, Equifax, or Experian) that issued the "
            "report, or with the lender that reported the information. The bureau "
            "typically investigates within 30 days and corrects the record if the "
            "lender confirms the error. Keep any supporting documents, like "
            "payment receipts, handy when you raise the dispute."
        ),
        tags=("dispute", "credit report error", "correction"),
    ),
    KnowledgeDocument(
        id="kb-010",
        title="What does GoodScore's subscription include?",
        content=(
            "A GoodScore subscription gives you ongoing access to your credit "
            "score and report from a linked bureau, monthly refreshes, a score "
            "simulator to see how actions like paying down a card might affect "
            "your score, and alerts if something changes on your report. You can "
            "check your plan status, renewal date, and autopay settings any time "
            "from the app's account settings."
        ),
        tags=("subscription", "goodscore plan", "monitoring"),
    ),
    KnowledgeDocument(
        id="kb-011",
        title="What is a NOC and why does it matter after closing a loan?",
        content=(
            "A No Objection Certificate (NOC) is a document your lender issues "
            "after you've fully repaid a loan, confirming you have no further "
            "dues. It's important to collect and keep the NOC because it's the "
            "proof lenders and bureaus rely on to correctly mark the account as "
            "closed on your credit report — without it, an account can "
            "occasionally keep showing as active even after you've paid it off."
        ),
        tags=("noc", "loan closure", "no objection certificate"),
    ),
    KnowledgeDocument(
        id="kb-012",
        title="Does GoodScore ever ask for my OTP?",
        content=(
            "No. GoodScore will never call, message, or email you asking for an "
            "OTP, PIN, or password. Anyone claiming to be from GoodScore and "
            "asking for your OTP is attempting fraud — do not share it, and "
            "report the contact through the app's support option."
        ),
        tags=("otp", "fraud", "security"),
    ),
)


_WORD_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_PATTERN.findall(text)}


def search(query: str, top_k: int = 3) -> list[dict]:
    """Rank dummy documents by keyword overlap with the query.

    Deliberately simple (word-set intersection, no embeddings/vector
    index) — this only needs to exercise the tool-call path realistically
    for local testing, not to be a good search algorithm. See module
    docstring: this is not a stand-in for the real S3 Vectors semantic
    search production would use.
    """
    query_words = _tokenize(query)
    if not query_words:
        return []

    scored: list[tuple[int, KnowledgeDocument]] = []
    for doc in _DOCUMENTS:
        doc_words = _tokenize(doc.title) | _tokenize(doc.content) | {t.lower() for t in doc.tags}
        overlap = len(query_words & doc_words)
        # Small bonus for tag matches — tags are curated topic labels, so a
        # hit there is a stronger signal than an incidental word in the body.
        tag_words = {t.lower() for t in doc.tags}
        overlap += len(query_words & tag_words)
        if overlap > 0:
            scored.append((overlap, doc))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {"id": doc.id, "title": doc.title, "content": doc.content}
        for _, doc in scored[:top_k]
    ]
