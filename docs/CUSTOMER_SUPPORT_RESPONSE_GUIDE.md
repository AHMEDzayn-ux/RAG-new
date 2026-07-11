# Customer Support Response Generation Guide

## Overview

This guide covers the **Generation & Synthesis** layer of the RAG system - how responses are crafted for customer support with strict accuracy, proper citations, and empathetic tone.

---

## 1. Strict "I Don't Know" Protocol

### The Problem

**Hallucinations are worse than no answer** in customer support. Giving wrong advice can:

- Violate policies (refund amounts, SLAs, legal terms)
- Damage customer trust
- Create internal escalations
- Lead to liability issues

### Our Solution: Fail-Safe Protocol

The system is programmed with **explicit "I don't know" triggers** that prevent hallucination:

```
❌ NEVER make up information
❌ NEVER guess or speculate
❌ NEVER use knowledge outside the provided context
✅ If context doesn't contain the answer, respond with escalation offer
```

### Automatic Response Template

When the system lacks information, it responds:

> _"I don't have enough information in our documentation to answer that question accurately. I'd be happy to:_
>
> 1. _Connect you with a human support agent who can help_
> 2. _Research this further and get back to you_
>
> _Would you prefer option 1 or 2?"_

### Implementation Details

**In LLM Prompts** ([llm_service.py](backend/services/llm_service.py)):

```python
STRICT "I DON'T KNOW" PROTOCOL (CRITICAL):
❌ NEVER make up information
❌ NEVER guess or speculate
❌ NEVER use knowledge outside the provided context
✅ If the context doesn't contain the answer, respond EXACTLY:
   "I don't have enough information..."
```

**Automated Detection** ([rag_pipeline.py](backend/services/rag_pipeline.py)):

```python
def _detect_uncertainty(response: str) -> bool:
    """Detect if LLM response indicates uncertainty."""
    uncertainty_phrases = [
        "don't have enough information",
        "i don't know",
        "cannot answer",
        "connect you with a human",
        "escalate to"
    ]
    return any(phrase in response.lower() for phrase in uncertainty_phrases)
```

**Frontend Warning Display**:

```jsx
{
  msg.is_uncertain && (
    <div className="uncertainty-warning">
      ⚠️ Limited Information: This answer may be incomplete or require human
      verification.
    </div>
  );
}
```

---

## 2. Citation & Grounding

### The Problem

Users don't trust unsourced information. They need to verify claims, especially for:

- Policy interpretations
- Technical procedures
- Pricing/billing
- Legal compliance

### Our Solution: Mandatory Citations

**Every factual statement must cite its source:**

✅ Good Example:

> "According to [Source 1], enterprise users can request refunds within 30 days of purchase. The refund policy [Source 2] states that shipping fees are non-refundable."

❌ Bad Example:

> "Enterprise users can request refunds within 30 days. Shipping fees are non-refundable."

### Implementation

**LLM Prompt Instructions**:

```
CITATION & GROUNDING:
- ALWAYS cite sources when stating facts: "According to [Source 1]..."
- Reference specific section/document names in citations
- If multiple sources say the same thing, cite all: [Source 1][Source 2]
- Format: [Context 1], [Context 2], matching the numbered contexts provided
```

**Context Numbering** ([llm_service.py](backend/services/llm_service.py)):

```python
def _build_user_message(self, query: str, context: Optional[List[str]] = None):
    context_str = "\n\n".join([
        f"[Context {i+1}]\n{ctx}"
        for i, ctx in enumerate(context)
    ])
```

**Source Formatting** ([rag_pipeline.py](backend/services/rag_pipeline.py)):

```python
def _format_sources_for_citations(self, documents) -> List[Dict]:
    """Format sources with citation IDs and relevance."""
    for idx, doc in enumerate(documents, start=1):
        source_info = {
            'citation_id': idx,
            'citation_label': f"Source {idx}",
            'text': doc['text'],
            'relevance': self._distance_to_relevance(doc['distance']),
            'source_file': metadata.get('filename', 'Unknown'),
            'section': metadata.get('section', 'General')
        }
```

### Frontend Display

**Citation Badges**:

```jsx
<span className="citation-label">
    [Source 1]
</span>
<span className="relevance-badge relevance-highly-relevant">
    Highly Relevant
</span>
```

**Expandable Sources**:
Users can click "View Sources" to see:

- Full source text
- Filename and section
- Relevance score
- Distance metric (for debugging)

---

## 3. Confidence Scoring

### The Problem

Not all answers are equally reliable. Users need to know **how confident** the system is.

### Our Solution: Automatic Confidence Calculation

Confidence is based on **semantic distance** of retrieved documents:

- **Lower distance = Higher confidence** (closer match)
- **Higher distance = Lower confidence** (weaker match)

### Confidence Levels

| Distance | Confidence | Label | Color     | Meaning                        |
| -------- | ---------- | ----- | --------- | ------------------------------ |
| < 0.3    | 0.95       | 95%   | 🟢 Green  | Very high - near-perfect match |
| 0.3-0.5  | 0.85       | 85%   | 🟢 Green  | High - strong match            |
| 0.5-0.8  | 0.70       | 70%   | 🟡 Yellow | Moderate - acceptable match    |
| 0.8-1.2  | 0.50       | 50%   | 🟠 Orange | Low - weak match               |
| > 1.2    | 0.30       | 30%   | 🔴 Red    | Very low - poor match          |

### Implementation

```python
def _calculate_confidence(self, documents: List[Dict]) -> float:
    """Calculate confidence from retrieval distances."""
    if not documents:
        return 0.0

    # Get average distance of top 3 documents
    distances = [doc.get('distance', 1.0) for doc in documents[:3]]
    avg_distance = sum(distances) / len(distances)

    # Map distance to confidence
    if avg_distance < 0.3:
        return 0.95  # Very high
    elif avg_distance < 0.5:
        return 0.85  # High
    # ... etc
```

### Frontend Display

**Confidence Bar**:

```jsx
<div className="confidence-indicator confidence-high">
  <span>Confidence:</span>
  <div className="confidence-bar">
    <div className="confidence-fill" style={{ width: "85%" }}></div>
  </div>
  <span>85%</span>
</div>
```

**Color-Coded Indicator**:

- 🟢 **Green** (≥ 70%): Trust this answer
- 🟡 **Yellow** (50-69%): Verify if critical
- 🔴 **Red** (< 50%): Needs human review

---

## 4. Empathetic Tone Matching

### The Problem

Technical accuracy isn't enough. Customer support requires:

- **Empathy** for frustrated users
- **Clarity** for confused users
- **Warmth** to build trust
- **Professionalism** to maintain credibility

### Our Solution: Context-Aware Tone

**System Prompt** ([llm_service.py](backend/services/llm_service.py)):

```
EMPATHY & TONE:
- Read the user's emotional state (frustrated, confused, urgent)
- Acknowledge emotions first: "I understand how frustrating this must be..."
- Use clear, simple language - avoid technical jargon
- Be warm: "I'd be happy to help!", "Great question!"
- Stay professional but friendly
```

### Tone Guidelines

**For Frustrated Users:**

```
Bad: "The refund policy is in article 5.2.3."
Good: "I understand how frustrating this must be. Let me help you with that refund right away."
```

**For Confused Users:**

```
Bad: "Navigate to Settings > API Keys > Generate Token."
Good: "No worries! Here's a simple step-by-step guide to set that up..."
```

**For Technical Users:**

```
Mirror their level:
User: "What's the rate limit on the /v1/users endpoint?"
Bot: "The /v1/users endpoint is rate-limited to 100 req/min per API key."
```

**For Non-Technical Users:**

```
Simplify jargon:
Instead of: "Your OAuth token expired."
Say: "Your login session timed out. Let's get you logged back in!"
```

### Conversation Flow

The system is trained to:

1. **Acknowledge** the user's situation
2. **Provide** clear, actionable answer
3. **Offer** next steps or alternatives
4. **Check** if user needs more help

Example:

> "I understand you're having trouble accessing your account. Let me help you reset that right away.
>
> According to [Source 1], you can reset your password by clicking 'Forgot Password' on the login page...
>
> Would you like me to walk you through the steps, or would you prefer a direct link?"

---

## 5. Response Structure

Every assistant response follows this structure:

### 1. Uncertainty Warning (if applicable)

```
⚠️ Limited Information: This answer may be incomplete...
```

### 2. Confidence Indicator (if applicable)

```
Confidence: [████████░░] 85%
```

### 3. Summary (first line, highlighted)

```
[Highlighted] Short, direct answer in 1-2 sentences
```

### 4. Details (remaining text)

```
Detailed explanation with citations [Source 1][Source 2]
```

### 5. Sources (expandable)

```
📚 View Sources (3)
  [Source 1] [Highly Relevant]
  "Refund policy states..."
  📄 refund_policy_eu.pdf → Billing
```

---

## 6. Example Responses

### High Confidence, Clear Answer

**User:** "What's your refund policy for enterprise users?"

**Response:**

```
Confidence: [███████████] 95%

Enterprise users can request full refunds within 30 days of purchase.

According to [Source 1], enterprise customers are eligible for full refunds
within 30 days of purchase for any reason. The refund [Source 2] typically
processes within 5-7 business days to the original payment method.

To request a refund, please contact your account manager or email
billing@company.com with your order number.

📚 View Sources (2)
```

### Low Confidence, Uncertain

**User:** "Do you support SSO with Azure AD version 2.0?"

**Response:**

```
⚠️ Limited Information: This answer may require human verification.
Confidence: [████░░░░░░] 40%

I don't have enough information in our documentation to answer that specific
question about Azure AD 2.0 support accurately.

I'd be happy to:
1. Connect you with our technical support team who can verify this
2. Research this further and get back to you within 24 hours

Which would you prefer?
```

### Frustrated User, Empathetic Response

**User:** "This is ridiculous! I've been trying to download my invoice for 2 hours!"

**Response:**

```
Confidence: [██████████░] 90%

I completely understand how frustrating that must be - let me help you
get that invoice right away.

According to [Source 1], you can download invoices by going to Account >
Billing > Invoice History. If you're having trouble accessing that page,
it might be a browser cache issue.

Here's what you can try:
1. Clear your browser cache (Ctrl+Shift+Delete)
2. Try in an incognito/private window
3. Or I can email the invoice directly to you right now

Which option works best for you?

📚 View Sources (1)
```

---

## 7. API Response Structure

### Query/Chat Response

```json
{
  "question": "What's the refund policy?",
  "answer": "Enterprise users can request full refunds...",
  "confidence": 0.85,
  "is_uncertain": false,
  "num_sources": 3,
  "sources_available": true,
  "sources": [
    {
      "citation_id": 1,
      "citation_label": "Source 1",
      "text": "Refund policy states...",
      "relevance": "Highly Relevant",
      "distance": 0.245,
      "source_file": "refund_policy_eu.pdf",
      "section": "Billing"
    }
  ]
}
```

---

## 8. Testing Your System

### Test Cases

**1. Test "I Don't Know" Protocol:**

```
Query: "What's the CEO's favorite color?"
Expected: "I don't have enough information... connect you with a human..."
```

**2. Test Citation Grounding:**

```
Query: "What's the refund timeframe?"
Expected: Response must contain [Source X] citations
```

**3. Test Confidence Scoring:**

```
Query: Exact match from docs → Should be 85-95% confidence
Query: Vague/ambiguous → Should be 30-50% confidence
```

**4. Test Empathy:**

```
Query: "This doesn't work! So frustrated!!!"
Expected: Response starts with acknowledgment: "I understand..."
```

---

## 9. Monitoring & Analytics

### Key Metrics to Track

1. **Uncertainty Rate**: % of responses with `is_uncertain: true`
   - Target: < 10% for well-indexed docs

2. **Average Confidence**: Mean confidence across all responses
   - Target: > 0.75

3. **Low Confidence Alerts**: Responses with confidence < 0.5
   - Action: Review and improve documentation

4. **Escalation Rate**: How often "connect with human" is offered
   - Target: < 5%

5. **Citation Coverage**: % of responses with citations
   - Target: > 90% for factual queries

---

## 10. Best Practices

### DO ✅

- Always cite sources for factual claims
- Acknowledge user emotions before answering
- Use confidence indicators for transparency
- Offer escalation when uncertain
- Keep answers clear and jargon-free

### DON'T ❌

- Never guess or speculate
- Don't provide advice not in documentation
- Don't ignore low confidence warnings
- Don't use technical jargon without explanation
- Don't give answers without citations

---

## Summary

This RAG system implements **production-grade customer support** with:

✅ **Strict "I Don't Know" Protocol**: Prevents hallucinations  
✅ **Mandatory Citations**: Every claim is sourced  
✅ **Confidence Scoring**: Transparency about reliability  
✅ **Empathetic Tone**: Acknowledges emotions  
✅ **Uncertainty Warnings**: Flags incomplete answers

The result: **Trustworthy, accurate support responses** that build customer confidence and reduce escalations.
