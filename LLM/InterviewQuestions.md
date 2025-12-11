# LLM Questions
### 1. How to handle LLM hallucinations, broken links, and incorrect data
To handle LLM hallucinations and incorrect or broken outputs, I follow a **three-layer strategy: Prevention, Detection, and Correction**.

 **1. Prevention (Reduce hallucinations before they happen)**

* **Use RAG (Retrieval-Augmented Generation)** to ground the model in verified data from internal knowledge bases, APIs, or databases.
* **Strong prompt instructions** such as “answer only from the provided context” or “say I don’t know if unsure.”
* **Provide high-quality, structured context** instead of relying on the model’s memory.

 **2. Detection (Catch issues after generation)**

* **Self-verification:** Ask the model to re-check its own response for unsupported claims.
* **Cross-model or dual-pass verification:** Use a second model or second pass to validate facts.
* **Rule-based validators:**

  * URL checker for broken links
  * Schema/format validators for structured data
  * Regex rules to detect invented numbers or citations

**3. Correction (Fix and regenerate safely)**

* **Regenerate answers with strict grounding** when validation fails.
* **Replace or remove broken links** using authoritative sources only.
* Implement **fallback responses** like “Not found in the provided data.”

 **4. Continuous Improvement**
 * Log incorrect outputs, user feedback, and error patterns.
 * Refresh the knowledge base regularly to avoid stale information.

### 2. Concise Answer – Output Guardrails

To enforce output guardrails, I use a layered approach:

1. **Schema Enforcement:** Validate outputs against a fixed JSON or text structure so the agent cannot produce uncontrolled formats.
2. **Rule-Based Filters:** Use regex, allow/deny lists, and content filters to block unsafe, sensitive, or out-of-scope responses.
3. **Verification Pass:** A second LLM pass checks for policy violations, hallucinations, broken links, or unsupported claims before sending the output.
4. **Fallback Logic:** If validation fails, regenerate the response with stricter instructions or return a safe fallback message.

**In short:** *Constrain the format, filter the content, verify the answer, and fall back safely when needed.*


# Agents Questions
### 1. How to handle Agent hallucinations, broken links, and incorrect data 
When designing LLM Agents, I handle hallucinations and incorrect outputs by using a tool-centric, verification-driven architecture.
I ground the agent using RAG or external tools (search, API, database) so it fetches facts instead of inventing them.
I add verification loops—self-check, second-pass validation, and schema/URL validators—to catch wrong facts or broken links.
Any failed validation triggers regeneration with strict grounding or fallback responses like “data not available.”
Finally, I monitor errors and update the knowledge base to prevent repeated hallucinations.

In short: Ground with tools, verify every step, validate outputs, and use fallback logic to ensure reliable agent behavior.

* Log incorrect outputs, user feedback, and error patterns.
* Refresh the knowledge base regularly to avoid stale information.

**I use a robust workflow of grounding (RAG), verification, rule-based validation, and feedback loops to minimize hallucinations and ensure the model outputs factual, reliable, and verifiable data.**

