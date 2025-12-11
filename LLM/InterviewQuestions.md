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

