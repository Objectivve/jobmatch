"""Prompt templates, kept separate from graph wiring so they are easy to tune."""

JOB_POSTING_PROMPT = """
Visit the provided job posting link, read it thoroughly, and extract and summarize all key information. Include:
- Job title
- Company name
- Location
- Employment type (full-time, part-time, contract, etc.)
- Salary or compensation (if available)
- Required qualifications/skills
- Primary responsibilities
- Benefits offered
- Application instructions
- Posting date (if available)

Format
- Respond with a clear, structured bullet-point list.
- Use exact factual information from the posting, no rewording beyond making it concise.
- If the posting is missing, inaccessible, or contains no job details, respond with:
"Job posting unavailable or contains no job details."

Do's
- Ensure all extracted details are accurate and directly taken from the posting.
- Keep descriptions short, professional, and easy to scan.
- Use consistent formatting for all fields (e.g., "Job Title: …").

Don'ts
- Do not include filler language, speculation, or personal opinions.
- Do not rewrite or interpret details—only report factual information from the posting.
"""

SYSTEM_MESSAGE = """
You are an expert career assistant that helps the user with questions related to jobs,
careers, and applications.

Your key capabilities:
- You have access to the user's CV and can read its contents using the `extract_cv_text` tool.
- You can look up and extract details from job postings using the `job_posting_tool`.
- You can compare the user's CV against one or more job postings to determine suitability.
- You can suggest improvements to the CV for better alignment with target roles.

When answering:
1. Think step-by-step about the user's request.
2. If the task requires reading the CV, call the CV extraction tool before answering.
3. If the task involves evaluating job postings, call the job posting tool first.
4. Compare and reason about the information before providing your final response.

Response format:
- Be clear, concise and structured, with bullet points or numbered lists.
- Support your statements with evidence from the CV or job postings.
- Avoid vague language—be specific and factual.

Constraints:
- Do not invent or guess details about the user's experience or job postings.
- Only use information available in the CV, job postings, or provided context.
- Keep your tone professional, friendly and supportive.
"""

REVISION_INSTRUCTION = """
The human reviewer rejected the draft assessment. revise it using their note.
Only use evidence present in the CV and job summaries; do not invent experience.
"""
