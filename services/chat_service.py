
import os
import re
import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChatService:
    """
    Chat service that answers user questions using the full podcast transcript.
    Uses keyword-based chunk retrieval for long transcripts and LLM self-evaluation
    for confidence scoring.
    """

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )

    def _get_relevant_chunks(self, full_text, query, max_chunks=4):
        """
        Split transcript into chunks and rank by keyword relevance to the query.
        Returns the top-N most relevant chunks concatenated.
        """
        if not full_text or len(full_text) < 3000:
            # Short transcript — return entire text
            return full_text or ""

        # Split into chunks
        chunks = self.splitter.split_text(full_text)

        if len(chunks) <= max_chunks:
            return full_text

        # Simple keyword relevance scoring
        query_words = set(re.findall(r'\w+', query.lower()))
        # Remove very short / stop words
        query_words = {w for w in query_words if len(w) > 2}

        scored = []
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            score = sum(1 for word in query_words if word in chunk_lower)
            scored.append((score, i, chunk))

        # Sort by score descending, then by position (to preserve context flow)
        scored.sort(key=lambda x: (-x[0], x[1]))

        # Pick top chunks, then re-sort by position for coherent context
        selected = scored[:max_chunks]
        selected.sort(key=lambda x: x[1])

        return "\n\n---\n\n".join(s[2] for s in selected)

    def get_chat_response(self, full_text, summary, user_query, language='en'):
        """
        Generates a response using the full transcript context and user query.  Replies in the
        specified language (default english) and returns the response, confidence score and
        language the model used.

        Returns a dict: { 'answer': str, 'confidence_score': int, 'language': str }
        """
        try:
            # Guard: empty query
            if not user_query or not user_query.strip():
                return {
                    'answer': "Please ask me a question about the podcast!",
                    'confidence_score': 0,
                    'language': language
                }

            # Guard: no podcast content at all
            if not full_text and not summary:
                return {
                    'answer': "I don't have any podcast content loaded yet. Please make sure the podcast has been processed first.",
                    'confidence_score': 0,
                    'language': language
                }

            # Detect language of the user query if possible
            try:
                from langdetect import detect, LangDetectException
                query_lang = detect(user_query)
            except Exception:
                query_lang = None
            # If query language differs from transcript language, default to English
            use_lang = language
            if query_lang and query_lang != language:
                use_lang = 'en'

            # Retrieve the most relevant chunks from the transcript
            context = self._get_relevant_chunks(full_text or summary, user_query)

            llm = ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=0.3  # Lower temperature for more factual answers
            )

            prompt = PromptTemplate.from_template("""You are PodcastAI, a helpful and friendly AI assistant that helps users understand podcast content.

PODCAST TRANSCRIPT EXCERPTS:
{context}

PODCAST SUMMARY (for additional context):
{summary}

USER QUESTION:
{question}

GUIDELINES:
1. If the user sends a greeting or a conversational message (e.g., "hi", "hello", "how are you", "thanks"), respond warmly and briefly — mention that you are ready to answer questions about this podcast.
2. For specific questions about the podcast content, answer thoroughly using the transcript excerpts and summary provided above.
3. If a question cannot be answered from the podcast content, politely say so and suggest the user ask something related to the podcast topics.
4. Keep your answers clear, helpful, and directly relevant to what was asked.
5. Do NOT fabricate information that is not in the transcript or summary.
6. Respond in this language: {language}. If the user writes in a different language, prefer English.

After your answer, on a new line, add a confidence rating in this EXACT format:
CONFIDENCE: [number from 0 to 100]

Confidence scale:
- 90-100: Answer is directly and clearly stated in the transcript
- 70-89: Answer is strongly supported by the transcript
- 50-69: Answer is partially supported, some inference needed
- 1-49: Answer is weakly supported or not found in transcript
- 0: This was a greeting or general conversational message""",
                                           )

            response = llm.invoke(prompt.format(
                context=context,
                summary=summary or "No summary available.",
                question=user_query,
                language=use_lang
            ))

            response_text = response.content.strip()

            # Parse language of the final answer in case model switched
            detected_lang = language
            try:
                # Very simple heuristic: if the answer contains non-ascii letters,
                # we delegate to langdetect if available
                from langdetect import detect, LangDetectException
                try:
                    detected_lang = detect(response_text)
                except LangDetectException:
                    pass
            except ImportError:
                pass

            # Parse confidence score from response
            confidence_score = 70  # default
            confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', response_text, re.IGNORECASE)
            if confidence_match:
                confidence_score = int(confidence_match.group(1))
                confidence_score = max(0, min(100, confidence_score))
                # Remove the confidence line from the answer
                answer = re.sub(r'\n*CONFIDENCE:\s*\d+.*$', '', response_text, flags=re.IGNORECASE).strip()
            else:
                answer = response_text

            # If the answer indicates no knowledge, set low confidence
            if "i don't know based on this podcast" in answer.lower():
                confidence_score = min(confidence_score, 20)

            return {
                'answer': answer,
                'confidence_score': confidence_score,
                'language': detected_lang
            }

        except Exception as e:
            print(f"Chat error: {e}")
            return {
                'answer': "Sorry, I encountered an error processing your request. Please try again.",
                'confidence_score': 0,
                'language': language
            }
