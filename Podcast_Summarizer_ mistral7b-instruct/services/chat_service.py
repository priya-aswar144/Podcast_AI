
import os
import re
import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Language code to name mapping (must match home.py)
LANGUAGE_NAMES = {
    'en': 'English', 'hi': 'Hindi', 'mr': 'Marathi',
    'ta': 'Tamil', 'te': 'Telugu', 'bn': 'Bengali',
    'gu': 'Gujarati', 'kn': 'Kannada', 'ml': 'Malayalam',
    'pa': 'Punjabi', 'ur': 'Urdu', 'es': 'Spanish',
    'fr': 'French', 'de': 'German', 'ja': 'Japanese',
    'ko': 'Korean', 'zh-cn': 'Chinese', 'ar': 'Arabic',
    'pt': 'Portuguese', 'ru': 'Russian', 'it': 'Italian',
}

def _get_language_name(code, default='the same language as the question'):
    """Convert a 2-letter ISO language code to a human-readable name."""
    return LANGUAGE_NAMES.get(code, default)


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

    def get_chat_response(self, full_text, summary, user_query, language='en', video_title=''):
        """
        Generates a response using the full transcript context and user query.
        Responds in the language of the user's query (detected automatically).
        Returns a dict: { 'answer': str, 'confidence_score': int, 'language': str }
        """
        try:
            if not full_text and not summary:
                return {
                    'answer': "I don't have enough information to answer that.",
                    'confidence_score': 0,
                    'language': language
                }

            # Detect language of the user query — respond in that language
            query_lang = language  # default to transcript language
            try:
                from langdetect import detect, LangDetectException
                detected = detect(user_query)
                if detected:
                    query_lang = detected
            except Exception:
                pass

            response_language_name = _get_language_name(query_lang)

            # Retrieve the most relevant chunks from the transcript
            context = self._get_relevant_chunks(full_text or summary, user_query)

            llm = ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "mistral:7b-instruct"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=0.3  # Lower temperature for more factual answers
            )

            # Build video metadata section from title (contains speaker names, topic, etc.)
            video_meta = f"Video Title: {video_title}" if video_title else ""

            prompt = PromptTemplate.from_template("""You are PodcastAI, an assistant that answers questions ONLY using the provided podcast information.

VIDEO METADATA (use this for speaker names, topic, channel info):
{video_meta}

PODCAST TRANSCRIPT EXCERPTS:
{context}

PODCAST SUMMARY (for additional context):
{summary}

USER QUESTION:
{question}

STRICT RULES:
1. Answer ONLY based on the video metadata, transcript excerpts, and summary provided above.
2. Do NOT use any external knowledge, assumptions, or information not found in the provided content.
3. If the question cannot be answered from the provided content, respond with EXACTLY: "This information is not available in the podcast."
4. Keep your answer clear, concise, and directly relevant to the question.
5. If the user greets you (e.g., "hi", "hello"), respond politely and briefly.
6. You MUST respond in {response_language}. This is critical — do not switch to another language.

After your answer, on a new line, provide a confidence rating in this EXACT format:
CONFIDENCE: [number from 0 to 100]

The confidence score should reflect how well the transcript content supports your answer:
- 90-100: Answer is directly and clearly stated in the transcript
- 70-89: Answer is strongly supported by the transcript
- 50-69: Answer is partially supported, some inference needed
- 0-49: Answer is weakly supported or not found in transcript""",
                                           )

            response = llm.invoke(prompt.format(
                video_meta=video_meta,
                context=context,
                summary=summary or "No summary available.",
                question=user_query,
                response_language=response_language_name
            ))

            response_text = response.content.strip()

            # parse language of the final answer in case model switched
            detected_lang = query_lang
            try:
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
            no_info_phrases = [
                "this information is not available in the podcast",
                "i don't know based on this podcast"
            ]
            if any(phrase in answer.lower() for phrase in no_info_phrases):
                confidence_score = min(confidence_score, 20)

            return {
                'answer': answer,
                'confidence_score': confidence_score,
                'language': detected_lang
            }

        except Exception as e:
            print(f"Chat error: {e}")
            return {
                'answer': "Sorry, I encountered an error processing your request.",
                'confidence_score': 0
            }

