"""
MedAI RAG Pipeline
PDF -> Chunks -> Embeddings -> Pinecone -> Retrieval -> Groq LLM
"""

import os
import logging
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader
)

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter
)

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

from langchain_pinecone import PineconeVectorStore

from pinecone import (
    Pinecone,
    ServerlessSpec
)


logger = logging.getLogger(__name__)


# =========================================================
# MEDICAL PROMPT
# =========================================================

MEDICAL_SYSTEM_PROMPT = """
You are MedAI, a friendly and knowledgeable medical information assistant.

Answer the user's question using ONLY the medical information supplied below.

STYLE:
- Speak naturally and conversationally.
- Answer the user's question directly.
- Keep simple questions concise.
- For simple factual questions, prefer 2-5 clear sentences.
- Use bullet points when listing several items.
- Do not automatically use tables.
- Do not repeatedly give disclaimers.
- Do not say "based on the context".
- Do not mention Pinecone, RAG, embeddings, vector databases,
  chunks, retrieval, documents, page numbers, or internal processes.
- Do not mention where the information came from.
- Do not expose internal sources.

MEDICAL SAFETY:
- Do not invent medical facts that are not supported by the supplied
  medical information.
- If the supplied information is insufficient, clearly say that the
  available medical information is not sufficient to answer accurately.
- Do not provide prescription dosages.
- Do not create personalized treatment plans.
- If symptoms could indicate an emergency, recommend seeking appropriate
  medical care.

IMPORTANT ANSWERING RULE:
- Use the supplied medical information as the factual basis.
- Do not add unsupported medical facts simply because they are commonly
  known.
- If only part of the user's question is supported, answer that part
  and clearly state what information is not available.

MEDICAL INFORMATION:
{context}

CONVERSATION:
{chat_history}

USER QUESTION:
{question}

Answer naturally and helpfully:
"""


class RAGPipeline:

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(self):

        print(
            "RAG: Initializing...",
            flush=True
        )

        self.index_name = os.getenv(
            "PINECONE_INDEX_NAME",
            "medai-knowledge"
        )

        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.chunk_size = int(
            os.getenv(
                "CHUNK_SIZE",
                "500"
            )
        )

        self.chunk_overlap = int(
            os.getenv(
                "CHUNK_OVERLAP",
                "20"
            )
        )

        self.top_k = int(
            os.getenv(
                "RETRIEVAL_TOP_K",
                "6"
            )
        )

        self.similarity_threshold = float(
            os.getenv(
                "RETRIEVAL_SCORE_THRESHOLD",
                "0.35"
            )
        )

        print(
            "RAG: Loading embedding model...",
            flush=True
        )

        self.embeddings = self._init_embeddings()

        print(
            "RAG: Connecting to Pinecone...",
            flush=True
        )

        self.vectorstore = self._init_vectorstore()

        print(
            "RAG: Initializing LLM...",
            flush=True
        )

        self.llm = self._init_llm()

        print(
            "RAG: Pipeline ready.",
            flush=True
        )

    # =====================================================
    # EMBEDDINGS
    # =====================================================

    def _init_embeddings(self):

        return HuggingFaceEmbeddings(

            model_name=self.embedding_model_name,

            model_kwargs={
                "device": "cpu"
            },

            encode_kwargs={
                "normalize_embeddings": True
            }
        )

    # =====================================================
    # PINECONE
    # =====================================================

    def _init_vectorstore(self):

        api_key = os.getenv(
            "PINECONE_API_KEY"
        )

        if not api_key:

            raise ValueError(
                "PINECONE_API_KEY environment variable not set."
            )

        pc = Pinecone(
            api_key=api_key
        )

        existing_indexes = [
            index.name
            for index in pc.list_indexes()
        ]

        if self.index_name not in existing_indexes:

            print(
                f"RAG: Creating Pinecone index "
                f"'{self.index_name}'...",
                flush=True
            )

            pc.create_index(

                name=self.index_name,

                dimension=384,

                metric="cosine",

                spec=ServerlessSpec(

                    cloud=os.getenv(
                        "PINECONE_CLOUD",
                        "aws"
                    ),

                    region=os.getenv(
                        "PINECONE_REGION",
                        "us-east-1"
                    )
                )
            )

            print(
                "RAG: Pinecone index created.",
                flush=True
            )

        return PineconeVectorStore(

            index_name=self.index_name,

            embedding=self.embeddings
        )

    # =====================================================
    # LLM
    # =====================================================

    def _init_llm(self):

        provider = os.getenv(
            "LLM_PROVIDER",
            "groq"
        ).lower()

        if provider == "groq":

            from langchain_groq import ChatGroq

            api_key = os.getenv(
                "GROQ_API_KEY"
            )

            if not api_key:

                raise ValueError(
                    "GROQ_API_KEY not set."
                )

            return ChatGroq(

                api_key=api_key,

                model_name="openai/gpt-oss-120b",

                temperature=0.2,

                max_tokens=700
            )

        else:

            from langchain_community.chat_models import ChatOpenAI

            api_key = os.getenv(
                "OPENAI_API_KEY"
            )

            if not api_key:

                raise ValueError(
                    "OPENAI_API_KEY not set."
                )

            return ChatOpenAI(

                api_key=api_key,

                model_name=os.getenv(
                    "OPENAI_MODEL",
                    "gpt-4-turbo-preview"
                ),

                temperature=0.2,

                max_tokens=700
            )

    # =====================================================
    # PDF INGESTION
    # =====================================================

    def ingest_documents(
        self,
        pdf_path: str
    ) -> Dict[str, Any]:

        print(
            f"RAG: Starting ingestion: {pdf_path}",
            flush=True
        )

        print(
            "RAG: Loading PDF...",
            flush=True
        )

        if os.path.isdir(pdf_path):

            loader = DirectoryLoader(

                pdf_path,

                glob="**/*.pdf",

                loader_cls=PyPDFLoader
            )

        else:

            if not os.path.exists(pdf_path):

                raise FileNotFoundError(
                    f"PDF not found: {pdf_path}"
                )

            loader = PyPDFLoader(
                pdf_path
            )

        documents = loader.load()

        print(
            f"RAG: Loaded {len(documents)} pages.",
            flush=True
        )

        print(
            "RAG: Creating text chunks...",
            flush=True
        )

        splitter = RecursiveCharacterTextSplitter(

            chunk_size=self.chunk_size,

            chunk_overlap=self.chunk_overlap,

            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ],

            length_function=len
        )

        chunks = splitter.split_documents(
            documents
        )

        print(
            f"RAG: Created {len(chunks)} chunks.",
            flush=True
        )

        if not chunks:

            raise ValueError(
                "No text chunks were created from the PDF."
            )

        print(
            "RAG: Creating vector IDs...",
            flush=True
        )

        ids = []

        for i, doc in enumerate(chunks):

            page = doc.metadata.get(
                "page",
                "unknown"
            )

            source = os.path.basename(
                doc.metadata.get(
                    "source",
                    "medical_book"
                )
            )

            source = (
                source
                .replace(" ", "_")
                .replace(".", "_")
            )

            ids.append(
                f"{source}_page_{page}_chunk_{i}"
            )

        print(
            "RAG: Starting embedding + Pinecone upload...",
            flush=True
        )

        vectorstore = PineconeVectorStore(

            index_name=self.index_name,

            embedding=self.embeddings
        )

        batch_size = 50

        total = len(chunks)

        for start in range(
            0,
            total,
            batch_size
        ):

            end = min(
                start + batch_size,
                total
            )

            batch_docs = chunks[
                start:end
            ]

            batch_ids = ids[
                start:end
            ]

            print(
                f"RAG: Embedding/uploading "
                f"{start + 1}-{end} of {total}...",
                flush=True
            )

            vectorstore.add_documents(

                documents=batch_docs,

                ids=batch_ids
            )

            print(
                f"RAG: Completed {end}/{total}.",
                flush=True
            )

        print(
            "RAG: PDF ingestion completed successfully.",
            flush=True
        )

        return {

            "status": "success",

            "pages_loaded": len(
                documents
            ),

            "chunks_created": len(
                chunks
            ),

            "index": self.index_name
        }

    # =====================================================
    # QUERY VARIANT GENERATION
    # =====================================================

    def _build_query_variants(
        self,
        question: str
    ) -> List[str]:

        question_lower = question.lower()

        variants = [
            question
        ]

        # -------------------------------------------------
        # TYPE 2 DIABETES
        # -------------------------------------------------

        if any(
            term in question_lower
            for term in [
                "type 2",
                "type ii",
                "type two",
                "t2d"
            ]
        ):

            variants.extend([

                "type II diabetes mellitus insulin resistance",

                "type 2 diabetes mellitus",

                "non insulin dependent diabetes mellitus",

                "type II diabetes clinical features",

                "type II diabetes signs symptoms"
            ])

        # -------------------------------------------------
        # TYPE 1 DIABETES
        # -------------------------------------------------

        if any(
            term in question_lower
            for term in [
                "type 1",
                "type i",
                "type one",
                "t1d"
            ]
        ):

            variants.extend([

                "type I diabetes mellitus insulin deficiency",

                "type 1 diabetes mellitus",

                "insulin dependent diabetes mellitus",

                "type I diabetes clinical features",

                "type I diabetes signs symptoms"
            ])

        # -------------------------------------------------
        # DIABETES
        # -------------------------------------------------

        if "diabetes" in question_lower:

            variants.append(
                "diabetes mellitus blood glucose insulin"
            )

        # -------------------------------------------------
        # SYMPTOMS
        # -------------------------------------------------

        if any(
            term in question_lower
            for term in [
                "symptom",
                "symptoms",
                "sign",
                "signs",
                "clinical feature",
                "clinical features",
                "manifestation",
                "manifestations"
            ]
        ):

            variants.extend([

                "diabetes mellitus signs symptoms",

                "diabetes clinical manifestations",

                "diabetes symptoms clinical features"
            ])

        # -------------------------------------------------
        # CAUSES
        # -------------------------------------------------

        if any(
            word in question_lower
            for word in [
                "cause",
                "causes",
                "caused",
                "why"
            ]
        ):

            variants.append(
                "diabetes causes etiology"
            )

        # -------------------------------------------------
        # TREATMENT
        # -------------------------------------------------

        if any(
            word in question_lower
            for word in [
                "treatment",
                "treat",
                "therapy",
                "management"
            ]
        ):

            variants.append(
                "diabetes treatment therapy management"
            )

        # -------------------------------------------------
        # Remove duplicate variants
        # -------------------------------------------------

        unique_variants = []

        seen = set()

        for variant in variants:

            normalized = (
                " ".join(
                    variant.lower().split()
                )
            )

            if normalized not in seen:

                seen.add(
                    normalized
                )

                unique_variants.append(
                    variant
                )

        return unique_variants

    # =====================================================
    # RETRIEVAL + RERANKING
    # =====================================================

    def _retrieve_relevant_documents(
        self,
        question: str
    ):

        print(
            "RAG: Searching Pinecone...",
            flush=True
        )

        # -------------------------------------------------
        # IMPORTANT FIX
        # -------------------------------------------------

        question_lower = question.lower()

        # -------------------------------------------------
        # Generate query variants
        # -------------------------------------------------

        query_variants = (
            self._build_query_variants(
                question
            )
        )

        print(
            f"RAG: Using {len(query_variants)} "
            f"query variants.",
            flush=True
        )

        # -------------------------------------------------
        # Retrieve candidates
        # -------------------------------------------------

        candidates = {}

        for query in query_variants:

            print(
                f"RAG: Searching variant: {query}",
                flush=True
            )

            results = (
                self.vectorstore
                .similarity_search_with_score(
                    query,
                    k=30
                )
            )

            for doc, score in results:

                text = (
                    doc.page_content
                    .strip()
                )

                if not text:
                    continue

                normalized = (
                    " ".join(
                        text.lower().split()
                    )
                )

                if (
                    normalized not in candidates
                    or score >
                    candidates[
                        normalized
                    ]["semantic_score"]
                ):

                    candidates[
                        normalized
                    ] = {

                        "doc": doc,

                        "semantic_score": score
                    }

        print(
            f"RAG: Collected "
            f"{len(candidates)} "
            f"unique candidate chunks.",
            flush=True
        )

        # -------------------------------------------------
        # Prepare question words
        # -------------------------------------------------

        question_clean = (
            question_lower
            .replace("?", "")
            .replace(",", "")
            .replace(".", "")
            .replace("!", "")
            .replace(":", "")
            .replace(";", "")
        )

        question_words = set(

            word

            for word in question_clean.split()

            if len(word) > 2
        )

        # -------------------------------------------------
        # Rerank candidates
        # -------------------------------------------------

        ranked = []

        for item in candidates.values():

            doc = item["doc"]

            semantic_score = (
                item["semantic_score"]
            )

            text_lower = (
                doc.page_content.lower()
            )

            # ---------------------------------------------
            # Keyword overlap
            # ---------------------------------------------

            keyword_matches = sum(

                1

                for word in question_words

                if word in text_lower
            )

            lexical_score = min(
                keyword_matches / 8.0,
                1.0
            )

            # ---------------------------------------------
            # Concept boost
            # ---------------------------------------------

            concept_boost = 0.0

            # ---------------------------------------------
            # Type 2 diabetes
            # ---------------------------------------------

            if any(
                term in question_lower
                for term in [
                    "type 2",
                    "type ii",
                    "type two",
                    "t2d"
                ]
            ):

                if "type ii" in text_lower:
                    concept_boost += 0.35

                if "type 2" in text_lower:
                    concept_boost += 0.35

                if "insulin resistance" in text_lower:
                    concept_boost += 0.25

                if (
                    "non insulin-dependent"
                    in text_lower
                ):
                    concept_boost += 0.25

                if (
                    "non-insulin-dependent"
                    in text_lower
                ):
                    concept_boost += 0.25

                if (
                    "non insulin dependent"
                    in text_lower
                ):
                    concept_boost += 0.25

            # ---------------------------------------------
            # Type 1 diabetes
            # ---------------------------------------------

            if any(
                term in question_lower
                for term in [
                    "type 1",
                    "type i",
                    "type one",
                    "t1d"
                ]
            ):

                if "type i" in text_lower:
                    concept_boost += 0.35

                if "type 1" in text_lower:
                    concept_boost += 0.35

                if "insulin deficiency" in text_lower:
                    concept_boost += 0.25

                if "insulin" in text_lower:
                    concept_boost += 0.15

            # ---------------------------------------------
            # Diabetes
            # ---------------------------------------------

            if "diabetes" in question_lower:

                if "diabetes" in text_lower:
                    concept_boost += 0.15

                if "diabetes mellitus" in text_lower:
                    concept_boost += 0.15

            # ---------------------------------------------
            # Symptoms
            # ---------------------------------------------

            if any(
                term in question_lower
                for term in [
                    "symptom",
                    "symptoms",
                    "sign",
                    "signs",
                    "clinical feature",
                    "clinical features",
                    "manifestation"
                ]
            ):

                if "symptom" in text_lower:
                    concept_boost += 0.20

                if "symptoms" in text_lower:
                    concept_boost += 0.20

                if "clinical" in text_lower:
                    concept_boost += 0.10

                if "manifestation" in text_lower:
                    concept_boost += 0.10

            # ---------------------------------------------
            # Causes
            # ---------------------------------------------

            if any(
                word in question_lower
                for word in [
                    "cause",
                    "causes",
                    "caused",
                    "why"
                ]
            ):

                if "cause" in text_lower:
                    concept_boost += 0.15

                if "etiology" in text_lower:
                    concept_boost += 0.15

            # ---------------------------------------------
            # Treatment
            # ---------------------------------------------

            if any(
                word in question_lower
                for word in [
                    "treatment",
                    "treat",
                    "therapy",
                    "management"
                ]
            ):

                if "treatment" in text_lower:
                    concept_boost += 0.15

                if "therapy" in text_lower:
                    concept_boost += 0.15

                if "management" in text_lower:
                    concept_boost += 0.15

            # ---------------------------------------------
            # Final score
            # ---------------------------------------------

            final_score = (

                semantic_score * 0.60

                + lexical_score * 0.20

                + concept_boost * 0.20
            )

            ranked.append(

                (
                    final_score,
                    semantic_score,
                    doc
                )
            )

        # -------------------------------------------------
        # Sort
        # -------------------------------------------------

        ranked.sort(
            key=lambda x: x[0],
            reverse=True
        )

        # -------------------------------------------------
        # Select final documents
        # -------------------------------------------------

        relevant_docs = []

        for (
            final_score,
            semantic_score,
            doc
        ) in ranked:

            if (
                semantic_score
                < self.similarity_threshold
            ):
                continue

            relevant_docs.append(
                doc
            )

            if (
                len(relevant_docs)
                >= self.top_k
            ):
                break

        print(
            f"RAG: Retrieved "
            f"{len(relevant_docs)} "
            f"relevant chunks after reranking.",
            flush=True
        )

        # -------------------------------------------------
        # Development logging
        # -------------------------------------------------

        for i, doc in enumerate(
            relevant_docs,
            start=1
        ):

            print(
                f"RAG: Selected chunk {i} "
                f"- page "
                f"{doc.metadata.get('page', 'unknown')}",
                flush=True
            )

        return relevant_docs

    # =====================================================
    # CHAT QUERY
    # =====================================================

    def query(
        self,
        question: str,
        chat_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:

        if (
            not question
            or not question.strip()
        ):

            return {

                "answer": (
                    "Please enter a medical question "
                    "so MedAI can help."
                ),

                "sources": []
            }

        # -------------------------------------------------
        # Conversation history
        # -------------------------------------------------

        history_text = ""

        if chat_history:

            history_lines = []

            for message in chat_history:

                role = message.get(
                    "role",
                    ""
                )

                content = message.get(
                    "content",
                    ""
                )

                if role == "user":

                    history_lines.append(
                        f"User: {content}"
                    )

                elif role == "assistant":

                    history_lines.append(
                        f"MedAI: {content}"
                    )

            history_text = "\n".join(
                history_lines[-10:]
            )

        # -------------------------------------------------
        # Retrieve
        # -------------------------------------------------

        relevant_docs = (
            self._retrieve_relevant_documents(
                question
            )
        )

        # -------------------------------------------------
        # No relevant information
        # -------------------------------------------------

        if not relevant_docs:

            return {

                "answer": (
                    "The available medical information "
                    "does not contain enough information "
                    "to answer that accurately."
                ),

                "sources": []
            }

        # -------------------------------------------------
        # Build context
        # -------------------------------------------------

        context = "\n\n".join(

            doc.page_content

            for doc in relevant_docs
        )

        # -------------------------------------------------
        # Build prompt
        # -------------------------------------------------

        prompt = MEDICAL_SYSTEM_PROMPT.format(

            context=context,

            chat_history=history_text,

            question=question
        )

        print(
            "RAG: Asking LLM...",
            flush=True
        )

        # -------------------------------------------------
        # LLM
        # -------------------------------------------------

        response = self.llm.invoke(
            prompt
        )

        answer = (
            response.content
            .strip()
        )

        return {

            "answer": answer,

            "sources": []
        }


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    print(
        "Starting MedAI RAG Pipeline...",
        flush=True
    )

    pipeline = RAGPipeline()

    result = pipeline.ingest_documents(
        "data/medical_book.pdf"
    )

    print(
        result,
        flush=True
    )