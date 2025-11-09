from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import OllamaEmbeddings, ChatOllama

from langgraph.graph import StateGraph
from typing_extensions import TypedDict

import dspy

try:
    from mcp_use import MCPClient
except Exception:
    MCPClient = None 


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)5s | %(name)s | %(message)s"
)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))


DATA_DIR = Path(os.getenv("DATA_DIR", str(Path.home() / "MATH/train")))

INDEX_DIR = Path(os.environ.get("INDEX_DIR", Path(__file__).resolve().parent / ".index"))

SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", 0.60))

TOP_K = int(os.environ.get("TOP_K", 3))

TAVILY_API_KEY = os.environ.get(
    "TAVILY_API_KEY", "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"
)

# Input/Output Guardrails
BAD_INPUT_KEYWORDS = {
    "kill", "murder", "suicide", "attack", "bomb", "explosive", "weapon",
    "hack", "steal", "drug", "sex", "porn", "violent", "crime", "terror",
}

BAD_OUTPUT_KEYWORDS = {
    "kill", "murder", "suicide", "attack", "bomb", "explosive", "weapon",
    "hack", "steal", "drug", "sex", "porn", "violent", "crime", "terror",
}

def input_guardrail_ok(prompt: str) -> bool:
    text = (prompt or "").lower()
    for kw in BAD_INPUT_KEYWORDS:
        if kw in text:
            return False
    return True

def output_guardrail_ok(response: str) -> bool:
    text = (response or "").lower()
    for kw in BAD_OUTPUT_KEYWORDS:
        if kw in text:
            return False
    return True

def _looks_math_text(text: str) -> bool:
    if not text:
        return False
    t = text.lower()

    if any(ch in t for ch in "0123456789=+-*/^%"):
        return True

    math_keywords = [
        "solve", "equation", "fraction", "algebra", "geometry", "calculus",
        "integral", "derivative", "limit", "matrix", "vector", "sum", "product",
        "theorem", "proof", "probability", "median", "mean", "variance",
        "mod", "gcd", "lcm", "series"
    ]
    if any(kw in t for kw in math_keywords):
        return True

    if any(sym in t for sym in ("√", "π", "∑", "∫", "≤", "≥", "≈")):
        return True

    return False

# LLM-based math classification guardrail 
async def llm_is_math(llm: ChatOllama, text: str) -> bool:
    if _looks_math_text(text):
        return True

    classification_prompt = (
        "Respond with 'yes' if the following text solely involves mathematics or solving or"
        "mathematical problems. Respond with 'no' if it contains any non-mathematical content. "
        "Return only 'yes' or 'no'.\n\nText: {text}\nAnswer:"
    ).format(text=text)

    try:
        result = await llm.ainvoke(classification_prompt)
        raw = (result.content or "").strip().lower()
        normalized = re.sub(r"[^a-z]", "", raw)
        if "yes" in normalized and "no" not in normalized:
            return True
        if "no" in normalized and "yes" not in normalized:
            return _looks_math_text(text)
        return _looks_math_text(text)

    except Exception:
        return _looks_math_text(text)



# Data loading and indexing
def load_math_documents(data_dir: Path) -> List[Document]:
    documents: List[Document] = []

    logger.debug("load_math_documents: Loading documents from %s", data_dir)
    if not data_dir.exists():
        logger.warning("Data directory %s does not exist. No documents loaded.", data_dir)
        return documents
    for root, _, files in os.walk(data_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            path = Path(root) / file
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                logger.debug("load_math_documents: failed to parse JSON at %s", path)
                continue
            problem = data.get("problem", "").strip()
            solution = data.get("solution", "").strip()
            if not problem:
                logger.debug("load_math_documents: skipping empty 'problem' at %s", path) 
                continue
            content = problem
            if solution:
                content += "\n\nSolution: " + solution
            documents.append(Document(page_content=content, metadata={"source": str(path)}))
    logger.info("Loaded %d documents from %s", len(documents), data_dir)
    return documents


def build_or_load_index(index_dir: Path, data_dir: Path) -> FAISS:

    index_path = index_dir / "index.faiss"
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    if index_path.exists():

        logger.debug("Loading existing FAISS index from %s", index_path)
        vs = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)

        try:
            index_dim: Optional[int] = None
            try:
                index_dim = getattr(vs.index, "d", None)
            except Exception:
                index_dim = None

            emb_dim: Optional[int] = None
            try:
                dummy_vec = embeddings.embed_query("dimension check")
                if isinstance(dummy_vec, list):
                    emb_dim = len(dummy_vec)
            except Exception:
                emb_dim = None

            if index_dim is not None and emb_dim is not None and index_dim != emb_dim:
                logger.warning(
                    "FAISS index dimension %s does not match embedding dimension %s; rebuilding index.",
                    index_dim,
                    emb_dim,
                )

                try:
                    for f in index_dir.glob("*"):
                        f.unlink()
                except Exception:
                    logger.debug("Failed to clear existing index files; continuing anyway")

                documents = load_math_documents(data_dir)
                logger.info("Rebuilding FAISS index: documents=%d (dir=%s)", len(documents), data_dir)
                if not documents:
                    documents.append(
                        Document(page_content="Placeholder document.", metadata={"source": "dummy"})
                    )
                vector_store = FAISS.from_documents(documents, embeddings)
                try:
                    size_after = getattr(vector_store.index, "ntotal", None)
                    logger.info("FAISS rebuild: ntotal=%s", size_after)
                except Exception:
                    logger.debug("FAISS rebuild: couldn't read ntotal")
                vector_store.save_local(str(index_dir))
                return vector_store
        except Exception:
            logger.exception(
                "Failed to verify index/embedding dimension; proceeding with loaded index"
            )
        try:
            size = getattr(vs.index, "ntotal", None)  
            logger.info("FAISS load: ntotal=%s from %s", size, index_dir)  
        except Exception:
            logger.debug("FAISS load: couldn't read ntotal") 
        return vs
    logger.debug("Building new FAISS index in %s", index_dir)
    documents = load_math_documents(data_dir)
    logger.info("Index build: documents found=%d (dir=%s)", len(documents), data_dir)  
    if not documents:
        logger.warning("Index build: no documents found; inserting placeholder (KB will be empty)")  
        documents.append(Document(page_content="Placeholder document.", metadata={"source": "dummy"}))
    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store = FAISS.from_documents(documents, embeddings)
    try:
        size_after = getattr(vector_store.index, "ntotal", None)  
        logger.info("FAISS build: ntotal=%s", size_after)   
    except Exception:
        logger.debug("FAISS build: couldn't read ntotal")   
    vector_store.save_local(str(index_dir))
    logger.info("Built FAISS index with %d documents at %s", len(documents), index_dir)
    return vector_store

# Web search tool

import os
import asyncio
from typing import List

async def web_search(query: str, max_results: int = 5) -> str:
    try:
        logger.debug("web_search called with query: '%s', max_results: %d", query, max_results)
        logger.info("MCP: start search q=%r k=%d", query, max_results)   
    except Exception:
        pass

    if MCPClient is None:
        logger.error("MCP: mcp_use not installed")   
        raise RuntimeError("mcp_use not installed or failed to import.")
    if not TAVILY_API_KEY:
        logger.error("MCP: TAVILY_API_KEY missing")   
        raise RuntimeError("TAVILY_API_KEY is not set for MCP Tavily server.")

    npx = "npx.cmd" if os.name == "nt" else "npx"

    mcp_pkg = os.getenv("TAVILY_MCP_NPX", "@mcptools/mcp-tavily")
    logger.debug("MCP: using package %s via %s", mcp_pkg, npx)   

    config = {
        "mcpServers": {
            "tavily": {
                "command": npx,
                "args": ["-y", mcp_pkg],
                "env": {
                    "TAVILY_API_KEY": TAVILY_API_KEY,
                    "MCP_USE_ANONYMIZED_TELEMETRY": os.getenv("MCP_USE_ANONYMIZED_TELEMETRY", "false"),
                },
            }
        }
    }

    client = MCPClient(config)
    snippets: List[str] = []
    session = None

    try:
        await asyncio.wait_for(client.create_all_sessions(), timeout=8)
        logger.debug("MCP: session established")   
        session = client.get_session("tavily")

        tool_name = "search"
        try:
            tools = await asyncio.wait_for(session.list_tools(), timeout=4)
            logger.debug("MCP: available tools=%s", [t.name for t in tools])   
            for t in tools:
                if "search" in t.name.lower():
                    tool_name = t.name
                    break
        except Exception as e:
            logger.debug("MCP list_tools timeout/err: %s", e)

        args = {"query": query, "max_results": max_results}
        logger.info("MCP: calling tool=%s", tool_name)   
        result = await asyncio.wait_for(session.call_tool(name=tool_name, arguments=args), timeout=15)


        if getattr(result, "isError", False):
            logger.error("MCP: tool returned error result")   
            return ""
        for block in getattr(result, "content", []):
            text = getattr(block, "text", "")
            if text:
                snippets.append(text)

    except asyncio.TimeoutError:
        logger.debug("MCP Tavily call timed out.")
        logger.error("MCP: timeout")   
    except Exception as e:
        logger.debug("MCP Tavily call failed: %s", e)
        logger.exception("MCP: failure: %s", e)   
    finally:
        try:
            await client.close_all_sessions()
            logger.debug("MCP: sessions closed")   
        except Exception:
            pass
    try:
        logger.debug("web_search returning %d snippet(s)", len(snippets))
        logger.info("MCP: done, snippets=%d", len(snippets))   
    except Exception:
        pass

    return "\n\n".join(snippets)

class MathAgent:

    def __init__(self, vector_store: FAISS) -> None:
        self.last_route = "UNKNOWN"  

        logger.debug("Initializing MathAgent with provided vector store.")
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
        self.llm = ChatOllama(model="llama3.2", temperature=0.0)

        prompt = ChatPromptTemplate.from_template(
            "Use the following context to answer the math question with a concise, step‑by‑step solution. "
            "Avoid adding extra details or examples that would make the answer too long. "
            "If the answer is not contained in the context, say you don't know.\n\n"
            "{context}\n\nQuestion: {input}\n\nAnswer:"
        )
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        self.qa_chain = create_retrieval_chain(self.retriever, document_chain)

    async def answer_question(self, question: str) -> str:
        logger.info("Router: incoming question=%r", question)   
        question = question.strip()
        if not question:
            logger.info("Router: empty question")   
            return "Please ask a non-empty question."

        if not input_guardrail_ok(question):
            logger.info("Router: question blocked by input guardrail")   
            return "I'm sorry, your question was blocked due to safety policies."

        if not self._is_math_question(question):
            logger.info("Router: non-math question detected")   
            return "I'm sorry, I can only help with mathematical questions."

        try:
            if not await llm_is_math(self.llm, question):
                logger.info("Router: question blocked by LLM math guardrail")   
                return "I'm sorry, I can only help with mathematical questions."
        except Exception:
            logger.exception("Router: LLM math guardrail error")

        try:
            fcr = globals().get("feedback_context_retriever")
        except Exception:
            fcr = None
        if fcr is not None:
            try:
                feedback_docs = await fcr.ainvoke(question)
                
                if feedback_docs:
                    logger.info("Path: FEEDBACK_MEMORY (context-aware feedback hit)")
                    self.last_route = "FEEDBACK_MEMORY"
                    context = "\n\n".join(d.page_content for d in feedback_docs)
                    prompt_text = (
                        "You are a math tutor. The following feedback snippets contain corrected "
                        "solutions and explanations from earlier in this session. When they clearly "
                        "match the new question, you MUST reuse and adapt those solutions rather than "
                        "solving from scratch. If nothing matches, solve the question as usual.\n\n"
                        f"Feedback-based context:\n{context}\n\n"
                        f"Question: {question}\n\nAnswer:"
                    )
                    response = await self.llm.ainvoke(prompt_text)
                    return response.content
            except Exception:
                logger.exception("Router: feedback memory retrieval failed; falling back to normal routing")

        try:
            results: List = self.vector_store.similarity_search_with_relevance_scores(
                question, k=TOP_K
            )
        except AssertionError:
            logger.exception(
                "Similarity search failed due to dimension mismatch; falling back to search"
            )
            results = []
        except Exception:
            logger.exception("Similarity search failed; falling back to search")
            results = []

        results = [
            (d, s) for (d, s) in results if d.metadata.get("source") != "dummy"
        ]
        try:
            logger.info("Router: retrieved=%d", len(results))   
        except Exception:
            pass
        if results:
            docs = [doc for (doc, _score) in results]
            use_kb = False
            try:
                top_score = float(results[0][1])
                if 0.0 <= top_score <= 1.0 and top_score >= SIMILARITY_THRESHOLD:
                    use_kb = True
                else:
                    use_kb = False
            except Exception:
                use_kb = False

            logger.info("KB gate: top_score=%s thr=%.2f -> use_kb=%s",
                        results[0][1], SIMILARITY_THRESHOLD, use_kb)   

            if use_kb:
                self.last_route = "KB"
                logger.info("Path: KB (direct solution extract) docs=%d", len(docs))   

                direct = None
                try:
                    top_text = docs[0].page_content or ""
                    m = re.search(r"Solution:\s*(.*)\Z", top_text, flags=re.S)
                    if m:
                        direct = m.group(1).strip()
                except Exception:
                    direct = None

                if direct:
                    logger.info("KB: returning direct solution from JSON")   
                    return direct

                try:
                    logger.info("Path: KB->RetrievalChain (no explicit Solution found)")   
                    response = await self.qa_chain.ainvoke({"input": question})
                    return response.get("answer", "I'm sorry, I couldn't generate an answer.")
                except Exception:
                    logger.debug(
                        "Retrieval chain invocation failed; falling back to direct LLM prompt with context."
                    )
                    logger.info("Path: KB->DirectContextFallback (docs=%d)", len(docs))   
                    context = "\n\n".join(d.page_content for d in docs)
                    prompt_text = (
                        "Answer concisely (max 3 sentences). If the context lacks the answer, say you don't know.\n\n"
                        f"{context}\n\nQuestion: {question}\n\nAnswer:"
                    )
                    answer = (await self.llm.ainvoke(prompt_text)).content
                    logger.debug("Answer generated from direct context fallback: %s", answer)
                    return answer

        logger.info("Path: MCP (KB not confident or no results)")   
        self.last_route = "MCP" 
        search_context = await web_search(query=question, max_results=5)
        try:
            snippet_count = len(search_context.split("\n\n")) if search_context else 0   
            logger.info("MCP: snippet_count=%d", snippet_count)   
        except Exception:
            pass
        if not search_context:
            prompt = (
                "You are a helpful math tutor. The question is not covered in your knowledge base and no "
                "search results were found. Politely explain that you cannot answer. Answer concisely."
                "The question is not covered in your knowledge base and no search results were found."
                "If you cannot answer, say so plainly without any meta commentary.\n\n"
                f"Question: {question}\nAnswer: "
            )
            return (await self.llm.ainvoke(prompt)).content

        prompt = (
            "You are a helpful math tutor. Use the following web search snippets to answer the student's question.\n"
            "Provide a concise, step‑by‑step solution without adding extra details or examples. If the snippets contradict each other or are incomplete, "
            "state that you cannot provide a reliable answer. "
            "Use only the following web snippets; "
            "if they are insufficient or contradictory, say you cannot answer. "
            "Do not include any preface or notes—output only the final answer.\n\n"
            f"Search snippets:\n{search_context}\n\nQuestion: {question}\n\nAnswer:"
        )
        return (await self.llm.ainvoke(prompt)).content


    @staticmethod
    def _is_math_question(question: str) -> bool:
        q = question.lower()
        if re.search(r"\d", q):
            return True
        keywords = [
            "integral",
            "derivative",
            "solve",
            "sum",
            "fraction",
            "equation",
            "algebra",
            "geometry",
            "calculus",
            "+",
            "-",
            "*",
            "/",
        ]
        return any(kw in q for kw in keywords)

class MathState(TypedDict):
    question: str
    answer: Optional[str]


def build_graph(agent: MathAgent) -> StateGraph:
    """Create a simple LangGraph that answers the question using the agent."""
    async def solve_node(state: MathState) -> MathState:
        question = state["question"]
        answer = await agent.answer_question(question)
        return {"question": question, "answer": answer}
    builder = StateGraph(MathState)
    builder.add_node("solve", solve_node)
    builder.set_entry_point("solve")
    graph = builder.compile()
    return graph

# DSPy module for answer refinement

class AnswerRefiner(dspy.Module):


    def __init__(self, llm: ChatOllama) -> None:
        super().__init__()
        self.llm = llm

    async def forward(self, question: str, draft: str) -> str:

        prompt = (
            "You are a math tutor reviewing an assistant's draft solution.\n"
            "Your goal is to improve the draft so that it is clear, step‑by‑step, "
            "and mathematically accurate. If the draft contains mistakes, correct "
            "them. If the draft lacks sufficient information, politely say so. "
            "Keep the final answer direct and do not add extra details or examples that would make it longer. "
            "Output ONLY the final improved answer—no preface, no lists, no notes.\n\n"
            "Question: {question}\n\n"
            "Draft solution:\n{draft}\n\n"
            "Refined solution:".format(question=question, draft=draft)
        )
        try:
            logger.debug("AnswerRefiner: refining draft (len=%d)", len(draft) if draft else 0)   
            result = await self.llm.ainvoke(prompt)
            return result.content
        except Exception:
            logger.exception("AnswerRefiner: refinement failed; returning draft")   
            # Fall back to returning the original draft if refinement fails
            return draft


# FastAPI 
app = FastAPI(title="Math Tutor Agent")

vector_store: Optional[FAISS] = None
agent: Optional[MathAgent] = None
graph: Optional[StateGraph] = None
answer_refiner: Optional[AnswerRefiner] = None

# In-memory feedback store
feedback_records: Dict[str, Dict[str, Optional[str]]] = {}

feedback_vector_store: Optional[FAISS] = None
feedback_context_retriever: Optional[ContextualCompressionRetriever] = None


class ChatRequest(BaseModel):
    question: str = Field(..., description="The student's math question")


class ChatResponse(BaseModel):
    answer: str
    record_id: str
    tool_used: str


class FeedbackRequest(BaseModel):
    record_id: str
    rating: int = Field(..., ge=1, le=5)
    comments: Optional[str] = None


class FeedbackResponse(BaseModel):
    success: bool
    updated_answer: Optional[str] = Field(None, description="The revised answer after applying feedback, if applicable")


from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, agent, graph, answer_refiner

    logger.info("Startup: building/loading FAISS index...")   
    vector_store = build_or_load_index(INDEX_DIR, DATA_DIR)
    logger.debug("Vector store initialized.")

    agent = MathAgent(vector_store)
    logger.debug("MathAgent instance created.")

    graph = build_graph(agent)
    logger.debug("LangGraph compiled.")

    answer_refiner = AnswerRefiner(agent.llm)
    logger.debug("AnswerRefiner initialized.")
    logger.info("Application startup complete.")
    
    yield  
    logger.info("Shutting down...")

app = FastAPI(title="Math Tutor Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """
    Handle a chat request by invoking the graph to answer the question and then
    refining the answer if an AnswerRefiner is available.  Detailed debug logs
    are emitted to trace the processing steps and tools used.
    """
    logger.debug("/chat endpoint called with question: '%s'", req.question)
    logger.info("/chat: question=%r", req.question)   
    if graph is None:
        logger.error("Server not ready; graph is None")
        raise HTTPException(status_code=503, detail="Server not ready")
    state: MathState = {"question": req.question, "answer": None}
    try:
        result = await graph.ainvoke(state)
        answer = result.get("answer", "I'm sorry, I couldn't generate an answer.")
        logger.debug("Initial answer from graph: %s", answer)
        logger.info("/chat: graph answered (preview=%r)", answer[:80] if answer else "")   
    except Exception as exc:
        logger.exception("Error answering question via graph")
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        route = getattr(agent, "last_route", "UNKNOWN")
        if answer_refiner is not None and route != "KB": 
            logger.debug("Passing answer to AnswerRefiner for refinement.")
            logger.info("/chat: refining answer via AnswerRefiner (route=%s)", route)
            answer = await answer_refiner(question=req.question, draft=answer)
            logger.debug("Refined answer: %s", answer)
        else:
            logger.info("/chat: skipping refiner (route=%s)", route)

    except Exception as exc:
        logger.exception("Error refining answer; returning draft answer: %s", exc)

    try:
        if agent is not None:
            if not await llm_is_math(agent.llm, answer):
                logger.info("/chat: answer blocked by LLM math guardrail")   
                answer = "I'm sorry, the generated answer was blocked due to safety policies."
    except Exception:
        logger.exception("/chat: LLM math guardrail check failed")

    try:
        if not output_guardrail_ok(answer):
            logger.info("/chat: answer blocked by output guardrail")   
            answer = "I'm sorry, the generated answer was blocked due to safety policies."
    except Exception:
        logger.exception("/chat: output guardrail check failed")

    record_id = str(uuid.uuid4())
    feedback_records[record_id] = {
        "question": req.question,
        "answer": answer,
        "rating": None,
        "comments": None,
    }
    
    route = getattr(agent, "last_route", "UNKNOWN")
    if route == "KB":
        tool_used = "Knowledge Base"
    elif route == "MCP":
        tool_used = "MCP"
    else:
        tool_used = "Knowledge Base"
    
    logger.debug("Generated record_id: %s", record_id)
    logger.info("/chat: done record_id=%s tool_used=%s", record_id, tool_used)   
    return ChatResponse(answer=answer, record_id=record_id, tool_used=tool_used)


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback_endpoint(req: FeedbackRequest) -> FeedbackResponse:
    """
    Handle feedback from the user.  Update the stored feedback record with
    the provided rating and comments, then attempt to revise the previously
    generated answer using the feedback.  The revised answer is returned in
    the response.  This function lays the groundwork for self-learning by
    incorporating feedback into answer refinement.
    """
    logger.debug(
        "/feedback endpoint called with record_id: %s, rating: %s, comments: %s",
        req.record_id,
        req.rating,
        req.comments,
    )
    logger.info("/feedback: record_id=%s rating=%s", req.record_id, req.rating)   
    record = feedback_records.get(req.record_id)
    if not record:
        logger.error("Invalid record_id received: %s", req.record_id)
        raise HTTPException(status_code=404, detail="Invalid record_id")
    record["rating"] = req.rating
    record["comments"] = req.comments
    logger.debug("Feedback stored for record_id %s", req.record_id)

    has_text_feedback = bool((req.comments or "").strip())

    updated_answer: Optional[str] = None
    original_answer = record.get("answer")
    original_question = record.get("question")
    if has_text_feedback and original_answer and original_question:
        feedback_text = req.comments or ""
        revision_prompt = (
            "You are a math tutor. A student previously asked the following question: "
            f"{original_question}.\n"
            "The answer provided was:\n"
            f"{original_answer}\n\n"
            f"The student has provided the following feedback (rating {req.rating}):\n"
            f"{feedback_text}\n\n"
            "Please update or refine the original answer to address the feedback. "
            "If the feedback highlights errors, correct them. If the feedback suggests "
            "improvements or additional details, include them. If the feedback is positive and "
            "offers no specific suggestions, you may simply restate the original answer more clearly."
            " Return only the improved final answer. Do not include any explanations, bullets, or notes."
        )
        try:
            logger.info("/feedback: applying refinement to previous answer")   
            llm_to_use = None
            if answer_refiner is not None:
                llm_to_use = answer_refiner.llm
            elif agent is not None:
                llm_to_use = agent.llm
            if llm_to_use is None:
                raise RuntimeError("No LLM available to apply feedback")
            result = await llm_to_use.ainvoke(revision_prompt)
            updated_answer = result.content
            logger.debug("Revised answer generated via feedback: %s", updated_answer)
            record["answer"] = updated_answer
            logger.info("/feedback: refinement applied")   
        except Exception as exc:
            logger.exception("Error applying feedback to the answer: %s", exc)
            updated_answer = None

    try:
        global feedback_vector_store, feedback_context_retriever
        if has_text_feedback and original_question and (updated_answer or original_answer):
            best_answer = updated_answer or original_answer
            feedback_text = req.comments or ""
            doc_text = (
                f"Question: {original_question}\n"
                f"Correct Answer: {best_answer}\n"
                f"Feedback: {feedback_text}"
            )
            doc = Document(
                page_content=doc_text,
                metadata={"type": "feedback", "record_id": req.record_id},
            )

            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            if feedback_vector_store is None:
                feedback_vector_store = FAISS.from_documents([doc], embeddings)
            else:
                try:
                    idx_dim = getattr(feedback_vector_store.index, "d", None)
                    emb_sample = embeddings.embed_query("dimension check")
                    emb_dim = len(emb_sample) if isinstance(emb_sample, list) else None
                    if idx_dim is not None and emb_dim is not None and idx_dim != emb_dim:
                        logger.warning(
                            "Feedback FAISS index dimension %s does not match embedding dimension %s; rebuilding feedback index.",
                            idx_dim,
                            emb_dim,
                        )
                        existing_docs = []
                        try:
                            existing_docs = list(getattr(feedback_vector_store, "docs", []))
                        except Exception:
                            existing_docs = []
                        existing_docs.append(doc)
                        feedback_vector_store = FAISS.from_documents(existing_docs, embeddings)
                    else:
                        feedback_vector_store.add_documents([doc])
                except Exception:
                    try:
                        feedback_vector_store.add_documents([doc])
                    except Exception:
                        logger.exception("Failed to add document to feedback vector store")

            if feedback_vector_store is not None and (agent is not None or answer_refiner is not None):
                llm_for_compression = agent.llm if agent is not None else answer_refiner.llm
                base_retriever = feedback_vector_store.as_retriever(search_kwargs={"k": TOP_K})
                compressor = LLMChainExtractor.from_llm(llm_for_compression)
                feedback_context_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=base_retriever,
                )
    except Exception:
        logger.exception("Feedback memory: failed to update vector store")

    return FeedbackResponse(success=True, updated_answer=updated_answer)


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Welcome to the Math Tutor Agent"}
