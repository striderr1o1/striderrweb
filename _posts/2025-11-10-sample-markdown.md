---
layout: post
title: I Built a Research Agent using LangGraph
subtitle: power your researches with agentic ai
gh-repo: https://github.com/striderr1o1/research-agentai-backend
# gh-badge: [star, fork, follow]
# tags: [test]
comments: true
mathjax: true
author: Mustafa
---

# Building an Intelligent Research Agent: A LangGraph-Powered Multi-Agent System

## Introduction

In the age of information overload, researchers, students, and professionals often struggle to synthesize insights from multiple sources—academic papers, web articles, and documents. What if we could have an AI assistant that not only reads through PDFs but also searches the web, extracts key insights, and delivers a comprehensive research summary—all in real-time?

This is exactly what I built: **An Intelligent Research Agent** powered by **LangGraph**, a framework for creating sophisticated multi-agent workflows. In this blog post, I'll walk you through the architecture, implementation details, challenges faced, and lessons learned while building this project.

---

## Project Overview

The Research Agent is a full-stack application that combines:

- **Backend**: A Python-based multi-agent system using LangGraph, FastAPI, and Groq LLM
- **Frontend**: A React TypeScript application with modern UI components (built using Lovable.dev AI)
- **Key Features**:
  - PDF document upload and intelligent extraction
  - Web search integration using Tavily API
  - Multi-agent orchestration with streaming responses
  - Real-time agent status updates
  - Markdown-formatted research articles as output

---

## Architecture Deep Dive

### The Multi-Agent Workflow

The heart of this project is a **stateful workflow** powered by LangGraph. The system is designed around four specialized agents, each with a specific responsibility:

```
User Query + PDFs → Planner → Extractor → Searcher → Summarizer → Final Report
```

#### **1. State Management**

LangGraph uses a typed state object that flows through the entire workflow. Here's the state structure:

```python
class State(TypedDict):
    uploaded_pdfs: dict          # Uploaded PDF files and their paths
    query: str                    # User's research query
    plan: Dict                    # Task assignments from Planner
    retrieved_data: list          # Extracted text from PDFs
    extracted_insights: str       # Insights from PDF analysis
    search_results: str           # Raw web search results
    search_summary: str           # Summarized web search insights
    summarized_data: str          # Final research article
```

This state object is passed from node to node, allowing each agent to read previous outputs and add its own contributions.

#### **2. The Planner Agent**

The Planner is the "brain" of the operation. It analyzes the user's query and decides which agents need to be activated and what tasks they should perform.

**Key Implementation Details**:
- Uses Groq's LLM with structured JSON output
- Employs Pydantic models for type-safe responses
- Intelligently assigns tasks based on query semantics
- Can set tasks to `null` if an agent isn't needed

```python
class Plan(BaseModel):
    search_task: str
    extraction_task: str
    summarizer_task: str
```

**Example Planning Logic**:
- If user uploads PDFs → assigns `extraction_task`
- If query requires current information → assigns `search_task`
- Always assigns `summarizer_task` to combine insights

The Planner's system prompt is carefully crafted to make intelligent decisions:
> "You are a Planner Agent in a workflow system. Your job is to divide a user query into the appropriate tasks for downstream agents..."

#### **3. The Extractor Agent**

The Extractor is responsible for reading uploaded PDFs and extracting meaningful insights.

**Implementation Highlights**:
- Uses **PyPDF2** library for PDF text extraction
- Processes multiple PDFs concurrently
- Sends extracted text to Groq LLM for insight generation
- Only activates if Planner assigns an `extraction_task`

**Technical Flow**:
1. Check if `extraction_task` exists in plan
2. Iterate through uploaded PDFs
3. Extract text using PyPDF2's `PdfReader`
4. Send extracted text + task to LLM
5. Store generated insights in state

**Error Handling**: The system gracefully handles corrupted PDFs or extraction failures without crashing the entire workflow.

#### **4. The Searcher Agent**

The Searcher performs intelligent web searches using the **Tavily API** and summarizes findings.

**Key Features**:
- Integrates with Tavily for high-quality search results
- LLM-powered summarization of search results
- Includes source URLs in the summary for citation
- Conditional execution based on Planner's decision

**Implementation**:
```python
def Search(Query):
    tavily_client = TavilyClient(api_key=os.environ.get('TAVILY_API_KEY'))
    response = tavily_client.search(f"{Query}")
    return response['results']
```

The Searcher's LLM system prompt ensures:
> "You are a search results summarizer. You provide detailed guidance from search results targeted towards the next agent... You must also include supporting URLs for your insights only from the search results."

#### **5. The Summarizer Agent**

The final agent combines all insights into a cohesive research article.

**Responsibilities**:
- Synthesizes PDF insights and web search summaries
- Generates well-structured markdown articles
- Includes references and citations
- Handles cases where PDF or search data is unavailable

**Output Format**: The Summarizer produces markdown-formatted research articles with:
- Hierarchical headings (H1, H2, H3)
- Bullet points for key findings
- Bold text for emphasis
- Inline citations with URLs

---

## Backend Implementation

### FastAPI Server Design

The backend exposes two main endpoints:

#### **1. `/run` - Synchronous Execution**
Returns the final result after all agents complete their work.

```python
@app.post("/run")
async def run_agent(query: str = Form(...), 
                   files: Optional[List[UploadFile]] = File(None)):
    # Process files, create state, run workflow
    result = graph.invoke(new_state)
    return {"result": result.get("summarized_data")}
```

#### **2. `/run-streaming` - Server-Sent Events (SSE)**
Streams real-time updates as each agent completes its work.

```python
@app.post("/run-streaming")
async def run_agent_streaming(query: str = Form(...), 
                              files: Optional[List[UploadFile]] = File(None)):
    return StreamingResponse(stream_generator(new_state), 
                           media_type="text/event-stream")
```

**Streaming Implementation**: The system uses `graph.astream()` to yield updates asynchronously:

```python
async for item in graph.astream(new_state):
    for agent_name, agent_output in item.items():
        output = {"agent": agent_name, "output": agent_output}
        yield f"data: {json.dumps(output)}\n\n"
```

This allows the frontend to display real-time progress as each agent works.

### LangGraph Workflow Compilation

The workflow is defined using LangGraph's `StateGraph`:

```python
graph_builder.add_node("planner", RunPlannerAgent)
graph_builder.add_node("extractor", RunExtractionAgent)
graph_builder.add_node("searcher", RunsearchAgent)
graph_builder.add_node("summarizer", RunSummarizerAgent)

graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "extractor")
graph_builder.add_edge("extractor", "searcher")
graph_builder.add_edge("searcher", "summarizer")
graph_builder.add_edge("summarizer", END)

graph = graph_builder.compile()
```

**Note**: While this implementation uses a linear flow, LangGraph supports conditional edges, allowing for more complex routing logic (e.g., skipping the Extractor if no PDFs are uploaded).

### Technologies Used in Backend

| Technology | Purpose |
|------------|---------|
| **LangGraph** | Multi-agent workflow orchestration |
| **FastAPI** | RESTful API and streaming endpoints |
| **Groq** | Fast LLM inference with llama models |
| **PyPDF2** | PDF text extraction |
| **Tavily API** | Intelligent web search |
| **Python dotenv** | Environment variable management |
| **Pydantic** | Type-safe data validation |

---

## Frontend Implementation

### UI Architecture

The frontend is a **React TypeScript** application built with modern tooling and design systems. 

**⚠️ Important Note**: The frontend UI and the FastAPI portion was **not hand-coded by me**. I used **Lovable.dev**, an AI-powered web development platform, to generate the React components and UI design. This allowed me to focus on the backend logic and agent orchestration while still delivering a polished user experience.

### Key Components

#### **1. Index Page (Main Application)**
The main page orchestrates the entire user experience:

```typescript
const Index = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [query, setQuery] = useState("");
  const [agentUpdates, setAgentUpdates] = useState<AgentUpdate[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  
  const handleSubmit = async () => {
    // Streaming SSE connection
    const reader = response.body?.getReader();
    // Process streaming updates...
  }
}
```

**Features**:
- Two-column layout with file upload and query input
- Real-time stats showing uploaded PDFs and query status
- Streaming response handling using browser's Fetch API
- Toast notifications for user feedback

#### **2. FileUpload Component**
Handles PDF upload with drag-and-drop functionality:

```typescript
export const FileUpload = ({ files, setFiles }) => {
  const handleDrop = useCallback((e: React.DragEvent) => {
    const droppedFiles = Array.from(e.dataTransfer.files)
      .filter(file => file.type === "application/pdf");
    setFiles([...files, ...droppedFiles]);
  }, [files, setFiles]);
}
```

**Features**:
- Drag-and-drop zone with hover effects
- PDF validation
- File list with size information
- Individual file removal

#### **3. QueryInput Component**
Text area for research queries with keyboard shortcuts:

```typescript
const handleKeyDown = (e: React.KeyboardEvent) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    onSubmit();
  }
};
```

**UX Enhancements**:
- Cmd/Ctrl + Enter to submit
- Loading state with spinner
- Disabled state during processing

#### **4. ResultsDisplay Component**
The most complex component, showing real-time agent progress and final results:

```typescript
export const ResultsDisplay = ({ agentUpdates, isLoading }) => {
  const [expandedAgents, setExpandedAgents] = useState<string[]>([]);
  
  // Agent progress indicators
  const agentIcons = {
    planner: Brain,
    extractor: FileSearch,
    searcher: Search,
    summarizer: FileText,
  };
}
```

**Features**:
- Real-time agent status indicators
- Expandable agent outputs (click to see raw data)
- Markdown rendering for final results
- Link formatting with external link icons
- Copy-to-clipboard functionality

**Markdown Rendering**: The component includes a custom `formatResult` function that parses markdown-like syntax:
- Headers (`#`, `##`, `###`)
- Bold text (`**text**`)
- Bullet points (`-`, `*`)
- URLs with clickable links
- Proper spacing and typography

### UI Component Library

The frontend uses **shadcn/ui**, a collection of re-usable components built with:
- **Radix UI** primitives (headless, accessible components)
- **Tailwind CSS** for styling
- **CVA** (Class Variance Authority) for component variants
- **Lucide React** for icons

**Key shadcn/ui components used**:
- `Button`, `Card`, `Badge`, `Textarea`
- `Toast`/`Sonner` for notifications
- `ThemeToggle` for dark/light mode

### Technologies Used in Frontend

| Technology | Purpose |
|------------|---------|
| **React 18** | UI framework |
| **TypeScript** | Type safety |
| **Vite** | Build tool and dev server |
| **Tailwind CSS** | Utility-first styling |
| **shadcn/ui** | Component library |
| **TanStack Query** | Server state management (setup) |
| **React Router** | Routing |
| **next-themes** | Dark mode support |

---

## Development Experience

### Setting Up the Project

**Backend Setup**:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

**Frontend Setup**:
```bash
cd frontend
npm install
npx vite
```

### Environment Variables

The project requires several API keys:

```env
GROQ_API_KEY=your_groq_api_key
CHAT_GROQ_MODEL=llama-3.1-70b-versatile  # or another Groq model
TAVILY_API_KEY=your_tavily_api_key
```

### CORS Configuration

For local development, the FastAPI backend is configured to accept requests from any origin:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Production Note**: In production, this should be restricted to specific origins.

---

## Challenges and Solutions

### Challenge 1: Streaming State Updates

**Problem**: How to stream incremental updates from LangGraph to the frontend while maintaining state consistency?

**Solution**: 
- Used FastAPI's `StreamingResponse` with Server-Sent Events
- Implemented `graph.astream()` for async iteration over agent outputs
- Structured streaming data as JSON with agent names and outputs

### Challenge 2: Conditional Agent Execution

**Problem**: Not every query needs all agents (e.g., no PDFs uploaded means no extraction needed).

**Solution**:
- Made Planner the "orchestrator" that assigns tasks conditionally
- Each agent checks if its task is `None` before executing
- Allows for flexible workflows without changing graph structure

**Future Enhancement**: Implement conditional edges in LangGraph to skip agents entirely rather than having them check for null tasks.

### Challenge 3: LLM Hallucinations and JSON Parsing

**Problem**: LLMs sometimes produce invalid JSON or hallucinate when generating structured outputs.

**Solution**:
- Used Groq's structured output feature with JSON schemas
- Leveraged Pydantic models for validation
- Added error handling for JSON parsing failures

### Challenge 4: PDF Extraction Quality

**Problem**: PyPDF2 sometimes produces garbled text from complex PDFs with images, tables, or unusual encodings.

**Solution**:
- Added error handling to continue processing even if one PDF fails
- **Potential Improvements**: Could integrate OCR for scanned PDFs, or use more robust libraries like `pdfplumber` or `unstructured.io`

### Challenge 5: Frontend Real-Time Updates

**Problem**: Parsing Server-Sent Events in the browser and updating React state efficiently.

**Solution**:
```typescript
const reader = response.body?.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.substring(6));
      setAgentUpdates(prev => [...prev, data]);
    }
  }
}
```

---

## What I Learned

### 1. **LangGraph is Powerful for Multi-Agent Systems**

LangGraph's state-based approach makes it incredibly intuitive to build complex agent workflows. The ability to define nodes, edges, and state transformations declaratively is a game-changer compared to manually managing agent communication.

### 2. **Streaming Improves UX Dramatically**

Users don't want to stare at a loading spinner for 30 seconds. Streaming agent updates provides transparency, builds trust, and makes the application feel responsive even when processing takes time.

### 3. **AI-Assisted UI Development is Viable**

Using Lovable.dev to generate the frontend saved me days of work on UI components and styling. While the backend required careful custom logic, the frontend was mostly presentational—a perfect use case for AI-generated code.

### 4. **Groq is Blazingly Fast**

Groq's LLM inference speed is impressive. Responses come back in 1-2 seconds even for complex prompts, making the streaming workflow feel snappy.

### 5. **Prompt Engineering Matters**

The quality of agent outputs depends heavily on well-crafted system prompts. I iterated on the Planner's prompt multiple times to get intelligent task assignments.

---

## Future Improvements

### Short-Term Enhancements

1. **Conditional Graph Edges**: Implement LangGraph conditional routing to skip unnecessary agents
2. **Better PDF Extraction**: Integrate OCR for scanned documents, use `pdfplumber` for tables
3. **Citation Management**: Add proper citation formatting (APA, MLA, etc.)
4. **User Authentication**: Add user accounts to save research history
5. **Export Options**: Allow exporting results as PDF, Word, or markdown files

### Long-Term Vision

1. **Conversational Follow-Ups**: Allow users to ask follow-up questions on the research
2. **Multi-Modal Inputs**: Support image uploads, video transcripts, audio files
3. **Collaborative Research**: Enable teams to work on shared research projects
4. **Agent Parallelization**: Run Extractor and Searcher in parallel for faster results
5. **Custom Agent Plugins**: Allow users to create custom agents for specific research domains

---

## Conclusion

Building this Research Agent has been an incredible learning experience. It demonstrates the power of combining modern AI frameworks (LangGraph), fast LLM inference (Groq), and thoughtful UX design (streaming updates, real-time feedback).

The project showcases how **specialized agents working in concert** can solve complex problems that would be difficult for a single monolithic AI model. The Planner decides the strategy, the Extractor and Searcher gather evidence, and the Summarizer synthesizes everything into a coherent narrative—just like how humans collaborate on research tasks.

### Key Takeaways

✅ **LangGraph** makes multi-agent orchestration intuitive  
✅ **Streaming responses** dramatically improve user experience  
✅ **AI-generated UI** (via Lovable.dev) accelerates frontend development  
✅ **Structured outputs** with Pydantic ensure reliability  
✅ **Modular agent design** enables easy extension and debugging  

---

## Technical Acknowledgments

### Technologies I Coded Myself

- **Backend Logic**: All Python code in `app.py`, `planner.py`, `retriever.py`, `search.py`, `summarizer.py`, and `main.py`
- **LangGraph Workflow**: Agent orchestration, state management, and graph compilation
- **API Design**: FastAPI endpoints, streaming implementation, and CORS configuration
- **Agent Prompts**: System prompts for each specialized agent
- **Error Handling**: Graceful failures and edge case management

### Technologies I Used (But Didn't Create)

- **Frontend UI**: Generated using **Lovable.dev AI** (React components, Tailwind styling, shadcn/ui integration)
- **LangGraph Framework**: Open-source by LangChain team
- **Groq API**: Third-party LLM inference service
- **Tavily API**: Third-party search API
- **shadcn/ui Components**: Open-source component library
- **PyPDF2**: Open-source PDF parsing library

---

## Project Links

📺 **Demo Video**: [Google Drive Link](https://drive.google.com/file/d/15YWnCZEneZ_KCw789Dz-_RLlVMctIt0K/view)  
🎥 **Technical Walkthrough**: [Google Drive Link](https://drive.google.com/file/d/1DPYQvFHMy6Rt9gKHgggyyU8ZpRZ0bYvw/view?usp=sharing)  
💻 **GitHub Repository**: [striderr1o1/research-agentai-backend](https://github.com/striderr1o1/research-agentai-backend)

---

## Repository Structure

```
research-agent/
├── backend/
│   ├── app.py              # LangGraph workflow & state
│   ├── main.py             # FastAPI endpoints
│   ├── planner.py          # Planner Agent
│   ├── retriever.py        # Extractor Agent
│   ├── search.py           # Searcher Agent
│   ├── summarizer.py       # Summarizer Agent
│   └── uploaded_pdfs/      # PDF storage
├── frontend/
│   ├── src/
│   │   ├── components/     # React components (AI-generated)
│   │   ├── pages/          # Page layouts
│   │   └── lib/            # Utilities
│   └── package.json        # Dependencies
└── requirements.txt        # Python dependencies
```

---

## Final Thoughts

This project represents the intersection of several cutting-edge technologies: multi-agent AI systems, streaming APIs, modern web frameworks, and AI-assisted development. It's been a joy to build, and I hope this deep dive inspires others to explore LangGraph and multi-agent architectures.

The future of AI applications isn't just bigger models—it's **smarter orchestration** of specialized agents working together. This Research Agent is just the beginning.

Happy researching! 🚀📚

---

**Author**: Built as part of the Applied Data Science and AI Specialization Hackathon  
**License**: MIT  
**Date**: November 2025
