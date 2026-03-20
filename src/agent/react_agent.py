import json
from typing import Optional, Any
from src.generation.llm_client import LLMClient
from src.retrieval.qdrant_storage import QdrantStorage
from src.retrieval.graph_retriever import GraphRetriever
from src.graph.entity_extractor import MedicalEntityExtractor
from src.ingestion.pdf_loader import embed_texts

class MedicalAgent:
    """
    Agentic orchestrator that uses ReAct reasoning to autonomously
    query and combine data from vector and graph databases.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        qdrant: QdrantStorage,
        graph_retriever: GraphRetriever,
        entity_extractor: MedicalEntityExtractor,
        reranker: Any = None
    ):
        self.llm = llm_client
        self.qdrant = qdrant
        self.graph = graph_retriever
        self.extractor = entity_extractor
        self.reranker = reranker
        
        self.system_prompt = """You are an advanced medical research AI agent.
Your goal is to answer a clinical or scientific question by analyzing multiple highly-technical medical sources.
You have access to tools that can search academic literature and a Knowledge Graph.

TOOLS AT YOUR DISPOSAL:
1. `search_vector_database`: Use this to find semantic matches for complex ideas (e.g. "What are the cardiovascular effects of COVID-19?").
2. `extract_clinical_entities`: Use this first if you have a chunk of text or a question and want to know the unified clinical entities inside it.
3. `query_knowledge_graph`: Use this to look up a precise clinical entity (e.g. "SARS-CoV-2", "mechanical ventilation") to see its direct relationships (TREATS, CAUSES, etc.) and related literature.

YOUR REASONING PROCESS:
1. Call one or more tools to gather evidence. You can call multiple tools if necessary.
2. Read the tool results (the "Observations").
3. Critically evaluate if you have enough clinical evidence to definitively answer the user's question.
4. If you DO NOT have enough evidence, formulate a refined query and call another tool (Multi-Hop Reasoning).
5. If you DO have enough evidence, output your final, comprehensive answer. DO NOT call any tools once you are ready to answer the end user.

Important:
- Never guess, extrapolate, or hallucinate. ONLY synthesize facts retrieved from your tools.
- Your final answer must cite which tool/entity provided the insight if possible."""
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_vector_database",
                    "description": "Search the literature database for semantic concepts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The concept or question to search for in vector space."},
                            "top_k": {"type": "integer", "description": "Number of context snippets to return (default 5)."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_knowledge_graph",
                    "description": "Query the Neo4j graph for exact medical entity relationships.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "The exact medical entity name to look up in the graph."}
                        },
                        "required": ["entity"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_clinical_entities",
                    "description": "Extract and normalize medical entities from a sentence. Yields exact entity names you can pass to the graph.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to analyze for entity extraction."}
                        },
                        "required": ["text"]
                    }
                }
            }
        ]

    def _execute_tool(self, tool_call) -> str:
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return "Error: arguments must be valid JSON."
            
        print(f"Agent Action: {name}({args})")
        
        try:
            if name == "search_vector_database":
                query_str = args.get("query") or args.get("q") or args.get("text")
                if not query_str:
                    return "Error: Missing 'query' parameter."
                top_k = int(args.get("top_k", 5))
                
                # Embed and search
                try:
                    query_embedding = embed_texts([query_str])[0]
                except Exception as e:
                    return f"Embedding failed: {e}"
                
                fetch_k = int(top_k) * 3
                results = self.qdrant.search(query_embedding, top_k=fetch_k, min_score=0.2)
                
                contexts = results.get("contexts", [])
                sources = results.get("sources", [])
                scores = results.get("scores", [])
                
                # Rerank if available
                if self.reranker and contexts:
                    try:
                        reranked = self.reranker.rerank(
                            query=query_str, 
                            documents=contexts,
                            sources=sources,
                            scores=scores,
                            top_k=top_k
                        )
                        if isinstance(reranked, dict) and "contexts" in reranked:
                            # Parse output format from Reranker class
                            contexts = []
                            for ctx, score in zip(reranked["contexts"], reranked.get("rerank_scores", [])):
                                if score >= -2.0:
                                    contexts.append(ctx)
                        else:
                            # Fallback if using a different reranker implementation
                            contexts = [r.get('text', '') for r in reranked if r.get('score', 0) >= -2.0]
                    except Exception as e:
                        print(f"Reranking error: {e}")
                        contexts = contexts[:top_k]
                else:
                    contexts = contexts[:top_k]
                    
                if not contexts:
                    return "No relevant contexts found in vector database."
                return json.dumps({"retrieved_contexts": contexts})
                
            elif name == "query_knowledge_graph":
                entity = args.get("entity") or args.get("q") or args.get("query")
                if not entity:
                    return "Error: Missing 'entity' parameter."
                res = self.graph.retrieve_by_entity(entity, top_k=3)
                
                if not res.contexts and not res.graph_paths:
                    return f"No nodes or contexts found for entity '{entity}'."
                    
                output = {
                    "graph_relationships": [],
                    "entity_contexts": res.contexts
                }
                
                # Format graph paths
                for record in res.graph_paths:
                    path = record.get("path")
                    if path:
                        path_str = ""
                        for i in range(len(path.nodes) - 1):
                            n1 = path.nodes[i].get("name", "Unknown")
                            rel = path.relationships[i].type
                            n2 = path.nodes[i+1].get("name", "Unknown")
                            path_str += f"({n1}) -[{rel}]-> ({n2}) "
                        output["graph_relationships"].append(path_str.strip())
                        
                return json.dumps(output)
                
            elif name == "extract_clinical_entities":
                text = args.get("text") or args.get("query") or args.get("q")
                if not text:
                    return "Error: Missing 'text' parameter."
                res = self.extractor.extract_entities(text)
                entities = [e.text for e in res.entities]
                if not entities:
                    return "No clinical entities recognized."
                return json.dumps({"extracted_entities": entities})
                
            else:
                return f"Error: Tool {name} not found."
                
        except Exception as e:
            return f"Error executing tool {name}: {str(e)}"

    def run(self, user_query: str, max_steps: int = 7) -> dict:
        """
        Run the agent loop to answer the user's query.
        Returns a dictionary with the final 'answer' and the 'thought_process' (list of messages).
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        thought_process = []
        
        for step in range(max_steps):
            response_msg = self.llm.chat_completion(
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            if not response_msg:
                return {
                    "answer": "Error: Failed to get response from generation model.",
                    "thought_process": thought_process
                }
                
            # Convert response message to dict to append to history
            msg_dict = {"role": "assistant"}
            if hasattr(response_msg, "content") and response_msg.content:
                msg_dict["content"] = response_msg.content
                if response_msg.content.strip():
                    thought_process.append({"type": "thought", "content": response_msg.content})
            else:
                msg_dict["content"] = None
                
            active_tool_calls = getattr(response_msg, "tool_calls", None) or []
            
            # Fallback text parsing for LLMs that inline JSON tool calls
            if not active_tool_calls and msg_dict.get("content"):
                import re, uuid
                content = msg_dict["content"]
                pattern = r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"parameters"\s*:\s*\{[^}]*\}\s*\}'
                for match in re.finditer(pattern, content):
                    try:
                        parsed = json.loads(match.group(0))
                        
                        class MockFunction:
                            def __init__(self, name, arguments):
                                self.name = name
                                self.arguments = arguments
                        class MockToolCall:
                            def __init__(self, id, type, function):
                                self.id = id
                                self.type = type
                                self.function = function

                        tc = MockToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            type="function",
                            function=MockFunction(
                                name=parsed["name"],
                                arguments=json.dumps(parsed.get("parameters", {}))
                            )
                        )
                        active_tool_calls.append(tc)
                    except Exception:
                        pass
                        
            if active_tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in active_tool_calls
                ]
                
            messages.append(msg_dict)
            
            # Check if LLM wants to call tools
            if active_tool_calls:
                for tool_call in active_tool_calls:
                    # Log action for UI
                    action_log = f"Using API: `{tool_call.function.name}` -> {tool_call.function.arguments}"
                    thought_process.append({"type": "action", "content": action_log})
                    
                    # Execute
                    result = self._execute_tool(tool_call)
                    
                    # Log observation for UI
                    if len(result) > 400:
                        obs_log = f"{result[:397]}... [truncated]"
                    else:
                        obs_log = result
                    thought_process.append({"type": "observation", "content": obs_log})
                    
                    # Add to history
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": result
                    })
            else:
                # No tools called -> Phase complete
                return {
                    "answer": msg_dict.get("content", "I am unable to answer this question."),
                    "thought_process": thought_process
                }
                
        # Handle running out of steps
        # Strip tools and force output
        final_msg = self.llm.chat_completion(
            messages=messages,
            tools=None 
        )
        final_content = final_msg.content if final_msg and hasattr(final_msg, "content") else "Max reasoning steps reached before completing research."
        return {
            "answer": final_content,
            "thought_process": thought_process
        }
