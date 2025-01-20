import streamlit as st
import os
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import openai
from pinecone.grpc import PineconeGRPC as Pinecone
import time
from datetime import datetime
import logging
import json
from dataclasses import dataclass
import tiktoken
import extra_streamlit_components as stx
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.stateful_button import button
from streamlit_lottie import st_lottie
import requests
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class SearchResult:
    """Data class for search results."""
    chunk_id: str
    score: float
    metadata: Dict[str, Any]

def load_lottie_url(url: str) -> Dict:
    """Load Lottie animation from URL."""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

class StreamlitRAGSystem:
    def __init__(self):
        """Initialize the RAG system with Streamlit configuration."""
        # Initialize session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'llm_outputs' not in st.session_state:
            st.session_state.llm_outputs = []
        
        # Load configurations
        self._load_api_keys()
        self._initialize_apis()
        self._configure_system()
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Set Streamlit theme
        self._set_streamlit_theme()
        
        # Load animations
        self.loading_animation = load_lottie_url("https://assets6.lottiefiles.com/packages/lf20_kxq5bhml.json")

    def _set_streamlit_theme(self):
        """Configure Streamlit theme and styling."""
        st.set_page_config(
            page_title="WTS Tax Advisory",
            page_icon="🏢",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom CSS
        st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }
        .stButton>button {
            background-color: #1e88e5;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px 24px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1565c0;
            transform: translateY(-2px);
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
            max-width: 80%;
            animation: fadeIn 0.5s ease-in;
        }
        .user-message {
            background-color: #1e88e5;
            margin-left: auto;
        }
        .assistant-message {
            background-color: #424242;
            margin-right: auto;
        }
        .search-result {
            border: 1px solid #1e88e5;
            border-radius: 5px;
            padding: 1rem;
            margin-bottom: 1rem;
            background-color: rgba(30, 136, 229, 0.1);
            transition: all 0.3s ease;
        }
        .search-result:hover {
            transform: translateY(-2px);
            box-shadow: 0 2px 5px rgba(30, 136, 229, 0.2);
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .sidebar .sidebar-content {
            background-color: #1a1a1a;
        }
        .system-metrics {
            background-color: #242424;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        .stProgress .st-bo {
            background-color: #1e88e5;
        }
        </style>
        """, unsafe_allow_html=True)

    def _load_api_keys(self):
        """Load API keys from environment variables."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_host = os.getenv("PINECONE_HOST")
        
        if not all([self.openai_api_key, self.pinecone_api_key, self.pinecone_host]):
            st.error("Missing required API keys. Please check your environment variables.")
            st.stop()

    def _initialize_apis(self):
        """Initialize OpenAI and Pinecone connections."""
        try:
            openai.api_key = self.openai_api_key
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            self.index = self.pc.Index(host=self.pinecone_host)
            self.namespace = "default"
        except Exception as e:
            st.error(f"Failed to initialize APIs: {str(e)}")
            st.stop()

    def _configure_system(self):
        """Configure system parameters."""
        # Load configuration from sidebar
        with st.sidebar:
            colored_header(label="System Configuration", description="Adjust system parameters", color_name="blue-70")
            
            self.initial_chunks = st.slider("Initial Search Chunks", 5, 30, 15)
            self.refined_chunks = st.slider("Refined Search Chunks", 5, 20, 10)
            self.relevance_threshold = st.slider("Relevance Threshold", 0.0, 1.0, 0.61)
            self.temperature = st.slider("Temperature", 0.0, 1.0, 0.0)
            self.max_history_tokens = st.number_input("Max History Tokens", 1000, 100000, 10000)
            
            add_vertical_space(2)
            
            # Model configuration
            colored_header(label="Model Configuration", description="Select models", color_name="blue-70")
            self.embedding_model = st.selectbox("Embedding Model", ["text-embedding-3-large"])
            self.filter_model = st.selectbox("Filter Model", ["gpt-4", "gpt-3.5-turbo"])
            self.response_model = st.selectbox("Response Model", ["gpt-4", "gpt-3.5-turbo"])
            
            add_vertical_space(2)
            
            # System metrics
            colored_header(label="System Metrics", description="Current system status", color_name="blue-70")
            with st.container():
                st.markdown("""<div class="system-metrics">""", unsafe_allow_html=True)
                st.metric("Total Messages", len(st.session_state.chat_history))
                st.metric("Unique Sources", len(set(msg.get('source_file', '') 
                    for msg in st.session_state.llm_outputs if isinstance(msg, dict))))
                st.markdown("""</div>""", unsafe_allow_html=True)
            
            # System actions
            colored_header(label="System Actions", description="Clear history and save logs", color_name="blue-70")
            if st.button("Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.llm_outputs = []
                st.success("Chat history cleared!")
            
            if st.button("Save Session Logs", use_container_width=True):
                self.save_interaction_logs()
                st.success("Session logs saved!")

    def get_embedding(self, text: str, retries: int = 3) -> List[float]:
        """Get embeddings with retry logic."""
        for attempt in range(retries):
            try:
                response = openai.Embedding.create(
                    input=text,
                    model=self.embedding_model
                )
                return response['data'][0]['embedding']
            except Exception as e:
                if attempt == retries - 1:
                    st.error(f"Failed to get embedding: {str(e)}")
                    return []
                time.sleep(1 * (attempt + 1))

    def search_documents(self, query: str, top_k: int = None) -> List[SearchResult]:
        """Enhanced document search with retries and error handling."""
        if top_k is None:
            top_k = self.initial_chunks
            
        try:
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return []
                
            results = self.index.query(
                namespace=self.namespace,
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            return [
                SearchResult(
                    chunk_id=match.id,
                    score=match.score,
                    metadata=match.metadata
                )
                for match in results.matches
            ]
            
        except Exception as e:
            st.error(f"Error in document search: {str(e)}")
            return []

    def refine_query(self, original_query: str, initial_results: List[SearchResult]) -> str:
        """Refine query based on initial search results."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # Format initial chunks for context
            context_chunks = []
            for result in initial_results[:3]:
                context_chunks.append(
                    f"Title: {result.metadata.get('title', 'N/A')}\n"
                    f"Content: {result.metadata.get('text', 'N/A')}\n"
                    f"Keywords: {result.metadata.get('key_keywords', 'N/A')}\n"
                    f"Category: {result.metadata.get('ato_tax_category', 'N/A')}"
                )
            
            context = "\n\n---\n\n".join(context_chunks)
            
            refinement_prompt = f"""As an expert tax researcher, analyze the user's original query and the initial search results to create a more detailed and contextually rich search query.

            Original Query: {original_query}

            Initial Search Results Context:
            {context}

            Based on these initial results, formulate a more specific and detailed query that will help find the most relevant tax information. The new query should:
            1. Include relevant technical terms from the found documents
            2. Specify any relevant tax categories or sections mentioned
            3. Add context-specific qualifiers
            4. Maintain focus on the user's original intent
            5. Include any relevant legislation or ruling references found

            Return only the refined query text, no other explanation."""

            response = openai.ChatCompletion.create(
                model=self.filter_model,
                messages=[
                    {"role": "system", "content": "You are an expert tax researcher specializing in query refinement."},
                    {"role": "user", "content": refinement_prompt}
                ],
                temperature=0.2
            )
            
            refined_query = response.choices[0].message.content.strip()
            
            # Log refinement
            st.session_state.llm_outputs.append({
                'timestamp': timestamp,
                'type': 'query_refinement',
                'original_query': original_query,
                'refined_query': refined_query,
                'context_used': context,
                'model': self.filter_model
            })
            
            return refined_query
            
        except Exception as e:
            st.error(f"Query refinement error: {str(e)}")
            return original_query

    def filter_chunks(self, query: str, chunks: List[SearchResult]) -> List[SearchResult]:
        """Enhanced chunk filtering with comprehensive scoring."""
        if not chunks:
            return []

        try:
            # Format chunks for LLM analysis
            formatted_chunks = []
            for i, chunk in enumerate(chunks):
                metadata = chunk.metadata
                formatted_chunk = (
                    f"=== CHUNK {i+1} ANALYSIS ===\n"
                    f"1. CORE INFORMATION:\n"
                    f"   - Title: {metadata.get('title', 'N/A')}\n"
                    f"   - Category: {metadata.get('ato_tax_category', 'N/A')}\n"
                    f"   - Initial Score: {chunk.score:.4f}\n\n"
                    f"2. CONTENT:\n"
                    f"   - Text: {metadata.get('text', 'N/A')}\n"
                    f"   - Summary: {metadata.get('document_summary', 'N/A')}\n"
                    f"   - Reference: {metadata.get('section_reference', 'N/A')}\n\n"
                    f"3. APPLICABILITY:\n"
                    f"   - Scope: {metadata.get('applicability', 'N/A')}\n"
                    f"   - Exceptions: {metadata.get('exceptions', 'N/A')}\n"
                    f"   - Industry: {metadata.get('industry_relevance', 'N/A')}"
                )
                formatted_chunks.append(formatted_chunk)

            filter_prompt = f"""Analyze these document chunks for relevance to the query with systematic scoring.

            Query: {query}

            Document chunks to analyze:
            {formatted_chunks}

            Scoring Criteria (100 points total):
            1. Content Relevance (50 points):
               - Direct topic match (20 points)
               - Semantic relevance (20 points)
               - Context applicability (10 points)

            2. Authority & Quality (30 points):
               - Source authority (10 points)
               - Section specificity (10 points)
               - Documentation completeness (10 points)

            3. Technical Alignment (20 points):
               - Industry relevance (10 points)
               - Technical accuracy (10 points)

            Return scores as: chunk_index:score,chunk_index:score,...
            Example: "0:87,1:92,2:45,3:78"
            Only return the scores, no other text."""

            response = openai.ChatCompletion.create(
                model=self.filter_model,
                messages=[
                    {"role": "system", "content": "You are an expert tax document analyst."},
                    {"role": "user", "content": filter_prompt}
                ],
                temperature=0
            )
            
            # Parse scores and rank chunks
            scores = {}
            for score_pair in response.choices[0].message.content.split(','):
                idx, score = score_pair.split(':')
                scores[int(idx.strip())] = float(score.strip()) / 100

            # Combine scores with vector similarity
            final_chunks = []
            for idx, chunk in enumerate(chunks):
                if idx in scores:
                    combined_score = (0.7 * scores[idx]) + (0.3 * chunk.score)
                    if combined_score >= self.relevance_threshold:
                        final_chunks.append(chunk)

            # Sort by combined score and limit
            final_chunks.sort(key=lambda x: x.score, reverse=True)
            return final_chunks[:self.max_chunks]

        except Exception as e:
            st.error(f"Chunk filtering error: {str(e)}")
            return chunks[:self.min_chunks]

    def two_pass_search(self, query: str) -> Tuple[List[SearchResult], str]:
        """Perform two-pass search with query refinement."""
        try:
            # First pass
            with st.spinner("🔍 Performing initial search..."):
                initial_results = self.search_documents(query, self.initial_chunks)
                if not initial_results:
                    return [], query

            # Refine query
            with st.spinner("🔄 Refining search query..."):
                refined_query = self.refine_query(query, initial_results)
            
            # Second pass with refined query
            with st.spinner("🎯 Performing refined search..."):
                refined_results = self.search_documents(refined_query, self.refined_chunks)
            
            # Combine and deduplicate results
            seen_ids = set()
            combined_results = []
            
            # Add refined results first
            for result in refined_results:
                if result.chunk_id not in seen_ids:
                    seen_ids.add(result.chunk_id)
                    combined_results.append(result)
            
            # Add high-scoring initial results
            for result in initial_results:
                if result.chunk_id not in seen_ids and result.score >= self.relevance_threshold:
                    seen_ids.add(result.chunk_id)
                    combined_results.append(result)
            
            # Filter combined results
            with st.spinner("⚖️ Filtering and ranking results..."):
                filtered_results = self.filter_chunks(refined_query, combined_results)
            
            return filtered_results, refined_query
            
        except Exception as e:
            st.error(f"Two-pass search error: {str(e)}")
            return [], query

    def generate_response(self, query: str, chunks: List[SearchResult]) -> str:
        """Generate response with enhanced context handling."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # Format chunks into context
            context_parts = []
            for chunk in chunks:
                metadata = chunk.metadata
                context_parts.append(
                    f"Source: {metadata.get('source_file', 'N/A')}\n"
                    f"Title: {metadata.get('title', 'N/A')}\n"
                    f"Category: {metadata.get('ato_tax_category', 'N/A')}\n"
                    f"Content: {metadata.get('text', 'N/A')}\n"
                    f"Reference: {metadata.get('section_reference', 'N/A')}"
                )
            
            context = "\n\n---\n\n".join(context_parts)
            
            system_prompt = f"""You are an expert tax advisor at WTS Australia, known for providing accurate, comprehensive, and well-structured tax advice. Base your response ONLY on the provided context.

            Guidelines:
            1. Focus on accuracy and specificity
            2. Include relevant section references and citations
            3. Structure the response clearly
            4. Address all aspects of the query
            5. Highlight any important caveats or conditions

            Context:
            {context}"""

            messages = [
                {"role": "system", "content": system_prompt},
                *[{"role": msg["role"], "content": msg["content"]} 
                  for msg in st.session_state.chat_history[-4:]],  # Include recent context
                {"role": "user", "content": query}
            ]

            response = openai.ChatCompletion.create(
                model=self.response_model,
                messages=messages,
                temperature=self.temperature
            )

            answer = response.choices[0].message.content
            
            # Log response generation
            st.session_state.llm_outputs.append({
                'timestamp': timestamp,
                'type': 'response_generation',
                'query': query,
                'context': context,
                'response': answer,
                'model': self.response_model
            })

            return answer

        except Exception as e:
            st.error(f"Response generation error: {str(e)}")
            return "I apologize, but I encountered an error while generating the response. Please try again."

    def update_chat_history(self, user_message: str, assistant_response: str):
        """Update chat history with token management."""
        try:
            # Add new messages
            st.session_state.chat_history.append({"role": "user", "content": user_message})
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
            
            # Calculate total tokens
            total_tokens = sum(len(self.tokenizer.encode(msg["content"])) 
                             for msg in st.session_state.chat_history)
            
            # Remove oldest pairs if exceeding token limit
            while total_tokens > self.max_history_tokens and len(st.session_state.chat_history) >= 2:
                # Remove oldest Q&A pair
                st.session_state.chat_history = st.session_state.chat_history[2:]
                total_tokens = sum(len(self.tokenizer.encode(msg["content"])) 
                                 for msg in st.session_state.chat_history)
                
        except Exception as e:
            st.error(f"Error updating chat history: {str(e)}")

    def display_search_results(self, chunks: List[SearchResult], refined_query: str = None):
        """Display search results in a structured format."""
        if refined_query:
            st.markdown(f"**🔄 Refined Query:** {refined_query}")
        
        for chunk in chunks:
            with st.container():
                st.markdown(f"""
                    <div class="search-result">
                        <h4>{chunk.metadata.get('title', 'N/A')}</h4>
                        <p><strong>Source:</strong> {chunk.metadata.get('source_file', 'N/A')}</p>
                        <p><strong>Category:</strong> {chunk.metadata.get('ato_tax_category', 'N/A')}</p>
                        <p><strong>Score:</strong> {chunk.score:.4f}</p>
                        <details>
                            <summary>View Content</summary>
                            <p>{chunk.metadata.get('text', 'N/A')}</p>
                        </details>
                    </div>
                """, unsafe_allow_html=True)

    def save_interaction_logs(self):
        """Save detailed interaction logs with enhanced metadata."""
        try:
            if not st.session_state.llm_outputs:
                st.warning("No interactions to save.")
                return
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = Path("logs") / f"interaction_logs_{timestamp}"
            log_path.mkdir(parents=True, exist_ok=True)
            
            # Save main interaction log
            with open(log_path / "interactions.json", 'w', encoding='utf-8') as f:
                json.dump(st.session_state.llm_outputs, f, indent=2)
            
            # Save chat history
            with open(log_path / "chat_history.md", 'w', encoding='utf-8') as f:
                f.write("# Chat History\n\n")
                for msg in st.session_state.chat_history:
                    f.write(f"## {msg['role'].capitalize()}\n")
                    f.write(f"{msg['content']}\n\n")
            
            # Save system configuration
            config = {
                'initial_chunks': self.initial_chunks,
                'refined_chunks': self.refined_chunks,
                'relevance_threshold': self.relevance_threshold,
                'temperature': self.temperature,
                'embedding_model': self.embedding_model,
                'filter_model': self.filter_model,
                'response_model': self.response_model
            }
            
            with open(log_path / "config.json", 'w') as f:
                json.dump(config, f, indent=2)
            
            st.success(f"Interaction logs saved to {log_path}")
            
        except Exception as e:
            st.error(f"Error saving interaction logs: {str(e)}")

    def display_chat_interface(self):
        """Display the main chat interface."""
        st.title("🏢 WTS Tax Advisory System")
        st.markdown("Ask your tax-related questions and get expert answers backed by authoritative sources.")
        
        # Chat container
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for message in st.session_state.chat_history:
                role = message["role"]
                content = message["content"]
                
                message_class = "user-message" if role == "user" else "assistant-message"
                st.markdown(f"""
                    <div class="chat-message {message_class}">
                        <div class="message-content">{content}</div>
                    </div>
                """, unsafe_allow_html=True)

        # Input container
        with st.container():
            # Create two columns for input and button
            col1, col2 = st.columns([5, 1])
            
            with col1:
                user_input = st.text_input("Enter your tax query:", key="user_input", 
                                         placeholder="e.g., What are the requirements for claiming a home office deduction?")
            
            with col2:
                send_button = st.button("Send", use_container_width=True)

            if send_button and user_input:
                # Show loading animation
                with st.spinner("🤔 Processing your query..."):
                    # Perform search and generate response
                    filtered_chunks, refined_query = self.two_pass_search(user_input)
                    
                    if filtered_chunks:
                        # Display search results in an expander
                        with st.expander("📚 Search Results", expanded=False):
                            self.display_search_results(filtered_chunks, refined_query)
                        
                        # Generate and display response
                        response = self.generate_response(refined_query, filtered_chunks)
                        
                        # Update chat history
                        self.update_chat_history(user_input, response)
                        
                        # Clear input
                        st.session_state.user_input = ""
                        
                        # Rerun to update chat display
                        st.experimental_rerun()
                    else:
                        st.warning("No relevant information found. Please try a different query.")

    def run(self):
        """Main method to run the Streamlit application."""
        try:
            self.display_chat_interface()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error(f"Application error: {str(e)}")

def main():
    """Main entry point for the Streamlit application."""
    st.set_page_config(
        page_title="WTS Tax Advisory",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize and run the system
    rag_system = StreamlitRAGSystem()
    rag_system.run()

if __name__ == "__main__":
    main()