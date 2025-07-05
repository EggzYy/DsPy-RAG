"""
Test script for multi-iteration research with the new implementation.
"""

import logging
import sys
from local_file_research.research_system import ResearchSystem
from local_file_research.llamaindex_vector_store import get_llamaindex_vector_store

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def test_multi_iteration():
    """Test multi-iteration research with the new implementation."""
    # Initialize vector store
    vector_store = get_llamaindex_vector_store()
    
    # Initialize research system
    research_system = ResearchSystem(vector_store)
    
    # Run research with multi-iteration mode
    results = research_system.run(
        query="What are the key features of Python?",
        research_mode="multi_iteration",
        report_mode="normal",
        top_k=3,
        max_iterations=2,
        max_k=10,
        relevance_threshold=0.6
    )
    
    # Print report
    print("\n\n=== REPORT ===\n")
    print(results["report"])
    
    # Print metrics
    if "findings" in results and results["findings"]:
        print(f"\nTotal findings: {len(results['findings'])}")
    
    if "sources" in results and results["sources"]:
        print(f"Total sources: {len(results['sources'])}")
    
    print(f"Elapsed time: {results.get('elapsed_time', 'N/A')}")

if __name__ == "__main__":
    test_multi_iteration()
