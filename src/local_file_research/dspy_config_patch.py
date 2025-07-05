"""
Patch for dspy_config.py to fix the agent registry issue.

This patch ensures that the query_expansion and text_generator agents are properly registered
and accessible when needed.
"""

import logging
import importlib
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_patch():
    """Apply the patch to fix the agent registry issue."""
    logger.info("Applying patch to fix agent registry issue...")
    
    try:
        # Import the necessary modules
        import dspy
        
        # Import the DSPyAgentRegistry
        from dspy_config import DSPyAgentRegistry, initialize_dspy, DSPY_CONFIGURED
        
        # Initialize DSPy if not already initialized
        if not DSPY_CONFIGURED:
            logger.info("DSPy not configured. Initializing...")
            initialize_dspy()
        
        # Get all registered agents
        registered_agents = list(DSPyAgentRegistry.agents.keys())
        logger.info(f"Found {len(registered_agents)} registered agents: {registered_agents}")
        
        # Check if the agents mentioned in the logs are registered
        focus_agents = ["query_expansion", "text_generator"]
        
        missing_agents = [agent for agent in focus_agents if agent not in registered_agents]
        
        if missing_agents:
            logger.info(f"Missing agents: {missing_agents}")
            
            # Register the missing agents
            for agent_name in missing_agents:
                logger.info(f"Registering {agent_name}...")
                
                if agent_name == "query_expansion":
                    class QueryExpansionSignature(dspy.Signature):
                        """Expand a search query with related terms."""
                        instructions = 'Expand a search query with related terms.'
                        query: str = dspy.InputField(desc="The original search query")
                        expanded_query: str = dspy.OutputField(desc="The expanded search query with related terms")
                    
                    DSPyAgentRegistry.register_agent(agent_name, dspy.Predict(QueryExpansionSignature))
                    
                elif agent_name == "text_generator":
                    class TextGeneratorSignature(dspy.Signature):
                        """Generate text based on a prompt."""
                        instructions = 'Generate text based on a prompt.'
                        prompt: str = dspy.InputField(desc="The prompt to generate text from")
                        query: str = dspy.InputField(desc="Optional query for context", default="", required=False)
                        context: str = dspy.InputField(desc="Optional additional context", default="", required=False)
                        text: str = dspy.OutputField(desc="The generated text")
                    
                    DSPyAgentRegistry.register_agent(agent_name, dspy.Predict(TextGeneratorSignature))
            
            # Check if all agents are now registered
            registered_agents = list(DSPyAgentRegistry.agents.keys())
            still_missing = [agent for agent in focus_agents if agent not in registered_agents]
            
            if still_missing:
                logger.error(f"Still missing agents after registration: {still_missing}")
                return False
            
            logger.info("Successfully registered all missing agents.")
        else:
            logger.info("All required agents are already registered.")
        
        # Verify that the agents can be retrieved
        for agent_name in focus_agents:
            agent = DSPyAgentRegistry.get_agent(agent_name)
            if not agent:
                logger.error(f"Failed to retrieve agent: {agent_name}")
                return False
            
            logger.info(f"Successfully verified agent: {agent_name}")
        
        # Patch the initialize_dspy function to ensure agents are registered
        original_initialize_dspy = initialize_dspy
        
        def patched_initialize_dspy():
            """Patched version of initialize_dspy that ensures all agents are registered."""
            # Call the original function
            original_initialize_dspy()
            
            # Check if the agents are registered
            registered_agents = list(DSPyAgentRegistry.agents.keys())
            focus_agents = ["query_expansion", "text_generator"]
            missing_agents = [agent for agent in focus_agents if agent not in registered_agents]
            
            if missing_agents:
                logger.info(f"After initialization, still missing agents: {missing_agents}")
                
                # Register the missing agents
                for agent_name in missing_agents:
                    logger.info(f"Registering {agent_name} after initialization...")
                    
                    if agent_name == "query_expansion":
                        class QueryExpansionSignature(dspy.Signature):
                            """Expand a search query with related terms."""
                            instructions = 'Expand a search query with related terms.'
                            query: str = dspy.InputField(desc="The original search query")
                            expanded_query: str = dspy.OutputField(desc="The expanded search query with related terms")
                        
                        DSPyAgentRegistry.register_agent(agent_name, dspy.Predict(QueryExpansionSignature))
                        
                    elif agent_name == "text_generator":
                        class TextGeneratorSignature(dspy.Signature):
                            """Generate text based on a prompt."""
                            instructions = 'Generate text based on a prompt.'
                            prompt: str = dspy.InputField(desc="The prompt to generate text from")
                            query: str = dspy.InputField(desc="Optional query for context", default="", required=False)
                            context: str = dspy.InputField(desc="Optional additional context", default="", required=False)
                            text: str = dspy.OutputField(desc="The generated text")
                        
                        DSPyAgentRegistry.register_agent(agent_name, dspy.Predict(TextGeneratorSignature))
        
        # Replace the original function with the patched version
        import dspy_config
        dspy_config.initialize_dspy = patched_initialize_dspy
        
        logger.info("Successfully patched initialize_dspy function.")
        
        # Also patch the setup_default_agents method to ensure it registers all agents
        original_setup_default_agents = DSPyAgentRegistry.setup_default_agents
        
        @classmethod
        def patched_setup_default_agents(cls):
            """Patched version of setup_default_agents that ensures all agents are registered."""
            # Call the original method
            original_setup_default_agents()
            
            # Check if the agents are registered
            registered_agents = list(cls.agents.keys())
            focus_agents = ["query_expansion", "text_generator"]
            missing_agents = [agent for agent in focus_agents if agent not in registered_agents]
            
            if missing_agents:
                logger.info(f"After setup_default_agents, still missing agents: {missing_agents}")
                
                # Import DSPy
                import dspy
                
                # Register the missing agents
                for agent_name in missing_agents:
                    logger.info(f"Registering {agent_name} after setup_default_agents...")
                    
                    if agent_name == "query_expansion":
                        class QueryExpansionSignature(dspy.Signature):
                            """Expand a search query with related terms."""
                            instructions = 'Expand a search query with related terms.'
                            query: str = dspy.InputField(desc="The original search query")
                            expanded_query: str = dspy.OutputField(desc="The expanded search query with related terms")
                        
                        cls.register_agent(agent_name, dspy.Predict(QueryExpansionSignature))
                        
                    elif agent_name == "text_generator":
                        class TextGeneratorSignature(dspy.Signature):
                            """Generate text based on a prompt."""
                            instructions = 'Generate text based on a prompt.'
                            prompt: str = dspy.InputField(desc="The prompt to generate text from")
                            query: str = dspy.InputField(desc="Optional query for context", default="", required=False)
                            context: str = dspy.InputField(desc="Optional additional context", default="", required=False)
                            text: str = dspy.OutputField(desc="The generated text")
                        
                        cls.register_agent(agent_name, dspy.Predict(TextGeneratorSignature))
        
        # Replace the original method with the patched version
        DSPyAgentRegistry.setup_default_agents = patched_setup_default_agents
        
        logger.info("Successfully patched setup_default_agents method.")
        
        return True
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error applying patch: {e}")
        return False

def main():
    """Apply the patch and verify that it works."""
    result = apply_patch()
    if result:
        logger.info("Successfully applied patch to fix agent registry issue.")
    else:
        logger.error("Failed to apply patch to fix agent registry issue.")

if __name__ == "__main__":
    main()
