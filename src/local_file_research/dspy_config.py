import os
import sys
import inspect
import traceback
from .config import ( # Import defaults from config.py
    DEFAULT_DSPY_LLM_PROVIDER,
    DEFAULT_DSPY_LLM_MODEL,
    DEFAULT_DSPY_TEMPERATURE,
    DEFAULT_DSPY_MAX_TOKENS,
    LOG_LEVEL, # +++ ADD THIS +++
    OLLAMA_API_BASE # +++ ADD THIS +++
)
import logging
import warnings
import dspy
import re
from typing import List, Dict, Any, Optional, Union, Type, get_origin, get_args, Tuple # Added Tuple

# Apply LiteLLM patch to fix __annotations__ error
DSPY_CONFIGURED = False
try:
    from .litellm_patch import apply_patch, patch_dspy
    apply_patch()
    patch_dspy()
    print("Applied LiteLLM patch in dspy_config.py")
except ImportError:
    try:
        from .litellm_patch import apply_patch, patch_dspy
        apply_patch()
        patch_dspy()
        print("Applied LiteLLM patch in dspy_config.py")
    except ImportError:
        print("Warning: Could not import litellm_patch module in dspy_config.py")

# *** MODIFY LOGGING CONFIGURATION SECTION ***
# Configure logging using level from config.py
log_level_to_use = getattr(logging, LOG_LEVEL, logging.INFO)
# Configure root logger if it has no handlers (to avoid conflicts)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=log_level_to_use, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print(f"Configured root logger with level {LOG_LEVEL}")
else:
    # If already configured, just set the level for this logger
    logging.getLogger(__name__).setLevel(log_level_to_use)
    print(f"Root logger already configured. Setting dspy_config logger level to {LOG_LEVEL}")

logger = logging.getLogger(__name__)

# Create special loggers for DSPy warnings
dspy_warning_logger = logging.getLogger("dspy.warnings")
dspy_predict_logger = logging.getLogger("dspy.predict.predict")
dspy_base_logger = logging.getLogger("dspy") # Catch general dspy logs

# Ensure handlers are only added once
# Create a basic formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Configure dspy_warning_logger
if not dspy_warning_logger.hasHandlers():
     handler = logging.StreamHandler()
     handler.setFormatter(formatter)
     dspy_warning_logger.addHandler(handler)
     dspy_warning_logger.propagate = False # Prevent double logging to root
     dspy_warning_logger.setLevel(logging.WARNING) # Log warnings and above
     print("DSPy warning logging configured for console only")

# Configure dspy_predict_logger
if not dspy_predict_logger.hasHandlers():
     predict_handler = logging.StreamHandler()
     predict_handler.setFormatter(formatter)
     dspy_predict_logger.addHandler(predict_handler)
     dspy_predict_logger.propagate = False # Prevent double logging
     dspy_predict_logger.setLevel(logging.WARNING) # Log predict warnings
     print("DSPy predict warning logging configured")

# Custom warning handler for DSPy warnings
def log_dspy_warning(message, category, filename, lineno, file=None, line=None):
    """Custom warning handler that logs DSPy warnings with stack trace to identify the source."""
    if isinstance(message, Warning): msg_text = str(message)
    else: msg_text = message
    # Filter warnings originating from DSPy or related internal calls
    if 'dspy' in filename.lower() or 'predict' in filename.lower():
        stack = traceback.extract_stack()
        caller_info = "Unknown location"; agent_info = "Unknown agent"
        try:
            # Try to find the immediate caller outside dspy/predict/logging/warnings
            for frame in reversed(stack[:-1]): # Exclude the current warning frame
                if all(lib not in frame.filename for lib in ['dspy/', 'predict.py', 'warnings.py', 'logging/']):
                    caller_info = f"{os.path.basename(frame.filename)}:{frame.lineno} in {frame.name}"; break
            # Try to find agent context
            for frame in reversed(stack[:-1]):
                 # Look for files where agents are typically defined or used
                if any(mod_name in frame.filename for mod_name in [
                    'dspy_config.py', 'dspy_agents.py', 'multi_iteration_research.py',
                    'advanced_reporting.py', 'research_system.py', 'document_analysis.py', 'advanced_search.py'
                    ]):
                    try:
                        # Access the frame object to get local variables
                        frame_obj = sys._getframe(len(stack) - 1 - stack.index(frame))
                        if frame_obj and frame_obj.f_locals:
                            # Look for common variable names holding the agent instance
                            agent_instance = frame_obj.f_locals.get('self', frame_obj.f_locals.get('agent', frame_obj.f_locals.get('module')))
                            if agent_instance and hasattr(agent_instance, '__class__'):
                                agent_info = f"Agent Context: {agent_instance.__class__.__name__}"; break
                    except Exception as stack_e:
                         # Ignore errors during stack inspection, it's best effort
                         logger.debug(f"Minor error inspecting stack frame for agent context: {stack_e}")
                    break # Stop searching after finding a relevant frame
        except Exception as stack_err: logger.debug(f"Error analyzing stack trace for warning context: {stack_err}")

        warning_msg = f"{msg_text} | Triggered near: {caller_info} | {agent_info}"

        # Route to specific logger based on origin and content
        if 'predict' in filename.lower() and 'not all input fields were provided' in msg_text.lower():
            dspy_predict_logger.warning(warning_msg) # Use the predict logger
        else:
            dspy_warning_logger.warning(warning_msg) # Use the general dspy warning logger

# Install the custom warning handler
warnings.showwarning = log_dspy_warning
logger.info("Custom DSPy warning handler installed.")
# --- END OF MODIFIED LOGGING SECTION ---

# DSPY_CONFIG holds the *current* configuration
DSPY_CONFIG = {
    "llm_provider": os.environ.get("DSPY_LLM_PROVIDER", DEFAULT_DSPY_LLM_PROVIDER),
    "llm_model": os.environ.get("DSPY_LLM_MODEL", DEFAULT_DSPY_LLM_MODEL),
    "temperature": float(os.environ.get("DSPY_TEMPERATURE", DEFAULT_DSPY_TEMPERATURE)),
    "max_tokens": int(os.environ.get("DSPY_MAX_TOKENS", DEFAULT_DSPY_MAX_TOKENS)),
    "api_base": os.environ.get("DSPY_API_BASE", None), # Keep this line
}

def configure_dspy(**kwargs):
    """
    Configures the DSPy framework based on provided arguments or DSPY_CONFIG.
    Updates the global DSPY_CONFIG and DSPY_CONFIGURED flag.
    """
    global DSPY_CONFIGURED, DSPY_CONFIG

    config_to_use = DSPY_CONFIG.copy()
    config_to_use.update(kwargs)

    # +++ ADD THIS: Use OLLAMA_API_BASE if needed +++
    if config_to_use["llm_provider"] == "ollama" and not config_to_use.get("api_base"):
        config_to_use["api_base"] = OLLAMA_API_BASE
        logger.info(f"Using OLLAMA_API_BASE from config: {OLLAMA_API_BASE}")

    if DSPY_CONFIGURED and config_to_use == DSPY_CONFIG:
         logger.info("DSPy configuration unchanged. Skipping reconfiguration.")
         return

    logger.info(f"Attempting to configure DSPy with settings: {config_to_use}") # Keep this log
    # +++ ADD/MODIFY LOGS +++
    logger.info(f"Provider: {config_to_use['llm_provider']}")
    logger.info(f"Model: {config_to_use['llm_model']}")
    logger.info(f"API Base: {config_to_use.get('api_base')}") # Log the API base being used

    try:
        import dspy
        from packaging import version

        # Check DSPy version
        min_dspy_version = "2.0.0" # Keep requirement
        dspy_version_str = getattr(dspy, "__version__", "0.0.0")
        if version.parse(dspy_version_str) < version.parse(min_dspy_version):
            logger.error(f"DSPy version {min_dspy_version} or higher is required. Found version: {dspy_version_str}. Please run: pip install --upgrade dspy-ai. Agent features might be disabled.")
            DSPY_CONFIGURED = False # Mark as not configured
            sys.exit("DSPy version requirement not met.") # Consider exiting if critical
            # return # Stop configuration attempt # Exit instead

        logger.info(f"DSPy version {dspy_version_str} meets requirement (>= {min_dspy_version}).")

        provider = config_to_use["llm_provider"]
        model = config_to_use["llm_model"]
        temperature = config_to_use["temperature"]
        max_tokens = config_to_use["max_tokens"]
        api_base = config_to_use.get("api_base", None)
        api_key = None # Initialize api_key

        lm_instance = None

        # *** MODIFY PROVIDER SELECTION LOGIC ***
        # Use dspy.LM generic class for better compatibility
        model_kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        if api_base: model_kwargs["api_base"] = api_base

        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key: logger.warning("OPENAI_API_KEY missing.")
            openai_model = f"openai/{model}" if not model.startswith("openai/") else model
            try:
                lm_instance = dspy.LM(model=openai_model, api_key=api_key, **model_kwargs)
                logger.info(f"Configuring DSPy with OpenAI model: {openai_model} at {api_base or 'default OpenAI'}")
            except Exception as e: logger.error(f"Failed configuring dspy.OpenAI: {e}", exc_info=True)

        elif provider == "ollama":
            try:
                ollama_base_url = api_base if api_base else OLLAMA_API_BASE
                if not ollama_base_url:
                     logger.error("Ollama base URL (DSPY_API_BASE or OLLAMA_API_BASE) is not set.")
                     DSPY_CONFIGURED = False; return
                # Update kwargs for Ollama
                model_kwargs["base_url"] = ollama_base_url
                if "api_base" in model_kwargs: del model_kwargs["api_base"] # Ollama uses base_url
                lm_instance = dspy.LM(model=model, **model_kwargs)
                logger.info(f"Configuring DSPy with Ollama model: {model} at {ollama_base_url}")
            except Exception as e: logger.error(f"Failed configuring dspy.OllamaLocal: {e}", exc_info=True)

        elif provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key: logger.warning("ANTHROPIC_API_KEY missing.")
            try:
                lm_instance = dspy.LM(model=model, api_key=api_key, **model_kwargs)
                logger.info(f"Configuring DSPy with Anthropic model: {model} at {api_base or 'default Anthropic'}")
            except Exception as e: logger.error(f"Failed configuring dspy.Anthropic: {e}", exc_info=True)

        else:
            logger.error(f"Unsupported DSPy LLM provider: {provider}. DSPy cannot be configured.")
            DSPY_CONFIGURED = False
            return

        # --- Final DSPy Configuration ---
        if lm_instance:
            try:
                 dspy.settings.configure(lm=lm_instance)
                 DSPY_CONFIG = config_to_use.copy()
                 DSPY_CONFIGURED = True
                 logger.info(f"DSPy configured successfully for provider '{provider}'. LM: {dspy.settings.lm}") # Log LM
            except Exception as config_e:
                 logger.error(f"Failed during final dspy.settings.configure: {config_e}", exc_info=True)
                 DSPY_CONFIGURED = False
        else:
             logger.error(f"LM instance for provider '{provider}' could not be created.")
             DSPY_CONFIGURED = False

    except ImportError:
        logger.error("DSPy not installed (dspy-ai package). Please run: pip install dspy-ai>=2.0.0. Agent features disabled.")
        DSPY_CONFIGURED = False
    except Exception as e:
        logger.error(f"An unexpected error occurred during DSPy configuration: {e}", exc_info=True)
        DSPY_CONFIGURED = False


class DSPyAgentRegistry:
    """Registry for DSPy agents and chains."""
    agents: Dict[str, dspy.Module] = {} # Type hint for agents
    chains: Dict[str, List[str]] = {}

    @classmethod
    def register_agent(cls, name: str, agent: dspy.Module):
        """Register a DSPy agent module."""
        if not isinstance(agent, dspy.Module):
             logger.warning(f"Attempted to register non-DSPy module '{name}' of type {type(agent)}. Skipping.")
             return
        cls.agents[name] = agent
        logger.info(f"Successfully registered agent: {name}")

    @classmethod
    def get_agent(cls, name: str) -> Optional[dspy.Module]:
        """Get a registered agent by name."""
        agent = cls.agents.get(name)
        if not agent:
            logger.warning(f"Agent '{name}' not found in registry.")

            # Check if this is a critical agent that should be registered on-demand
            critical_agents = ["query_expansion", "text_generator"]

            # Only attempt to register critical agents if DSPy is configured
            if name in critical_agents and DSPY_CONFIGURED:
                logger.warning(f"Critical agent '{name}' not found. Registering it on-demand...")

                # Register the missing agent
                import dspy

                if name == "query_expansion":
                    class QueryExpansionSignature(dspy.Signature):
                        """Expand a query with related terms/concepts."""
                        #instructions = 'Expand a query with related terms/concepts.'
                        query: str = dspy.InputField(desc="Query to expand")
                        context: str = dspy.InputField(desc="Optional context", default="")
                        content: str = dspy.InputField(desc="Content relevant to the question (optional)", default="")
                        document: str = dspy.InputField(desc="Alternative document content (optional)", default="")
                        expanded_query: str = dspy.OutputField(desc="Expanded query string")

                    cls.register_agent(name, dspy.Predict(QueryExpansionSignature))
                    logger.info(f"Registered critical agent on-demand: {name}")
                    return cls.agents.get(name)

                elif name == "text_generator":
                    class TextGeneratorSignature(dspy.Signature):
                        """Generic text generation based on a prompt."""
                        #instructions = 'Generic text generation based on a prompt.'
                        prompt: str = dspy.InputField(desc="The input prompt")
                        context: str = dspy.InputField(desc="Optional context", default="")
                        content: str = dspy.InputField(desc="Content relevant to the question (optional)", default="")
                        document: str = dspy.InputField(desc="Alternative document content (optional)", default="")
                        query: str = dspy.InputField(desc="Optional query focus", default="")
                        text: str = dspy.OutputField(desc="Generated text")

                    cls.register_agent(name, dspy.Predict(TextGeneratorSignature))
                    logger.info(f"Registered critical agent on-demand: {name}")
                    return cls.agents.get(name)

        return agent

    @classmethod
    def register_chain(cls, name: str, agent_names: List[str]):
        """Register a chain as a list of agent names."""
        # Validate agent names
        for agent_name in agent_names:
            if agent_name not in cls.agents:
                 logger.error(f"Agent '{agent_name}' in chain '{name}' is not registered yet. Chain might fail.")
        cls.chains[name] = agent_names
        logger.info(f"Registered chain: {name} -> {agent_names}")

    @classmethod
    def get_chain(cls, name: str) -> Optional[List[str]]:
        """Get the list of agent names for a chain."""
        chain = cls.chains.get(name)
        if not chain:
             logger.warning(f"Chain '{name}' not found in registry.")
        return chain

    @classmethod
    def _get_signature_input_fields(cls, signature: Type[dspy.Signature]) -> Dict[str, Tuple[Type, bool]]:
        """
        Extracts input fields, their base types, and requirement status.

        Returns:
            Dict where key is field name, value is tuple (field_type, is_required)
        """
        input_fields = {}
        # Check if signature is valid
        if not signature:
            logger.warning(f"Provided signature '{signature}' is not valid. Cannot extract fields.")
            return input_fields

        try:
            # Check if the signature has an input_fields attribute (newer DSPy versions)
            if hasattr(signature, 'input_fields'):
                # Extract fields from the input_fields attribute
                for field_name, field_obj in signature.input_fields.items():
                    # Check if it's an input field
                    if hasattr(field_obj, 'json_schema_extra') and field_obj.json_schema_extra.get('__dspy_field_type') == 'input':
                        # In DSPy, all input fields need values for the model to work properly
                        is_required = True

                        # Log the requirement status for debugging
                        logger.debug(f"Field '{field_name}' is an input field for DSPy")

                        # Extract type hint from annotation if available
                        field_type_annotation = signature.__annotations__.get(field_name, Any)
                        actual_type = Any # Default to Any

                        if field_type_annotation:
                            origin = get_origin(field_type_annotation)
                            args = get_args(field_type_annotation)

                            if origin is Union and type(None) in args: # Optional[T]
                                non_none_args = [arg for arg in args if arg is not type(None)]
                                if len(non_none_args) == 1:
                                    base_type_inner = non_none_args[0]
                                    origin_inner = get_origin(base_type_inner)
                                    actual_type = origin_inner if origin_inner else base_type_inner
                                elif len(non_none_args) > 1:
                                    actual_type = Union[tuple(non_none_args)]
                            elif origin is list or origin is List: actual_type = list
                            elif origin is dict or origin is Dict: actual_type = dict
                            elif origin: actual_type = origin # Other generic types like Tuple
                            elif isinstance(field_type_annotation, type): actual_type = field_type_annotation # Basic type
                            else: logger.debug(f"Complex annotation '{field_type_annotation}' for field '{field_name}'. Using Any.")

                        # Store the extracted type and requirement status
                        input_fields[field_name] = (actual_type, is_required)
                        logger.debug(f"Extracted Input Field: '{field_name}' -> Type: {actual_type}, Required: {is_required}")
            else:
                # Fallback to the original method using __dict__ (older DSPy versions)
                # Check if signature is valid and has attributes we can inspect
                if not isinstance(signature, type) or not issubclass(signature, dspy.Signature):
                    logger.warning(f"Provided signature '{signature}' is not a valid dspy.Signature subclass. Cannot extract fields.")
                    return input_fields

                # Iterate through the signature's defined fields (attributes)
                for field_name, field_obj in signature.__dict__.items():
                    # Check if it's a Field object by looking at its attributes rather than using isinstance
                    if hasattr(field_obj, 'json_schema_extra') and hasattr(field_obj, 'required'):
                        # Check if it's an input field using its metadata
                        is_input = field_obj.json_schema_extra.get('__dspy_field_type') == 'input'

                        if is_input:
                            # In DSPy, all input fields need values for the model to work properly
                            is_required = True

                            # Log the requirement status for debugging
                            logger.debug(f"Field '{field_name}' is an input field for DSPy")

                            # Extract type hint from annotation if available
                            field_type_annotation = signature.__annotations__.get(field_name)
                            actual_type = Any # Default to Any

                            if field_type_annotation:
                                origin = get_origin(field_type_annotation)
                                args = get_args(field_type_annotation)

                                if origin is Union and type(None) in args: # Optional[T]
                                    non_none_args = [arg for arg in args if arg is not type(None)]
                                    if len(non_none_args) == 1:
                                        base_type_inner = non_none_args[0]
                                        origin_inner = get_origin(base_type_inner)
                                        actual_type = origin_inner if origin_inner else base_type_inner
                                    elif len(non_none_args) > 1:
                                        actual_type = Union[tuple(non_none_args)]
                                elif origin is list or origin is List: actual_type = list
                                elif origin is dict or origin is Dict: actual_type = dict
                                elif origin: actual_type = origin # Other generic types like Tuple
                                elif isinstance(field_type_annotation, type): actual_type = field_type_annotation # Basic type
                                else: logger.debug(f"Complex annotation '{field_type_annotation}' for field '{field_name}'. Using Any.")

                            else:
                                 # Fallback if no annotation found (less ideal)
                                 logger.warning(f"No type annotation found for input field '{field_name}' in {signature.__name__}. Defaulting to type Any.")

                            # Store the extracted type and requirement status
                            input_fields[field_name] = (actual_type, is_required)
                            logger.debug(f"Extracted Input Field: '{field_name}' -> Type: {actual_type}, Required: {is_required}")

        except Exception as e:
            logger.error(f"Failed to inspect signature for input fields: {e}", exc_info=True)

        if not input_fields:
             logger.warning(f"Could not extract any input fields for signature: {signature}") # Keep this warning

        return input_fields

    @classmethod
    def _ensure_required_fields(cls, agent_name: str, agent: dspy.Module, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures required fields are present, using extracted type and requirement info."""
        signature = getattr(agent, 'signature', None)
        if not signature:
            logger.debug(f"No signature found for agent '{agent_name}'. Passing data as is.")
            return data.copy() # Return a copy

        input_field_info = cls._get_signature_input_fields(signature)
        if not input_field_info:
            logger.debug(f"No input fields found in signature for agent '{agent_name}'. Passing data as is.")
            return data.copy() # Return a copy

        updated_data = data.copy() # Work on a copy
        missing_populated = []
        fields_with_fallbacks = set() # Keep track of fields populated with fallbacks

        # Prepare potential fallback values from input data
        query_val = updated_data.get("query", updated_data.get("question", ""))
        content_val = updated_data.get("content", "")
        document_val = updated_data.get("document", "")
        context_val = updated_data.get("context", "")
        summary_val = updated_data.get("summary", "")
        statement_val = updated_data.get("statement", "")
        prompt_val = updated_data.get("prompt", "")

        primary_text = content_val or document_val or context_val or summary_val or statement_val or query_val or prompt_val or ""
        fallback_context = context_val or summary_val or document_val or content_val or ""

        for field, (field_type, is_required) in input_field_info.items():
            is_missing = field not in updated_data
            # Check for None OR empty string/list/dict
            is_empty = not is_missing and (updated_data[field] is None or (isinstance(updated_data[field], (str, list, dict)) and not updated_data[field]))

            # Log field status for debugging
            logger.debug(f"Field '{field}': is_required={is_required}, is_missing={is_missing}, is_empty={is_empty}")

            # Populate all input fields that are missing or empty
            # In DSPy, all input fields need values for the model to work properly
            if is_missing or is_empty:
                missing_populated.append(field)
                fields_with_fallbacks.add(field)
                fallback_value = None # Start with None

                # --- Specific Fallbacks by Field Name ---
                # Important: Prioritize name-based fallbacks for fields with specific meanings
                # Try to extract meaningful values from existing data first

                # For query/question fields
                if field in ("query", "question"):
                    # Try to extract a query from various sources
                    fallback_value = query_val or updated_data.get("title", "") or "Analyze this content."
                    if not fallback_value and primary_text:
                        # Generate a query based on the content
                        fallback_value = f"Analyze the following: {primary_text[:50]}..."
                    if not fallback_value:
                        fallback_value = "No query provided."

                # For content fields
                elif field == "content":
                    fallback_value = content_val or document_val or context_val or summary_val or statement_val or query_val or prompt_val
                    if not fallback_value:
                        fallback_value = "No content provided."

                # For document fields
                elif field == "document":
                    fallback_value = document_val or content_val or context_val or summary_val
                    if not fallback_value:
                        fallback_value = "No document provided."

                # For context fields
                elif field == "context":
                    fallback_value = context_val or summary_val or document_val or content_val
                    if not fallback_value:
                        fallback_value = "No context provided."

                # For summary fields
                elif field == "summary":
                    fallback_value = summary_val
                    if not fallback_value and primary_text:
                        # Generate a brief summary from the primary text
                        fallback_value = primary_text[:200] + ("..." if len(primary_text) > 200 else "")
                    if not fallback_value:
                        fallback_value = "No summary provided."

                # For statement fields
                elif field == "statement":
                    fallback_value = statement_val or query_val or content_val
                    if not fallback_value:
                        fallback_value = "No statement provided."

                # For prompt fields
                elif field == "prompt":
                    fallback_value = prompt_val
                    if not fallback_value and fallback_context:
                        fallback_value = f"Process the following context: {fallback_context[:100]}..."
                    elif not fallback_value and primary_text:
                        fallback_value = f"Process the following: {primary_text[:100]}..."
                    if not fallback_value:
                        fallback_value = "Please process the input."

                # For code fields
                elif field == "code":
                    fallback_value = updated_data.get("code", "") or content_val or document_val
                    if not fallback_value:
                        fallback_value = "# No code provided"

                # For language fields
                elif field == "language":
                    fallback_value = updated_data.get("language", "python")

                # For data fields
                elif field == "data":
                    fallback_value = updated_data.get("data", "") or content_val or document_val
                    if not fallback_value:
                        fallback_value = "No data provided."

                # For document_type fields
                elif field == "document_type":
                    fallback_value = updated_data.get("document_type", "general")

                # For numeric fields
                elif field == "num_queries":
                    fallback_value = updated_data.get("num_queries", 3)

                elif field == "iteration":
                    fallback_value = updated_data.get("iteration", 1)

                # For paper_id fields (used in academic contexts)
                elif field == "paper_id":
                    fallback_value = updated_data.get("paper_id", "") or "unknown_paper_id"
                # Add other specific field fallbacks here
                else:
                    # --- Type-based Fallbacks (if no specific name match) ---
                    # Try to infer the field type and provide an appropriate fallback

                    # First, check if the field name gives us clues about its purpose
                    field_lower = field.lower()

                    # Check for common field name patterns
                    if any(term in field_lower for term in ["query", "question", "prompt", "instruction"]):
                        # This is likely a query or instruction field
                        fallback_value = f"Analyze the following information: {primary_text[:50]}..." if primary_text else "Please analyze this information."

                    elif any(term in field_lower for term in ["content", "text", "document", "input", "data"]):
                        # This is likely a content field
                        fallback_value = primary_text or "No content provided."

                    elif any(term in field_lower for term in ["context", "background", "info"]):
                        # This is likely a context field
                        fallback_value = context_val or primary_text or "No context available."

                    elif any(term in field_lower for term in ["summary", "overview", "abstract"]):
                        # This is likely a summary field
                        if primary_text:
                            fallback_value = primary_text[:150] + ("..." if len(primary_text) > 150 else "")
                        else:
                            fallback_value = "No summary available."

                    # Now check the field type
                    elif isinstance(field_type, type):
                        if issubclass(field_type, list):
                            # Try to infer what kind of list based on field name
                            if any(term in field_lower for term in ["option", "choice", "select"]):
                                fallback_value = ["Option 1", "Option 2"]  # Default options
                            else:
                                fallback_value = []  # Default empty list
                        elif issubclass(field_type, int):
                            # Try to infer a reasonable default based on field name
                            if "count" in field_lower or "num" in field_lower:
                                fallback_value = 1  # Default count is usually at least 1
                            elif "index" in field_lower or "position" in field_lower:
                                fallback_value = 0  # Default index is usually 0
                            else:
                                fallback_value = 0  # Generic integer default
                        elif issubclass(field_type, float): fallback_value = 0.0  # Default float
                        elif issubclass(field_type, dict): fallback_value = {}  # Default dict
                        elif issubclass(field_type, bool):
                            # Try to infer a reasonable default based on field name
                            if any(term in field_lower for term in ["is_", "has_", "should_", "can_", "enable", "active"]):
                                fallback_value = True  # Default to True for fields that sound like flags
                            else:
                                fallback_value = False  # Default boolean
                        elif issubclass(field_type, str):
                            # For string fields, try to use a relevant value from the data
                            fallback_value = primary_text or "No text provided."
                        else: fallback_value = "" # Default for other types
                    elif field_type is Any: fallback_value = primary_text or "" # Handle Any type
                    else: fallback_value = primary_text or "" # Default catch-all

                # --- Type Casting ---
                try:
                    if isinstance(field_type, type):
                        if issubclass(field_type, int): fallback_value = int(fallback_value)
                        elif issubclass(field_type, float): fallback_value = float(fallback_value)
                        elif issubclass(field_type, bool): fallback_value = bool(fallback_value)
                        # Ensure string conversion if the expected type is str
                        elif issubclass(field_type, str) and not isinstance(fallback_value, str): fallback_value = str(fallback_value)
                        # Add casting for list/dict if needed, though defaults are usually fine
                except (ValueError, TypeError) as cast_err:
                     logger.error(f"Could not cast fallback value '{fallback_value}' to type {field_type} for field '{field}'. Using original fallback. Error: {cast_err}")

                updated_data[field] = fallback_value
                logger.error(f"Populated missing input field '{field}' with fallback value: '{str(fallback_value)[:50]}...' (Type: {type(fallback_value)})")

            # We've already handled all fields in the if block above


        if missing_populated:
            logger.error(f"Agent '{agent_name}': Populated missing input fields: {missing_populated}. Original keys: {list(data.keys())}")

        # --- Final check: Log fields present vs expected (helps debug the dspy warning) ---
        expected_input_names = set(input_field_info.keys())
        fields_present_for_dspy = {k: v for k, v in updated_data.items() if k in expected_input_names}

        # This log should show exactly what's being passed to the DSPy agent's `forward` method
        logger.debug(f"Final prepared data for Agent '{agent_name}': {list(fields_present_for_dspy.keys())}")


        return updated_data # Return the prepared data


    @classmethod
    def run_chain(cls, chain_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        agent_names = cls.get_chain(chain_name)
        if not agent_names:
            logger.error(f"Chain '{chain_name}' not found.")
            input_data['error'] = f"Chain '{chain_name}' not found."; input_data['chain_status'] = 'failed'
            return input_data
        current_data = input_data.copy(); current_data['chain_status'] = 'running'
        logger.info(f"Running agent chain: {chain_name} ({' -> '.join(agent_names)})")
        logger.debug(f"Initial data for chain '{chain_name}': {list(current_data.keys())}")
        for agent_idx, agent_name in enumerate(agent_names):
            agent = cls.get_agent(agent_name)
            if agent is None:
                error_msg = f"Agent '{agent_name}' (step {agent_idx+1}/{len(agent_names)}) not found in registry for chain '{chain_name}'"
                logger.error(error_msg); current_data['error'] = current_data.get('error', '') + error_msg + "; "; current_data['chain_status'] = 'failed'; break
            if not isinstance(current_data, dict):
                 error_msg = f"Input to agent '{agent_name}' (step {agent_idx+1}) in chain '{chain_name}' is not a dict ({type(current_data)}). Chain aborted."
                 logger.error(error_msg); current_data['error'] = current_data.get('error', '') + error_msg + "; "; current_data['chain_status'] = 'failed'; break
            try: prepared_data = cls._ensure_required_fields(agent_name, agent, current_data)
            except Exception as prep_e:
                 error_msg = f"Data prep error for agent '{agent_name}' (step {agent_idx+1}) in chain '{chain_name}': {prep_e}"
                 logger.error(error_msg, exc_info=True); current_data['error'] = current_data.get('error', '') + error_msg + "; "; current_data['chain_status'] = 'failed'; break
            logger.debug(f"Calling agent '{agent_name}' (step {agent_idx+1}) in chain '{chain_name}'. Passing fields: {list(prepared_data.keys())}")
            try:
                # --- CRITICAL: ENSURE LM IS SET ---
                try:
                    lm_exists = hasattr(dspy, 'settings') and hasattr(dspy.settings, 'lm') and dspy.settings.lm is not None
                    if not lm_exists:
                        logger.error(f"DSPy LM is not configured in settings before calling agent '{agent_name}'. Aborting chain.")
                        current_data['error'] = f"DSPy LM not configured for agent {agent_name}; "; current_data['chain_status'] = 'failed'
                        break
                    # ---------------------------------
                    with dspy.context(lm=dspy.settings.lm): result = agent(**prepared_data)
                except Exception as lm_e:
                    logger.error(f"Error accessing DSPy LM for agent '{agent_name}': {lm_e}")
                    current_data['error'] = f"Error accessing DSPy LM for agent {agent_name}: {lm_e}; "; current_data['chain_status'] = 'failed'
                    break
                if isinstance(result, dspy.Prediction):
                    result_dict = {}
                    # Extract ALL fields from the Prediction object, not just OutputFields
                    # Use .keys() if available, otherwise fallback to dir()
                    result_keys = result.keys() if hasattr(result, 'keys') else dir(result)
                    for key in result_keys:
                        if not key.startswith('_') and not callable(getattr(result, key, None)):
                            result_dict[key] = getattr(result, key)
                    logger.debug(f"Agent '{agent_name}' output (Prediction): {list(result_dict.keys())}"); current_data.update(result_dict)
                elif isinstance(result, dict): logger.debug(f"Agent '{agent_name}' output (dict): {list(result.keys())}"); current_data.update(result)
                else: logger.warning(f"Agent '{agent_name}' produced non-standard output type ({type(result)}). Storing in key '{agent_name}_output'."); current_data[f'{agent_name}_output'] = result
                logger.debug(f"Data after agent '{agent_name}': {list(current_data.keys())}")
            except Exception as call_e:
                present_fields = list(prepared_data.keys()); signature = getattr(agent, 'signature', None); expected_fields = list(cls._get_signature_input_fields(signature).keys()) if signature else 'unknown'
                error_msg = (f"Agent call error for '{agent_name}' (step {agent_idx+1}) in chain '{chain_name}'. Error: {call_e}. Present fields: {present_fields}. Expected fields (approx): {expected_fields}")
                logger.error(error_msg, exc_info=True); current_data['error'] = current_data.get('error', '') + f"Agent call error for {agent_name}: {call_e}; "; current_data['chain_status'] = 'failed'
                stack_trace = traceback.format_exc(); logger.error(f"Stack trace for agent call error ({agent_name}):\n{stack_trace}"); break
        if 'chain_status' not in current_data or current_data['chain_status'] == 'running': current_data['chain_status'] = 'completed'
        logger.info(f"Chain '{chain_name}' finished with status: {current_data['chain_status']}.")
        logger.debug(f"Final data after chain '{chain_name}': {list(current_data.keys())}")
        return current_data


    @classmethod
    def setup_default_agents(cls):
        """Defines and registers default DSPy agents and chains."""
        if not DSPY_CONFIGURED:
             logger.error("Cannot setup default agents: DSPy is not configured.")
             return
        # +++ ADD LM CHECK +++
        try:
            lm_exists = hasattr(dspy, 'settings') and hasattr(dspy.settings, 'lm') and dspy.settings.lm is not None
            if not lm_exists:
                logger.error("Cannot setup default agents: DSPy LM is not configured in settings.")
                return # Crucial check
        except Exception as e:
            logger.error(f"Error checking DSPy LM status: {e}")
            return # Cannot proceed without LM

        try:
            logger.info(f"Setting up default agents with LM: {dspy.settings.lm}...") # Log LM
        except Exception as e:
            logger.info("Setting up default agents...") # Fallback log

        try:
            # --- Define Signatures ---
            # (Signatures remain the same, but ensure required=False for optional fields)
            # --- Basic Task Signatures ---
            class SummarizerSignature(dspy.Signature):
                """Summarize the provided content."""
                #instructions = 'Summarize the provided content.'
                content: str = dspy.InputField(desc="The text content to summarize")
                query: str = dspy.InputField(desc="Optional query to focus the summary", default="")
                document: str = dspy.InputField(desc="Alternative document content (optional)", default="")
                context: str = dspy.InputField(desc="Alternative context (optional)", default="")
                summary: str = dspy.OutputField(desc="A concise summary")

            class AnswererSignature(dspy.Signature):
                """Answer a question based on the provided context."""
                #instructions = 'Answer the question based on the provided context.'
                context: str = dspy.InputField(desc="Context relevant to the question")
                query: str = dspy.InputField(desc="The question to answer")
                document: str = dspy.InputField(desc="Alternative document content (optional)", default="")
                content: str = dspy.InputField(desc="Alternative content source (optional)", default="")
                answer: str = dspy.OutputField(desc="The answer derived from the context")

            class ExtractorSignature(dspy.Signature):
                """Extract specific information (e.g., entities, keywords) from content."""
                #instructions = 'Extract specific information based on the query from the content.'
                content: str = dspy.InputField(desc="The text content to extract from")
                query: str = dspy.InputField(desc="Description of the information to extract (e.g., 'names of people')")
                context: str = dspy.InputField(desc="Context relevant to the question (optional)", default="")
                document: str = dspy.InputField(desc="Alternative document content (optional)", default="")
                info: str = dspy.OutputField(desc="The extracted information")

            class ChainOfThoughtSignature(dspy.Signature):
                """Generate step-by-step reasoning to answer a query based on content."""
                #instructions = 'Generate step-by-step reasoning to answer a query based on content.'
                content: str = dspy.InputField(desc="Content to analyze")
                query: str = dspy.InputField(desc="The query or task")
                context: str = dspy.InputField(desc="Context relevant to the question (optional)", default="")
                document: str = dspy.InputField(desc="Alternative document content (optional)", default="")
                cot: str = dspy.OutputField(desc="Step-by-step reasoning process")
                conclusion: str = dspy.OutputField(desc="Final conclusion based on reasoning")

            class FactCheckerSignature(dspy.Signature):
                """Assess the factual consistency of a statement against provided context."""
                #instructions = 'Assess the factual consistency of the statement against the provided context.'
                statement: str = dspy.InputField(desc="The statement to fact-check")
                context: str = dspy.InputField(desc="The context to check against")
                query: str = dspy.InputField(desc="Optional focus for fact-checking", default="Check consistency")
                content: str = dspy.InputField(desc="Content relevant to the question (optional)", default="")
                document: str = dspy.InputField(desc="Alternative document content (optional)", default="")
                fact_check: str = dspy.OutputField(desc="Assessment of factual consistency (e.g., 'Consistent', 'Inconsistent', 'Needs More Info') with explanation")

            class TextGeneratorSignature(dspy.Signature):
                """Generic text generation based on a prompt."""
                #instructions = 'Generic text generation based on a prompt.'
                prompt: str = dspy.InputField(desc="The input prompt")
                context: str = dspy.InputField(desc="Optional context", default="")
                content: str = dspy.InputField(desc="Content relevant to the question (optional)", default="")
                document: str = dspy.InputField(desc="Alternative document content (optional)", default="")
                query: str = dspy.InputField(desc="Optional query focus", default="")
                text: str = dspy.OutputField(desc="Generated text")

            # --- Research & Reporting Signatures ---
            class QueryExpansionSignature(dspy.Signature):
                """Expand a query with related terms/concepts."""
                #instructions = 'Expand a query with related terms/concepts.'
                query: str = dspy.InputField(desc="Query to expand")
                context: str = dspy.InputField(desc="Optional context", default="")
                content: str = dspy.InputField(desc="Content relevant to the question (optional)", default="")
                document: str = dspy.InputField(desc="Alternative document content (optional)", default="")
                expanded_query: str = dspy.OutputField(desc="Expanded query string")

            class QueryRefinementSignature(dspy.Signature):
                """Generate refined/follow-up queries."""
                #instructions = 'Generate refined/follow-up queries.'
                query: str = dspy.InputField(desc="Original query")
                context: str = dspy.InputField(desc="Current research context/knowledge")
                content: str = dspy.InputField(desc="Content relevant to the question (optional)", default="")
                document: str = dspy.InputField(desc="Alternative document content (optional)", default="")
                num_queries: int = dspy.InputField(desc="Number of queries to generate")
                iteration: int = dspy.InputField(desc="Current research iteration")
                related_queries: str = dspy.OutputField(desc="List of refined/follow-up queries, one per line") # Specify format

            class FactVerificationSignature(dspy.Signature):
                """Verify content consistency against a summary."""
                #instructions = 'Verify content consistency against a summary.'
                content: str = dspy.InputField(desc="Content snippet to verify")
                summary: str = dspy.InputField(desc="Overall summary to verify against")
                context: str = dspy.InputField(desc="Context relevant to the question (optional)", default="")
                document: str = dspy.InputField(desc="Alternative document content (optional)", default="")
                query: str = dspy.InputField(desc="Optional focus for verification", default="Verify facts")
                is_consistent: str = dspy.OutputField(desc="Boolean indicating consistency (e.g., 'Yes', 'No', 'Uncertain')") # Suggest specific outputs
                confidence: str = dspy.OutputField(desc="Confidence score (0.0-1.0)")
                notes: str = dspy.OutputField(desc="Explanation for consistency assessment")

            class DocumentAnalysisSignature(dspy.Signature):
                """Provide a structured analysis of a document."""
                #instructions = 'Provide a structured analysis of a document.'
                # Make document optional, primary input should be 'content' if that's what's passed
                document: str = dspy.InputField(desc="The document text (optional)", default="")
                content: str = dspy.InputField(desc="Content relevant to the question (primary input)") # Make content the primary input
                query: str = dspy.InputField(desc="Optional query to guide analysis", default="Analyze this document")
                context: str = dspy.InputField(desc="Optional additional context", default="")
                summary: str = dspy.OutputField(desc="Overall summary")
                key_points: str = dspy.OutputField(desc="List of key points, one per line") # Specify format
                entities: str = dspy.OutputField(desc="List of important entities, comma-separated") # Specify format
                sentiment: str = dspy.OutputField(desc="Overall sentiment (e.g., Positive, Negative, Neutral)")

            class InterpreterSignature(dspy.Signature):
                """Interpret research findings contextually."""
                #instructions = 'Interpret research findings contextually.'
                query: str = dspy.InputField(desc="The original research query")
                context: str = dspy.InputField(desc="Aggregated research findings")
                content: str = dspy.InputField(desc="Content relevant to the question (optional)", default="")
                document: str = dspy.InputField(desc="Alternative document content (optional)", default="")
                interpretation: str = dspy.OutputField(desc="Main interpretation")
                insights: str = dspy.OutputField(desc="List of key insights, one per line") # Specify format
                limitations: str = dspy.OutputField(desc="List of limitations, one per line") # Specify format
                confidence: str = dspy.OutputField(desc="Confidence score (0.0-1.0)")

            class ProposalGeneratorSignature(dspy.Signature):
                """Generate actionable proposals from findings. Format each recommendation, next step, and alternative as a separate line. Do not use bullet points or numbering."""
                #instructions = 'Generate actionable proposals from findings. Format each recommendation, next step, and alternative as a separate line. Do not use bullet points or numbering.'
                query: str = dspy.InputField(desc="Original research query")
                context: str = dspy.InputField(desc="Research findings context")
                content: str = dspy.InputField(desc="Content relevant to the question (optional)", default="")
                document: str = dspy.InputField(desc="Alternative document content (optional)", default="")
                recommendations: str = dspy.OutputField(desc="List of specific recommendations, with each recommendation on a separate line. Do not use bullet points or numbering.")
                next_steps: str = dspy.OutputField(desc="List of concrete next steps, with each step on a separate line. Do not use bullet points or numbering.")
                alternatives: str = dspy.OutputField(desc="List of alternative approaches, with each alternative on a separate line. Do not use bullet points or numbering.")
                rationale: str = dspy.OutputField(desc="Justification for proposals")

            class TechnicalAnalyzerSignature(dspy.Signature):
                """Provide technical analysis of findings."""
                #instructions = 'Provide technical analysis of findings.'
                query: str = dspy.InputField(desc="Original research query")
                context: str = dspy.InputField(desc="Research findings context")
                content: str = dspy.InputField(desc="Content relevant to the question (optional)", default="")
                document: str = dspy.InputField(desc="Alternative document content (optional)", default="")
                analysis: str = dspy.OutputField(desc="Technical analysis summary")
                details: str = dspy.OutputField(desc="List of technical details, one per line") # Specify format
                challenges: str = dspy.OutputField(desc="List of technical challenges, one per line") # Specify format
                solutions: str = dspy.OutputField(desc="List of potential solutions, one per line") # Specify format

            class ContentSynthesizerSignature(dspy.Signature):
                """Synthesize findings into a structured article."""
                #instructions = 'Synthesize findings into a structured article.'
                query: str = dspy.InputField(desc="Original research query")
                context: str = dspy.InputField(desc="Research findings context")
                content: str = dspy.InputField(desc="Content relevant to the question (optional)", default="")
                document: str = dspy.InputField(desc="Alternative document content (optional)", default="")
                article: str = dspy.OutputField(desc="Synthesized article content")
                article_type: str = dspy.OutputField(desc="Type of article (e.g., summary, analysis, blog post)")
                key_themes: str = dspy.OutputField(desc="List of key themes discussed, one per line") # Specify format
                word_count: str = dspy.OutputField(desc="Approximate word count")

            # --- Document Type Specific Signatures (from original dspy_agents.py) ---
            class CodeAnalysisSignature(dspy.Signature):
                """Analyze code documents and extract key information."""
                #instructions = 'Analyze code documents and extract key information.'
                code: str = dspy.InputField(desc="The code content to analyze")
                language: str = dspy.InputField(desc="The programming language of the code")
                query: str = dspy.InputField(desc="The query or task to focus the analysis on")
                document: str = dspy.InputField(desc="Alternative document content (optional)", default="")
                content: str = dspy.InputField(desc="Alternative content source (optional)", default="")
                context: str = dspy.InputField(desc="Additional context for analysis (optional)", default="")
                summary: str = dspy.OutputField(desc="A concise summary of what the code does")
                functions: str = dspy.OutputField(desc="List of key functions/methods in the code, comma-separated") # Specify format
                classes: str = dspy.OutputField(desc="List of key classes in the code, comma-separated") # Specify format
                dependencies: str = dspy.OutputField(desc="List of external dependencies or imports, comma-separated") # Specify format
                complexity: str = dspy.OutputField(desc="Assessment of code complexity")
                issues: str = dspy.OutputField(desc="Potential issues or bugs in the code, one per line") # Specify format
                suggestions: str = dspy.OutputField(desc="Suggestions for improvement, one per line") # Specify format

            class SpreadsheetAnalysisSignature(dspy.Signature):
                """Analyze spreadsheet data and extract key information."""
                #instructions = 'Analyze spreadsheet data and extract key information.'
                data: str = dspy.InputField(desc="The spreadsheet content in text format")
                query: str = dspy.InputField(desc="The query or task to focus the analysis on")
                document: str = dspy.InputField(desc="Alternative document content (optional)", default="")
                content: str = dspy.InputField(desc="Alternative content source (optional)", default="")
                context: str = dspy.InputField(desc="Additional context for analysis (optional)", default="")
                summary: str = dspy.OutputField(desc="A concise summary of the data")
                structure: str = dspy.OutputField(desc="Description of the data structure (columns, sheets)")
                key_metrics: str = dspy.OutputField(desc="Key metrics or statistics from the data, one per line") # Specify format
                patterns: str = dspy.OutputField(desc="Patterns or trends identified in the data")
                anomalies: str = dspy.OutputField(desc="Anomalies or outliers in the data")
                insights: str = dspy.OutputField(desc="Key insights derived from the data, one per line") # Specify format

            class PDFAnalysisSignature(dspy.Signature):
                """Analyze PDF documents and extract key information."""
                #instructions = 'Analyze PDF documents and extract key information.'
                content: str = dspy.InputField(desc="The PDF content in text format")
                query: str = dspy.InputField(desc="The query or task to focus the analysis on")
                document: str = dspy.InputField(desc="Alternative document content (optional)", default="")
                context: str = dspy.InputField(desc="Additional context for analysis (optional)", default="")
                summary: str = dspy.OutputField(desc="A concise summary of the document")
                key_points: str = dspy.OutputField(desc="Key points from the document, one per line") # Specify format
                entities: str = dspy.OutputField(desc="Important entities mentioned in the document, comma-separated") # Specify format
                topics: str = dspy.OutputField(desc="Main topics covered in the document, comma-separated") # Specify format
                structure: str = dspy.OutputField(desc="Document structure (sections, headings)")
                citations: str = dspy.OutputField(desc="Citations or references in the document, one per line") # Specify format

            class TechnicalDocAnalysisSignature(dspy.Signature):
                """Analyze technical documents and extract key information."""
                #instructions = 'Analyze technical documents and extract key information.'
                content: str = dspy.InputField(desc="The technical document content")
                document_type: str = dspy.InputField(desc="The type of technical document (e.g., API doc, whitepaper)")
                query: str = dspy.InputField(desc="The query or task to focus the analysis on")
                document: str = dspy.InputField(desc="Alternative document content (optional)", default="")
                context: str = dspy.InputField(desc="Additional context for analysis (optional)", default="")
                summary: str = dspy.OutputField(desc="A concise summary of the technical document")
                key_concepts: str = dspy.OutputField(desc="Key technical concepts explained in the document, one per line") # Specify format
                technical_details: str = dspy.OutputField(desc="Important technical details or specifications, one per line") # Specify format
                requirements: str = dspy.OutputField(desc="Requirements or prerequisites mentioned, one per line") # Specify format
                examples: str = dspy.OutputField(desc="Code examples or usage examples")
                limitations: str = dspy.OutputField(desc="Limitations or constraints mentioned, one per line") # Specify format

            class ResearchPaperAnalysisSignature(dspy.Signature):
                """Analyze research papers and extract key information."""
                #instructions = 'Analyze research papers and extract key information.'
                content: str = dspy.InputField(desc="The research paper content")
                query: str = dspy.InputField(desc="The query or task to focus the analysis on")
                document: str = dspy.InputField(desc="Alternative document content (optional)", default="")
                context: str = dspy.InputField(desc="Additional context for analysis (optional)", default="")
                summary: str = dspy.OutputField(desc="A concise summary of the research paper")
                research_question: str = dspy.OutputField(desc="The main research question or objective")
                methodology: str = dspy.OutputField(desc="The methodology used in the research")
                findings: str = dspy.OutputField(desc="Key findings or results")
                limitations: str = dspy.OutputField(desc="Limitations of the research")
                implications: str = dspy.OutputField(desc="Implications or applications of the research")
                future_work: str = dspy.OutputField(desc="Suggested future work")

            class ChainOfThoughtAnalysisSignature(dspy.Signature): # Renamed for clarity
                """Perform step-by-step reasoning on a document."""
                #instructions = 'Perform step-by-step reasoning on a document.'
                content: str = dspy.InputField(desc="The document content to analyze")
                query: str = dspy.InputField(desc="The query or task to focus the analysis on")
                document: str = dspy.InputField(desc="Alternative document content (optional)", default="")
                context: str = dspy.InputField(desc="Additional context for analysis (optional)", default="")
                reasoning: str = dspy.OutputField(desc="Step-by-step reasoning process")
                key_insights: str = dspy.OutputField(desc="Key insights derived from the reasoning, one per line") # Specify format
                conclusion: str = dspy.OutputField(desc="Conclusion based on the reasoning")

            class MultiDocumentSynthesisSignature(dspy.Signature):
                """Synthesize information from multiple documents."""
                #instructions = 'Synthesize information from multiple documents based on the query.'
                documents: str = dspy.InputField(desc="JSON string representing a list of document contents and metadata (e.g., [{'title': 'Doc1', 'content': '...'}, ...])") # Changed to string input
                query: str = dspy.InputField(desc="The query or task to focus the synthesis on")
                document: str = dspy.InputField(desc="Single document content (optional)", default="")
                content: str = dspy.InputField(desc="Alternative content source (optional)", default="")
                synthesis: str = dspy.OutputField(desc="Synthesized information from all documents")
                common_themes: str = dspy.OutputField(desc="Common themes across documents, comma-separated") # Specify format
                contradictions: str = dspy.OutputField(desc="Contradictions or disagreements between documents, one per line") # Specify format
                unique_insights: str = dspy.OutputField(desc="Unique insights from specific documents")
                integrated_view: str = dspy.OutputField(desc="Integrated view of the information")


            # --- Register Agents --- (Using dspy.Predict for simplicity)
            try:
                # Register each agent individually with detailed error logging
                agent_signatures = [
                    ("summarizer", SummarizerSignature),
                    ("answerer", AnswererSignature),
                    ("extractor", ExtractorSignature),
                    ("chain_of_thought", ChainOfThoughtSignature),
                    ("fact_checker", FactCheckerSignature),
                    ("document_analyzer", DocumentAnalysisSignature),
                    ("interpreter", InterpreterSignature),
                    ("proposal_generator", ProposalGeneratorSignature),
                    ("technical_analyzer", TechnicalAnalyzerSignature),
                    ("query_refinement", QueryRefinementSignature),
                    ("fact_verification", FactVerificationSignature),
                    ("text_generator", TextGeneratorSignature),
                    ("query_expansion", QueryExpansionSignature),
                    ("content_synthesizer", ContentSynthesizerSignature),
                    ("code_analyzer", CodeAnalysisSignature),
                    ("spreadsheet_analyzer", SpreadsheetAnalysisSignature),
                    ("pdf_analyzer", PDFAnalysisSignature),
                    ("tech_doc_analyzer", TechnicalDocAnalysisSignature),
                    ("research_paper_analyzer", ResearchPaperAnalysisSignature),
                    ("chain_of_thought_analyzer", ChainOfThoughtAnalysisSignature),
                    ("multi_doc_synthesizer", MultiDocumentSynthesisSignature)
                ]

                for agent_name, signature_class in agent_signatures:
                    try:
                        logger.info(f"Registering agent: {agent_name}")
                        cls.register_agent(agent_name, dspy.Predict(signature_class))
                        logger.info(f"Successfully registered agent: {agent_name}")
                    except Exception as e:
                        logger.error(f"Error registering agent {agent_name}: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Error in agent registration block: {e}", exc_info=True)

            # --- ADD VERIFICATION LOGGING ---
            registered_agents = list(cls.agents.keys())
            logger.info(f"Registered Agents: {registered_agents}")
            if not registered_agents:
                 logger.error("!!! No agents were registered during setup_default_agents !!!")
            # --- END VERIFICATION LOGGING ---

            # Log the configured LM (should be set by now)
            try:
                lm_exists = hasattr(dspy, 'settings') and hasattr(dspy.settings, 'lm') and dspy.settings.lm is not None
                if lm_exists:
                    logger.info(f"DSPy agents registered with LM: {dspy.settings.lm}")
                else:
                    logger.warning("DSPy LM is not configured during agent registration!")
            except Exception as e:
                logger.warning(f"Error checking DSPy LM status during agent registration: {e}")


            # --- Register Chains ---
            cls.register_chain("summarize_then_fact_check", ["summarizer", "fact_checker"])
            cls.register_chain("cot_then_answer", ["chain_of_thought", "answerer"])
            cls.register_chain("analyze_document", ["document_analyzer"]) # Simple analysis chain
            cls.register_chain("enhanced_analysis", ["document_analyzer", "fact_checker"])
            # Deep research: Analyze, CoT for depth, Summarize findings
            cls.register_chain("deep_research", ["document_analyzer", "chain_of_thought", "summarizer"])
            # Advanced reporting chains
            cls.register_chain("interpretation_chain", ["interpreter"]) # Assumes context is prepared
            cls.register_chain("proposal_chain", ["proposal_generator"]) # Assumes context is prepared
            cls.register_chain("technical_chain", ["technical_analyzer"]) # Assumes context is prepared
            cls.register_chain("verification_chain", ["fact_verification"]) # Assumes content/summary prepared
            # Multi-iteration support chain (can be called iteratively)
            cls.register_chain("multi_iteration_step", ["query_refinement", "answerer"]) # Refine query, get answer based on context
            # Comprehensive analysis chain - used potentially by 'enhanced' report mode implicitly
            cls.register_chain("comprehensive_report_gen", ["interpreter", "proposal_generator", "technical_analyzer"]) #, "content_synthesizer"
            cls.register_chain("comprehensive_analysis", ["interpreter", "proposal_generator", "technical_analyzer"]) #, "content_synthesizer"
            cls.register_chain("code_analysis_chain", ["code_analyzer"])
            cls.register_chain("spreadsheet_analysis_chain", ["spreadsheet_analyzer"])
            cls.register_chain("pdf_analysis_chain", ["pdf_analyzer"])
            cls.register_chain("tech_doc_analysis_chain", ["tech_doc_analyzer"])
            cls.register_chain("research_paper_analysis_chain", ["research_paper_analyzer"])
            cls.register_chain("cot_analysis_chain", ["cot_analyzer"])
            cls.register_chain("multi_doc_synthesis_chain", ["multi_doc_synthesizer"])

            # --- ADD VERIFICATION LOGGING ---
            registered_chains = list(cls.chains.keys())
            logger.info(f"Registered Chains: {registered_chains}")
            if not registered_chains:
                 logger.error("!!! No chains were registered during setup_default_agents !!!")
            # --- END VERIFICATION LOGGING ---

            logger.info("Successfully registered default DSPy agents and chains")
        except ImportError:
            logger.error("DSPy not installed. Default agents and chains cannot be registered.")
        except Exception as e:
            logger.error(f"Failed to setup default DSPy agents/chains: {e}", exc_info=True)

# Call this at startup to configure DSPy and register agents/chains
def initialize_dspy():
    """
    Initialize DSPy with proper configuration and set up default agents.
    Safe to call multiple times.
    """
    global DSPY_CONFIGURED
    if not DSPY_CONFIGURED:
        configure_dspy() # Attempt configuration

    # Setup agents regardless of whether configure_dspy just ran or not
    # This ensures agents are registered even if config was set previously
    try:
        # Check if DSPy is properly configured with an LM
        lm_exists = False
        try:
            import dspy
            lm_exists = hasattr(dspy, 'settings') and hasattr(dspy.settings, 'lm') and dspy.settings.lm is not None
            if lm_exists:
                logger.info(f"DSPy LM is configured: {dspy.settings.lm}")
                DSPY_CONFIGURED = True
            else:
                logger.error("DSPy LM is not configured in settings.")
                DSPY_CONFIGURED = False
        except Exception as e:
            logger.error(f"Error checking DSPy initialization: {e}")
            DSPY_CONFIGURED = False

        if DSPY_CONFIGURED: # Only setup agents if configuration is successful
            if not DSPyAgentRegistry.agents:
                logger.info("Agent registry is empty. Setting up default agents...") # Added log
                DSPyAgentRegistry.setup_default_agents()
            else:
                logger.info("Default agents already set up.") # Log if already done

                # All agents are considered critical
                # Define the complete list of expected agents
                all_critical_agents = [
                    "summarizer", "answerer", "extractor", "chain_of_thought",
                    "fact_checker", "document_analyzer", "interpreter",
                    "proposal_generator", "technical_analyzer", "query_refinement",
                    "fact_verification", "text_generator", "query_expansion",
                    "content_synthesizer", "code_analyzer", "spreadsheet_analyzer",
                    "pdf_analyzer", "tech_doc_analyzer", "research_paper_analyzer",
                    "chain_of_thought_analyzer", "multi_doc_synthesizer"
                ]

                # Check which critical agents are missing
                missing_critical = [agent for agent in all_critical_agents if agent not in DSPyAgentRegistry.agents]

                if missing_critical:
                    logger.warning(f"Critical agents missing from registry: {missing_critical}. Registering them now...")
                    # We'll register the missing agents in the code block below
    except Exception as e:
        logger.error(f"Error during DSPy initialization check: {e}")

    # Define the list of critical agents in the outer scope as well
    all_critical_agents = [
        "summarizer", "answerer", "extractor", "chain_of_thought",
        "fact_checker", "document_analyzer", "interpreter",
        "proposal_generator", "technical_analyzer", "query_refinement",
        "fact_verification", "text_generator", "query_expansion",
        "content_synthesizer", "code_analyzer", "spreadsheet_analyzer",
        "pdf_analyzer", "tech_doc_analyzer", "research_paper_analyzer",
        "chain_of_thought_analyzer", "multi_doc_synthesizer"
    ]

    # Register missing critical agents if needed
    if DSPY_CONFIGURED and 'missing_critical' in locals() and missing_critical:
        # If there are any missing critical agents, it's simplest to just call setup_default_agents()
        # This will register all agents with their proper signatures
        logger.info(f"Registering {len(missing_critical)} missing critical agents: {missing_critical}")
        try:
            # Log the current state before re-registering
            logger.info(f"Current agents before re-registering: {list(DSPyAgentRegistry.agents.keys())}")

            # Re-register all agents
            DSPyAgentRegistry.setup_default_agents()

            # Log the state after re-registering
            logger.info(f"Agents after re-registering: {list(DSPyAgentRegistry.agents.keys())}")

            # Check if any agents are still missing
            still_missing = [agent for agent in all_critical_agents if agent not in DSPyAgentRegistry.agents]
            if still_missing:
                logger.error(f"Failed to register some critical agents: {still_missing}")
            else:
                logger.info("All critical agents successfully registered")
        except Exception as e:
            logger.error(f"Error during agent re-registration: {e}", exc_info=True)
    elif not DSPY_CONFIGURED:
        logger.error("DSPy initialization failed or skipped; default agents not set up.")

    # +++ ADD FINAL CHECK LOG +++
    try:
        if DSPY_CONFIGURED:
            # Check if dspy.settings.lm exists and is accessible
            lm_exists = hasattr(dspy, 'settings') and hasattr(dspy.settings, 'lm') and dspy.settings.lm is not None

            if lm_exists:
                logger.info(f"DSPy initialization check complete. LM: {dspy.settings.lm}")
            else:
                logger.warning("DSPy initialization check: LM is None or not accessible despite configured flag.")
        else:
            logger.error("DSPy initialization check failed: Not configured.")
    except Exception as e:
        logger.error(f"Error during DSPy initialization check: {e}")