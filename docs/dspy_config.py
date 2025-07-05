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
        from src.local_file_research.litellm_patch import apply_patch, patch_dspy
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
        # Check if signature is valid and has attributes we can inspect
        if not signature or not isinstance(signature, type) or not issubclass(signature, dspy.Signature):
            logger.warning(f"Provided signature '{signature}' is not a valid dspy.Signature subclass. Cannot extract fields.")
            return input_fields

        try:
            # Iterate through the signature's defined fields (attributes)
            for field_name, field_obj in signature.__dict__.items():
                if isinstance(field_obj, dspy.Field):
                    # Check if it's an input field using its metadata
                    is_input = field_obj.json_schema_extra.get('__dspy_field_type') == 'input'

                    if is_input:
                        # Extract requirement status (default to True if not explicitly set)
                        # The 'required' attribute is directly on the Field object
                        is_required = getattr(field_obj, 'required', True)

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
            logger.error(f"Failed to inspect signature {signature.__name__} for input fields: {e}", exc_info=True)

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

            if is_required and (is_missing or is_empty):
                missing_populated.append(field)
                fields_with_fallbacks.add(field)
                fallback_value = None # Start with None

                # --- Specific Fallbacks by Field Name ---
                # Important: Prioritize name-based fallbacks for fields with specific meanings
                if field in ("query", "question"): fallback_value = query_val or "No query provided."
                elif field == "content": fallback_value = content_val or primary_text or "No content provided." # Ensure not empty
                elif field == "document": fallback_value = document_val or primary_text or "No document provided."# Ensure not empty
                elif field == "context": fallback_value = fallback_context or "No context provided." # Ensure not empty
                elif field == "summary": fallback_value = summary_val or (primary_text[:200] + "..." if len(primary_text) > 200 else primary_text) or "No summary provided." # Ensure not empty
                elif field == "statement": fallback_value = statement_val or "No statement provided."
                elif field == "prompt": fallback_value = prompt_val or f"Process the following context: {fallback_context}" or "Please process the input." # Ensure not empty
                elif field == "code": fallback_value = content_val or document_val or "No code provided."
                elif field == "language": fallback_value = updated_data.get("language", "python")
                elif field == "data": fallback_value = content_val or document_val or "No data provided."
                elif field == "document_type": fallback_value = updated_data.get("document_type", "general")
                elif field == "num_queries": fallback_value = 3 # Default to int
                elif field == "iteration": fallback_value = 1 # Default to int
                # Add other specific field fallbacks here
                else:
                    # --- Type-based Fallbacks (if no specific name match) ---
                    if isinstance(field_type, type):
                        if issubclass(field_type, list): fallback_value = []
                        elif issubclass(field_type, int): fallback_value = 0
                        elif issubclass(field_type, float): fallback_value = 0.0
                        elif issubclass(field_type, dict): fallback_value = {}
                        elif issubclass(field_type, bool): fallback_value = False
                        elif issubclass(field_type, str): fallback_value = "" # Explicit fallback for str
                        else: fallback_value = "" # Default for other types
                    elif field_type is Any: fallback_value = "" # Handle Any type
                    else: fallback_value = "" # Default catch-all

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
                     logger.warning(f"Could not cast fallback value '{fallback_value}' to type {field_type} for field '{field}'. Using original fallback. Error: {cast_err}")

                updated_data[field] = fallback_value
                logger.debug(f"Populated REQUIRED field '{field}' with fallback value: '{str(fallback_value)[:50]}...' (Type: {type(fallback_value)})")

            elif not is_required and is_missing:
                # If an OPTIONAL field is missing, DO NOT add a fallback here.
                # Get the default value from the Field object itself if possible.
                field_definition = getattr(signature, field, None)
                if isinstance(field_definition, dspy.Field) and hasattr(field_definition, 'default'):
                    # Only populate if the default is not the special Pydantic Undefined type
                    if field_definition.default is not dspy.pydantic_form.PydanticUndefined:
                         updated_data[field] = field_definition.default
                         logger.debug(f"Populated OPTIONAL missing field '{field}' with default from Field definition: {field_definition.default}")
                    else:
                         logger.debug(f"Optional field '{field}' is missing and has no explicit default in Field definition. DSPy will handle.")
                else:
                     logger.debug(f"Optional field '{field}' is missing. DSPy should handle.")


        if missing_populated:
            logger.warning(f"Agent '{agent_name}': Populated missing/empty REQUIRED fields: {missing_populated}. Original keys: {list(data.keys())}")

        # --- Final check: Log fields present vs expected (helps debug the dspy warning) ---
        present_fields = set(updated_data.keys())
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
                if not dspy.settings.lm:
                    logger.error(f"DSPy LM is not configured in settings before calling agent '{agent_name}'. Aborting chain.")
                    current_data['error'] = f"DSPy LM not configured for agent {agent_name}; "; current_data['chain_status'] = 'failed'
                    break
                # ---------------------------------
                with dspy.context(lm=dspy.settings.lm): result = agent(**prepared_data)
                if isinstance(result, dspy.Prediction):
                    result_dict = {}; sig = getattr(agent, 'signature', None)
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
        if not dspy.settings.lm:
             logger.error("Cannot setup default agents: DSPy LM is not configured in settings.")
             return # Crucial check

        logger.info(f"Setting up default agents with LM: {dspy.settings.lm}...") # Log LM

        try:
            # --- Define Signatures ---
            # (Signatures remain the same, but ensure required=False for optional fields)
            class SummarizerSignature(dspy.Signature):
                """Summarize the provided content."""
                content = dspy.InputField(desc="The text content to summarize")
                query = dspy.InputField(desc="Optional query to focus the summary", default="", required=False) # Optional
                document = dspy.InputField(desc="Alternative document content (optional)", default="", required=False) # Optional
                context = dspy.InputField(desc="Alternative context (optional)", default="", required=False) # Optional
                summary = dspy.OutputField(desc="A concise summary")

            class AnswererSignature(dspy.Signature):
                """Answer a question based on the provided context."""
                context = dspy.InputField(desc="Context relevant to the question")
                query = dspy.InputField(desc="The question to answer")
                document = dspy.InputField(desc="Alternative document content (optional)", default="", required=False) # Optional
                content = dspy.InputField(desc="Alternative content source (optional)", default="", required=False) # Optional
                answer = dspy.OutputField(desc="The answer derived from the context")

            class ExtractorSignature(dspy.Signature):
                """Extract specific information (e.g., entities, keywords) from content."""
                content = dspy.InputField(desc="The text content to extract from")
                query = dspy.InputField(desc="Description of the information to extract (e.g., 'names of people')")
                context = dspy.InputField(desc="Context relevant to the question (optional)", default="", required=False) # Optional
                document = dspy.InputField(desc="Alternative document content (optional)", default="", required=False) # Optional
                info = dspy.OutputField(desc="The extracted information")

            class ChainOfThoughtSignature(dspy.Signature):
                """Generate step-by-step reasoning to answer a query based on content."""
                content = dspy.InputField(desc="Content to analyze")
                query = dspy.InputField(desc="The query or task")
                context = dspy.InputField(desc="Context relevant to the question (optional)", default="", required=False) # Optional
                document = dspy.InputField(desc="Alternative document content (optional)", default="", required=False) # Optional
                cot = dspy.OutputField(desc="Step-by-step reasoning process")
                conclusion = dspy.OutputField(desc="Final conclusion based on reasoning")

            class FactCheckerSignature(dspy.Signature):
                """Assess the factual consistency of a statement against provided context."""
                statement = dspy.InputField(desc="The statement to fact-check")
                context = dspy.InputField(desc="The context to check against")
                query = dspy.InputField(desc="Optional focus for fact-checking", default="", required=False) # Optional
                content = dspy.InputField(desc="Content relevant to the question (optional)", default="", required=False) # Optional
                document = dspy.InputField(desc="Alternative document content (optional)", default="", required=False) # Optional
                fact_check = dspy.OutputField(desc="Assessment of factual consistency (e.g., 'Consistent', 'Inconsistent', 'Needs More Info') with explanation")

            class DocumentAnalysisSignature(dspy.Signature):
                """Provide a structured analysis of a document."""
                document = dspy.InputField(desc="The document text", default="", required=False) # Changed to optional with fallback
                query = dspy.InputField(desc="Optional query to guide analysis", default="Analyze this document", required=False) # Optional
                context = dspy.InputField(desc="Optional additional context", default="", required=False) # Optional
                content = dspy.InputField(desc="Content relevant to the question (optional)", default="", required=False) # Optional
                summary = dspy.OutputField(desc="Overall summary")
                key_points = dspy.OutputField(desc="List of key points")
                entities = dspy.OutputField(desc="List of important entities")
                sentiment = dspy.OutputField(desc="Overall sentiment (e.g., Positive, Negative, Neutral)")

            class InterpreterSignature(dspy.Signature):
                """Interpret research findings contextually."""
                query = dspy.InputField(desc="The original research query")
                context = dspy.InputField(desc="Aggregated research findings")
                content = dspy.InputField(desc="Content relevant to the question (optional)", default="", required=False) # Optional
                document = dspy.InputField(desc="Alternative document content (optional)", default="", required=False) # Optional
                interpretation = dspy.OutputField(desc="Main interpretation")
                insights = dspy.OutputField(desc="List of key insights")
                limitations = dspy.OutputField(desc="List of limitations")
                confidence = dspy.OutputField(desc="Confidence score (0.0-1.0)")

            class ProposalGeneratorSignature(dspy.Signature):
                """Generate actionable proposals from findings."""
                query = dspy.InputField(desc="Original research query")
                context = dspy.InputField(desc="Research findings context")
                content = dspy.InputField(desc="Content relevant to the question (optional)", default="", required=False) # Optional
                document = dspy.InputField(desc="Alternative document content (optional)", default="", required=False) # Optional
                recommendations = dspy.OutputField(desc="List of specific recommendations")
                next_steps = dspy.OutputField(desc="List of concrete next steps")
                alternatives = dspy.OutputField(desc="List of alternative approaches")
                rationale = dspy.OutputField(desc="Justification for proposals")

            class TechnicalAnalyzerSignature(dspy.Signature):
                """Provide technical analysis of findings."""
                query = dspy.InputField(desc="Original research query")
                context = dspy.InputField(desc="Research findings context")
                content = dspy.InputField(desc="Content relevant to the question (optional)", default="", required=False) # Optional
                document = dspy.InputField(desc="Alternative document content (optional)", default="", required=False) # Optional
                analysis = dspy.OutputField(desc="Technical analysis summary")
                details = dspy.OutputField(desc="List of technical details")
                challenges = dspy.OutputField(desc="List of technical challenges")
                solutions = dspy.OutputField(desc="List of potential solutions")

            # --- CORRECTED QueryRefinementSignature ---
            class QueryRefinementSignature(dspy.Signature):
                """Generate refined/follow-up queries."""
                query = dspy.InputField(desc="Original query")
                context = dspy.InputField(desc="Current research context/knowledge")
                content = dspy.InputField(desc="Content relevant to the question (optional)", default="", required=False) # Optional
                document = dspy.InputField(desc="Alternative document content (optional)", default="", required=False) # Optional
                num_queries: int = dspy.InputField(desc="Number of queries to generate") # Use int annotation
                iteration: int = dspy.InputField(desc="Current research iteration") # Use int annotation
                related_queries = dspy.OutputField(desc="List of refined/follow-up queries")

            class FactVerificationSignature(dspy.Signature):
                """Verify content consistency against a summary."""
                content = dspy.InputField(desc="Content snippet to verify")
                summary = dspy.InputField(desc="Overall summary to verify against")
                context = dspy.InputField(desc="Context relevant to the question (optional)", default="", required=False) # Optional
                document = dspy.InputField(desc="Alternative document content (optional)", default="", required=False) # Optional
                query = dspy.InputField(desc="Optional focus for verification", default="Verify facts", required=False) # Optional
                is_consistent = dspy.OutputField(desc="Boolean indicating consistency")
                confidence = dspy.OutputField(desc="Confidence score (0.0-1.0)")
                notes = dspy.OutputField(desc="Explanation for consistency assessment")

            class TextGeneratorSignature(dspy.Signature):
                """Generic text generation based on a prompt."""
                prompt = dspy.InputField(desc="The input prompt")
                context = dspy.InputField(desc="Optional context", default="", required=False) # Optional
                content = dspy.InputField(desc="Content relevant to the question (optional)", default="", required=False) # Optional
                document = dspy.InputField(desc="Alternative document content (optional)", default="", required=False) # Optional
                query = dspy.InputField(desc="Optional query focus", default="", required=False) # Optional
                text = dspy.OutputField(desc="Generated text")

            class ContentSynthesizerSignature(dspy.Signature):
                """Synthesize findings into a structured article."""
                query = dspy.InputField(desc="Original research query")
                context = dspy.InputField(desc="Research findings context")
                content = dspy.InputField(desc="Content relevant to the question (optional)", default="", required=False) # Optional
                document = dspy.InputField(desc="Alternative document content (optional)", default="", required=False) # Optional
                article = dspy.OutputField(desc="Synthesized article content")
                article_type = dspy.OutputField(desc="Type of article (e.g., summary, analysis, blog post)")
                key_themes = dspy.OutputField(desc="List of key themes discussed")
                word_count = dspy.OutputField(desc="Approximate word count")

            class QueryExpansionSignature(dspy.Signature):
                """Expand a query with related terms/concepts."""
                query = dspy.InputField(desc="Query to expand")
                context = dspy.InputField(desc="Optional context", default="", required=False) # Optional
                content = dspy.InputField(desc="Content relevant to the question (optional)", default="", required=False) # Optional
                document = dspy.InputField(desc="Alternative document content (optional)", default="", required=False) # Optional
                expanded_query = dspy.OutputField(desc="Expanded query string")


            # --- Register Agents --- (Using dspy.Predict for simplicity)
            cls.register_agent("summarizer", dspy.Predict(SummarizerSignature))
            cls.register_agent("answerer", dspy.Predict(AnswererSignature))
            cls.register_agent("extractor", dspy.Predict(ExtractorSignature))
            cls.register_agent("chain_of_thought", dspy.Predict(ChainOfThoughtSignature))
            cls.register_agent("fact_checker", dspy.Predict(FactCheckerSignature))
            cls.register_agent("document_analyzer", dspy.Predict(DocumentAnalysisSignature))
            cls.register_agent("interpreter", dspy.Predict(InterpreterSignature))
            cls.register_agent("proposal_generator", dspy.Predict(ProposalGeneratorSignature))
            cls.register_agent("technical_analyzer", dspy.Predict(TechnicalAnalyzerSignature))
            cls.register_agent("query_refinement", dspy.Predict(QueryRefinementSignature))
            cls.register_agent("fact_verification", dspy.Predict(FactVerificationSignature))
            cls.register_agent("text_generator", dspy.Predict(TextGeneratorSignature))
            cls.register_agent("query_expansion", dspy.Predict(QueryExpansionSignature))
            cls.register_agent("content_synthesizer", dspy.Predict(ContentSynthesizerSignature))

            # --- ADD VERIFICATION LOGGING ---
            registered_agents = list(cls.agents.keys())
            logger.info(f"Registered Agents: {registered_agents}")
            if not registered_agents:
                 logger.error("!!! No agents were registered during setup_default_agents !!!")
            # --- END VERIFICATION LOGGING ---

            # Log the configured LM (should be set by now)
            if dspy.settings.lm:
                logger.info(f"DSPy agents registered with LM: {dspy.settings.lm}")
            else:
                logger.error("DSPy LM is not configured during agent registration!")


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
            cls.register_chain("comprehensive_report_gen", ["interpreter", "proposal_generator", "technical_analyzer", "content_synthesizer"])

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
    if DSPY_CONFIGURED: # Only setup agents if configuration is successful
        if not DSPyAgentRegistry.agents:
            logger.info("Agent registry is empty. Setting up default agents...") # Added log
            DSPyAgentRegistry.setup_default_agents()
        else:
            logger.info("Default agents already set up.") # Log if already done
    else:
        logger.error("DSPy initialization failed or skipped; default agents not set up.")

    # +++ ADD FINAL CHECK LOG +++
    if DSPY_CONFIGURED and dspy.settings.lm:
         logger.info(f"DSPy initialization check complete. LM: {dspy.settings.lm}")
    elif not DSPY_CONFIGURED:
         logger.error("DSPy initialization check failed: Not configured.")
    elif not dspy.settings.lm:
         logger.error("DSPy initialization check failed: LM is None despite configured flag.")