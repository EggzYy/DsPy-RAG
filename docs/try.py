
**`dspy_config.py`:**
    *   `_get_signature_input_fields`: Changed to use `inspect.getmembers` and `signature.__annotations__` for more reliable field discovery. It now also extracts the `default` value from the `dspy.Field` definition. Caching was added for performance. Type extraction logic was refined to handle `Union`, `Optional`, `List`, `Dict`, and `ForwardRef`.
    *   `_ensure_required_fields`: Updated to use the new info from `_get_signature_input_fields`. It now explicitly handles *optional* missing fields by adding their default values found in the signature definition. If no explicit default exists for an optional field, it adds a type-appropriate default (like `None`, `""`, `[]`, `{}`) to prevent the DSPy warning. Fallback logic for *required* fields remains similar but uses the improved field info. It now returns *only* the fields expected by the signature's input definition.
    *   `QueryRefinementSignature`: Corrected the type hints for `num_queries` and `iteration` to use standard Python type annotations (`int`).
    *   `setup_default_agents`: Imports signatures from `dspy_agents.py` instead of potentially defining them inline. Adds verification logging. Checks `dspy.settings.lm` before proceeding.
    *   `initialize_dspy`: Added more checks and logging around configuration success and LM setting.
```python name=src/local_file_research/dspy_config.py
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
# +++ Correct Type Hint Imports +++
from typing import List, Dict, Any, Optional, Union, Type, get_origin, get_args, Tuple, ForwardRef, _GenericAlias

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
                # Use dspy.OpenAI directly for clarity if using specific OpenAI features
                lm_instance = dspy.OpenAI(model=openai_model, api_key=api_key, **model_kwargs)
                # lm_instance = dspy.LM(model=openai_model, api_key=api_key, **model_kwargs) # Generic alternative
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
                # Use dspy.OllamaLocal
                lm_instance = dspy.OllamaLocal(model=model, **model_kwargs)
                # lm_instance = dspy.LM(model=model, **model_kwargs) # Generic alternative
                logger.info(f"Configuring DSPy with Ollama model: {model} at {ollama_base_url}")
            except Exception as e: logger.error(f"Failed configuring dspy.OllamaLocal: {e}", exc_info=True)

        elif provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key: logger.warning("ANTHROPIC_API_KEY missing.")
            try:
                # Use dspy.Anthropic
                lm_instance = dspy.Anthropic(model=model, api_key=api_key, **model_kwargs)
                # lm_instance = dspy.LM(model=model, api_key=api_key, **model_kwargs) # Generic alternative
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
    # +++ Cache for signature info +++
    _signature_cache: Dict[Type[dspy.Signature], Dict[str, Tuple[Type, bool, Any]]] = {}

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
    def _get_signature_input_fields(cls, signature: Type[dspy.Signature]) -> Dict[str, Tuple[Type, bool, Any]]:
        """
        Extracts input fields, their base types, requirement status, and default value.
        Uses caching for efficiency.

        Returns:
            Dict where key is field name, value is tuple (field_type, is_required, default_value)
        """
        # +++ Use Cache +++
        if signature in cls._signature_cache:
            return cls._signature_cache[signature]

        input_fields = {}
        if not signature or not isinstance(signature, type) or not issubclass(signature, dspy.Signature):
            logger.warning(f"Provided signature '{signature}' is not a valid dspy.Signature subclass. Cannot extract fields.")
            cls._signature_cache[signature] = input_fields # Cache empty result
            return input_fields

        try:
            # +++ Use inspect.getmembers for robustness +++
            # Also get annotations directly
            annotations = getattr(signature, '__annotations__', {})
            logger.debug(f"Inspecting signature: {signature.__name__}, Annotations: {annotations}") # Debug log
            for field_name, field_obj in inspect.getmembers(signature):
                if isinstance(field_obj, dspy.Field):
                    is_input = field_obj.json_schema_extra.get('__dspy_field_type') == 'input'
                    logger.debug(f"  Field: {field_name}, Is dspy.Field: True, Is Input: {is_input}") # Debug log

                    if is_input:
                        # Extract requirement status (default to True if not explicitly set)
                        is_required = getattr(field_obj, 'required', True)

                        # Extract default value
                        default_value = getattr(field_obj, 'default', dspy.pydantic_form.PydanticUndefined)
                        if default_value is dspy.pydantic_form.PydanticUndefined:
                             default_value = None # Use None if no explicit default

                        # Extract type hint from annotation if available
                        field_type_annotation = annotations.get(field_name)
                        actual_type = Any # Default to Any

                        # +++ Refined Type Extraction +++
                        if field_type_annotation:
                             origin = get_origin(field_type_annotation)
                             args = get_args(field_type_annotation)

                             if origin is Union and type(None) in args: # Optional[T] or Union[T, None]
                                 non_none_args = [arg for arg in args if arg is not type(None)]
                                 if len(non_none_args) == 1:
                                     base_type_inner = non_none_args[0]
                                     # Handle ForwardRefs if they occur
                                     if isinstance(base_type_inner, (str, ForwardRef)): base_type_inner = Any
                                     origin_inner = get_origin(base_type_inner)
                                     actual_type = origin_inner if origin_inner else base_type_inner
                                 elif len(non_none_args) > 1:
                                     # Filter out ForwardRefs from Union
                                     actual_args = tuple(a for a in non_none_args if not isinstance(a, (str, ForwardRef)))
                                     if len(actual_args) == 1: actual_type = actual_args[0]
                                     elif len(actual_args) > 1: actual_type = Union[actual_args]
                                     else: actual_type = Any # If only ForwardRefs were left
                             elif origin is list or origin is List or isinstance(origin, _GenericAlias) and origin.__origin__ is list: actual_type = list
                             elif origin is dict or origin is Dict or isinstance(origin, _GenericAlias) and origin.__origin__ is dict: actual_type = dict
                             elif origin: actual_type = origin # Other generic types like Tuple
                             # Handle ForwardRefs directly
                             elif isinstance(field_type_annotation, (str, ForwardRef)): actual_type = Any
                             elif isinstance(field_type_annotation, type): actual_type = field_type_annotation # Basic type
                             else: logger.debug(f"Complex annotation '{field_type_annotation}' for field '{field_name}'. Using Any.")
                        else:
                             logger.warning(f"No type annotation found for input field '{field_name}' in {signature.__name__}. Defaulting to type Any.")

                        # Store the extracted type, requirement status, and default value
                        input_fields[field_name] = (actual_type, is_required, default_value)
                        logger.debug(f"Extracted Input Field: '{field_name}' -> Type: {actual_type}, Required: {is_required}, Default: {default_value}")
                # Debug log for non-field attributes
                # else: logger.debug(f"  Field: {field_name}, Is dspy.Field: False")

        except Exception as e:
            logger.error(f"Failed to inspect signature {signature.__name__} for input fields: {e}", exc_info=True)

        if not input_fields:
             logger.warning(f"Could not extract any input fields for signature: {signature}")

        cls._signature_cache[signature] = input_fields # Cache the result
        return input_fields

    @classmethod
    def _ensure_required_fields(cls, agent_name: str, agent: dspy.Module, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures required fields are present and populates missing optional fields with defaults.
        Returns only the fields defined as input in the signature.
        """
        signature = getattr(agent, 'signature', None)
        if not signature:
            logger.debug(f"No signature found for agent '{agent_name}'. Passing data as is.")
            return data.copy()

        input_field_info = cls._get_signature_input_fields(signature)
        if not input_field_info:
            logger.debug(f"No input fields found in signature for agent '{agent_name}'. Passing data as is.")
            return data.copy()

        updated_data = data.copy() # Work on a copy
        final_data = {} # Store only the fields needed by the signature
        missing_required_populated = [] # Track populated required fields
        missing_optional_populated = [] # Track populated optional fields

        # --- Prepare potential fallback values ---
        query_val = updated_data.get("query", updated_data.get("question", ""))
        content_val = updated_data.get("content", "")
        document_val = updated_data.get("document", "")
        context_val = updated_data.get("context", "")
        summary_val = updated_data.get("summary", "")
        statement_val = updated_data.get("statement", "")
        prompt_val = updated_data.get("prompt", "")
        primary_text = content_val or document_val or context_val or summary_val or statement_val or query_val or prompt_val or ""
        fallback_context = context_val or summary_val or document_val or content_val or ""

        # --- Iterate and Populate ---
        for field, (field_type, is_required, default_value) in input_field_info.items():
            field_value = updated_data.get(field) # Get value from input data
            is_missing = field not in updated_data
            is_empty = not is_missing and (field_value is None or (isinstance(field_value, (str, list, dict)) and not field_value))

            if is_required and (is_missing or is_empty):
                 missing_required_populated.append(field)
                 fallback_value = None # Default fallback

                 # --- Specific Fallbacks for REQUIRED fields ---
                 # (Same logic as before)
                 if field in ("query", "question"): fallback_value = query_val or "No query provided."
                 elif field == "content": fallback_value = content_val or primary_text or "No content provided."
                 elif field == "document": fallback_value = document_val or primary_text or "No document provided."
                 elif field == "context": fallback_value = fallback_context or "No context provided."
                 elif field == "summary": fallback_value = summary_val or (primary_text[:200] + "..." if len(primary_text) > 200 else primary_text) or "No summary provided."
                 elif field == "statement": fallback_value = statement_val or "No statement provided."
                 elif field == "prompt": fallback_value = prompt_val or f"Process the following context: {fallback_context}" or "Please process the input."
                 elif field == "code": fallback_value = content_val or document_val or "No code provided."
                 elif field == "language": fallback_value = "python" # Common default
                 elif field == "data": fallback_value = content_val or document_val or "No data provided."
                 elif field == "document_type": fallback_value = "general"
                 elif field == "num_queries": fallback_value = 3
                 elif field == "iteration": fallback_value = 1
                 else:
                      # --- Type-based Fallbacks for REQUIRED fields ---
                      if field_type is list: fallback_value = []
                      elif field_type is int: fallback_value = 0
                      elif field_type is float: fallback_value = 0.0
                      elif field_type is dict: fallback_value = {}
                      elif field_type is bool: fallback_value = False
                      else: fallback_value = "" # Default to empty string

                 # Type Casting (Attempt to cast fallback to expected type)
                 try:
                      if field_type is int: fallback_value = int(fallback_value)
                      elif field_type is float: fallback_value = float(fallback_value)
                      elif field_type is bool: fallback_value = bool(fallback_value)
                      elif field_type is str and not isinstance(fallback_value, str): fallback_value = str(fallback_value)
                 except (ValueError, TypeError) as cast_err:
                      logger.warning(f"Could not cast fallback value '{fallback_value}' to type {field_type} for REQUIRED field '{field}'. Using original fallback. Error: {cast_err}")

                 final_data[field] = fallback_value # Add to final data
                 logger.debug(f"Populated REQUIRED field '{field}' with fallback: '{str(fallback_value)[:50]}...'")

            elif not is_required and is_missing:
                 # --- Handle OPTIONAL missing fields ---
                 if default_value is not None:
                      final_data[field] = default_value # Use signature default
                      missing_optional_populated.append(field)
                      logger.debug(f"Populated OPTIONAL field '{field}' with default: {default_value}")
                 else:
                      # Provide type-based default if no explicit default
                      fallback_optional = None
                      if field_type is list: fallback_optional = []
                      elif field_type is dict: fallback_optional = {}
                      elif field_type is str: fallback_optional = ""
                      elif field_type is int: fallback_optional = 0
                      elif field_type is float: fallback_optional = 0.0
                      elif field_type is bool: fallback_optional = False
                      else: fallback_optional = None

                      final_data[field] = fallback_optional # Add to final data
                      missing_optional_populated.append(field)
                      logger.debug(f"Populated OPTIONAL field '{field}' with type-based default: {fallback_optional} (as no explicit default was found)")
            else:
                 # Field is present, add it to final data
                 final_data[field] = field_value

        # Log which fields were populated
        if missing_required_populated:
            logger.warning(f"Agent '{agent_name}': Populated missing/empty REQUIRED fields: {missing_required_populated}.")
        if missing_optional_populated:
             logger.info(f"Agent '{agent_name}': Populated missing OPTIONAL fields with defaults: {missing_optional_populated}.")

        # Final check log (as before)
        present_fields = set(final_data.keys())
        expected_input_names = set(input_field_info.keys())
        logger.debug(f"Final prepared data for Agent '{agent_name}': {list(present_fields)}")
        if present_fields != expected_input_names:
             logger.warning(f"Mismatch between prepared fields ({present_fields}) and expected signature fields ({expected_input_names}) for agent '{agent_name}'. This might still cause DSPy warnings.")

        return final_data # Return only the fields expected by the signature


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
            try:
                 # +++ Use the refined _ensure_required_fields +++
                 prepared_data = cls._ensure_required_fields(agent_name, agent, current_data)
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
                with dspy.context(lm=dspy.settings.lm):
                     # +++ Pass only the prepared data +++
                     result = agent(**prepared_data)
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
            # (Make sure these are defined correctly, preferably in dspy_agents.py)
            # --- Assuming signatures are defined elsewhere (e.g., dspy_agents.py) ---
            # --- We just need to import and register them ---
            try:
                # +++ Import ALL signatures +++
                from .dspy_agents import (
                    SummarizerSignature, AnswererSignature, ExtractorSignature,
                    ChainOfThoughtSignature, FactCheckerSignature, DocumentAnalysisSignature,
                    InterpreterSignature, ProposalGeneratorSignature, TechnicalAnalyzerSignature,
                    QueryRefinementSignature, FactVerificationSignature, TextGeneratorSignature,
                    QueryExpansionSignature, ContentSynthesizerSignature,
                    CodeAnalysisSignature, SpreadsheetAnalysisSignature, PDFAnalysisSignature,
                    TechnicalDocAnalysisSignature, ResearchPaperAnalysisSignature,
                    MultiDocumentSynthesisSignature
                )

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

                # Register agents from dspy_agents.py
                cls.register_agent("code_analyzer", dspy.Predict(CodeAnalysisSignature))
                cls.register_agent("spreadsheet_analyzer", dspy.Predict(SpreadsheetAnalysisSignature))
                cls.register_agent("pdf_analyzer", dspy.Predict(PDFAnalysisSignature))
                cls.register_agent("tech_doc_analyzer", dspy.Predict(TechnicalDocAnalysisSignature))
                cls.register_agent("research_paper_analyzer", dspy.Predict(ResearchPaperAnalysisSignature))
                # Reuse the existing ChainOfThought agent for cot_analyzer
                # No need to register cot_analyzer separately if it's the same signature
                cls.register_agent("multi_doc_synthesizer", dspy.Predict(MultiDocumentSynthesisSignature))

            except ImportError as ie:
                 logger.error(f"Failed to import one or more signatures from dspy_agents.py: {ie}")
                 # Decide how to handle this - exit, or proceed with fewer agents?
                 # For now, log the error and potentially continue with what was registered.
                 # return # Or raise an error

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
            # Chains from dspy_agents.py
            cls.register_chain("code_analysis_chain", ["code_analyzer"])
            cls.register_chain("spreadsheet_analysis_chain", ["spreadsheet_analyzer"])
            cls.register_chain("pdf_analysis_chain", ["pdf_analyzer"])
            cls.register_chain("tech_doc_analysis_chain", ["tech_doc_analyzer"])
            cls.register_chain("research_paper_analysis_chain", ["research_paper_analyzer"])
            # No need for cot_analysis_chain if using chain_of_thought
            cls.register_chain("multi_doc_synthesis_chain", ["multi_doc_synthesizer"])
            cls.register_chain("code_review_chain", ["code_analyzer", "chain_of_thought"]) # Use chain_of_thought
            cls.register_chain("data_analysis_chain", ["spreadsheet_analyzer", "chain_of_thought"]) # Use chain_of_thought
            cls.register_chain("research_review_chain", ["research_paper_analyzer", "chain_of_thought"]) # Use chain_of_thought


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
```

































**`dspy_agents.py`:**
    *   Moved *all* signature definitions here for better organization.
    *   Ensured type hints are used correctly (e.g., `query: str = dspy.InputField(...)`).
    *   Added `instructions` to signatures where appropriate.
    *   Specified expected output formats for list-like fields in descriptions (e.g., "one per line", "comma-separated") to guide the LLM.
    *   Made fields like `document`, `content`, `context` explicitly optional (`required=False`, `default=""`) in most signatures where they are likely supplementary.

```python name=src/local_file_research/dspy_agents.py
# filename: src/local_file_research/dspy_agents.py
"""
Custom DSPy agents for Local File Deep Research.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
import json

# Configure logging
logger = logging.getLogger(__name__)

# Initialize DSPy
try:
    import dspy
    from .dspy_config import DSPyAgentRegistry # Import registry to potentially use later if needed
    DSPY_AVAILABLE = True
except ImportError:
    logger.warning("DSPy not available. Custom agents will not be functional.")
    DSPY_AVAILABLE = False
    # DSPY_CONFIGURED needs to be checked where agents are used, typically in dspy_config.py

class DocumentTypeError(Exception):
    """Exception raised when a document type is not supported."""
    pass

# --- Define ALL Signatures Here ---

if DSPY_AVAILABLE:
    # --- Basic Task Signatures ---
    class SummarizerSignature(dspy.Signature):
        """Summarize the provided content."""
        instructions = 'Summarize the provided content.'
        content: str = dspy.InputField(desc="The text content to summarize")
        query: str = dspy.InputField(desc="Optional query to focus the summary", default="", required=False)
        document: str = dspy.InputField(desc="Alternative document content (optional)", default="", required=False)
        context: str = dspy.InputField(desc="Alternative context (optional)", default="", required=False)
        summary: str = dspy.OutputField(desc="A concise summary")

    class AnswererSignature(dspy.Signature):
        """Answer a question based on the provided context."""
        instructions = 'Answer the question based on the provided context.'
        context: str = dspy.InputField(desc="Context relevant to the question")
        query: str = dspy.InputField(desc="The question to answer")
        document: str = dspy.InputField(desc="Alternative document content (optional)", default="", required=False)
        content: str = dspy.InputField(desc="Alternative content source (optional)", default="", required=False)
        answer: str = dspy.OutputField(desc="The answer derived from the context")

    class ExtractorSignature(dspy.Signature):
        """Extract specific information (e.g., entities, keywords) from content."""
        instructions = 'Extract specific information based on the query from the content.'
        content: str = dspy.InputField(desc="The text content to extract from")
        query: str = dspy.InputField(desc="Description of the information to extract (e.g., 'names of people')")
        context: str = dspy.InputField(desc="Context relevant to the question (optional)", default="", required=False)
        document: str = dspy.InputField(desc="Alternative document content (optional)", default="", required=False)
        info: str = dspy.OutputField(desc="The extracted information")

    class ChainOfThoughtSignature(dspy.Signature):
        """Generate step-by-step reasoning to answer a query based on content."""
        instructions = 'Generate step-by-step reasoning to answer a query based on content.'
        content: str = dspy.InputField(desc="Content to analyze")
        query: str = dspy.InputField(desc="The query or task")
        context: str = dspy.InputField(desc="Context relevant to the question (optional)", default="", required=False)
        document: str = dspy.InputField(desc="Alternative document content (optional)", default="", required=False)
        cot: str = dspy.OutputField(desc="Step-by-step reasoning process")
        conclusion: str = dspy.OutputField(desc="Final conclusion based on reasoning")

    class FactCheckerSignature(dspy.Signature):
        """Assess the factual consistency of a statement against provided context."""
        instructions = 'Assess the factual consistency of the statement against the provided context.'
        statement: str = dspy.InputField(desc="The statement to fact-check")
        context: str = dspy.InputField(desc="The context to check against")
        query: str = dspy.InputField(desc="Optional focus for fact-checking", default="Check consistency", required=False)
        content: str = dspy.InputField(desc="Content relevant to the question (optional)", default="", required=False)
        document: str = dspy.InputField(desc="Alternative document content (optional)", default="", required=False)
        fact_check: str = dspy.OutputField(desc="Assessment of factual consistency (e.g., 'Consistent', 'Inconsistent', 'Needs More Info') with explanation")

    class TextGeneratorSignature(dspy.Signature):
        """Generic text generation based on a prompt."""
        instructions = 'Generic text generation based on a prompt.'
        prompt: str = dspy.InputField(desc="The input prompt")
        context: str = dspy.InputField(desc="Optional context", default="", required=False)
        content: str = dspy.InputField(desc="Content relevant to the question (optional)", default="", required=False)
        document: str = dspy.InputField(desc="Alternative document content (optional)", default="", required=False)
        query: str = dspy.InputField(desc="Optional query focus", default="", required=False)
        text: str = dspy.OutputField(desc="Generated text")

    # --- Research & Reporting Signatures ---
    class QueryExpansionSignature(dspy.Signature):
        """Expand a query with related terms/concepts."""
        instructions = 'Expand a query with related terms/concepts.'
        query: str = dspy.InputField(desc="Query to expand")
        context: str = dspy.InputField(desc="Optional context", default="", required=False)
        content: str = dspy.InputField(desc="Content relevant to the question (optional)", default="", required=False)
        document: str = dspy.InputField(desc="Alternative document content (optional)", default="", required=False)
        expanded_query: str = dspy.OutputField(desc="Expanded query string")

    class QueryRefinementSignature(dspy.Signature):
        """Generate refined/follow-up queries."""
        instructions = 'Generate refined/follow-up queries.'
        query: str = dspy.InputField(desc="Original query")
        context: str = dspy.InputField(desc="Current research context/knowledge")
        content: str = dspy.InputField(desc="Content relevant to the question (optional)", default="", required=False)
        document: str = dspy.InputField(desc="Alternative document content (optional)", default="", required=False)
        num_queries: int = dspy.InputField(desc="Number of queries to generate")
        iteration: int = dspy.InputField(desc="Current research iteration")
        related_queries: str = dspy.OutputField(desc="List of refined/follow-up queries, one per line") # Specify format

    class FactVerificationSignature(dspy.Signature):
        """Verify content consistency against a summary."""
        instructions = 'Verify content consistency against a summary.'
        content: str = dspy.InputField(desc="Content snippet to verify")
        summary: str = dspy.InputField(desc="Overall summary to verify against")
        context: str = dspy.InputField(desc="Context relevant to the question (optional)", default="", required=False)
        document: str = dspy.InputField(desc="Alternative document content (optional)", default="", required=False)
        query: str = dspy.InputField(desc="Optional focus for verification", default="Verify facts", required=False)
        is_consistent: str = dspy.OutputField(desc="Boolean indicating consistency (e.g., 'Yes', 'No', 'Uncertain')") # Suggest specific outputs
        confidence: str = dspy.OutputField(desc="Confidence score (0.0-1.0)")
        notes: str = dspy.OutputField(desc="Explanation for consistency assessment")

    class DocumentAnalysisSignature(dspy.Signature):
        """Provide a structured analysis of a document."""
        instructions = 'Provide a structured analysis of a document.'
        # Make document optional, primary input should be 'content' if that's what's passed
        document: str = dspy.InputField(desc="The document text (optional)", default="", required=False)
        content: str = dspy.InputField(desc="Content relevant to the question (primary input)") # Make content the primary input
        query: str = dspy.InputField(desc="Optional query to guide analysis", default="Analyze this document", required=False)
        context: str = dspy.InputField(desc="Optional additional context", default="", required=False)
        summary: str = dspy.OutputField(desc="Overall summary")
        key_points: str = dspy.OutputField(desc="List of key points, one per line") # Specify format
        entities: str = dspy.OutputField(desc="List of important entities, comma-separated") # Specify format
        sentiment: str = dspy.OutputField(desc="Overall sentiment (e.g., Positive, Negative, Neutral)")

    class InterpreterSignature(dspy.Signature):
        """Interpret research findings contextually."""
        instructions = 'Interpret research findings contextually.'
        query: str = dspy.InputField(desc="The original research query")
        context: str = dspy.InputField(desc="Aggregated research findings")
        content: str = dspy.InputField(desc="Content relevant to the question (optional)", default="", required=False)
        document: str = dspy.InputField(desc="Alternative document content (optional)", default="", required=False)
        interpretation: str = dspy.OutputField(desc="Main interpretation")
        insights: str = dspy.OutputField(desc="List of key insights, one per line") # Specify format
        limitations: str = dspy.OutputField(desc="List of limitations, one per line") # Specify format
        confidence: str = dspy.OutputField(desc="Confidence score (0.0-1.0)")

    class ProposalGeneratorSignature(dspy.Signature):
        """Generate actionable proposals from findings."""
        instructions = 'Generate actionable proposals from findings.'
        query: str = dspy.InputField(desc="Original research query")
        context: str = dspy.InputField(desc="Research findings context")
        content: str = dspy.InputField(desc="Content relevant to the question (optional)", default="", required=False)
        document: str = dspy.InputField(desc="Alternative document content (optional)", default="", required=False)
        recommendations: str = dspy.OutputField(desc="List of specific recommendations, one per line") # Specify format
        next_steps: str = dspy.OutputField(desc="List of concrete next steps, one per line") # Specify format
        alternatives: str = dspy.OutputField(desc="List of alternative approaches, one per line") # Specify format
        rationale: str = dspy.OutputField(desc="Justification for proposals")

    class TechnicalAnalyzerSignature(dspy.Signature):
        """Provide technical analysis of findings."""
        instructions = 'Provide technical analysis of findings.'
        query: str = dspy.InputField(desc="Original research query")
        context: str = dspy.InputField(desc="Research findings context")
        content: str = dspy.InputField(desc="Content relevant to the question (optional)", default="", required=False)
        document: str = dspy.InputField(desc="Alternative document content (optional)", default="", required=False)
        analysis: str = dspy.OutputField(desc="Technical analysis summary")
        details: str = dspy.OutputField(desc="List of technical details, one per line") # Specify format
        challenges: str = dspy.OutputField(desc="List of technical challenges, one per line") # Specify format
        solutions: str = dspy.OutputField(desc="List of potential solutions, one per line") # Specify format

    class ContentSynthesizerSignature(dspy.Signature):
        """Synthesize findings into a structured article."""
        instructions = 'Synthesize findings into a structured article.'
        query: str = dspy.InputField(desc="Original research query")
        context: str = dspy.InputField(desc="Research findings context")
        content: str = dspy.InputField(desc="Content relevant to the question (optional)", default="", required=False)
        document: str = dspy.InputField(desc="Alternative document content (optional)", default="", required=False)
        article: str = dspy.OutputField(desc="Synthesized article content")
        article_type: str = dspy.OutputField(desc="Type of article (e.g., summary, analysis, blog post)")
        key_themes: str = dspy.OutputField(desc="List of key themes discussed, one per line") # Specify format
        word_count: str = dspy.OutputField(desc="Approximate word count")

    # --- Document Type Specific Signatures (from original dspy_agents.py) ---
    class CodeAnalysisSignature(dspy.Signature):
        """Analyze code documents and extract key information."""
        instructions = 'Analyze code documents and extract key information.'
        code: str = dspy.InputField(desc="The code content to analyze")
        language: str = dspy.InputField(desc="The programming language of the code")
        query: str = dspy.InputField(desc="The query or task to focus the analysis on")
        document: str = dspy.InputField(desc="Alternative document content (optional)", default="", required=False)
        content: str = dspy.InputField(desc="Alternative content source (optional)", default="", required=False)
        context: str = dspy.InputField(desc="Additional context for analysis (optional)", default="", required=False)
        summary: str = dspy.OutputField(desc="A concise summary of what the code does")
        functions: str = dspy.OutputField(desc="List of key functions/methods in the code, comma-separated") # Specify format
        classes: str = dspy.OutputField(desc="List of key classes in the code, comma-separated") # Specify format
        dependencies: str = dspy.OutputField(desc="List of external dependencies or imports, comma-separated") # Specify format
        complexity: str = dspy.OutputField(desc="Assessment of code complexity")
        issues: str = dspy.OutputField(desc="Potential issues or bugs in the code, one per line") # Specify format
        suggestions: str = dspy.OutputField(desc="Suggestions for improvement, one per line") # Specify format

    class SpreadsheetAnalysisSignature(dspy.Signature):
        """Analyze spreadsheet data and extract key information."""
        instructions = 'Analyze spreadsheet data and extract key information.'
        data: str = dspy.InputField(desc="The spreadsheet content in text format")
        query: str = dspy.InputField(desc="The query or task to focus the analysis on")
        document: str = dspy.InputField(desc="Alternative document content (optional)", default="", required=False)
        content: str = dspy.InputField(desc="Alternative content source (optional)", default="", required=False)
        context: str = dspy.InputField(desc="Additional context for analysis (optional)", default="", required=False)
        summary: str = dspy.OutputField(desc="A concise summary of the data")
        structure: str = dspy.OutputField(desc="Description of the data structure (columns, sheets)")
        key_metrics: str = dspy.OutputField(desc="Key metrics or statistics from the data, one per line") # Specify format
        patterns: str = dspy.OutputField(desc="Patterns or trends identified in the data")
        anomalies: str = dspy.OutputField(desc="Anomalies or outliers in the data")
        insights: str = dspy.OutputField(desc="Key insights derived from the data, one per line") # Specify format

    class PDFAnalysisSignature(dspy.Signature):
        """Analyze PDF documents and extract key information."""
        instructions = 'Analyze PDF documents and extract key information.'
        content: str = dspy.InputField(desc="The PDF content in text format")
        query: str = dspy.InputField(desc="The query or task to focus the analysis on")
        document: str = dspy.InputField(desc="Alternative document content (optional)", default="", required=False)
        context: str = dspy.InputField(desc="Additional context for analysis (optional)", default="", required=False)
        summary: str = dspy.OutputField(desc="A concise summary of the document")
        key_points: str = dspy.OutputField(desc="Key points from the document, one per line") # Specify format
        entities: str = dspy.OutputField(desc="Important entities mentioned in the document, comma-separated") # Specify format
        topics: str = dspy.OutputField(desc="Main topics covered in the document, comma-separated") # Specify format
        structure: str = dspy.OutputField(desc="Document structure (sections, headings)")
        citations: str = dspy.OutputField(desc="Citations or references in the document, one per line") # Specify format

    class TechnicalDocAnalysisSignature(dspy.Signature):
        """Analyze technical documents and extract key information."""
        instructions = 'Analyze technical documents and extract key information.'
        content: str = dspy.InputField(desc="The technical document content")
        document_type: str = dspy.InputField(desc="The type of technical document (e.g., API doc, whitepaper)")
        query: str = dspy.InputField(desc="The query or task to focus the analysis on")
        document: str = dspy.InputField(desc="Alternative document content (optional)", default="", required=False)
        context: str = dspy.InputField(desc="Additional context for analysis (optional)", default="", required=False)
        summary: str = dspy.OutputField(desc="A concise summary of the technical document")
        key_concepts: str = dspy.OutputField(desc="Key technical concepts explained in the document, one per line") # Specify format
        technical_details: str = dspy.OutputField(desc="Important technical details or specifications, one per line") # Specify format
        requirements: str = dspy.OutputField(desc="Requirements or prerequisites mentioned, one per line") # Specify format
        examples: str = dspy.OutputField(desc="Code examples or usage examples")
        limitations: str = dspy.OutputField(desc="Limitations or constraints mentioned, one per line") # Specify format

    class ResearchPaperAnalysisSignature(dspy.Signature):
        """Analyze research papers and extract key information."""
        instructions = 'Analyze research papers and extract key information.'
        content: str = dspy.InputField(desc="The research paper content")
        query: str = dspy.InputField(desc="The query or task to focus the analysis on")
        document: str = dspy.InputField(desc="Alternative document content (optional)", default="", required=False)
        context: str = dspy.InputField(desc="Additional context for analysis (optional)", default="", required=False)
        summary: str = dspy.OutputField(desc="A concise summary of the research paper")
        research_question: str = dspy.OutputField(desc="The main research question or objective")
        methodology: str = dspy.OutputField(desc="The methodology used in the research")
        findings: str = dspy.OutputField(desc="Key findings or results")
        limitations: str = dspy.OutputField(desc="Limitations of the research")
        implications: str = dspy.OutputField(desc="Implications or applications of the research")
        future_work: str = dspy.OutputField(desc="Suggested future work")

    class ChainOfThoughtAnalysisSignature(dspy.Signature): # Renamed for clarity
        """Perform step-by-step reasoning on a document."""
        instructions = 'Perform step-by-step reasoning on a document.'
        content: str = dspy.InputField(desc="The document content to analyze")
        query: str = dspy.InputField(desc="The query or task to focus the analysis on")
        document: str = dspy.InputField(desc="Alternative document content (optional)", default="", required=False)
        context: str = dspy.InputField(desc="Additional context for analysis (optional)", default="", required=False)
        reasoning: str = dspy.OutputField(desc="Step-by-step reasoning process")
        key_insights: str = dspy.OutputField(desc="Key insights derived from the reasoning, one per line") # Specify format
        conclusion: str = dspy.OutputField(desc="Conclusion based on the reasoning")

    class MultiDocumentSynthesisSignature(dspy.Signature):
        """Synthesize information from multiple documents."""
        instructions = 'Synthesize information from multiple documents based on the query.'
        documents: str = dspy.InputField(desc="JSON string representing a list of document contents and metadata (e.g., [{'title': 'Doc1', 'content': '...'}, ...])") # Changed to string input
        query: str = dspy.InputField(desc="The query or task to focus the synthesis on")
        document: str = dspy.InputField(desc="Single document content (optional)", default="", required=False)
        content: str = dspy.InputField(desc="Alternative content source (optional)", default="", required=False)
        synthesis: str = dspy.OutputField(desc="Synthesized information from all documents")
        common_themes: str = dspy.OutputField(desc="Common themes across documents, comma-separated") # Specify format
        contradictions: str = dspy.OutputField(desc="Contradictions or disagreements between documents, one per line") # Specify format
        unique_insights: str = dspy.OutputField(desc="Unique insights from specific documents")
        integrated_view: str = dspy.OutputField(desc="Integrated view of the information")


# --- Agent Implementations (Example - No need to define Predict here) ---

# Registration logic is handled in dspy_config.py

# --- Helper Functions (Can remain here or move to utils) ---
def analyze_document(content: str, document_type: str, query: str = None) -> Dict[str, Any]:
    """
    Analyze a document using the appropriate DSPy agent based on document type.
    (This function likely belongs in document_analysis.py or similar)
    """
    # Logic to select and call the appropriate agent from the registry
    # based on document_type. This requires DSPy to be configured.
    from .dspy_config import DSPY_CONFIGURED, DSPyAgentRegistry
    if not DSPY_CONFIGURED:
        logger.warning("DSPy not available or configured. Cannot perform agent-based analysis.")
        # Fallback to basic analysis or return error
        from .document_analysis import DocumentAnalyzer # Avoid circular import if possible
        analyzer = DocumentAnalyzer()
        return analyzer._basic_analysis(content, query) # Use fallback

    agent_name = f"{document_type}_analyzer" # Simple mapping
    agent = DSPyAgentRegistry.get_agent(agent_name)

    if not agent:
        logger.warning(f"No specific agent found for document type '{document_type}'. Using default analyzer.")
        agent = DSPyAgentRegistry.get_agent("document_analyzer")
        if not agent:
            logger.error("Default document analyzer agent not found.")
            return {"error": "Default document analyzer not available."}

    try:
        # Prepare data, ensuring all required fields for the *specific* agent are present
        # This is tricky without knowing the agent beforehand, rely on _ensure_required_fields
        input_data = {"content": content[:15000], "query": query or f"Analyze this {document_type} document"}
        # Add specific fields if known for the type
        if document_type == "code": input_data["code"] = content[:15000]; input_data["language"] = _detect_language(content)
        if document_type == "spreadsheet": input_data["data"] = content[:15000]
        # ... add others

        prepared_data = DSPyAgentRegistry._ensure_required_fields(agent_name, agent, input_data)
        result = agent(**prepared_data)

        # Convert result to dictionary
        if hasattr(result, "__dict__"):
            # Use keys() method if available (for dspy.Prediction)
            if hasattr(result, 'keys'):
                 return {k: getattr(result, k) for k in result.keys() if not k.startswith('_')}
            else:
                 return {k: v for k, v in result.__dict__.items() if not k.startswith("_")}
        elif isinstance(result, dict):
            return result
        else:
            return {"analysis_output": str(result)}
    except Exception as e:
        logger.error(f"Error analyzing document type '{document_type}' with agent '{agent_name}': {e}", exc_info=True)
        return {"error": f"Analysis failed: {e}"}


def synthesize_documents(documents: List[Dict[str, Any]], query: str = None) -> Dict[str, Any]:
    """
    Synthesize information from multiple documents.
    (This function likely belongs in multi_document_synthesis.py or similar)
    """
    from .dspy_config import DSPY_CONFIGURED, DSPyAgentRegistry
    if not DSPY_CONFIGURED:
        return {"error": "DSPy not available or configured."}

    agent = DSPyAgentRegistry.get_agent("multi_doc_synthesizer")
    if not agent:
        return {"error": "Multi-document synthesis agent not available"}

    # Format documents for the agent (limit content size)
    formatted_docs = []
    for doc in documents:
        formatted_docs.append({
            "title": doc.get("title", "Untitled"),
            "content": doc.get("content", "")[:5000] # Limit content size per doc
        })

    # Convert to JSON string for the agent
    docs_json = json.dumps(formatted_docs)

    try:
        input_data = {"documents": docs_json, "query": query or "Synthesize these documents"}
        prepared_data = DSPyAgentRegistry._ensure_required_fields("multi_doc_synthesizer", agent, input_data)
        result = agent(**prepared_data)

        if hasattr(result, "__dict__"):
             # Use keys() method if available (for dspy.Prediction)
             if hasattr(result, 'keys'):
                  return {k: getattr(result, k) for k in result.keys() if not k.startswith('_')}
             else:
                  return {k: v for k, v in result.__dict__.items() if not k.startswith("_")}
        elif isinstance(result, dict):
            return result
        else:
            return {"synthesis_output": str(result)}
    except Exception as e:
        logger.error(f"Error synthesizing documents: {e}", exc_info=True)
        return {"error": f"Synthesis failed: {e}"}


def _detect_language(content: str) -> str:
    """
    Detect the programming language from code content.
    (This helper function can stay here or move to utils)
    """
    # Simple language detection based on keywords and syntax
    content_lower = content.lower()

    if "def " in content and "import " in content and ("self" in content or ":" in content):
        return "python"
    elif "function " in content and ("{" in content and "}" in content):
        return "javascript"
    elif "public class " in content or "private class " in content:
        return "java"
    elif "#include" in content and ("int main" in content or "void main" in content):
        return "c/c++"
    elif "using namespace" in content or "std::" in content:
        return "c++"
    elif "<?php" in content:
        return "php"
    elif "<html" in content or "<!doctype html" in content_lower:
        return "html"
    elif "@interface" in content or "@implementation" in content:
        return "objective-c"
    elif "func " in content and ("let " in content or "var " in content):
        return "swift"
    elif "package " in content and "import " in content and "{" in content:
        return "go"
    else:
        return "unknown"
```

























**`multi_iteration_research.py`:**
    *   Added checks for `DSPY_CONFIGURED` before calling agents.
    *   Ensured `num_queries` and `iteration` are cast to `int` when passed to the agent call, matching the corrected signature.
    *   Improved parsing of the `related_queries` output string from the agents.
    *   Improved fallback logic for question generation.
    *   Corrected parsing of verification results.





```python name=src/local_file_research/multi_iteration_research.py
"""
Multi-iteration research for Local File Deep Research.

This module provides functionality for conducting multi-iteration research
with query expansion and follow-up questions based on collected context.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Union
from collections import Counter
import re # Import re for robust question parsing
import traceback # Import traceback

from .vector_store import VectorStore
from .document_analysis import DocumentAnalyzer
from .advanced_search import QueryExpander
from .dspy_config import DSPyAgentRegistry, DSPY_CONFIGURED # Import DSPY_CONFIGURED

# Configure logging
logger = logging.getLogger(__name__)

# Import safe_str from logging_config if available
try:
    from .logging_config import safe_str
except ImportError:
    # Fallback implementation if logging_config is not available
    def safe_str(obj):
        """Convert any object to a string that can be safely printed/logged."""
        if obj is None:
            return 'None'
        try:
            # First try normal string conversion
            return str(obj)
        except UnicodeEncodeError:
            # If that fails, try with explicit encoding/decoding
            try:
                if isinstance(obj, str):
                    # For strings, encode with replacement
                    return obj.encode('utf-8', errors='backslashreplace').decode('utf-8')
                else:
                    # For other objects, convert to string with replacement
                    return str(obj).encode('utf-8', errors='backslashreplace').decode('utf-8')
            except Exception:
                # Last resort - return a placeholder
                return "[Unicode conversion error]"

class MultiIterationResearch:
    """
    Multi-iteration research system that uses query expansion and follow-up questions.
    """

    def __init__(self, vector_store: VectorStore, max_iterations: int = 3, questions_per_iteration: int = 2, max_context_size: int = 10000):
        """
        Initialize the multi-iteration research system.

        Args:
            vector_store: Vector store for searching documents
            max_iterations: Maximum number of research iterations
            questions_per_iteration: Number of follow-up questions per iteration
            max_context_size: Maximum size of the context in characters (default: 10000)
        """
        self.vector_store = vector_store
        self.max_iterations = max_iterations
        self.questions_per_iteration = questions_per_iteration
        self.max_context_size = max_context_size
        self.document_analyzer = DocumentAnalyzer()
        self.query_expander = QueryExpander()
        self.expanded_query = "" # Initialize expanded_query


        # Performance tracking
        self.performance_metrics = {
            "search_time": 0,
            "analysis_time": 0,
            "question_generation_time": 0,
            "total_time": 0
        }

        # State tracking
        self.questions_by_iteration = {}
        self.all_findings = []
        self.current_knowledge = ""

        # Enhanced tracking for multi-iteration research
        self.findings_by_question = {}  # Dictionary to track findings by question
        self.context_by_question = {}   # Dictionary to track context by question

        # Tracking for deduplication
        self.seen_chunks = set()  # Set of chunk identifiers we've already processed

    def run(self, query: str, top_k: int = 5, context_filter: str = None) -> Dict[str, Any]:
        """
        Run multi-iteration research.

        Args:
            query: Original research query
            top_k: Number of results to retrieve per question
            context_filter: Optional filter for results

        Returns:
            Dictionary with research findings and metadata
        """
        start_time = time.time()
        logger.info(f"Starting multi-iteration research for query: '{safe_str(query)}'")

        # Reset state
        self.questions_by_iteration = {}
        self.all_findings = []
        self.current_knowledge = ""
        self.seen_chunks = set()
        self.findings_by_question = {}
        self.context_by_question = {}
        self.performance_metrics = {"search_time": 0, "analysis_time": 0, "question_generation_time": 0} # Reset timings

        # Initial query expansion
        self.expanded_query = self._expand_query(query) # Store expanded query
        logger.info(f"Expanded query: '{safe_str(self.expanded_query)}'")

        # First iteration: search with original and expanded query
        self._perform_initial_search(query, top_k, context_filter)

        # Track all questions asked so far to avoid duplicates
        all_questions = set()
        for q in self.questions_by_iteration.get('0', []):
            all_questions.add(q.lower())

        # Subsequent iterations with follow-up questions
        for iteration in range(1, self.max_iterations):
            iteration_num = iteration + 1
            logger.info(f"Starting research iteration {iteration_num}/{self.max_iterations}")

            # Generate follow-up questions, ensuring they're unique
            follow_up_questions = self._generate_follow_up_questions(query, iteration=iteration_num) # Pass correct iteration number

            # Filter out questions we've already asked
            unique_questions = []
            for q in follow_up_questions:
                if q.lower() not in all_questions:
                    unique_questions.append(q)
                    all_questions.add(q.lower())

            # If we don't have any unique questions, generate some generic ones
            if not unique_questions and iteration_num < self.max_iterations: # Only add generic if not last iteration
                logger.info("No unique questions generated, adding generic follow-ups")
                generic_questions = [
                    f"What are additional aspects of {query} not covered so far?",
                    f"What are alternative perspectives on {query}?"
                ]
                for q in generic_questions:
                    if q.lower() not in all_questions:
                        unique_questions.append(q)
                        all_questions.add(q.lower())
                        if len(unique_questions) >= self.questions_per_iteration: break # Stop if we have enough

            # If still no new questions, stop iterating
            if not unique_questions:
                 logger.info(f"No new unique questions generated for iteration {iteration_num}. Stopping iterations.")
                 break

            # Store the unique questions for this iteration
            self.questions_by_iteration[str(iteration)] = unique_questions
            logger.info(f"Iteration {iteration_num}: Using {len(unique_questions)} unique follow-up questions: {unique_questions}")

            # Search for each follow-up question
            iteration_findings = []
            for question in unique_questions:
                results = self._search_and_analyze(
                    question,
                    top_k=top_k,
                    context_filter=context_filter
                )
                iteration_findings.extend(results)

            # Update findings and knowledge only if new findings were made
            if iteration_findings:
                self.all_findings.extend(iteration_findings)
                self._update_current_knowledge()
            else:
                logger.info(f"No new findings in iteration {iteration_num}.")

            # Check if we have enough information (optional break condition)
            # if self._check_sufficient_information():
            #     logger.info(f"Sufficient information gathered after iteration {iteration_num}")
            #     break

        # Final verification and fact-checking
        self._verify_findings()

        # Limit the final context if it's too large
        max_final_context_size = 64000  # Maximum size for the final context
        if sum(len(f.get("content", "")) for f in self.all_findings) > max_final_context_size:
            logger.info(f"Limiting final context to approximately {max_final_context_size} characters")
            self._limit_final_context(max_final_context_size)

        # Calculate total time
        total_time = time.time() - start_time
        self.performance_metrics["total_time"] = total_time

        logger.info(f"Multi-iteration research complete in {total_time:.2f} seconds")

        # Pool all context from all questions for enhanced reporting
        all_pooled_context = self._pool_all_context()
        # Calculate total time again? (Already calculated above)
        # total_time = time.time() - start_time
        # self.performance_metrics["total_time"] = total_time
        logger.info(f"Multi-iteration research phase complete in {self.performance_metrics['total_time']:.2f} seconds")

        # --- Return data for ReportGenerator, NOT the report itself ---
        return {
            "findings": self.all_findings, # Raw findings from all iterations
            "pooled_context": all_pooled_context, # Pooled/deduplicated context and findings
            "questions_by_iteration": self.questions_by_iteration,
            "iterations": len(self.questions_by_iteration),
            "performance_metrics": self.performance_metrics,
            # Removed report generation from here
        }

    def _search_and_analyze(self, query: str, top_k: int, context_filter: str = None) -> List[Dict[str, Any]]:
        """
        Search for documents and analyze them.

        Args:
            query: Search query
            top_k: Number of results to retrieve
            context_filter: Optional filter for results

        Returns:
            List of analyzed search results
        """
        # Search
        search_start_time = time.time()

        # Get query embedding
        from .pipeline import embed_text # Import here to avoid circular deps?
        query_embedding = embed_text(query)

        # Log with safe_str to handle Unicode characters
        logger.info(f"Searching for query: {safe_str(query)}")

        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)

        # Apply context filter if provided
        if context_filter:
            logger.info(f"Applying context filter: {safe_str(context_filter)}")
            results = [r for r in results if context_filter.lower() in safe_str(r.get("file_path", "")).lower()]

        search_time = time.time() - search_start_time
        self.performance_metrics["search_time"] += search_time

        # Analyze results
        analysis_start_time = time.time()
        analyzed_results = []

        for result in results:
            try:
                # Get source information for chunk-level deduplication
                source_path = result.get("file_path", "")
                source_name = result.get("source_name", "")
                # +++ Use metadata fields if available +++
                start_pos = result.get("start", result.get("metadata", {}).get("start", 0))
                end_pos = result.get("end", result.get("metadata", {}).get("end", 0))
                chunk_index = result.get("chunk_index", result.get("metadata", {}).get("chunk_index", -1))

                # Create a unique chunk identifier using index if available, otherwise positions
                if chunk_index != -1:
                     chunk_id = f"{source_path}|{source_name}|chunk_{chunk_index}"
                else:
                     chunk_id = f"{source_path}|{source_name}|{start_pos}-{end_pos}"


                # Skip if we've already seen this exact chunk
                if chunk_id in self.seen_chunks:
                    logger.debug(f"Skipping duplicate chunk: {chunk_id}")
                    continue

                # Mark this chunk as seen
                self.seen_chunks.add(chunk_id)

                # Check if this document has already been analyzed (less reliable without doc ID)
                # Consider analyzing every unique chunk regardless of previous document analysis
                # existing_doc = next((d for d in self.all_findings if d.get("file_path") == source_path), None)
                # if existing_doc and "analysis" in existing_doc:
                #     result["analysis"] = existing_doc["analysis"]
                #     result["summary"] = existing_doc.get("summary", "")
                # else:

                # Perform new analysis on the chunk content
                analysis = self.document_analyzer.analyze_document(
                    result.get("content", ""),
                    source_path,
                    query
                )
                result["analysis"] = analysis
                result["summary"] = analysis.get("summary", "") # Add summary field

                # Add citation metadata
                result["citation"] = {
                    "source_path": source_path,
                    "source_name": source_name,
                    "source_type": result.get("source_type", result.get("metadata", {}).get("source_type")),
                    "start": start_pos,
                    "end": end_pos,
                    "score": result.get("score", result.get("similarity", 0.0)), # Use score or similarity
                    "chunk_index": chunk_index, # Add chunk index
                }

                # Add query information to track which question led to this finding
                # result["query"] = query # Already added?

                # Try to determine which iteration this is from
                current_iteration = None
                for iter_num_str, questions in self.questions_by_iteration.items():
                    if query in questions:
                        current_iteration = int(iter_num_str) if iter_num_str.isdigit() else 0
                        break
                current_iteration = current_iteration if current_iteration is not None else 0 # Default to 0 if not found

                # Add query info with iteration
                result["query_info"] = {
                    "query": query,
                    "iteration": current_iteration
                }

                # Add confidence level based on score
                result["confidence"] = result.get("score", result.get("similarity", 0.0))

                # Make sure content field exists
                if "content" not in result or not result["content"]:
                    result["content"] = "No content available"

                # Ensure required fields for tracking exist
                result.setdefault("query", query)
                result.setdefault("file_path", source_path) # Ensure file_path exists

                # Track findings by question for enhanced reporting
                if query not in self.findings_by_question:
                    self.findings_by_question[query] = []
                self.findings_by_question[query].append(result)

                # Track context by question for enhanced reporting
                if query not in self.context_by_question:
                    self.context_by_question[query] = []
                self.context_by_question[query].append(result.get("content", ""))

                analyzed_results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing document chunk: {e}", exc_info=True)

        analysis_time = time.time() - analysis_start_time
        self.performance_metrics["analysis_time"] += analysis_time

        return analyzed_results

    def _update_current_knowledge(self):
        """Update the current knowledge based on all findings."""
        # Extract summaries from all findings
        summaries = []
        seen_summaries = set() # Deduplicate summaries
        # Sort findings by iteration (desc) then score (desc) to prioritize recent, relevant info
        sorted_findings = sorted(self.all_findings, key=lambda x: (x.get("query_info", {}).get("iteration", 0), x.get("confidence", 0.0)), reverse=True)

        for finding in sorted_findings:
            summary = finding.get("summary", "")
            if summary and summary not in seen_summaries:
                # Include source reference if available
                ref_num = finding.get('reference_number') # Check if assigned later
                if ref_num:
                    summaries.append(f"{summary} [Ref: {ref_num}]")
                else:
                    summaries.append(summary)
                seen_summaries.add(summary)

        # Combine summaries
        combined_knowledge = "\n\n".join(summaries)

        # Limit the context size if it exceeds the maximum
        if len(combined_knowledge) > self.max_context_size:
            # Truncate the knowledge to the maximum size
            # Keep the most recent findings (already sorted) by taking from the start
            truncated_knowledge = combined_knowledge[:self.max_context_size]
            # Find the last full sentence or paragraph break
            last_break = max(truncated_knowledge.rfind("."), truncated_knowledge.rfind("\n\n"))
            if last_break > 0:
                 combined_knowledge = truncated_knowledge[:last_break+1] # Keep up to the break
            else:
                 combined_knowledge = truncated_knowledge # Keep truncated version if no break found
            logger.warning(f"Truncated current knowledge context to {len(combined_knowledge)} characters.")


        self.current_knowledge = combined_knowledge
        logger.debug(f"Updated current knowledge context (length: {len(self.current_knowledge)} chars)")


    def _generate_follow_up_questions(self, original_query: str, iteration: int = 1) -> List[str]:
        """
        Generate follow-up questions based on current knowledge.
        Relies on dspy_config._ensure_required_fields for handling agent inputs.
        """
        question_start_time = time.time()
        questions = []
        agent_used = "None" # Track which method succeeded

        # +++ Add check for DSPY config +++
        from .dspy_config import DSPY_CONFIGURED, DSPyAgentRegistry

        # Try DSPy query_refinement agent first
        if DSPY_CONFIGURED: # Check if DSPy is ready
            try:
                query_refinement_agent = DSPyAgentRegistry.get_agent("query_refinement")
                if query_refinement_agent:
                    logger.info("Attempting question generation with DSPy QueryRefinementAgent...")
                    context_val = self.current_knowledge if self.current_knowledge else "No knowledge gathered yet."

                    # +++ Ensure Integers are passed +++
                    agent_inputs = {
                        "query": original_query,
                        "context": context_val,
                        "num_queries": int(self.questions_per_iteration), # Cast to int
                        "iteration": int(iteration), # Cast to int
                    }
                    # The helper function will add defaults for 'content', 'document' if needed
                    prepared_data = DSPyAgentRegistry._ensure_required_fields("query_refinement", query_refinement_agent, agent_inputs)

                    # --- Execute the agent ---
                    logger.debug(f"Calling query_refinement with: {prepared_data}")
                    result = query_refinement_agent(**prepared_data)
                    # -------------------------

                    # +++ Improve question extraction +++
                    raw_output = result.get("related_queries", "") if result and hasattr(result, 'get') else ""
                    if isinstance(raw_output, str):
                        extracted_questions = []
                        # Split by newline and filter out empty/short lines, remove numbering/bullets
                        for line in raw_output.splitlines():
                            line = line.strip()
                            if len(line) > 5: # Basic check for meaningful question length
                                # Remove leading numbers, bullets, etc.
                                question = re.sub(r"^\s*[\d\.\-\*]+\s*", "", line).strip()
                                # Optional: Add question mark if missing
                                if question and not question.endswith('?'):
                                    question += '?'
                                extracted_questions.append(question)
                        questions = extracted_questions
                    elif isinstance(raw_output, list): # Handle if it returns a list directly
                        questions = [str(q).strip() for q in raw_output if isinstance(q, str) and q.strip()]
                    else:
                        questions = [] # Ensure questions is a list

                    if questions:
                        agent_used = "QueryRefinementAgent"
                        logger.info(f"Generated {len(questions)} questions via {agent_used}.")
                    else:
                        logger.warning("QueryRefinementAgent returned no questions or invalid format, attempting fallback.")
                else:
                     logger.warning("Query refinement agent not found, attempting fallback.")
            except Exception as e:
                logger.error(f"Error generating questions with DSPy QueryRefinementAgent: {e}", exc_info=True)
                logger.error(f"Traceback: {traceback.format_exc()}") # Log full traceback
                # Continue to fallback
        else: # DSPy not configured
            logger.warning("DSPy not configured. Skipping QueryRefinementAgent.")


        # --- Fallback logic (remains mostly the same, ensure DSPY_CONFIGURED check) ---
        # Fallback 1: Use TextGeneratorAgent if first agent failed or wasn't found/configured
        if not questions and DSPY_CONFIGURED:
            logger.info("Attempting fallback question generation with TextGeneratorAgent...")
            try:
                text_generator = DSPyAgentRegistry.get_agent("text_generator")
                if text_generator:
                     prompt = f"""Based on the original query "{original_query}" and the current knowledge context, generate {self.questions_per_iteration} specific follow-up questions to explore the topic further. Focus on gaps or related areas. Current knowledge: {self.current_knowledge[:1500]}...
Output ONLY the questions, one per line, numbered."""
                     agent_inputs = {"prompt": prompt, "query": original_query}
                     # Add context for the text generator if needed
                     if self.current_knowledge: agent_inputs["context"] = self.current_knowledge[:2000]
                     prepared_data = DSPyAgentRegistry._ensure_required_fields("text_generator", text_generator, agent_inputs)

                     # --- EXECUTE THE AGENT ---
                     result = text_generator(**prepared_data)
                     # -------------------------

                     response = result.get("text", "") if result and hasattr(result, 'get') else ""

                     # Extract questions from response more robustly
                     lines = [line.strip() for line in response.split('\n') if line.strip()]
                     extracted_questions = []
                     for line in lines:
                          if len(line) > 5:
                              question = re.sub(r"^\s*[\d\.\-\*]+\s*", "", line).strip()
                              if question and not question.endswith('?'):
                                  question += '?'
                              extracted_questions.append(question)
                     questions = extracted_questions

                     if questions:
                         agent_used = "TextGeneratorAgent"
                         logger.info(f"Generated {len(questions)} questions via {agent_used}.")
                     else:
                         logger.warning("TextGeneratorAgent returned no questions.")
                else:
                     logger.error("Fallback TextGenerator agent also not found.")
            except Exception as e:
                logger.error(f"Error generating questions with fallback TextGenerator: {e}", exc_info=True)
        elif not questions and not DSPY_CONFIGURED:
             logger.warning("DSPy not configured. Cannot use TextGeneratorAgent fallback.")


        # Fallback 2: Use DocumentAnalyzer's basic analysis if other methods fail
        if not questions:
             logger.warning("Using DocumentAnalyzer basic analysis as final fallback for question generation.")
             agent_used = "DocumentAnalyzer(basic)"
             try:
                  # Use a simple prompt for basic analysis
                  # NOTE: This calls the *basic analysis* which is non-LLM
                  analysis_input = f"Original query: {original_query}\nContext: {self.current_knowledge[:500]}..."
                  analysis = self.document_analyzer._basic_analysis(analysis_input, query=f"Generate {self.questions_per_iteration} follow-up questions") # Pass a query hint
                  # Basic analysis might put potential questions in key_points or summary
                  potential_questions = []
                  if "key_points" in analysis: potential_questions.extend([p for p in analysis["key_points"] if '?' in p])
                  if "summary" in analysis: potential_questions.extend([line.strip() for line in analysis["summary"].split('\n') if '?' in line])

                  if potential_questions:
                       questions = potential_questions[:self.questions_per_iteration]
                       logger.info(f"Extracted {len(questions)} questions from basic analysis.")
                  else:
                       logger.warning("Basic analysis did not yield questions.")

             except Exception as e:
                  logger.error(f"Error generating questions with basic analysis fallback: {e}", exc_info=True)

        # If still no questions, generate generic ones
        if not questions:
            logger.warning("No questions generated by any method, using generic questions.")
            agent_used = "GenericFallback"
            questions = [
                f"What are the key aspects of {original_query}?",
                f"What additional information is available about {original_query}?"
            ][:self.questions_per_iteration]

        question_time = time.time() - question_start_time
        self.performance_metrics["question_generation_time"] += question_time
        logger.info(f"Generated {len(questions)} follow-up questions using {agent_used} in {question_time:.2f}s")

        # Ensure only unique and correct number are returned
        final_questions = []
        seen_q = set()
        for q in questions:
             # Basic cleaning and lowercasing for uniqueness check
             q_clean = q.strip().lower()
             if q_clean and q_clean not in seen_q:
                  final_questions.append(q.strip()) # Store original stripped version
                  seen_q.add(q_clean)
             if len(final_questions) >= self.questions_per_iteration:
                  break

        # If still not enough questions after filtering, add generic ones back if needed
        while len(final_questions) < self.questions_per_iteration:
            generic_q = f"What else is important about {original_query}?"
            if generic_q.lower() not in seen_q:
                final_questions.append(generic_q)
                seen_q.add(generic_q.lower())
            else: # Avoid infinite loop if generic question already added
                 # Try another generic question
                 generic_q_2 = f"Can you elaborate on the findings related to {original_query}?"
                 if generic_q_2.lower() not in seen_q:
                      final_questions.append(generic_q_2)
                      seen_q.add(generic_q_2.lower())
                 else:
                      break # Give up if both generics already exist

        return final_questions


    def _check_sufficient_information(self) -> bool:
        """
        Check if we have sufficient information to answer the query.

        Returns:
            True if sufficient information has been gathered
        """
        # Check if we have reached the maximum context size
        if len(self.current_knowledge) >= self.max_context_size * 0.9:  # 90% of max size
            logger.info("Sufficient info check: Reached near max context size.")
            return True

        # Simple heuristic: check if we have enough content
        if len(self.current_knowledge) > 5000:
            logger.info("Sufficient info check: Context length > 5000 chars.")
            return True

        # Check if we have a diverse set of sources
        sources = set()
        for finding in self.all_findings:
            source = finding.get("file_path", "")
            if source:
                sources.add(source)

        # If we have at least 5 different sources, consider it sufficient
        if len(sources) >= 5:
            logger.info("Sufficient info check: Found >= 5 sources.")
            return True

        logger.debug("Sufficient info check: Conditions not met.")
        return False

    def _has_sufficient_information(self) -> bool:
        """Alias for _check_sufficient_information for backward compatibility."""
        return self._check_sufficient_information()

    def _perform_initial_search(self, query: str, top_k: int = 5, context_filter: str = None) -> None:
        """
        Perform initial search with both original and expanded queries.

        Args:
            query: The original query
            top_k: Number of results to retrieve
            context_filter: Optional filter for results
        """
        iteration_findings = []

        # Search with original query
        original_results = self._search_and_analyze(
            query,
            top_k=top_k,
            context_filter=context_filter
        )
        iteration_findings.extend(original_results)

        # Search with expanded query if different
        if hasattr(self, 'expanded_query') and self.expanded_query != query:
            expanded_results = self._search_and_analyze(
                self.expanded_query,
                top_k=top_k,
                context_filter=context_filter
            )
            iteration_findings.extend(expanded_results)

        # Update findings and knowledge
        self.all_findings.extend(iteration_findings)
        self._update_current_knowledge()

        # Store first iteration questions
        self.questions_by_iteration['0'] = [query]
        if hasattr(self, 'expanded_query') and self.expanded_query != query:
            self.questions_by_iteration['0'].append(self.expanded_query)

    def _verify_findings(self):
        """Verify findings for consistency and accuracy using fact_verification agent."""
        # +++ ADD CHECK +++
        from .dspy_config import DSPY_CONFIGURED, DSPyAgentRegistry # Import here
        if not DSPY_CONFIGURED:
             logger.warning("Skipping finding verification: DSPy not configured.")
             return

        try:
            fact_verification_agent = DSPyAgentRegistry.get_agent("fact_verification")
            if fact_verification_agent:
                logger.info("Verifying findings using DSPy FactVerificationAgent...")
                verified_count = 0
                for finding in self.all_findings:
                    try:
                         content = finding.get("content", "")
                         summary = finding.get("summary", "")
                         if not content or not summary: continue # Skip

                         agent_inputs = {"content": content, "summary": summary}
                         # Use helper
                         prepared_data = DSPyAgentRegistry._ensure_required_fields("fact_verification", fact_verification_agent, agent_inputs)
                         result = fact_verification_agent(**prepared_data)

                         # +++ MODIFY: Access attributes safely and parse result +++
                         is_consistent_str = getattr(result, "is_consistent", "Uncertain").lower()
                         is_consistent = is_consistent_str in ['yes', 'true', 'consistent'] # Basic parsing

                         confidence_str = getattr(result, "confidence", "0.5")
                         try: confidence = float(confidence_str)
                         except (ValueError, TypeError): confidence = 0.5

                         notes = getattr(result, "notes", "")

                         finding["verification"] = {
                             "is_consistent": is_consistent,
                             "confidence": confidence,
                             "notes": notes
                         }
                         verified_count += 1
                    except Exception as verify_e:
                         logger.warning(f"Error verifying finding {finding.get('citation', {}).get('source_path', '?')}: {verify_e}")
                logger.info(f"Verified {verified_count}/{len(self.all_findings)} findings.")
            else:
                 logger.warning("Fact verification agent not found. Skipping verification.")
        except Exception as e:
            logger.error(f"Error setting up finding verification: {e}", exc_info=True)


    def _expand_query(self, query: str) -> str:
        """Expand the query using the query expander."""
        try:
            # Use the QueryExpander instance directly
            expanded = self.query_expander.expand_query(query)
            self.expanded_query = expanded # Store for potential use
            return expanded
        except Exception as e:
            logger.error(f"Error expanding query: {e}", exc_info=True)
            self.expanded_query = query # Store original if expansion fails
            return query

    def _pool_all_context(self) -> Dict[str, Any]:
        """
        Pool and deduplicate context from all questions across all iterations.

        Returns:
            Dictionary with pooled context information
        """
        logger.info("Pooling and deduplicating context from all questions")

        # Get all unique questions asked
        all_questions_asked = set()
        for questions in self.questions_by_iteration.values():
            for q in questions:
                all_questions_asked.add(q)

        # Pool all findings associated with these questions
        pooled_findings = []
        seen_chunk_ids_for_pooling = set() # Use a separate set for this pooling operation

        for question in all_questions_asked:
            if question in self.findings_by_question:
                for finding in self.findings_by_question[question]:
                    # Create a unique identifier for this chunk
                    # +++ Use metadata fields if available +++
                    source_path = finding.get("file_path", finding.get("citation", {}).get("source_path", ""))
                    source_name = finding.get("source_name", finding.get("citation", {}).get("source_name", ""))
                    start_pos = finding.get("start", finding.get("citation", {}).get("start", 0)) or 0
                    end_pos = finding.get("end", finding.get("citation", {}).get("end", 0)) or 0
                    chunk_index = finding.get("chunk_index", finding.get("citation", {}).get("chunk_index", -1))

                    if chunk_index != -1:
                        chunk_id = f"{source_path}|{source_name}|chunk_{chunk_index}"
                    else:
                        chunk_id = f"{source_path}|{source_name}|{start_pos}-{end_pos}"

                    # Only add if we haven't seen this chunk during pooling
                    if chunk_id not in seen_chunk_ids_for_pooling:
                        seen_chunk_ids_for_pooling.add(chunk_id)
                        pooled_findings.append(finding)
                    else:
                        logger.debug(f"Skipping duplicate finding chunk during pooling: {chunk_id}")


        # Sort pooled findings by score (descending)
        pooled_findings = sorted(pooled_findings, key=lambda x: x.get("confidence", x.get("score", 0.0)), reverse=True)

        # Pool all context content
        pooled_context_content = []
        for finding in pooled_findings:
            content = finding.get("content", "")
            if content:
                pooled_context_content.append(content)

        # Create a summary of the pooling
        pooling_summary = {
            "total_questions": len(all_questions_asked),
            "total_findings_before_deduplication": sum(len(findings) for findings in self.findings_by_question.values()),
            "total_findings_after_deduplication": len(pooled_findings),
            "total_context_chunks": len(pooled_context_content),
            "questions": list(all_questions_asked)
        }

        logger.info(f"Pooled context summary: {pooling_summary['total_findings_after_deduplication']} unique findings from {pooling_summary['total_questions']} questions")

        # Ensure we have at least one item in pooled_context_content to avoid empty context errors
        if not pooled_context_content:
            pooled_context_content = ["No specific content found for the questions."]

        # Create a combined context string for DSPy agents (Limit length)
        combined_context_str = "\n\n---\n\n".join(pooled_context_content[:15]) # Limit to first 15 chunks
        MAX_COMBINED_LEN = 20000
        if len(combined_context_str) > MAX_COMBINED_LEN:
             combined_context_str = combined_context_str[:MAX_COMBINED_LEN] + "\n...[Truncated Pooled Context]"
             logger.warning(f"Truncated combined pooled context to {MAX_COMBINED_LEN} characters.")


        # Create a dictionary with essential fields
        result = {
            "findings": pooled_findings, # The deduplicated and sorted findings
            "context": combined_context_str, # The combined context string (potentially truncated)
            "summary": pooling_summary,
            "content": combined_context_str # Use the same combined string for 'content' field
        }

        # Make sure each finding has content
        for finding in pooled_findings:
            if "content" not in finding or not finding["content"]:
                finding["content"] = "No content available"

        return result

    def _limit_final_context(self, max_size: int) -> None:
        """
        Limit the final context (self.all_findings) to a maximum size
        while preserving the most relevant findings.

        Args:
            max_size: Maximum size in characters for the final context
        """
        if not self.all_findings:
            return

        # Sort findings by confidence/score (higher is better) then iteration (lower is better - keep initial results)
        def get_sort_key(finding):
             confidence = finding.get("confidence", finding.get("score", 0.0))
             iteration = finding.get("query_info", {}).get("iteration", self.max_iterations) # Default to last iteration if missing
             return (confidence, -iteration) # Sort descending by confidence, ascending by iteration (using negative)

        sorted_findings = sorted(self.all_findings, key=get_sort_key, reverse=True)

        kept_findings = []
        current_size = 0
        findings_to_keep = []

        for finding in sorted_findings:
            content_size = len(finding.get("content", ""))
            if current_size + content_size <= max_size:
                findings_to_keep.append(finding)
                current_size += content_size
            # Optionally break early if close to the limit
            # if current_size >= max_size * 0.95: break

        original_count = len(self.all_findings)
        self.all_findings = findings_to_keep
        new_count = len(self.all_findings)

        logger.info(f"Limited context from {original_count} to {new_count} findings ({current_size} characters)")

        # Update the current knowledge based on the limited findings
        self._update_current_knowledge()
```





















 **`advanced_reporting.py`:**
    *   Added checks for `DSPY_CONFIGURED`.
    *   Added `_ensure_list_from_string` helper for robustly parsing potentially stringified lists returned by the LLM (common issue). Used this helper when processing list-like output fields (`insights`, `limitations`, `recommendations`, etc.).
    *   Improved error handling and logging in agent calls and fallback methods.
    *   Corrected parsing of numeric fields like `confidence` and `word_count`.



```python name=src/local_file_research/advanced_reporting.py
# filename: src/local_file_research/advanced_reporting.py
"""
Advanced reporting module for Local File Deep Research.

This module provides enhanced reporting capabilities with interpretations,
proposals, and technical views.
"""

import logging
import time
from typing import List, Dict, Any, Optional
import json # Import json for robust parsing in fallbacks
import traceback # +++ ADD THIS +++
import re # +++ Add re for robust list parsing +++
# +++ Import DSPY_CONFIGURED +++
from .dspy_config import DSPyAgentRegistry, DSPY_CONFIGURED

# Configure logging
logger = logging.getLogger(__name__)

class AdvancedReportGenerator:
    """
    Advanced report generator that provides interpretations, proposals, and technical views.
    """

    def __init__(self):
        """Initialize the advanced report generator."""
        pass # Init can be empty now or do other setup

    def _call_dspy_agent(self, agent_name: str, required_fields: Dict[str, Any]) -> Optional[Any]:
        """Helper function to call a DSPy agent with robust error handling and input checking."""
        # +++ ADD CHECK +++
        if not DSPY_CONFIGURED:
             logger.error(f"Cannot call agent '{agent_name}': DSPy is not configured.")
             return None

        agent = DSPyAgentRegistry.get_agent(agent_name)
        if not agent:
            logger.warning(f"DSPy agent '{agent_name}' not found. Cannot generate this report section.")
            return None

        logger.info(f"Using DSPy {agent_name}Agent")

        # Ensure all required fields are present using the registry's helper
        # This will add fallbacks if necessary based on the signature
        try:
            prepared_data = DSPyAgentRegistry._ensure_required_fields(agent_name, agent, required_fields)
        except Exception as prep_e:
            logger.error(f"Error preparing data for agent '{agent_name}': {prep_e}", exc_info=True)
            return None # Cannot proceed if preparation fails

        try:
            # Log the exact fields being passed
            logger.debug(f"Calling {agent_name} with fields: {list(prepared_data.keys())}")
            result = agent(**prepared_data)
            logger.info(f"Successfully called DSPy agent '{agent_name}'. Result type: {type(result)}")
            return result
        except Exception as e:
            logger.error(f"Error calling DSPy agent '{agent_name}': {e}", exc_info=True)
            # Log the inputs that caused the error for debugging
            logger.error(f"Inputs provided to {agent_name}: {prepared_data}")
            return None # Return None on error

    # +++ Define _ensure_list_from_string helper +++
    def _ensure_list_from_string(self, value: Any) -> List[str]:
         """Ensures the value is a list of strings, attempting to parse if it's a string."""
         if isinstance(value, list):
             # Convert all items to string and strip whitespace
             return [str(item).strip() for item in value if str(item).strip()]
         elif isinstance(value, str):
             # Split by newline OR comma, remove bullets/numbering, filter empty strings
             items = re.split(r'\n|,', value)
             # Remove leading/trailing whitespace and common list markers/punctuation
             cleaned_items = [re.sub(r"^\s*[\d\.\-\*]+\s*|\.$", "", item).strip() for item in items if item.strip()]
             return [item for item in cleaned_items if item] # Filter out any now-empty strings
         else:
             # Return empty list for other types or None
             return []


    def generate_interpretations(self, findings: List[Dict], query: str) -> Dict[str, Any]:
        """Generate interpretations of the findings."""
        try:
            context = self._prepare_context_from_findings(findings)
            if not context:
                logger.warning("Prepared context is empty. Falling back for interpretations.")
                return self._generate_interpretations_with_text_generator(findings, query)

            agent_inputs = {"query": query if query else "Interpret these findings", "context": context}
            result = self._call_dspy_agent("interpreter", agent_inputs) # Use helper

            if result and hasattr(result, 'get'):
                interpretation = result.get("interpretation", "")
                if not interpretation or len(interpretation) < 10: # Adjusted check
                     logger.warning(f"InterpreterAgent returned short/empty interpretation: '{str(interpretation)[:50]}...'. Falling back.")
                     return self._generate_interpretations_with_text_generator(findings, query)

                # +++ Use helper for list fields +++
                key_insights = self._ensure_list_from_string(result.get("insights", []))
                limitations = self._ensure_list_from_string(result.get("limitations", []))

                # Parse confidence score
                confidence_str = result.get("confidence", "0.5")
                try: confidence = float(confidence_str)
                except (ValueError, TypeError): confidence = 0.5

                return {
                    "main_interpretation": interpretation,
                    "key_insights": key_insights,
                    "limitations": limitations,
                    "confidence": confidence
                }
            else:
                logger.warning(f"InterpreterAgent returned None or invalid type ({type(result)}): {result}. Falling back.")
                return self._generate_interpretations_with_text_generator(findings, query)
        except Exception as e:
            logger.error(f"Error generating interpretations: {e}", exc_info=True)
            return self._generate_interpretations_with_text_generator(findings, query)

    def generate_proposals(self, findings: List[Dict], query: str) -> Dict[str, Any]:
        """Generate proposals based on the findings."""
        try:
            context = self._prepare_context_from_findings(findings)
            if not context:
                logger.warning("Prepared context is empty. Falling back for proposals.")
                return self._generate_proposals_with_text_generator(findings, query)

            agent_inputs = {"query": query if query else "Generate proposals based on these findings", "context": context}
            result = self._call_dspy_agent("proposal_generator", agent_inputs) # Use helper

            if result and hasattr(result, 'get'):
                 # +++ Use helper for list fields +++
                 recommendations = self._ensure_list_from_string(result.get("recommendations", []))
                 next_steps = self._ensure_list_from_string(result.get("next_steps", []))
                 alternatives = self._ensure_list_from_string(result.get("alternatives", []))
                 rationale = result.get("rationale", "")

                 if not recommendations: # Check if list is empty after parsing
                      logger.warning("ProposalGeneratorAgent returned no meaningful recommendations. Falling back.")
                      return self._generate_proposals_with_text_generator(findings, query)

                 return {
                    "recommendations": recommendations, "next_steps": next_steps,
                    "alternatives": alternatives, "rationale": rationale
                 }
            else:
                logger.warning(f"ProposalGeneratorAgent returned None or invalid type ({type(result)}): {result}. Falling back.")
                return self._generate_proposals_with_text_generator(findings, query)
        except Exception as e:
            logger.error(f"Error generating proposals: {e}", exc_info=True)
            return self._generate_proposals_with_text_generator(findings, query)

    def generate_technical_view(self, findings: List[Dict], query: str) -> Dict[str, Any]:
        """Generate technical view of the findings."""
        try:
            context = self._prepare_context_from_findings(findings)
            if not context:
                logger.warning("Prepared context is empty. Cannot generate technical view.")
                return self._generate_technical_view_with_text_generator(findings, query) # Fallback

            agent_inputs = {"query": query if query else "Generate technical view based on these findings", "context": context}
            result = self._call_dspy_agent("technical_analyzer", agent_inputs)

            if result and hasattr(result, 'get'):
                analysis = result.get("analysis", "")
                if not analysis or len(analysis) < 10:
                     logger.warning(f"TechAnalyzer short/empty. Falling back.")
                     return self._generate_technical_view_with_text_generator(findings, query)

                # +++ Use helper for list fields +++
                details = self._ensure_list_from_string(result.get("details",[]))
                challenges = self._ensure_list_from_string(result.get("challenges",[]))
                solutions = self._ensure_list_from_string(result.get("solutions",[]))

                return {"technical_analysis": analysis, "technical_details": details, "technical_challenges": challenges, "technical_solutions": solutions}
            else:
                logger.warning(f"TechAnalyzer invalid result ({type(result)}): {result}. Falling back.")
                return self._generate_technical_view_with_text_generator(findings, query)
        except Exception as e:
            logger.error(f"Error generating tech view: {e}", exc_info=True)
            return self._generate_technical_view_with_text_generator(findings, query)

    def generate_comprehensive_synthesis(self, findings: List[Dict], query: str) -> Dict[str, Any]:
        """Generate a comprehensive synthesis of the findings."""
        try:
            context = self._prepare_context_from_findings(findings)
            if not context:
                logger.warning("Prepared context is empty. Cannot generate synthesis.")
                context = f"Query: {query}\nNo specific findings available." # Minimal context

            agent_inputs = {"query": query if query else "Synthesize these findings", "context": context}
            result = self._call_dspy_agent("content_synthesizer", agent_inputs)

            if result and hasattr(result, 'get'):
                article = result.get("article", "")
                if not article or len(article) < 20:
                     logger.warning(f"Synthesizer short/empty. Falling back.")
                     return self._generate_comprehensive_synthesis_with_text_generator(findings, query)

                # Parse word count
                word_count_str = result.get("word_count", "0")
                try: word_count = int(re.sub(r'[^\d]', '', word_count_str)) # Extract digits
                except (ValueError, TypeError): word_count = len(article.split()) # Fallback to splitting

                # +++ Use helper for list fields +++
                key_themes = self._ensure_list_from_string(result.get("key_themes", []))

                return {"article": article, "article_type": result.get("article_type", "general"), "key_themes": key_themes, "word_count": word_count}
            else:
                logger.warning(f"Synthesizer invalid result ({type(result)}): {result}. Falling back.")
                return self._generate_comprehensive_synthesis_with_text_generator(findings, query)
        except Exception as e:
            logger.error(f"Error generating synthesis: {e}", exc_info=True)
            return self._generate_comprehensive_synthesis_with_text_generator(findings, query)

    def _prepare_context_from_findings(self, findings: List[Dict]) -> str:
        """Prepare context string from findings for agent input."""
        context_parts = []
        logger.info(f"Preparing context from {len(findings)} findings for report generation.")

        # Limit number of findings used for context to avoid excessive length
        MAX_FINDINGS_FOR_CONTEXT = 50 # Adjust as needed
        findings_to_use = findings[:MAX_FINDINGS_FOR_CONTEXT]
        if len(findings) > MAX_FINDINGS_FOR_CONTEXT:
             logger.warning(f"Using top {MAX_FINDINGS_FOR_CONTEXT} findings (out of {len(findings)}) for context generation.")

        for i, finding in enumerate(findings_to_use, 1):
            if not isinstance(finding, dict):
                logger.warning(f"Skipping finding at index {i-1} as it is not a dictionary: {finding}")
                continue

            # Get the most informative content
            summary = finding.get('summary', '')
            analysis = finding.get('analysis', {})
            content = finding.get('content', '') # Get raw content as fallback
            # Ensure analysis is a dict before accessing key_points
            key_points = analysis.get('key_points', []) if isinstance(analysis, dict) else []
            # +++ Ensure key_points is a list using helper +++
            key_points_list = self._ensure_list_from_string(key_points)


            # Prioritize summary, then key points, then raw content snippet
            finding_text = ""
            if summary:
                finding_text = summary
            elif key_points_list: # Use the parsed list
                 finding_text = "Key points:\n" + "\n".join([f"- {point}" for point in key_points_list])
            elif content:
                 finding_text = content[:500] + ("..." if len(content) > 500 else "") # Use snippet of content

            if finding_text:
                 context_parts.append(f"--- Finding {i} ---")
                 # Add source information
                 citation = finding.get('citation', {})
                 source_name = citation.get('source_name', citation.get('source_path', 'Unknown Source'))
                 context_parts.append(f"Source: {source_name}")
                 # Add query info if available
                 query_info = finding.get('query_info', {})
                 if query_info and query_info.get('query'):
                      context_parts.append(f"Related Question: {query_info['query']}")

                 context_parts.append(f"Content: {finding_text}")
                 context_parts.append("------\n")

        # Add a clear separator and instruction
        generated_context = "Context based on research findings:\n\n" + "\n".join(context_parts).strip()

        # Limit total context length further if needed (e.g., 8k chars)
        MAX_CONTEXT_LEN = 30000
        if len(generated_context) > MAX_CONTEXT_LEN:
             generated_context = generated_context[:MAX_CONTEXT_LEN] + "\n... [Context Truncated]"
             logger.warning(f"Truncated prepared context to {MAX_CONTEXT_LEN} characters.")

        logger.info(f"Generated context string length: {len(generated_context)}")
        if not generated_context or len(generated_context) < 100:
             logger.warning("Generated context is very short or empty.")
             # Provide a minimal context if completely empty
             if not generated_context.strip():
                  return "No findings available to generate context."
        return generated_context


    # --- Fallback methods using Text Generator ---
    # (Keep fallbacks, but ensure they also use _ensure_list_from_string where needed)

    def _generate_interpretations_with_text_generator(self, findings: List[Dict], query: str) -> Dict[str, Any]:
        """Generate interpretations using text generator (Fallback)."""
        logger.info("Attempting fallback: interpretations with text_generator.")
        try:
            context = self._prepare_context_from_findings(findings)
            if not context:
                logger.warning("Context is empty for fallback interpretation generation.")
                context = "No findings available to interpret."

            prompt = f"""Based on the following research findings, provide an interpretation for the query: "{query}"

Findings Context:
{context}

Please provide ONLY the following sections, clearly marked:
Main Interpretation: [Your interpretation here (1-2 paragraphs)]
Key Insights: [List 3-5 key insights, one per line starting with '-']
Limitations: [List 2-3 limitations, one per line starting with '-']
Confidence: [A single number between 0.0 and 1.0]
"""
            agent_inputs = {"prompt": prompt, "query": query, "context": context}
            result = self._call_dspy_agent("text_generator", agent_inputs) # Use helper
            response = result.get("text", "") if result and hasattr(result, 'get') else ""
            if not response:
                 logger.warning("Fallback text_generator for interpretations returned no response.")
                 return {"main_interpretation": "Fallback failed: No response.", "key_insights": [], "limitations": [], "confidence": 0.0}

            logger.debug(f"Fallback Interpretation Response:\n{response[:200]}...")

            # Robust parsing
            main_interpretation = ""
            key_insights = []
            limitations = []
            confidence = 0.5
            lines = response.splitlines()
            current_section = None
            for line in lines:
                line_strip = line.strip()
                if line_strip.startswith("Main Interpretation:"):
                    current_section = "interpretation"
                    main_interpretation += line_strip[len("Main Interpretation:"):].strip()
                elif line_strip.startswith("Key Insights:"):
                    current_section = "insights"
                elif line_strip.startswith("Limitations:"):
                    current_section = "limitations"
                elif line_strip.startswith("Confidence:"):
                    current_section = "confidence"
                    try:
                        confidence_str = line_strip[len("Confidence:"):].strip()
                        confidence = float(confidence_str)
                    except ValueError:
                        confidence = 0.5 # Default on parsing error
                elif current_section == "interpretation" and not line_strip.startswith(("Key Insights:", "Limitations:", "Confidence:")):
                     main_interpretation += "\n" + line_strip if main_interpretation else line_strip
                elif current_section == "insights" and line_strip.startswith("-"):
                    key_insights.append(line_strip[1:].strip())
                elif current_section == "limitations" and line_strip.startswith("-"):
                    limitations.append(line_strip[1:].strip())

            if not main_interpretation: main_interpretation = "Interpretation generation failed via fallback."
            if not key_insights: key_insights = ["No insights extracted via fallback."]
            if not limitations: limitations = ["Analysis limitations unclear via fallback."]

            return {
                "main_interpretation": main_interpretation.strip(),
                "key_insights": key_insights, # Already a list of strings
                "limitations": limitations, # Already a list of strings
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Error generating interpretations with text generator: {e}", exc_info=True)
            return {
                "main_interpretation": "Error during fallback interpretation generation.",
                "key_insights": [], "limitations": [], "confidence": 0.1
            }

    def _generate_proposals_with_text_generator(self, findings: List[Dict], query: str) -> Dict[str, Any]:
        """Generate proposals using text generator (Fallback)."""
        logger.info("Attempting to generate proposals using fallback text generator.")
        try:
            context = self._prepare_context_from_findings(findings)
            if not context:
                 context = "No findings available to generate proposals."

            prompt = f"""Based on the research findings context below, provide proposals for the query: "{query}"

Findings Context:
{context}

Please provide ONLY the following sections, clearly marked:
Recommendations: [List 3-5 recommendations, one per line starting with '-']
Next Steps: [List 2-3 next steps, one per line starting with '-']
Alternatives: [List 1-2 alternatives, one per line starting with '-']
Rationale: [Your rationale here (1-2 paragraphs)]
"""
            agent_inputs = {"prompt": prompt, "query": query, "context": context}
            result = self._call_dspy_agent("text_generator", agent_inputs)
            response = result.get("text", "") if result and hasattr(result, 'get') else ""

            if not response:
                 logger.warning("Fallback text_generator proposals empty.")
                 return {"recommendations": ["Fallback failed."], "next_steps": [], "alternatives": [], "rationale": ""}

            logger.debug(f"Fallback Proposals Response:\n{response[:200]}...")

            # Robust parsing
            recommendations, next_steps, alternatives = [], [], []
            rationale = ""
            lines = response.splitlines()
            current_section = None
            for line in lines:
                line_strip = line.strip()
                if line_strip.startswith("Recommendations:"): current_section = "recommendations"
                elif line_strip.startswith("Next Steps:"): current_section = "next_steps"
                elif line_strip.startswith("Alternatives:"): current_section = "alternatives"
                elif line_strip.startswith("Rationale:"):
                    current_section = "rationale"
                    rationale += line_strip[len("Rationale:"):].strip()
                elif current_section == "rationale" and not line_strip.startswith(("Recommendations:", "Next Steps:", "Alternatives:")) :
                    rationale += "\n" + line_strip if rationale else line_strip
                elif current_section == "recommendations" and line_strip.startswith("-"): recommendations.append(line_strip[1:].strip())
                elif current_section == "next_steps" and line_strip.startswith("-"): next_steps.append(line_strip[1:].strip())
                elif current_section == "alternatives" and line_strip.startswith("-"): alternatives.append(line_strip[1:].strip())

            if not recommendations: recommendations = ["Proposal generation failed via fallback."]
            if not next_steps: next_steps = ["No next steps extracted via fallback."]
            if not alternatives: alternatives = ["No alternatives extracted via fallback."]
            if not rationale: rationale = "Rationale generation failed via fallback."

            return {
                "recommendations": recommendations, # Already lists of strings
                "next_steps": next_steps,
                "alternatives": alternatives,
                "rationale": rationale.strip()
            }
        except Exception as e:
            logger.error(f"Error generating proposals with text generator: {e}", exc_info=True)
            return {
                "recommendations": ["Error during fallback proposal generation."],
                "next_steps": [], "alternatives": [], "rationale": "Error."
            }

    def _generate_technical_view_with_text_generator(self, findings: List[Dict], query: str) -> Dict[str, Any]:
        """Generate technical view using text generator (Fallback)."""
        logger.info("Attempting to generate technical view using fallback text generator.")
        try:
            context = self._prepare_context_from_findings(findings)
            if not context: context = "No findings available for technical analysis."

            prompt = f"""Based on the research findings context below, provide a technical analysis for the query: "{query}"

Findings Context:
{context}

Please provide ONLY the following sections, clearly marked:
Technical Analysis: [Your analysis here (1-2 paragraphs)]
Technical Details: [List 3-5 details, one per line starting with '-']
Technical Challenges: [List 2-3 challenges, one per line starting with '-']
Technical Solutions: [List 2-3 solutions, one per line starting with '-']
"""
            agent_inputs = {"prompt": prompt, "query": query, "context": context}
            result = self._call_dspy_agent("text_generator", agent_inputs)
            response = result.get("text", "") if result else ""
            logger.debug(f"Fallback Technical View Response:\n{response[:500]}...")
            if not response:
                logger.warning("Fallback Technical View empty.")
                # Return the expected structure even on failure
                return {"technical_analysis": "Fallback failed: No response.", "technical_details": [], "technical_challenges": [], "technical_solutions": []}

            # Robust parsing
            technical_analysis = ""
            technical_details, technical_challenges, technical_solutions = [], [], []
            lines = response.splitlines()
            current_section = None
            for line in lines:
                line_strip = line.strip()
                if line_strip.startswith("Technical Analysis:"):
                     current_section = "analysis"; technical_analysis += line_strip[len("Technical Analysis:"):].strip()
                elif line_strip.startswith("Technical Details:"): current_section = "details"
                elif line_strip.startswith("Technical Challenges:"): current_section = "challenges"
                elif line_strip.startswith("Technical Solutions:"): current_section = "solutions"
                elif current_section == "analysis" and not line_strip.startswith(("Technical Details:", "Technical Challenges:", "Technical Solutions:")):
                     technical_analysis += "\n" + line_strip if technical_analysis else line_strip
                elif current_section == "details" and line_strip.startswith("-"): technical_details.append(line_strip[1:].strip())
                elif current_section == "challenges" and line_strip.startswith("-"): technical_challenges.append(line_strip[1:].strip())
                elif current_section == "solutions" and line_strip.startswith("-"): technical_solutions.append(line_strip[1:].strip())

            if not technical_analysis: technical_analysis = "Technical analysis generation failed via fallback."
            if not technical_details: technical_details = ["No details extracted."]
            if not technical_challenges: technical_challenges = ["No challenges identified."]
            if not technical_solutions: technical_solutions = ["No solutions proposed."]

            return {
                "technical_analysis": technical_analysis.strip(),
                "technical_details": technical_details, # Already lists of strings
                "technical_challenges": technical_challenges,
                "technical_solutions": technical_solutions
            }
        except Exception as e:
            logger.error(f"Error generating technical view with text generator: {e}", exc_info=True)
            return {
                "technical_analysis": "Error during fallback technical analysis generation.",
                "technical_details": [], "technical_challenges": [], "technical_solutions": []
            }

    def _generate_comprehensive_synthesis_with_text_generator(self, findings: List[Dict], query: str) -> Dict[str, Any]:
        """Generate comprehensive synthesis using text generator (Fallback)."""
        logger.info("Attempting to generate synthesis using fallback text generator.")
        try:
            context = self._prepare_context_from_findings(findings)
            if not context: context = "No findings available to synthesize."

            article_type = self._determine_article_type(query, findings)
            prompt = f"""Based on the research findings context below, write a comprehensive {article_type} about: "{query}"

Findings Context:
{context}

Write a well-structured, informative {article_type}. Include an introduction, key themes (use markdown headings like ## Theme), and a conclusion.
"""
            agent_inputs = {"prompt": prompt, "query": query, "context": context}
            result = self._call_dspy_agent("text_generator", agent_inputs)
            article = result.get("text", "") if result else ""
            logger.debug(f"Fallback Synthesis Response:\n{article[:500]}...")

            if not article:
                 logger.warning("Fallback Synthesis empty.")
                 article = "Comprehensive synthesis generation failed via fallback."

            # Basic metadata extraction
            actual_word_count = len(article.split())
            # Extract themes based on markdown headings
            key_themes = [line.strip().lstrip('#').strip() for line in article.splitlines() if line.strip().startswith("#")]
            if not key_themes: key_themes = ["Main Topic", "Key Findings", "Conclusion"] # Fallback themes

            return {
                "article": article, "article_type": article_type,
                "key_themes": key_themes[:5], # Already list of strings
                "word_count": actual_word_count
            }
        except Exception as e:
            logger.error(f"Error generating comprehensive synthesis with text generator: {e}", exc_info=True)
            return {
                "article": "Error during fallback synthesis generation.",
                "article_type": "general", "key_themes": [], "word_count": 0
            }

    def _determine_article_type(self, query: str, findings: List[Dict]) -> str:
        """Determine the appropriate article type based on the query and findings."""
        query_lower = query.lower()

        # Keywords for different types
        scientific_keywords = ["research", "study", "experiment", "scientific", "academic", "journal", "hypothesis", "theory", "data analysis", "methodology", "findings", "results", "conclusion", "physics", "chemistry", "biology", "medicine", "engineering", "technology", "algorithm", "computation"]
        news_keywords = ["current events", "breaking", "latest", "today", "yesterday", "this week", "this month", "recent", "update", "development", "announcement", "press release", "politics", "economy", "government", "election", "policy", "legislation", "regulation", "crisis", "conflict"]
        blog_keywords = ["opinion", "perspective", "view", "experience", "personal", "lifestyle", "trend", "popular", "culture", "entertainment", "media", "social media", "influencer", "celebrity", "fashion", "food", "travel", "hobby", "review"]
        technical_keywords = ["how-to", "tutorial", "guide", "walkthrough", "step-by-step", "instruction", "manual", "documentation", "code", "programming", "software", "hardware", "system", "network", "database", "security", "implementation", "configuration", "setup", "installation"]
        philosophical_keywords = ["philosophy", "spiritual", "spirituality", "religion", "belief", "faith", "meaning", "purpose", "existence", "consciousness", "mindfulness", "meditation", "wisdom", "enlightenment", "soul", "divine", "sacred", "ritual", "practice", "tradition", "teaching", "doctrine", "theology", "metaphysics", "ethics", "moral", "value", "virtue", "transcendence", "immanence", "being", "reality"]

        keywords_map = {
            "scientific article": scientific_keywords, "news article": news_keywords,
            "blog post": blog_keywords, "technical guide": technical_keywords,
            "philosophical essay": philosophical_keywords
        }

        counts = {type_name: 0 for type_name in keywords_map}

        # Check query
        for type_name, keywords in keywords_map.items():
            counts[type_name] += sum(1 for kw in keywords if kw in query_lower)

        # Check findings content (simplified)
        all_content = " ".join([f.get("content", "")[:200] for f in findings[:5]]).lower() # Check snippets
        for type_name, keywords in keywords_map.items():
            counts[type_name] += sum(1 for kw in keywords if kw in all_content) * 0.5

        # Determine max
        if not any(counts.values()): # If all counts are zero
            return "informative article" # Default

        max_type = max(counts, key=counts.get)
        return max_type

    # --- _pool_all_context method remains the same ---
    # --- _limit_final_context method remains the same ---









#####FOR dspy agents.py
#after
#class DocumentTypeError(Exception):
# --- Document Type Signatures ---

if DSPY_AVAILABLE:
    # Code Document Analysis Signature
    class CodeAnalysisSignature(dspy.Signature):
        """Analyze code documents and extract key information."""
        code = dspy.InputField(desc="The code content to analyze")
        language = dspy.InputField(desc="The programming language of the code")
        query = dspy.InputField(desc="The query or task to focus the analysis on")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")
        content = dspy.InputField(desc="Alternative content source (optional)", default="")
        context = dspy.InputField(desc="Additional context for analysis (optional)", default="")

        summary = dspy.OutputField(desc="A concise summary of what the code does")
        functions = dspy.OutputField(desc="List of key functions/methods in the code")
        classes = dspy.OutputField(desc="List of key classes in the code")
        dependencies = dspy.OutputField(desc="List of external dependencies or imports")
        complexity = dspy.OutputField(desc="Assessment of code complexity")
        issues = dspy.OutputField(desc="Potential issues or bugs in the code")
        suggestions = dspy.OutputField(desc="Suggestions for improvement")

    # Spreadsheet Analysis Signature
    class SpreadsheetAnalysisSignature(dspy.Signature):
        """Analyze spreadsheet data and extract key information."""
        data = dspy.InputField(desc="The spreadsheet content in text format")
        query = dspy.InputField(desc="The query or task to focus the analysis on")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")
        content = dspy.InputField(desc="Alternative content source (optional)", default="")
        context = dspy.InputField(desc="Additional context for analysis (optional)", default="")

        summary = dspy.OutputField(desc="A concise summary of the data")
        structure = dspy.OutputField(desc="Description of the data structure (columns, sheets)")
        key_metrics = dspy.OutputField(desc="Key metrics or statistics from the data")
        patterns = dspy.OutputField(desc="Patterns or trends identified in the data")
        anomalies = dspy.OutputField(desc="Anomalies or outliers in the data")
        insights = dspy.OutputField(desc="Key insights derived from the data")

    # PDF Document Analysis Signature
    class PDFAnalysisSignature(dspy.Signature):
        """Analyze PDF documents and extract key information."""
        content = dspy.InputField(desc="The PDF content in text format")
        query = dspy.InputField(desc="The query or task to focus the analysis on")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")
        context = dspy.InputField(desc="Additional context for analysis (optional)", default="")

        summary = dspy.OutputField(desc="A concise summary of the document")
        key_points = dspy.OutputField(desc="Key points from the document")
        entities = dspy.OutputField(desc="Important entities mentioned in the document")
        topics = dspy.OutputField(desc="Main topics covered in the document")
        structure = dspy.OutputField(desc="Document structure (sections, headings)")
        citations = dspy.OutputField(desc="Citations or references in the document")

    # Technical Document Analysis Signature
    class TechnicalDocAnalysisSignature(dspy.Signature):
        """Analyze technical documents and extract key information."""
        content = dspy.InputField(desc="The technical document content")
        document_type = dspy.InputField(desc="The type of technical document (e.g., API doc, whitepaper)")
        query = dspy.InputField(desc="The query or task to focus the analysis on")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")
        context = dspy.InputField(desc="Additional context for analysis (optional)", default="")

        summary = dspy.OutputField(desc="A concise summary of the technical document")
        key_concepts = dspy.OutputField(desc="Key technical concepts explained in the document")
        technical_details = dspy.OutputField(desc="Important technical details or specifications")
        requirements = dspy.OutputField(desc="Requirements or prerequisites mentioned")
        examples = dspy.OutputField(desc="Code examples or usage examples")
        limitations = dspy.OutputField(desc="Limitations or constraints mentioned")

    # Research Paper Analysis Signature
    class ResearchPaperAnalysisSignature(dspy.Signature):
        """Analyze research papers and extract key information."""
        content = dspy.InputField(desc="The research paper content")
        query = dspy.InputField(desc="The query or task to focus the analysis on")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")
        context = dspy.InputField(desc="Additional context for analysis (optional)", default="")

        summary = dspy.OutputField(desc="A concise summary of the research paper")
        research_question = dspy.OutputField(desc="The main research question or objective")
        methodology = dspy.OutputField(desc="The methodology used in the research")
        findings = dspy.OutputField(desc="Key findings or results")
        limitations = dspy.OutputField(desc="Limitations of the research")
        implications = dspy.OutputField(desc="Implications or applications of the research")
        future_work = dspy.OutputField(desc="Suggested future work")

    # Chain-of-Thought Document Analysis
    class ChainOfThoughtAnalysisSignature(dspy.Signature):
        """Perform step-by-step reasoning on a document."""
        content = dspy.InputField(desc="The document content to analyze")
        query = dspy.InputField(desc="The query or task to focus the analysis on")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")
        context = dspy.InputField(desc="Additional context for analysis (optional)", default="")

        reasoning = dspy.OutputField(desc="Step-by-step reasoning process")
        key_insights = dspy.OutputField(desc="Key insights derived from the reasoning")
        conclusion = dspy.OutputField(desc="Conclusion based on the reasoning")

    # Multi-Document Synthesis
    class MultiDocumentSynthesisSignature(dspy.Signature):
        """Synthesize information from multiple documents."""
        documents = dspy.InputField(desc="List of document contents and their metadata")
        query = dspy.InputField(desc="The query or task to focus the synthesis on")
        document = dspy.InputField(desc="Single document content (optional)", default="")
        content = dspy.InputField(desc="Alternative content source (optional)", default="")

        synthesis = dspy.OutputField(desc="Synthesized information from all documents")
        common_themes = dspy.OutputField(desc="Common themes across documents")
        contradictions = dspy.OutputField(desc="Contradictions or disagreements between documents")
        unique_insights = dspy.OutputField(desc="Unique insights from specific documents")
        integrated_view = dspy.OutputField(desc="Integrated view of the information")

    # --- ADDED MISSING SIGNATURES FROM dspy_config.py ---
    class SummarizerSignature(dspy.Signature):
        """Summarize the provided content."""
        content = dspy.InputField(desc="The text content to summarize")
        query = dspy.InputField(desc="Optional query to focus the summary", default="")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")
        context = dspy.InputField(desc="Alternative context (optional)", default="")

        summary = dspy.OutputField(desc="A concise summary")

    class AnswererSignature(dspy.Signature):
        """Answer a question based on the provided context."""
        context = dspy.InputField(desc="Context relevant to the question")
        query = dspy.InputField(desc="The question to answer")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")
        content = dspy.InputField(desc="Alternative content source (optional)", default="")

        answer = dspy.OutputField(desc="The answer derived from the context")

    class ExtractorSignature(dspy.Signature):
        """Extract specific information (e.g., entities, keywords) from content."""
        content = dspy.InputField(desc="The text content to extract from")
        query = dspy.InputField(desc="Description of the information to extract (e.g., 'names of people')")
        context = dspy.InputField(desc="Context relevant to the question (optional)", default="")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")

        info = dspy.OutputField(desc="The extracted information")

    class FactCheckerSignature(dspy.Signature):
        """Assess the factual consistency of a statement against provided context."""
        statement = dspy.InputField(desc="The statement to fact-check")
        context = dspy.InputField(desc="The context to check against")
        query = dspy.InputField(desc="Optional focus for fact-checking", default="")
        content = dspy.InputField(desc="Content relevant to the question (optional)", default="")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")

        fact_check = dspy.OutputField(desc="Assessment of factual consistency (e.g., 'Consistent', 'Inconsistent', 'Needs More Info') with explanation")

    class DocumentAnalysisSignature(dspy.Signature):
        """Provide a structured analysis of a document."""
        document = dspy.InputField(desc="The document text")
        query = dspy.InputField(desc="Optional query to guide analysis", default="Analyze this document")
        context = dspy.InputField(desc="Optional additional context", default="")
        content = dspy.InputField(desc="Content relevant to the question (optional)", default="")

        summary = dspy.OutputField(desc="Overall summary")
        key_points = dspy.OutputField(desc="List of key points")
        entities = dspy.OutputField(desc="List of important entities")
        sentiment = dspy.OutputField(desc="Overall sentiment (e.g., Positive, Negative, Neutral)")

    class InterpreterSignature(dspy.Signature):
        """Interpret research findings contextually."""
        query = dspy.InputField(desc="The original research query")
        context = dspy.InputField(desc="Aggregated research findings")
        content = dspy.InputField(desc="Content relevant to the question (optional)", default="")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")

        interpretation = dspy.OutputField(desc="Main interpretation")
        insights = dspy.OutputField(desc="List of key insights")
        limitations = dspy.OutputField(desc="List of limitations")
        confidence = dspy.OutputField(desc="Confidence score (0.0-1.0)")

    class ProposalGeneratorSignature(dspy.Signature):
        """Generate actionable proposals from findings."""
        query = dspy.InputField(desc="Original research query")
        context = dspy.InputField(desc="Research findings context")
        content = dspy.InputField(desc="Content relevant to the question (optional)", default="")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")

        recommendations = dspy.OutputField(desc="List of specific recommendations")
        next_steps = dspy.OutputField(desc="List of concrete next steps")
        alternatives = dspy.OutputField(desc="List of alternative approaches")
        rationale = dspy.OutputField(desc="Justification for proposals")

    class TechnicalAnalyzerSignature(dspy.Signature):
        """Provide technical analysis of findings."""
        query = dspy.InputField(desc="Original research query")
        context = dspy.InputField(desc="Research findings context")
        content = dspy.InputField(desc="Content relevant to the question (optional)", default="")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")

        analysis = dspy.OutputField(desc="Technical analysis summary")
        details = dspy.OutputField(desc="List of technical details")
        challenges = dspy.OutputField(desc="List of technical challenges")
        solutions = dspy.OutputField(desc="List of potential solutions")

    # --- CORRECTED QueryRefinementSignature ---
    class QueryRefinementSignature(dspy.Signature):
        """Generate refined/follow-up queries."""
        instructions = 'Generate refined/follow-up queries.' # Moved instructions here

        query = dspy.InputField(desc="Original query")
        context = dspy.InputField(desc="Current research context/knowledge")
        content = dspy.InputField(desc="Content relevant to the question (optional)", default="")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")
        num_queries: int = dspy.InputField(desc="Number of queries to generate") # Corrected type hint
        iteration: int = dspy.InputField(desc="Current research iteration") # Corrected type hint

        related_queries = dspy.OutputField(desc="List of refined/follow-up queries")
    # --- END CORRECTION ---

    class FactVerificationSignature(dspy.Signature):
        """Verify content consistency against a summary."""
        content = dspy.InputField(desc="Content snippet to verify")
        summary = dspy.InputField(desc="Overall summary to verify against")
        context = dspy.InputField(desc="Context relevant to the question (optional)", default="")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")
        query = dspy.InputField(desc="Optional focus for verification", default="Verify facts")

        is_consistent = dspy.OutputField(desc="Boolean indicating consistency")
        confidence = dspy.OutputField(desc="Confidence score (0.0-1.0)")
        notes = dspy.OutputField(desc="Explanation for consistency assessment")

    class TextGeneratorSignature(dspy.Signature):
        """Generic text generation based on a prompt."""
        prompt = dspy.InputField(desc="The input prompt")
        context = dspy.InputField(desc="Optional context", default="")
        content = dspy.InputField(desc="Content relevant to the question (optional)", default="")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")
        query = dspy.InputField(desc="Optional query focus", default="")

        text = dspy.OutputField(desc="Generated text")

    class ContentSynthesizerSignature(dspy.Signature):
        """Synthesize findings into a structured article."""
        query = dspy.InputField(desc="Original research query")
        context = dspy.InputField(desc="Research findings context")
        content = dspy.InputField(desc="Content relevant to the question (optional)", default="")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")

        article = dspy.OutputField(desc="Synthesized article content")
        article_type = dspy.OutputField(desc="Type of article (e.g., summary, analysis, blog post)")
        key_themes = dspy.OutputField(desc="List of key themes discussed")
        word_count = dspy.OutputField(desc="Approximate word count")

    class QueryExpansionSignature(dspy.Signature):
        """Expand a query with related terms/concepts."""
        query = dspy.InputField(desc="Query to expand")
        context = dspy.InputField(desc="Optional context", default="")
        content = dspy.InputField(desc="Content relevant to the question (optional)", default="")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")

        expanded_query = dspy.OutputField(desc="Expanded query string")

# --- Agent Implementation ---

def register_custom_agents():
    """Register custom DSPy agents for different document types."""
    if not DSPY_AVAILABLE or not DSPY_CONFIGURED:
        logger.warning("DSPy not available or configured. Cannot register custom agents.")
        return


    # Register code analysis agent
    code_analyzer = dspy.Predict(CodeAnalysisSignature)
    DSPyAgentRegistry.register_agent("code_analyzer", code_analyzer)

    # Register spreadsheet analysis agent
    spreadsheet_analyzer = dspy.Predict(SpreadsheetAnalysisSignature)
    DSPyAgentRegistry.register_agent("spreadsheet_analyzer", spreadsheet_analyzer)

    # Register PDF analysis agent
    pdf_analyzer = dspy.Predict(PDFAnalysisSignature)
    DSPyAgentRegistry.register_agent("pdf_analyzer", pdf_analyzer)

    # Register technical document analysis agent
    tech_doc_analyzer = dspy.Predict(TechnicalDocAnalysisSignature)
    DSPyAgentRegistry.register_agent("tech_doc_analyzer", tech_doc_analyzer)

    # Register research paper analysis agent
    research_paper_analyzer = dspy.Predict(ResearchPaperAnalysisSignature)
    DSPyAgentRegistry.register_agent("research_paper_analyzer", research_paper_analyzer)

    # Register chain-of-thought analysis agent
    cot_analyzer = dspy.Predict(ChainOfThoughtAnalysisSignature)
    DSPyAgentRegistry.register_agent("cot_analyzer", cot_analyzer)

    # Register multi-document synthesis agent
    multi_doc_synthesizer = dspy.Predict(MultiDocumentSynthesisSignature)
    DSPyAgentRegistry.register_agent("multi_doc_synthesizer", multi_doc_synthesizer)

    # Register agent chains
    DSPyAgentRegistry.register_chain("code_analysis_chain", ["code_analyzer"])
    DSPyAgentRegistry.register_chain("spreadsheet_analysis_chain", ["spreadsheet_analyzer"])
    DSPyAgentRegistry.register_chain("pdf_analysis_chain", ["pdf_analyzer"])
    DSPyAgentRegistry.register_chain("tech_doc_analysis_chain", ["tech_doc_analyzer"])
    DSPyAgentRegistry.register_chain("research_paper_analysis_chain", ["research_paper_analyzer"])
    DSPyAgentRegistry.register_chain("cot_analysis_chain", ["cot_analyzer"])
    DSPyAgentRegistry.register_chain("multi_doc_synthesis_chain", ["multi_doc_synthesizer"])

    # Register combined chains
    DSPyAgentRegistry.register_chain("code_review_chain", ["code_analyzer", "cot_analyzer"])
    DSPyAgentRegistry.register_chain("data_analysis_chain", ["spreadsheet_analyzer", "cot_analyzer"])
    DSPyAgentRegistry.register_chain("research_review_chain", ["research_paper_analyzer", "cot_analyzer"])

    logger.info("Custom DSPy agents registered successfully")