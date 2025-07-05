"""
Patch for LiteLLM library to fix the __annotations__ error and other issues.
"""

import logging
import sys
import os

# Configure logging
logger = logging.getLogger(__name__)

# Create a filter to suppress LiteLLM logging
class LiteLLMFilter(logging.Filter):
    def filter(self, record):
        # Filter out logs from litellm modules
        if record.name.startswith('litellm'):
            return False
        # Also filter out logs containing "__annotations__"
        if "__annotations__" in record.getMessage():
            return False
        return True

# Apply the filter to the root logger
root_logger = logging.getLogger()
root_logger.addFilter(LiteLLMFilter())

# Flag to track if patches have been applied
PATCHES_APPLIED = False

def apply_patch():
    """
    Apply patches to fix LiteLLM issues.
    """
    global PATCHES_APPLIED

    # Don't apply patches multiple times
    if PATCHES_APPLIED:
        logger.debug("LiteLLM patches already applied, skipping")
        return

    try:
        # Disable LiteLLM logging via environment variable
        os.environ["LITELLM_LOG"] = "ERROR"  # Set to ERROR instead of NONE

        # Try to import the problematic module
        try:
            from litellm.litellm_core_utils.model_param_helper import ModelParamHelper

            # Create a patched version of the problematic method
            def patched_get_litellm_supported_transcription_kwargs():
                return set()

            # Apply the patch
            ModelParamHelper._get_litellm_supported_transcription_kwargs = patched_get_litellm_supported_transcription_kwargs
            logger.info("Patched ModelParamHelper._get_litellm_supported_transcription_kwargs")
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not patch ModelParamHelper: {e}")

        # Patch other problematic modules
        try:
            import litellm

            # Disable callbacks
            litellm.success_callback = []
            litellm.failure_callback = []

            # Try to disable logging in different ways
            try:
                # Method 1: Set verbose to False
                if hasattr(litellm, "verbose"):
                    litellm.verbose = False

                # Method 2: Set log level to a high value
                if hasattr(litellm, "utils") and hasattr(litellm.utils, "logging"):
                    # Don't call disable_logging() as it might not exist
                    # Instead, try to set the log level directly
                    if hasattr(litellm.utils.logging, "setLevel"):
                        litellm.utils.logging.setLevel(logging.ERROR)
            except Exception as e:
                logger.debug(f"Error disabling litellm logging: {e}")

            # Patch the __annotations__ error in other modules
            for module_name in dir(litellm):
                module = getattr(litellm, module_name)
                if hasattr(module, "__annotations__"):
                    try:
                        # Skip functools.partial objects which are immutable
                        import functools
                        if isinstance(module, functools.partial):
                            logger.debug(f"Skipping immutable functools.partial object: {module_name}")
                            continue

                        # Make a copy of the annotations to avoid the error
                        annotations = getattr(module, "__annotations__").copy()
                        setattr(module, "__annotations__", annotations)
                    except Exception as e:
                        logger.debug(f"Could not copy annotations for {module_name}: {e}")
                        try:
                            # If we can't copy, try to set it to an empty dict
                            # But skip if it's a functools.partial
                            import functools
                            if isinstance(module, functools.partial):
                                logger.debug(f"Skipping immutable functools.partial object: {module_name}")
                                continue
                            setattr(module, "__annotations__", {})
                        except Exception as e:
                            logger.debug(f"Could not set empty annotations for {module_name}: {e}")

            logger.info("LiteLLM callbacks and logging disabled")
        except Exception as e:
            logger.warning(f"Could not fully patch LiteLLM: {e}")

        # Set the flag to indicate patches have been applied
        PATCHES_APPLIED = True
        logger.info("LiteLLM patches applied successfully")
    except Exception as e:
        # Check if it's the specific functools.partial error
        if "cannot set '__annotations__' attribute of immutable type 'functools.partial'" in str(e):
            logger.warning("Encountered functools.partial immutability error. This is expected and can be ignored.")
            # Set the flag to indicate patches have been partially applied
            PATCHES_APPLIED = True
            logger.info("LiteLLM patches partially applied (skipped functools.partial objects)")
        else:
            logger.error(f"Error applying LiteLLM patches: {e}")

def patch_dspy():
    """
    Apply patches to fix DSPy issues with LiteLLM.
    """
    try:
        # Try to import DSPy
        import dspy

        # Check if we need to patch DSPy's LiteLLM integration
        if hasattr(dspy, 'LM'):
            original_init = dspy.LM.__init__

            # Create a patched version of the LM.__init__ method
            def patched_init(self, *args, **kwargs):
                # Disable LiteLLM logging before initializing
                try:
                    # Set environment variable to disable LiteLLM logging
                    os.environ["LITELLM_LOG"] = "ERROR"  # Set to ERROR instead of NONE

                    # Import and patch LiteLLM
                    import litellm
                    litellm.success_callback = []
                    litellm.failure_callback = []

                    # Try to disable logging in different ways
                    # Method 1: Set verbose to False
                    if hasattr(litellm, "verbose"):
                        litellm.verbose = False

                    # Method 2: Set log level to a high value
                    if hasattr(litellm, "utils") and hasattr(litellm.utils, "logging"):
                        # Don't call disable_logging() as it might not exist
                        # Instead, try to set the log level directly
                        if hasattr(litellm.utils.logging, "setLevel"):
                            litellm.utils.logging.setLevel(logging.ERROR)
                except Exception:
                    pass

                # Call the original __init__
                original_init(self, *args, **kwargs)

            # Apply the patch
            dspy.LM.__init__ = patched_init
            logger.info("DSPy LM initialization patched to disable LiteLLM logging")

            # Also patch the LM.generate method if it exists
            if hasattr(dspy.LM, "generate"):
                original_generate = dspy.LM.generate

                def patched_generate(self, *args, **kwargs):
                    # Disable LiteLLM logging before generating
                    try:
                        os.environ["LITELLM_LOG"] = "ERROR"  # Set to ERROR instead of NONE

                        # Import and patch LiteLLM
                        import litellm
                        if hasattr(litellm, "verbose"):
                            litellm.verbose = False
                    except Exception:
                        pass

                    # Call the original generate method
                    return original_generate(self, *args, **kwargs)

                # Apply the patch
                dspy.LM.generate = patched_generate
                logger.info("DSPy LM.generate patched to disable LiteLLM logging")

    except ImportError:
        logger.warning("DSPy module not found, patch not applied")
    except Exception as e:
        logger.error(f"Error applying DSPy patch: {e}")

# Apply patches when the module is imported
try:
    apply_patch()
    patch_dspy()
except Exception as e:
    logger.error(f"Error applying patches on import: {e}")
