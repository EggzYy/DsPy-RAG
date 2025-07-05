"""
Command-line tool to optimize document metadata.
"""

import argparse
import logging
from .metadata_optimizer import optimize_document_metadata, add_additional_metadata_fields

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """Run the metadata optimization."""
    parser = argparse.ArgumentParser(description="Optimize document metadata.")
    parser.add_argument("--optimize", action="store_true", help="Remove unnecessary metadata fields")
    parser.add_argument("--enhance", action="store_true", help="Add additional metadata fields")
    parser.add_argument("--all", action="store_true", help="Run both optimization and enhancement")
    parser.add_argument("--remove-content", action="store_true", help="Remove content field entirely")
    parser.add_argument("--keep-excerpt", action="store_true", help="Keep a small excerpt of the content")
    parser.add_argument("--excerpt-length", type=int, default=500, help="Length of the excerpt to keep (in characters)")

    args = parser.parse_args()

    # Determine whether to keep an excerpt
    keep_excerpt = True
    if args.remove_content:
        keep_excerpt = False
    elif args.keep_excerpt:
        keep_excerpt = True

    if args.all or args.optimize:
        logger.info("Running metadata optimization...")
        logger.info(f"Content handling: {'Keeping excerpt' if keep_excerpt else 'Removing entirely'}")

        result = optimize_document_metadata(
            keep_excerpt=keep_excerpt,
            excerpt_length=args.excerpt_length
        )

        logger.info(f"Metadata optimization complete: {result['status']}")
        logger.info(f"Processed {result.get('total_documents_processed', 0)} documents")
        logger.info(f"Saved {result.get('human_readable_bytes_saved', '0B')}")
        logger.info(f"Removed fields: {', '.join(result.get('fields_removed', []))}")

    if args.all or args.enhance:
        logger.info("Running metadata enhancement...")
        result = add_additional_metadata_fields()
        logger.info(f"Metadata enhancement complete: {result['status']}")
        logger.info(f"Processed {result.get('total_documents_processed', 0)} documents")
        logger.info(f"Added fields: {', '.join(result.get('fields_added', []))}")

    if not (args.all or args.optimize or args.enhance):
        parser.print_help()

if __name__ == "__main__":
    main()
