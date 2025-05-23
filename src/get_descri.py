from crawl4ai import (
    AsyncWebCrawler, 
    CrawlerRunConfig,
    LLMConfig)

from crawl4ai.extraction_strategy import LLMExtractionStrategy  
from pydantic import BaseModel, Field
import json
import asyncio
import os
import re

async def main() -> None:
    """
    Web scraper for extracting disability support information from French university websites.
    
    This script performs the following operations:
    1. Configures an LLM (Language Learning Model) for intelligent content extraction
    2. Sets up an asynchronous web scraper with JavaScript rendering capabilities
    3. Visits specific university pages about handicap/disability services
    4. Extracts structured information using AI-powered content analysis
    5. Saves the extracted data as JSON files for each university
    
    Requirements:
    - Environment variable 'Deepseek_API' must be set with a valid API key
    - Python packages: crawl4ai, pydantic, asyncio
    
    For detailed documentation: https://docs.crawl4ai.com/
    """
    # Initialize LLM configuration with Deepseek API credentials
    # The API key should be stored as an environment variable for security
    api = os.getenv("Deepseek_API")
    provider = "deepseek/deepseek-chat"
    llm_config = LLMConfig(provider=provider, api_token=api)


    class LLM(BaseModel):
        """
        Pydantic model defining the structure of extracted content.
        
        Each extracted item will contain:
        - title: The heading or section title describing the handicap service
        - description: The detailed content/paragraph about the service
        
        This schema ensures consistent output format from the LLM extraction.
        """
        title: str = Field(...,description="Title for each paragraph")
        description: str = Field(...,description="Paragraph")

    # Configure the LLM extraction strategy with specific instructions
    # for identifying and extracting handicap-related content
    llm_strategy = LLMExtractionStrategy(
        llm_config=llm_config,
        schema=LLM.model_json_schema(), # Define expected output structure
        extraction_type="schema",
        verbose=True,  # Enable detailed logging for debugging
        instruction="""Extract ALL content about handicap/disability services for students.
        For each section or paragraph you find, create an item with 'title' and 'description'.
        Include contact information, procedures, services, and any related content.
        Return as a list called 'items' where each item has 'title' and 'description' keys."""
    )


    # Configure the web scraper behavior for optimal content extraction
    crawler_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        delay_before_return_html=2, # Wait 2 seconds to avoid being lacked
        wait_for="body",  # Ensure the page body is fully loaded before extraction
        js_code="""
        // JavaScript code executed on the page to handle dynamic content
        // Scrolls to bottom to trigger lazy-loaded content
        window.scrollTo(0, document.body.scrollHeight);
        """
    )

    # Initialize asynchronous web scraper with context manager for proper resource handling
    async with AsyncWebCrawler() as crawler:
        # List of target university URLs containing handicap/disability information
        universities = ["https://www.nanterre.fr/annuaires/catalogue-des-demarches/detail/handicaps-au-quotidien", 
                        "https://www.inalco.fr/etudiant-en-situation-de-handicap", \
                        "https://www.sorbonne-nouvelle.fr/qu-est-ce-que-le-handicap--402898.kjsp?RH=1474288330206"]
        
        # Regex pattern to extract university name from URL
        # Captures the subdomain (e.g., 'nanterre' from 'www.nanterre.fr')
        pattern = re.compile(r"www\.([\w\-]+)\.fr")

        # Process each university URL sequentially
        for uni in universities:
            # Execute web scraping with configured settings
            result = await crawler.arun(url=uni,config=crawler_config)
            
            # Parse the JSON string returned by LLM extraction into Python object
            json_Data = json.loads(result.extracted_content)
            
            # Extract university name from URL for filename generation
            name = re.search(pattern, uni)
            
            # Save extracted data to JSON file named after the university
            with open(f"{name.group(1)}.json", "w") as f:
                json.dump(json_Data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())