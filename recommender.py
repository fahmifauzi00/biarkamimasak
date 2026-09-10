import os
import re
import time
import json
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import Optional, List, AsyncIterator
from datetime import datetime

class RecipeRecommender:
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 1000):
        """
        Initialize the RecipeRecommender with custom settings.
        
        Args:
            api_key (str, optional): API key (OpenRouter or OpenAI). If None, loads from environment.
            model (str, optional): Model to use for recommendations (defaults to google/gemma-4-31b-it:free)
            base_url (str, optional): Base URL for the API (defaults to https://openrouter.ai/api/v1)
            temperature (float): Temperature setting for response generation
            max_tokens (int): Maximum tokens in the response
        """
        # Load environment variables if needed
        load_dotenv()
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("No API key provided and neither OPENROUTER_API_KEY nor OPENAI_API_KEY found in environment variables")

        self.api_key = api_key
        self.model = model or os.getenv("OPENROUTER_MODEL") or "google/gemma-4-31b-it:free"
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.default_headers = {
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://biarkamimasak.vercel.app"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Biar Kami Masak"),
        }

        # Fallback model configuration (e.g. openrouter/free router)
        fallback_model = os.getenv("OPENROUTER_FALLBACK_MODEL", "openrouter/free")
        self.fallback_model = fallback_model

        # Extra body for OpenRouter (disable reasoning mode to ensure full recipe generation without token waste)
        extra_body = {}
        if "openrouter.ai" in self.base_url:
            models_list = [self.model]
            if fallback_model and fallback_model != self.model:
                models_list.append(fallback_model)
            extra_body = {
                "models": models_list,
                "route": "fallback",
                "reasoning": {"effort": "none"}
            }

        # Initialize the language model (pass extra_body directly to avoid UserWarning)
        self.llm = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens or 2000,
            default_headers=self.default_headers,
            extra_body=extra_body if extra_body else None,
        )

        # Fallback LLM instance for application-level failover on 429
        self.fallback_llm = None
        if fallback_model and fallback_model != self.model:
            fallback_extra_body = {"reasoning": {"effort": "none"}} if "openrouter.ai" in self.base_url else None
            try:
                self.fallback_llm = ChatOpenAI(
                    model=fallback_model,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens or 2000,
                    default_headers=self.default_headers,
                    extra_body=fallback_extra_body,
                )
            except Exception as fb_err:
                print(f"Notice: Fallback LLM initialization skipped: {fb_err}")

        # Define the recipe prompt template
        self.recipe_prompt = PromptTemplate(
            input_variables=["ingredients", "servings"],
            template="""You are a goofy chef and recipe recommender. You are very friendly, funny and helpful. 
            Given the following information:

            Main Ingredients Available: {ingredients}
            Servings: {servings}

            Please provide a recipe that uses these ingredients (you can suggest additional basic ingredients).
            Format your response EXACTLY as follows:

            TITLE: [Recipe Name]

            INGREDIENTS:
            - [ingredient 1 with quantity]
            - [ingredient 2 with quantity]
            ...

            INSTRUCTIONS:
            1. [First step]
            2. [Second step]
            ...

            COOKING TIME: [time in minutes]

            DIFFICULTY: [Easy/Medium/Hard]

            NOTES: [Add any helpful tips, substitutions, or humorous notes]

            Remember to be funny and engaging, but follow this EXACT format. Make some jokes in the notes section.
            Also reply with user's input language (Malay/English)."""
        )
        
        self.recipe_detailed_prompt = PromptTemplate(
            input_variables=["context"],
            template="""You are a goofy chef and recipe recommender. You are very friendly, funny and helpful. 
            Given the following information:

            {context}

            Please provide a recipe that:
            1. Uses the provided ingredients (additional basic ingredients can be suggested)
            2. Respects all dietary restrictions
            3. Matches the cuisine preference if specified
            4. Can be prepared within the time limit if specified

            Format your response EXACTLY as follows:

            TITLE: [Recipe Name]

            INGREDIENTS:
            - [ingredient 1 with quantity]
            - [ingredient 2 with quantity]
            ...

            INSTRUCTIONS:
            1. [First step]
            2. [Second step]
            ...

            COOKING TIME: [time in minutes]

            DIFFICULTY: [Easy/Medium/Hard]

            NOTES: [Add any helpful tips, substitutions, or humorous notes]

            Remember to be funny and engaging, but follow this EXACT format. Make some jokes in the notes section.
            Also reply with user's input language (Malay/English).
            """
        )

        # Create the recipe chain
        self.recipe_chain = self.recipe_prompt | self.llm
        self.recipe_detailed_chain = self.recipe_detailed_prompt | self.llm
        if self.fallback_llm:
            self.fallback_recipe_chain = self.recipe_prompt | self.fallback_llm
            self.fallback_recipe_detailed_chain = self.recipe_detailed_prompt | self.fallback_llm
        else:
            self.fallback_recipe_chain = None
            self.fallback_recipe_detailed_chain = None

    def _invoke_chain_with_fallback(self, chain, fallback_chain, input_data: dict, max_retries: int = 2):
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return chain.invoke(input_data)
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                is_rate_limited = "429" in err_str or "rate" in err_str or "temporarily" in err_str
                if is_rate_limited and attempt < max_retries:
                    sleep_time = (attempt + 1) * 2
                    print(f"Rate limited upstream on {self.model}. Retrying in {sleep_time}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(sleep_time)
                    continue
                break

        # If rate limited and fallback chain exists, try fallback
        if fallback_chain and ("429" in str(last_error).lower() or "rate" in str(last_error).lower()):
            try:
                print(f"Primary model {self.model} rate-limited, failing over to {self.fallback_model}...")
                return fallback_chain.invoke(input_data)
            except Exception as fb_err:
                print(f"Fallback model also failed: {fb_err}")

        raise last_error

    async def _stream_with_fallback(self, prompt: str):
        try:
            async for chunk in self.llm.astream(prompt):
                if hasattr(chunk, "content"):
                    yield str(chunk.content)
                elif isinstance(chunk, str):
                    yield chunk
        except Exception as e:
            err_str = str(e).lower()
            if self.fallback_llm and ("429" in err_str or "rate" in err_str):
                print(f"Primary streaming rate-limited, failing over to {self.fallback_model}...")
                async for chunk in self.fallback_llm.astream(prompt):
                    if hasattr(chunk, "content"):
                        yield str(chunk.content)
                    elif isinstance(chunk, str):
                        yield chunk
            else:
                raise
        
    def _extract_text_content(self, content) -> str:
        """Extract plain text string from str, list of content blocks, or dict."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return "\n".join(texts)
        return str(content) if content is not None else ""

    def extract_recipe_parts(self, raw_content) -> dict:
        """
        Extract and structure recipe components from the LLM output.
        Handles markdown, plain text, JSON, Malay and English headings, and thought tags.
        """
        recipe_text = self._extract_text_content(raw_content)

        if not recipe_text or not recipe_text.strip():
            print(f"WARNING: extract_recipe_parts received empty recipe_text (raw_content was: {repr(raw_content)})")
            return {
                'title': "Cadangan Resepi Masakan",
                'ingredients': [],
                'instructions': [],
                'cooking_time': "Not specified",
                'difficulty': "Not specified",
                'notes': "No recipe content generated. Please try again."
            }

        print(f"DEBUG: recipe output length={len(recipe_text)}, snippet={repr(recipe_text[:200])}")

        # Remove thinking/reasoning tags if present
        clean_text = re.sub(r'<(thought|think)>.*?</\1>', '', recipe_text, flags=re.DOTALL).strip()
        if not clean_text:
            clean_text = recipe_text.strip()
        else:
            recipe_text = clean_text

        # 1. Check if model returned JSON
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', recipe_text, flags=re.DOTALL)
        possible_json = json_match.group(1) if json_match else recipe_text.strip()
        if possible_json.startswith('{') and possible_json.endswith('}'):
            try:
                data = json.loads(possible_json)
                if isinstance(data, dict):
                    title = data.get('title') or data.get('recipe_name') or data.get('tajuk') or ""
                    ingredients = data.get('ingredients') or data.get('bahan_bahan') or data.get('bahan') or []
                    instructions = data.get('instructions') or data.get('steps') or data.get('arahan') or data.get('cara_cara') or []
                    cooking_time = data.get('cooking_time') or data.get('time') or data.get('masa_memasak') or "Not specified"
                    difficulty = data.get('difficulty') or data.get('tahap_kesukaran') or "Easy"
                    notes = data.get('notes') or data.get('nota') or "No additional notes."

                    if isinstance(ingredients, str):
                        ingredients = [i.strip() for i in ingredients.split('\n') if i.strip()]
                    if isinstance(instructions, str):
                        instructions = [i.strip() for i in instructions.split('\n') if i.strip()]

                    if title or ingredients or instructions:
                        return {
                            'title': str(title).strip(),
                            'ingredients': [str(i) for i in ingredients],
                            'instructions': [str(i) for i in instructions],
                            'cooking_time': str(cooking_time).strip(),
                            'difficulty': str(difficulty).strip(),
                            'notes': str(notes).strip()
                        }
            except Exception as json_err:
                print(f"DEBUG: JSON parse skipped: {json_err}")

        # 2. Text line-by-line parsing
        # Strip markdown bold/headers for matching
        clean_lines = recipe_text.replace('**', '').replace('__', '').replace('#', '').strip().split('\n')

        title = ""
        ingredients = []
        instructions = []
        cooking_time = ""
        difficulty = ""
        notes = ""

        current_section = None

        # Regex patterns supporting English & Malay
        title_patterns = r'^(?:TITLE|TAJUK|RECIPE NAME|NAMA RESEPI)\s*[:=\-]\s*(.*)$'
        ingredients_patterns = r'^(?:INGREDIENTS|BAHAN[\- ]BAHAN|BAHAN)\s*[:=\-]?\s*$'
        instructions_patterns = r'^(?:INSTRUCTIONS|ARAHAN|LANGKAH[\- ]LANGKAH|CARA[\- ]CARA|STEPS|METHOD|DIRECTIONS)\s*[:=\-]?\s*$'
        cooking_time_patterns = r'^(?:COOKING TIME|MASA MEMASAK|MASA|TIME)\s*[:=\-]\s*(.*)$'
        difficulty_patterns = r'^(?:DIFFICULTY|TAHAP KESUKARAN|KESUKARAN)\s*[:=\-]\s*(.*)$'
        notes_patterns = r'^(?:NOTES|NOTA|TIPS|CATATAN)\s*[:=\-]?\s*(.*)$'

        for raw_line in clean_lines:
            line = raw_line.strip()
            if not line:
                continue

            line_upper = line.upper()

            # Check title
            m = re.match(title_patterns, line, flags=re.IGNORECASE)
            if m:
                title = m.group(1).strip()
                current_section = None
                continue
            elif 'TITLE:' in line_upper or 'TAJUK:' in line_upper:
                title = re.sub(r'^(?:TITLE|TAJUK)\s*[:=\-]\s*', '', line, flags=re.IGNORECASE).strip()
                current_section = None
                continue

            # Check cooking time
            m = re.match(cooking_time_patterns, line, flags=re.IGNORECASE)
            if m:
                cooking_time = m.group(1).strip()
                continue

            # Check difficulty
            m = re.match(difficulty_patterns, line, flags=re.IGNORECASE)
            if m:
                difficulty = m.group(1).strip()
                continue

            # Check ingredients section
            if re.match(ingredients_patterns, line, flags=re.IGNORECASE) or line_upper.startswith('INGREDIENTS:') or line_upper.startswith('BAHAN-BAHAN:'):
                current_section = 'ingredients'
                continue

            # Check instructions section
            if re.match(instructions_patterns, line, flags=re.IGNORECASE) or line_upper.startswith('INSTRUCTIONS:') or line_upper.startswith('ARAHAN:'):
                current_section = 'instructions'
                continue

            # Check notes section
            m = re.match(notes_patterns, line, flags=re.IGNORECASE)
            if m or line_upper.startswith('NOTES:') or line_upper.startswith('NOTA:'):
                current_section = 'notes'
                if m and m.group(1).strip():
                    notes = m.group(1).strip()
                else:
                    notes = re.sub(r'^(?:NOTES|NOTA|TIPS|CATATAN)\s*[:=\-]?\s*', '', line, flags=re.IGNORECASE).strip()
                continue

            # Content collection
            if current_section == 'ingredients':
                item = re.sub(r'^[-*•\d+.]\s*', '', line).strip()
                if item and not any(k in line_upper for k in ['INSTRUCTIONS:', 'ARAHAN:', 'COOKING TIME:', 'DIFFICULTY:', 'NOTES:']):
                    ingredients.append(item)
            elif current_section == 'instructions':
                item = re.sub(r'^(?:Step\s*\d+|\d+)[.)\-:]\s*', '', line, flags=re.IGNORECASE).strip()
                if item and not any(k in line_upper for k in ['COOKING TIME:', 'DIFFICULTY:', 'NOTES:', 'NOTA:']):
                    instructions.append(item)
            elif current_section == 'notes':
                if notes:
                    notes += " " + line
                else:
                    notes = line

        # Heuristic fallbacks if sections were not recognized
        if not title and clean_lines:
            title = re.sub(r'^[-*•#\d+.]\s*', '', clean_lines[0]).strip()

        if not ingredients and not instructions:
            for line in clean_lines:
                if line.startswith(('-', '*', '•')):
                    ingredients.append(re.sub(r'^[-*•]\s*', '', line).strip())
                elif re.match(r'^\d+[.)]\s*', line):
                    instructions.append(re.sub(r'^\d+[.)]\s*', '', line).strip())

        # Default fallbacks
        if not notes:
            notes = "No additional notes."
        if not difficulty:
            difficulty = "Easy"
        if not cooking_time:
            cooking_time = "30 minutes"
        if not title:
            title = "Delicious Recipe"

        print(f"DEBUG: Parsed recipe: title='{title}', ingredients={len(ingredients)}, instructions={len(instructions)}")

        return {
            'title': title,
            'ingredients': ingredients,
            'instructions': instructions,
            'cooking_time': cooking_time,
            'difficulty': difficulty,
            'notes': notes
        }
    
    # Simple recipe
    def get_recipe(self, ingredients: List[str], servings: int = 2) -> dict:
        """
        Get a structured recipe recommendation based on the provided ingredients.
    
        Args:
            ingredients (List[str]): List of available ingredients
            servings (int): Number of servings (default: 2)
    
        Returns:
            dict: Structured recipe data
        """
        ingredients_str = ", ".join(ingredients)
    
        response = self._invoke_chain_with_fallback(
            chain=self.recipe_chain,
            fallback_chain=self.fallback_recipe_chain,
            input_data={
                "ingredients": ingredients_str,
                "servings": servings
            }
        )
    
        recipe_data = self.extract_recipe_parts(response.content)
        recipe_data['timestamp'] = datetime.now()
        return recipe_data
    
    # Detailed recipe
    def get_recipe_with_parameters(self,
                               ingredients: List[str],
                               servings: Optional[int] = 2,
                               dietary_restrictions: Optional[list] = None,
                               cuisine_preference: Optional[str] = None,
                               cooking_time: Optional[int] = None) -> dict:
        """
        Get a structured recipe recommendation based on the provided parameters.
        """
        ingredients_str = ", ".join(ingredients)

        # Build the context
        context_parts = [
            f"Main Ingredients Available: {ingredients_str}",
            f"Servings: {servings or 2}"
        ]

        if dietary_restrictions:
            context_parts.append(f"Dietary Restrictions: {', '.join(dietary_restrictions)}")
        if cuisine_preference:
            context_parts.append(f"Cuisine Preference: {cuisine_preference}")
        if cooking_time:
            context_parts.append(f"Maximum Cooking Time: {cooking_time} minutes")

        full_context = "\n".join(context_parts)

        response = self._invoke_chain_with_fallback(
            chain=self.recipe_detailed_chain,
            fallback_chain=self.fallback_recipe_detailed_chain,
            input_data={"context": full_context}
        )
        recipe_data = self.extract_recipe_parts(response.content)
        recipe_data['timestamp'] = datetime.now()
        return recipe_data
    
    async def get_recipe_stream(self, ingredients: List[str], servings: int = 2) -> AsyncIterator[str]:
        """
        Get a streaming recipe recommendation based on the provided ingredients.
        """
        prompt = self.recipe_prompt.format(
            ingredients=", ".join(ingredients),
            servings=servings
        )

        try:
            async for token in self._stream_with_fallback(prompt):
                yield token
        except Exception as e:
            print(f"Streaming error: {e}")
            raise

    async def get_recipe_with_parameters_stream(
        self,
        ingredients: List[str],
        servings: Optional[int] = 2,
        dietary_restrictions: Optional[list] = None,
        cuisine_preference: Optional[str] = None,
        cooking_time: Optional[int] = None
    ) -> AsyncIterator[str]:
        """
        Get a streaming recipe recommendation with detailed parameters.
        """
        # Build the context
        context_parts = [
            f"Main Ingredients Available: {', '.join(ingredients)}",
            f"Servings: {servings or 2}"
        ]

        if dietary_restrictions:
            context_parts.append(f"Dietary Restrictions: {', '.join(dietary_restrictions)}")
        if cuisine_preference:
            context_parts.append(f"Cuisine Preference: {cuisine_preference}")
        if cooking_time:
            context_parts.append(f"Maximum Cooking Time: {cooking_time} minutes")

        full_context = "\n".join(context_parts)
        prompt = self.recipe_detailed_prompt.format(context=full_context)

        try:
            async for token in self._stream_with_fallback(prompt):
                yield token
        except Exception as e:
            print(f"Streaming error: {e}")
            raise