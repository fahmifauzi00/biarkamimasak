import os
import re
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.schema import HumanMessage
from dotenv import load_dotenv
from typing import Optional, List, AsyncIterator
from datetime import datetime

class RecipeRecommender:
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.7,
                 max_tokens: int = 1000):
        """
        Initialize the RecipeRecommender with custom settings.
        
        Args:
            api_key (str, optional): OpenAI API key. If None, loads from environment.
            model (str): Model to use for recommendations
            base_url (str): Base URL for the API
            temperature (float): Temperature setting for response generation
            max_tokens (int): Maximum tokens in the response
        """
        # Load environment variables if no API key provided
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("No API key provided and none found in environment variables")

        # Initialize the language model
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

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
        
        # To extract title and main content    
    def extract_recipe_parts(self, recipe_text: str) -> dict:
        """
        Extract and structure recipe components from the LLM output.
    
        Args:
            recipe_text (str): Complete recipe text from LLM
    
        Returns:
            dict: Structured recipe data
        """
        # Remove markdown formatting
        recipe_text = recipe_text.replace('*', '').replace('#', '')
    
        # Initialize components
        title = ""
        ingredients = []
        instructions = []
        cooking_time = ""
        difficulty = ""
        notes = ""
    
        # Split into sections
        current_section = None
        lines = recipe_text.strip().split('\n')
    
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract title
            if 'TITLE:' in line.upper():
                title = re.sub(r'^TITLE:\s*', '', line, flags=re.IGNORECASE)
                continue
            
            # Identify sections
            if 'INGREDIENTS:' in line.upper():
                current_section = 'ingredients'
                continue
            elif 'INSTRUCTIONS:' in line.upper():
                current_section = 'instructions'
                continue
            elif 'COOKING TIME:' in line.upper():
                cooking_time = re.sub(r'^COOKING TIME:\s*', '', line, flags=re.IGNORECASE)
                continue
            elif 'DIFFICULTY:' in line.upper():
                difficulty = re.sub(r'^DIFFICULTY:\s*', '', line, flags=re.IGNORECASE)
                continue
            elif 'NOTES:' in line.upper():
                current_section = 'notes'
                notes = re.sub(r'^NOTES:\s*', '', line, flags=re.IGNORECASE)
                continue
            
            # Process line based on current section
            if current_section == 'ingredients' and line.strip():
                if line.startswith('-'):
                    ingredients.append(line.replace('-', '').strip())
                elif line.strip() and not any(section in line.upper() for section in ['TITLE:', 'INSTRUCTIONS:', 'COOKING TIME:', 'DIFFICULTY:', 'NOTES:']):
                    ingredients.append(line.strip())
                
            elif current_section == 'instructions':
                # Remove numbering and clean up
                if line.strip():
                    # Check if line starts with a number followed by a dot
                    match = re.match(r'^\d+\.\s*(.+)$', line)
                    if match:
                        instructions.append(match.group(1).strip())
                    elif not any(section in line.upper() for section in ['TITLE:', 'INGREDIENTS:', 'COOKING TIME:', 'DIFFICULTY:', 'NOTES:']):
                        instructions.append(line.strip())
                    
            elif current_section == 'notes' and line.strip():
                if notes:
                    notes += " " + line.strip()
                else:
                    notes = line.strip()

        # Set default values if sections are empty
        if not notes:
            notes = "No additional notes."
        if not difficulty:
            difficulty = "Not specified"
        if not cooking_time:
            cooking_time = "Not specified"

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
    
        response = self.recipe_chain.invoke({
            "ingredients": ingredients_str,
            "servings": servings
        })
    
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

        response = self.recipe_detailed_chain.invoke({"context": full_context})
        recipe_data = self.extract_recipe_parts(response.content)
        recipe_data['timestamp'] = datetime.now()
        return recipe_data
    
    async def get_recipe_stream(self, ingredients: List[str], servings: int = 2) -> AsyncIterator[str]:
        """
        Get a streaming recipe recommendation based on the provided ingredients.
        """
        callback = AsyncIteratorCallbackHandler()
        llm = ChatOpenAI(
            streaming=True,
            callbacks=[callback],
            model=self.llm.model_name,
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
        )

        prompt = self.recipe_prompt.format(
            ingredients=", ".join(ingredients),
            servings=servings
        )

        task = asyncio.create_task(
            llm.agenerate([[HumanMessage(content=prompt)]])
        )

        try:
            async for token in callback.aiter():
                yield token
        except Exception as e:
            print(f"Streaming error: {e}")
            raise
        finally:
            callback.done.set()

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
        callback = AsyncIteratorCallbackHandler()
        llm = ChatOpenAI(
            streaming=True,
            callbacks=[callback],
            model=self.llm.model_name,
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
        )

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

        task = asyncio.create_task(
            llm.agenerate([[HumanMessage(content=prompt)]])
        )

        try:
            async for token in callback.aiter():
                yield token
        except Exception as e:
            print(f"Streaming error: {e}")
            raise
        finally:
            callback.done.set()