import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import Optional, List

load_dotenv()
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
            template="""You are a goofy chef and recipe recommender. You are very friendly, funny and helpful. Given the following information:
            
            Main Ingredients Available: {ingredients}
            Servings: {servings}
            
            Recommend a detailed recipe that:
            1. Uses the provided ingredients (additional basic ingredients can be suggested)
            2. Respects all dietary restrictions
            3. Matches the cuisine preference if specified
            4. Can be prepared within the time limit if specified
            
            Provide the recipe in the following format:
            - Title
            - Complete ingredients list with measurements
            - Step-by-step instructions
            - Cooking time
            - Difficulty level
            
            Reply with humorous and helpful responses. Make some jokes where appropriate. Also reply with users input language.
            """
        )
        
        self.recipe_detailed_prompt = PromptTemplate(
            input_variables=["servings", "ingredients", "diery_restrictions", "cuisine_preference", "cooking_time"],
            template="""
            You are a goofy chef and recipe recommender. You are very friendly, funny and helpful. Given the following information:
            
            {context}
            
            Recommend a detailed recipe that:
            1. Uses the provided ingredients (additional basic ingredients can be suggested)
            2. Respects all dietary restrictions
            3. Matches the cuisine preference if specified
            4. Can be prepared within the time limit if specified
            
            Provide the recipe in the following format:
            - Title
            - Complete ingredients list with measurements
            - Step-by-step instructions
            - Cooking time
            - Difficulty level
            
            Reply with humorous and helpful responses. Make some jokes where appropriate. Also reply with users input language.
            """
        )

        # Create the recipe chain
        self.recipe_chain = self.recipe_prompt | self.llm
        self.recipe_detailed_chain = self.recipe_detailed_prompt | self.llm
        
    # To extract title and main content    
    def extract_recipe_parts(self, recipe_text: str) -> tuple:
        """
        Extract title and content from the recipe text.
        
        Args:
            recipe_text (str): Complete recipe text from LLM
        
        Returns:
            tuple: (title, content)
        """
        lines = recipe_text.strip().split('\n')
        title = ""
        content = recipe_text
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('-'):
                title = line.replace('Title:', '').strip()
                break
        
        if not title:
            # Fallback: use first line if no clear title found
            title = lines[0].strip()
        
        return title, content
    
    # Simple recipe
    def get_recipe(self, ingredients: List[str], servings: int = 2) -> tuple:
        """
        Get a recipe recommendation based on the provided ingredients.
        
        Args:
            ingredients (List[str]): List of available ingredients
            servings (int): Number of servings (default: 2)
        
        Returns:
            tuple: (title, content) of the recipe
        """
        ingredients_str = ", ".join(ingredients)
        
        response = self.recipe_chain.invoke({
            "ingredients": ingredients_str,
            "servings": servings
        })
        return self.extract_recipe_parts(response.content)
    
    # Detailed recipe
    def get_recipe_with_parameters(self,
                                   ingredients: List[str],
                                   servings: Optional[int] = 2,
                                   dietary_restrictions: Optional[list] = None,
                                   cuisine_preference: Optional[str] = None,
                                   cooking_time: Optional[int] = None) -> str:
        """
        Get a recipe recommendation based on the provided parameters.
        
        Args:
            servings (int): Number of people to serve
            dietary_restrictions (list, optional): List of dietary restrictions
            cuisine_preference (str, optional): Preferred cuisine type
            cooking_time (int, optional): Cooking time in minutes
            ingredients (list, optional): List of ingredients to use
        
        Returns:
            str: Detailed recipe recommendation
        """
        ingredients_str = ", ".join(ingredients)
    
        # Build the context for the prompt
        context_parts = [
            f"Main Ingredients Available: {ingredients_str}",
            f"Servings: {servings or 2}"
        ]
    
        if dietary_restrictions:
            context_parts.append(f"Dietary Restrictions: {', '.join(dietary_restrictions)}")
        else:
            context_parts.append("Dietary Restrictions: None specified")
        
        if cuisine_preference:
            context_parts.append(f"Cuisine Preference: {cuisine_preference}")
        else:
            context_parts.append("Cuisine Preference: Any")
        
        if cooking_time:
            context_parts.append(f"Maximum Cooking Time: {cooking_time} minutes")
        else:
            context_parts.append("Cooking Time: Not specified")
    
        full_context = "\n".join(context_parts)
    
    
        response = self.recipe_detailed_chain.invoke({"context": full_context})
        return self.extract_recipe_parts(response.content)