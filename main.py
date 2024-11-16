import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from recommender import RecipeRecommender
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

load_dotenv()

RECIPE_API_KEY = os.getenv("RECIPE_API_KEY")
if not RECIPE_API_KEY:
    raise Exception("RECIPE_API_KEY not found in environment variables")

# API Key security scheme
api_key_header = APIKeyHeader(name="X-Recipe-API-Key", auto_error=True)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == RECIPE_API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=403,
        detail="Could not validate API Key",
        headers={"WWW-Authenticate": "API key"},
    )

app = FastAPI(
    title="Biar Kami Masak API",
    description="API untuk dapatkan cadangan resepi masakan dengan LLM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    recommender = RecipeRecommender()
except ValueError as e:
    print(f"Error initializing RecipeRecommender: {e}")
    
    
class SimpleQuery(BaseModel):
    ingredients: List[str]
    servings: Optional[int] = 2
    
    class Config:
        json_schema_extra = {
            "example": {
                "ingredients": ["chicken", "onion", "rice"],
                "servings": 2
            }
        }
    
    
class DetailedQuery(BaseModel):
    ingredients: List[str]
    servings: Optional[int] = 2
    dietary_restrictions: Optional[List[str]] = None
    cuisine_preference: Optional[str] = None
    cooking_time: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "ingredients": ["chicken", "onion", "rice"],
                "servings": 2,
                "dietary_restrictions": ["diabetes"],
                "cuisine_preference": "asian",
                "cooking_time": 30
            }
        }
    
        
class RecipeResponse(BaseModel):
    title: str
    ingredients: List[str]
    instructions: List[str]
    cooking_time: str
    difficulty: str
    notes: str
    timestamp: Optional[datetime] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Chicken Curry",
                "ingredients": ["2 chicken breasts", "1 onion", "1 cup rice",],
                "instructions": ["Cook the chicken", "Add the onions", "Serve with rice"],
                "cooking_time": "30 minutes",
                "difficulty": "easy",
                "notes": "Add some soy sauce",
            }
        }
    
    
# Root endpoint
@app.get('/')
def root(request: Request):
    client_host = request.client.host if request.client else None
    return {
        "message": "Selamat datang ke 'Biar Kami Masak API'!",
        "version": "1.0.0",
        "client_host": client_host
    }

# Health check endpoint
@app.get("/health", dependencies=[Depends(get_api_key)])
async def health_check():
    return {"status": "healthy"}

# Simple query endpoint
@app.post("/v1/recipe/simple", response_model=RecipeResponse)
async def get_recipe_simple(
    query: SimpleQuery,
    api_key: str = Security(api_key_header)
):
    try:
        recipe_data = recommender.get_recipe(
            ingredients=query.ingredients,
            servings=query.servings
        )
        return RecipeResponse(**recipe_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Detailed query endpoint
@app.post("/v1/recipe/detailed", response_model=RecipeResponse)
async def get_recipe_detailed(
    query: DetailedQuery,
    api_key: str = Security(api_key_header)
):
    try:
        recipe_data = recommender.get_recipe_with_parameters(
            ingredients=query.ingredients,
            servings=query.servings,
            dietary_restrictions=query.dietary_restrictions,
            cuisine_preference=query.cuisine_preference,
            cooking_time=query.cooking_time
        )
        return RecipeResponse(**recipe_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Streaming recipe endpoint
@app.post("/v1/recipe/simple/stream")
async def get_recipe_simple_stream(
    query: SimpleQuery,
    api_key: str = Security(api_key_header)
):
    async def generate():
        try:
            async for token in recommender.get_recipe_stream(
                ingredients=query.ingredients,
                servings=query.servings
            ):
                yield token
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

@app.post("/v1/recipe/detailed/stream")
async def get_recipe_detailed_stream(
    query: DetailedQuery,
    api_key: str = Security(api_key_header)
):
    async def generate():
        try:
            async for token in recommender.get_recipe_with_parameters_stream(
                ingredients=query.ingredients,
                servings=query.servings,
                dietary_restrictions=query.dietary_restrictions,
                cuisine_preference=query.cuisine_preference,
                cooking_time=query.cooking_time
            ):
                yield token
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
    
# Error handling
@app.exception_handler(HTTPException)
async def generic_exception_handler(request, exc):
    return {
        "status": "error",
        "message": str(exc)
    }