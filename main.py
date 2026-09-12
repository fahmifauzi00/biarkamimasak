import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from recommender import RecipeRecommender
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

load_dotenv()

RECIPE_API_KEY = os.getenv("RECIPE_API_KEY")
if not RECIPE_API_KEY:
    print("WARNING: RECIPE_API_KEY is not set in environment variables. API requests requiring authentication will fail until set.")

# API Key security scheme
api_key_header = APIKeyHeader(name="X-Recipe-API-Key", auto_error=True)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not RECIPE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: RECIPE_API_KEY is not set on the server.",
        )
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

# Origins allowed to call this API from a browser.
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:4173",
    "https://biarkamimasak.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "X-Recipe-API-Key"],
    expose_headers=["Content-Type"],
)

recommender: Optional[RecipeRecommender] = None

def get_recommender() -> RecipeRecommender:
    global recommender
    if recommender is None:
        try:
            recommender = RecipeRecommender()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Recommender initialization error: {str(e)}. Check OPENROUTER_API_KEY in server environment.",
            )
    return recommender

try:
    recommender = RecipeRecommender()
except Exception as e:
    print(f"Notice: RecipeRecommender deferred initialization: {e}")
    
    
class SimpleQuery(BaseModel):
    ingredients: List[str]
    servings: Optional[int] = 2
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ingredients": ["chicken", "onion", "rice"],
                "servings": 2
            }
        }
    )
    
    
class DetailedQuery(BaseModel):
    ingredients: List[str]
    servings: Optional[int] = 2
    dietary_restrictions: Optional[List[str]] = None
    cuisine_preference: Optional[str] = None
    cooking_time: Optional[int] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ingredients": ["chicken", "onion", "rice"],
                "servings": 2,
                "dietary_restrictions": ["diabetes"],
                "cuisine_preference": "asian",
                "cooking_time": 30
            }
        }
    )
    
        
class RecipeResponse(BaseModel):
    title: str
    ingredients: List[str]
    instructions: List[str]
    cooking_time: str
    difficulty: str
    notes: str
    timestamp: Optional[datetime] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Chicken Curry",
                "ingredients": ["2 chicken breasts", "1 onion", "1 cup rice"],
                "instructions": ["Cook the chicken", "Add the onions", "Serve with rice"],
                "cooking_time": "30 minutes",
                "difficulty": "easy",
                "notes": "Add some soy sauce",
            }
        }
    )
    
    
# Root endpoint
@app.get('/')
def root(request: Request):
    client_host = request.client.host if request.client else None
    return {
        "message": "Selamat datang ke 'Biar Kami Masak API'!",
        "version": "1.0.0",
        "client_host": client_host
    }

# Health check endpoint (unauthenticated for Railway / container health checks)
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

def handle_endpoint_exception(e: Exception):
    if isinstance(e, HTTPException):
        raise e
    err_str = str(e)
    if "429" in err_str or "rate-limit" in err_str.lower() or "rate limit" in err_str.lower():
        raise HTTPException(
            status_code=429,
            detail=f"AI model provider is temporarily rate-limited. Please retry in a few moments. Details: {err_str}"
        )
    raise HTTPException(status_code=500, detail=err_str)

# Simple query endpoint
@app.post("/v1/recipe/simple", response_model=RecipeResponse)
async def get_recipe_simple(
    query: SimpleQuery,
    api_key: str = Security(api_key_header)
):
    try:
        rec = get_recommender()
        recipe_data = rec.get_recipe(
            ingredients=query.ingredients,
            servings=query.servings
        )
        return RecipeResponse(**recipe_data)
    except Exception as e:
        handle_endpoint_exception(e)
    
# Detailed query endpoint
@app.post("/v1/recipe/detailed", response_model=RecipeResponse)
async def get_recipe_detailed(
    query: DetailedQuery,
    api_key: str = Security(api_key_header)
):
    try:
        rec = get_recommender()
        recipe_data = rec.get_recipe_with_parameters(
            ingredients=query.ingredients,
            servings=query.servings,
            dietary_restrictions=query.dietary_restrictions,
            cuisine_preference=query.cuisine_preference,
            cooking_time=query.cooking_time
        )
        return RecipeResponse(**recipe_data)
    except Exception as e:
        handle_endpoint_exception(e)
    
# Streaming recipe endpoint
@app.post("/v1/recipe/simple/stream")
async def get_recipe_simple_stream(
    query: SimpleQuery,
    api_key: str = Security(api_key_header)
):
    async def generate():
        try:
            rec = get_recommender()
            async for token in rec.get_recipe_stream(
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
            rec = get_recommender()
            async for token in rec.get_recipe_with_parameters_stream(
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
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail
        },
        headers=getattr(exc, "headers", None)
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": str(exc)
        }
    )

if __name__ == "__main__":
    import uvicorn
    raw_port = os.getenv("PORT", "8000")
    try:
        port = int(raw_port)
    except ValueError:
        print(f"WARNING: Invalid PORT environment variable '{raw_port}', falling back to port 8000")
        port = 8000
    uvicorn.run("main:app", host="0.0.0.0", port=port)