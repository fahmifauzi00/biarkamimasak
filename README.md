# Biar Kami Masak API

A fun and intelligent recipe recommendation API powered by LLMs that suggests recipes based on available ingredients and preferences. The name "Biar Kami Masak" means "Let Us Cook" in Malay.

## Overview

Biar Kami Masak API is a FastAPI-based service that leverages Large Language Models (LLMs) to provide personalized recipe recommendations. The system features a goofy, friendly chef persona that makes cooking more enjoyable by adding humor to the recipe recommendations while ensuring they remain practical and useful.

## Features

- **Simple Recipe Generation**: Get recipe recommendations based on available ingredients and desired servings
- **Detailed Recipe Generation**: Advanced recipe recommendations considering:
  - Available ingredients
  - Number of servings
  - Dietary restrictions
  - Cuisine preferences
  - Maximum cooking time
- **API Key Authentication**: Secure endpoints with API key validation
- **CORS Support**: Built-in Cross-Origin Resource Sharing support
- **Health Check Endpoint**: Monitor API health status
- **Error Handling**: Comprehensive error handling with informative messages

## Technology Stack

- FastAPI
- LangChain
- OpenAI GPT Models
- Python 3.x
- Pydantic for data validation
- python-dotenv for environment management

## Prerequisites

- Python 3.x
- OpenAI API key
- Recipe API key (for authentication)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/biarkamimasak.git
cd biarkamimasak
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with the following variables:
```env
OPENAI_API_KEY=your_openai_api_key
RECIPE_API_KEY=your_recipe_api_key
```

## Usage

### Starting the Server

Run the FastAPI server using uvicorn:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Root Endpoint
```http
GET /
```
Returns basic API information and welcome message.

#### 2. Health Check
```http
GET /health
```
Requires API key authentication. Returns API health status.

#### 3. Simple Recipe Recommendation
```http
POST /v1/recipe/simple
```
Request body:
```json
{
    "ingredients": ["chicken", "rice", "onion"],
    "servings": 2
}
```

#### 4. Detailed Recipe Recommendation
```http
POST /v1/recipe/detailed
```
Request body:
```json
{
    "ingredients": ["chicken", "rice", "onion"],
    "servings": 4,
    "dietary_restrictions": ["gluten-free"],
    "cuisine_preference": "asian",
    "cooking_time": 60
}
```

### Authentication

All API endpoints (except the root endpoint) require an API key to be included in the request headers:
```http
X-Recipe-API-Key: your_recipe_api_key
```

### Example Response

```json
{
    "title": "Goofy Chef's Amazing Chicken Rice",
    "recipe": "... detailed recipe content ...",
    "status": "success",
    "timestamp": "2024-11-13T12:44:36.278Z"
}
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- 403: Invalid API key
- 500: Internal server error with detailed message
- Other standard HTTP status codes as appropriate

## Development

### Project Structure
```
biarkamimasak/
├── main.py           # FastAPI application and endpoints
├── recommender.py    # Recipe recommendation logic
├── requirements.txt  # Project dependencies
└── README.md        # Project documentation
```

### Adding New Features

1. Modify the `RecipeRecommender` class in `recommender.py` to add new recommendation features
2. Update the API endpoints in `main.py` to expose new features
3. Update the documentation and tests accordingly

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Submit a pull request with a clear description of your changes

## License

[Add your chosen license here]

## Acknowledgments

- OpenAI for providing the LLM capabilities
- FastAPI for the excellent web framework
- LangChain for the LLM integration tools