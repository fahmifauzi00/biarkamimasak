# Biar Kami Masak 🧑‍🍳

A fun and intelligent recipe recommendation system powered by LLMs that suggests recipes based on available ingredients and preferences. The name "Biar Kami Masak" means "Let Us Cook" in Malay.

## 🌟 Live Demo

Visit our web application: [https://biarkamimasak.vercel.app/](https://biarkamimasak.vercel.app/)

## 🔍 Overview

Biar Kami Masak is a full-stack application consisting of:
- A FastAPI backend service deployed on Railway
- A frontend web application hosted on Vercel
- Integration with OpenAI's GPT models for intelligent recipe generation

The system features a goofy, friendly chef persona that makes cooking more enjoyable by adding humor to the recipe recommendations while ensuring they remain practical and useful.

## ✨ Features

### Core Features
- **Ingredient-Based Recipe Generation**: Get personalized recipes based on your available ingredients
- **Smart Recipe Recommendations**: Takes into account:
  - Number of servings
  - Dietary restrictions
  - Cuisine preferences
  - Maximum cooking time
- **Streaming Responses**: Real-time recipe generation with streamed responses for better user experience
- **Bilingual Support**: Handles both English and Malay inputs

### Technical Features
- **API Security**: Protected endpoints with API key authentication
- **CORS Support**: Built-in Cross-Origin Resource Sharing support
- **Health Monitoring**: Dedicated health check endpoint
- **Error Handling**: Comprehensive error handling with informative messages
- **Streaming Capability**: Both simple and detailed recipe endpoints support streaming responses

## 🛠 Technology Stack

### Backend
- FastAPI
- LangChain
- OpenAI GPT Models
- Python 3.x
- Pydantic
- python-dotenv

### Frontend
- React
- Vite
- TypeScript
- Tailwind CSS
- Vercel Hosting
[Frontend Source Code](https://github.com/wamofi97/biarkamimasak)

### Deployment
- Railway (API)
- Vercel (Web Application)

## 📋 Prerequisites

- Python 3.x
- OpenAI API key
- Recipe API key (for authentication)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/biarkamimasak.git
cd biarkamimasak
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key
RECIPE_API_KEY=your_recipe_api_key
```

## 💻 API Usage

### Starting the Server Locally

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Simple Recipe Recommendation
```http
POST /v1/recipe/simple
Content-Type: application/json
X-Recipe-API-Key: your_api_key

{
    "ingredients": ["chicken", "rice", "onion"],
    "servings": 2
}
```

#### 2. Detailed Recipe Recommendation
```http
POST /v1/recipe/detailed
Content-Type: application/json
X-Recipe-API-Key: your_api_key

{
    "ingredients": ["chicken", "rice", "onion"],
    "servings": 4,
    "dietary_restrictions": ["gluten-free"],
    "cuisine_preference": "asian",
    "cooking_time": 60
}
```

#### 3. Streaming Endpoints
```http
POST /v1/recipe/simple/stream
POST /v1/recipe/detailed/stream
```
These endpoints provide real-time streaming responses for a better user experience.

### Example Response

```json
{
    "title": "Goofy Chef's Amazing Nasi Goreng",
    "ingredients": [
        "2 cups cooked rice",
        "2 chicken breasts, diced",
        "1 onion, chopped"
    ],
    "instructions": [
        "Heat oil in a wok over medium heat",
        "Sauté chopped onions until fragrant",
        "Add diced chicken and cook until golden"
    ],
    "cooking_time": "20 minutes",
    "difficulty": "Easy",
    "notes": "Want to make it fancy? Just add a fried egg on top! 🍳",
    "timestamp": "2024-11-16T12:44:36.278Z"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👥 Team

- Frontend Development: [Wan Firdaus](https://github.com/wamofi97)
- Backend API Development: [Fahmi Fauzi](https://github.com/fahmifauzi00)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing the LLM capabilities
- FastAPI for the excellent web framework
- LangChain for the LLM integration tools
- Railway for hosting the API
- Vercel for hosting the web application