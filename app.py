from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# API Configuration - Store in environment variables for security
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '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')
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_SITE_URL = os.getenv('SITE_URL', 'http://localhost:8000')
OPENROUTER_SITE_NAME = os.getenv('SITE_NAME', 'Eatlytic')

# Language mapping
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi"
}


@app.route('/api/analyze', methods=['POST'])
def analyze_food():
    """
    Endpoint to analyze food images
    Expects JSON with 'imageDataUrl' and 'language' fields
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        image_data_url = data.get('imageDataUrl')
        language_code = data.get('language', 'en')
        
        if not image_data_url:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Get language name
        language_name = LANGUAGE_NAMES.get(language_code, "English")
        
        # Debug logging
        print(f"\nðŸ” Processing request:")
        print(f"   Language: {language_name} ({language_code})")
        print(f"   API Key: {OPENROUTER_API_KEY[:20]}...{OPENROUTER_API_KEY[-4:]}")
        print(f"   Image size: {len(image_data_url)} characters")
        
        # Prepare the request to OpenRouter API
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": OPENROUTER_SITE_URL,
            "X-Title": OPENROUTER_SITE_NAME,
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": f"You are Eatlytic, an intelligent food companion. Analyze the provided image of a food item. Provide a detailed nutritional breakdown (estimated calories, macronutrients (fat, protein, carbs), and fiber if applicable). Then, explain its potential impact on the heart, muscles, and energy levels. Finally, offer smart consumption tips and important awareness notes for healthy eating. Format the response using markdown headings (##, ###) and bullet points for readability. Provide the analysis in {language_name}."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this food item:"},
                        {"type": "image_url", "image_url": {"url": image_data_url}}
                    ]
                }
            ]
        }
        
        # Make request to OpenRouter API
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
        
        # Check if request was successful
        if not response.ok:
            error_data = response.json()
            return jsonify({
                'error': f'API error: {response.status_code}',
                'details': error_data
            }), response.status_code
        
        # Parse and return the response
        result = response.json()
        analysis_content = result['choices'][0]['message']['content']
        
        return jsonify({
            'success': True,
            'analysis': analysis_content
        }), 200
        
    except requests.exceptions.RequestException as e:
        return jsonify({
            'error': 'Network error',
            'details': str(e)
        }), 500
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Eatlytic backend is running'
    }), 200


@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API information"""
    return jsonify({
        'name': 'Eatlytic API',
        'version': '1.0.0',
        'endpoints': {
            '/api/analyze': 'POST - Analyze food images',
            '/api/health': 'GET - Health check'
        }
    }), 200


if __name__ == '__main__':
    # Check and display API key status
    if OPENROUTER_API_KEY and OPENROUTER_API_KEY != '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':
        print(f"âœ… API Key loaded: {OPENROUTER_API_KEY[:15]}...{OPENROUTER_API_KEY[-4:]}")
    else:
        print("\nâš ï¸  WARNING: Please set your OPENROUTER_API_KEY in a .env file")
        print("Create a .env file with: OPENROUTER_API_KEY=your-actual-key\n")
    
    # Run the Flask app
    print("ðŸš€ Starting Eatlytic Backend Server...")
    print("ðŸ“ Server will run on http://localhost:5000")
    print("ðŸ“ Make sure to update your HTML to use this backend URL\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)