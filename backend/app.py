from urllib import response
import spacy
import os
import re
import logging
import werkzeug
import requests
from flask import Flask, request, jsonify, abort
from pdfminer.high_level import extract_text
from flask_cors import CORS
from transformers import pipeline
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from dateutil import parser as dparser
from enum import Enum
from sqlalchemy import or_, func, case
from dotenv import load_dotenv
from json import JSONDecodeError


# Load environment variables
load_dotenv()

# Flask App Initialization
app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "http://localhost:3000",
        "methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///requirements.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)

# Enums for requirement attributes
class PriorityEnum(str, Enum):
    HIGH = 'High'
    MEDIUM = 'Medium'
    LOW = 'Low'

class ComplexityEnum(str, Enum):
    HIGH = 'High'
    MODERATE = 'Moderate'
    LOW = 'Low'

class StatusEnum(str, Enum):
    APPROVED = 'Approved'
    DISAPPROVED = 'Disapproved'
    REVIEW = 'Review'
    DRAFT = 'Draft'

# Counter model for sequential IDs
class RequirementCounter(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    count = db.Column(db.Integer, default=0)

# Requirement model
class Requirement(db.Model):
    id = db.Column(db.String(10), primary_key=True)  # Reduced size for rX format
    requirement = db.Column(db.Text, nullable=False)
    categories = db.Column(db.Text, nullable=False)
    status = db.Column(db.Enum(StatusEnum), default=StatusEnum.REVIEW)
    priority = db.Column(db.Enum(PriorityEnum), default=PriorityEnum.MEDIUM)
    author = db.Column(db.String(100), default='System')
    ddate = db.Column(db.DateTime, default=datetime.now)
    complexity = db.Column(db.Enum(ComplexityEnum), default=ComplexityEnum.MODERATE)
    estimated_time = db.Column(db.Integer)

    __table_args__ = (
        db.Index('idx_ddate', 'ddate'),
        db.Index('idx_status', 'status'),
        db.Index('idx_priority', 'priority'),
        db.Index('idx_author', 'author'),
    )

# Create database tables
with app.app_context():
    db.create_all()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Helper function to generate sequential IDs
def generate_requirement_id():
    try:
        counter = RequirementCounter.query.filter_by(id=1).first()
        if not counter:
            counter = RequirementCounter(id=1, count=0)
            db.session.add(counter)
            db.session.commit()
        
        counter.count += 1
        db.session.commit()
        return f"r{counter.count}"
    except Exception as e:
        db.session.rollback()
        logging.error(f"ID generation failed: {str(e)}")
        raise RuntimeError("Failed to generate requirement ID")

# Load NLP models
try:
    nlp = spacy.load("en_core_web_sm")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        multi_label=True
    )
    logging.info("✅ AI models loaded successfully")
except Exception as e:
    logging.error(f"❌ Error loading models: {e}")
    nlp = None
    classifier = None

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Uploads Directory
UPLOAD_DIR = "uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Classification Labels
candidate_labels = ["Functional", "Non-Functional", "UI", "Security", "Performance"]
complexity_labels = ["High", "Moderate", "Low"]
priority_labels = ["High priority", "Medium priority", "Low priority"]

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'md'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_metadata(text):
    metadata = {'author': 'System', 'date': datetime.now()}
    try:
        date_match = dparser.parse(text, fuzzy=True)
        metadata['date'] = date_match
    except Exception:
        pass
    
    author_patterns = [
        r"Prepared by:\s*(.+)",
        r"Author:\s*(.+)",
        r"By\s+(.+)",
        r"Created by:\s*(.+)",
        r"Submitted by:\s*(.+)"
    ]
    for pattern in author_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metadata['author'] = match.group(1).strip()
            break
    return metadata

def clean_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[•\t\n]+", " ", text)).strip()

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        filename = werkzeug.utils.secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_DIR, filename)
        file.save(file_path)
        return jsonify({'message': 'File uploaded successfully', 'file_path': file_path}), 200
    except Exception as e:
        logging.error(f"File upload error: {e}")
        return jsonify({'error': 'File upload failed'}), 500

@app.route("/api/analyze", methods=["POST"])
def analyze_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    file_path = None
    try:
        filename = werkzeug.utils.secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_DIR, filename)
        file.save(file_path)

        if filename.endswith(".pdf"):
            text = extract_text(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

        if not text.strip():
            return jsonify({"error": "No text extracted from file"}), 500

        metadata = extract_metadata(text)
        requirements = []
        doc = nlp(text) if nlp else None
        sentences = doc.sents if doc else [text]

        for sent in sentences:
            cleaned = clean_text(sent.text if doc else sent)
            if len(cleaned.split()) < 3:
                continue

            classification = classifier(cleaned, candidate_labels)
            categories = ', '.join(classification['labels'][:3])
            
            priority_result = classifier(
                cleaned, 
                priority_labels,
                hypothesis_template="This requirement has {} priority."
            )
            priority = PriorityEnum(priority_result['labels'][0].split()[0])

            complexity_result = classifier(
                cleaned,
                complexity_labels,
                hypothesis_template="This requirement has {} complexity."
            )
            complexity = ComplexityEnum(complexity_result['labels'][0])

            complexity_time_map = {
                ComplexityEnum.HIGH: 8,
                ComplexityEnum.MODERATE: 4,
                ComplexityEnum.LOW: 2
            }
            estimated_time = complexity_time_map.get(complexity, 4)

            requirement = Requirement(
                id=generate_requirement_id(),
                requirement=cleaned,
                categories=categories,
                status=StatusEnum.REVIEW,
                priority=priority,
                complexity=complexity,
                estimated_time=estimated_time,
                author=metadata['author'],
                ddate=metadata['date']
            )

            db.session.add(requirement)
            requirements.append({
                "id": requirement.id,
                "requirement": cleaned,
                "categories": categories,
                "status": requirement.status.value,
                "priority": requirement.priority.value,
                "complexity": requirement.complexity.value,
                "estimated_time": estimated_time,
                "author": requirement.author,
                "date": requirement.ddate.isoformat()
            })

        db.session.commit()
        return jsonify({"requirements": requirements, "total": len(requirements)})

    except Exception as e:
        logging.error(f"Analysis error: {str(e)}", exc_info=True)
        db.session.rollback()
        return jsonify({"error": "Analysis failed"}), 500
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

def validate_filters(filters):
    valid_enums = {
        'status': StatusEnum,
        'priority': PriorityEnum,
        'complexity': ComplexityEnum
    }
    
    for key, values in filters.items():
        if key in valid_enums:
            for value in values:
                try:
                    valid_enums[key](value)
                except ValueError:
                    abort(400, f"Invalid {key} value: {value}")

@app.route("/api/requirements", methods=["GET", "POST"])
def handle_requirements():
    if request.method == "GET":
        try:
            search_query = request.args.get('search', '')
            types = request.args.getlist('type')
            statuses = request.args.getlist('status')
            complexities = request.args.getlist('complexity')
            priorities = request.args.getlist('priority')
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 10, type=int)

            validate_filters({
                'status': statuses,
                'priority': priorities,
                'complexity': complexities
            })

            query = Requirement.query

            if search_query:
                query = query.filter(
                    Requirement.requirement.ilike(f'%{search_query}%') |
                    Requirement.categories.ilike(f'%{search_query}%')
                )

            if types:
                type_filters = [Requirement.categories.ilike(f'%{t}%') for t in types]
                query = query.filter(or_(*type_filters))

            if statuses:
                query = query.filter(Requirement.status.in_([StatusEnum(s) for s in statuses]))

            if complexities:
                query = query.filter(Requirement.complexity.in_([ComplexityEnum(c) for c in complexities]))

            if priorities:
                query = query.filter(Requirement.priority.in_([PriorityEnum(p) for p in priorities]))

            # Calculate statistics before pagination
            stats_query = query.with_entities(
                func.count(Requirement.id),
                func.count(case((Requirement.status == StatusEnum.APPROVED, 1))),
                func.count(case((Requirement.status == StatusEnum.REVIEW, 1))),
                func.count(case((Requirement.status == StatusEnum.DISAPPROVED, 1)))
            ).one()

            stats = {
                "total": stats_query[0],
                "approved": stats_query[1],
                "inReview": stats_query[2],
                "disapproved": stats_query[3]
            }

            pagination = query.paginate(page=page, per_page=per_page, error_out=False)

            return jsonify({
                "requirements": [{
                    "id": req.id,
                    "requirement": req.requirement,
                    "categories": req.categories,
                    "status": req.status.value,
                    "priority": req.priority.value,
                    "complexity": req.complexity.value,
                    "estimated_time": req.estimated_time,
                    "author": req.author,
                    "date": req.ddate.isoformat()
                } for req in pagination.items],
                "stats": stats,
                "total": pagination.total,
                "page": pagination.page,
                "pages": pagination.pages
            })

        except Exception as e:
            logging.error(f"Error fetching requirements: {str(e)}")
            return jsonify({"error": "Failed to fetch requirements"}), 500
    
    elif request.method == "POST":
        try:
            data = request.get_json()
            cleaned = clean_text(data['requirement'])
            
            classification = classifier(cleaned, candidate_labels)
            categories = ', '.join(classification['labels'][:3])

            new_req = Requirement(
                id=generate_requirement_id(),
                requirement=cleaned,
                categories=categories,
                status=StatusEnum(data.get('status', 'Review')),
                priority=PriorityEnum(data.get('priority', 'Medium')),
                complexity=ComplexityEnum(data.get('complexity', 'Moderate')),
                estimated_time=data.get('estimated_time', 4),
                author=data.get('author', 'System'),
                ddate=datetime.now()
            )
            
            db.session.add(new_req)
            db.session.commit()
            return jsonify({
                "id": new_req.id,
                "message": "Requirement created successfully"
            }), 201
        
        except ValueError as e:
            db.session.rollback()
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": str(e)}), 500

@app.route("/api/requirements/<string:req_id>", methods=["GET", "PUT", "DELETE"])
def handle_single_requirement(req_id):
    requirement = Requirement.query.get(req_id)
    if not requirement:
        return jsonify({"error": "Requirement not found"}), 404

    if request.method == "GET":
        return jsonify({
            "id": requirement.id,
            "requirement": requirement.requirement,
            "categories": requirement.categories,
            "status": requirement.status.value,
            "priority": requirement.priority.value,
            "complexity": requirement.complexity.value,
            "estimated_time": requirement.estimated_time,
            "author": requirement.author,
            "date": requirement.ddate.isoformat()
        })
    
    elif request.method == "PUT":
        try:
            data = request.get_json()
            if 'requirement' in data:
                requirement.requirement = clean_text(data['requirement'])
            if 'categories' in data:
                requirement.categories = data['categories']
            if 'status' in data:
                requirement.status = StatusEnum(data['status'])
            if 'priority' in data:
                requirement.priority = PriorityEnum(data['priority'])
            if 'complexity' in data:
                requirement.complexity = ComplexityEnum(data['complexity'])
            if 'estimated_time' in data:
                requirement.estimated_time = int(data['estimated_time'])
            if 'author' in data:
                requirement.author = data['author']
            
            db.session.commit()
            return jsonify({
                "message": "Requirement updated successfully",
                "requirement": {
                    "id": requirement.id,
                    "author": requirement.author,
                    "status": requirement.status.value
                }
            })
        
        except ValueError as e:
            db.session.rollback()
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": str(e)}), 500
    
    elif request.method == "DELETE":
        db.session.delete(requirement)
        db.session.commit()
        return jsonify({"message": "Requirement deleted successfully"})

@app.route("/api/requirements/<string:req_id>/status", methods=["PATCH"])
def update_status(req_id):
    requirement = Requirement.query.get(req_id)
    if not requirement:
        return jsonify({"error": "Requirement not found"}), 404
    
    try:
        data = request.get_json()
        new_status = StatusEnum(data['status'])
        requirement.status = new_status
        db.session.commit()
        return jsonify({
            "message": "Status updated successfully",
            "new_status": new_status.value
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Status update failed"}), 500
    
def get_system_stats():
    """Get current system statistics from database"""
    try:
        return {
            "total": Requirement.query.count(),
            "approved": Requirement.query.filter_by(status=StatusEnum.APPROVED).count(),
            "inReview": Requirement.query.filter_by(status=StatusEnum.REVIEW).count(),
            "disapproved": Requirement.query.filter_by(status=StatusEnum.DISAPPROVED).count(),
        }
    except Exception as e:
        logging.error(f"Error getting system stats: {str(e)}")
        return {
            "total": 0,
            "approved": 0,
            "inReview": 0,
            "disapproved": 0
        }
    
@app.route('/api/requirements/stats', methods=['GET'])
def get_stats():
    try:
        total = Requirement.query.count()
        approved = Requirement.query.filter_by(status=StatusEnum.APPROVED).count()
        in_review = Requirement.query.filter_by(status=StatusEnum.REVIEW).count()
        disapproved = Requirement.query.filter_by(status=StatusEnum.DISAPPROVED).count()

        return jsonify({
            "total": total,
            "approved": approved,
            "inReview": in_review,
            "disapproved": disapproved
        })
    except Exception as e:
        logging.error(f"Error fetching stats: {e}")
        return jsonify({"error": "Failed to fetch stats"}), 500
    
@app.route('/api/chat', methods=['POST'])
def handle_chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message received'}), 400

        # Handle casual greetings immediately
        if is_casual_greeting(user_message):
            return jsonify({
                "response": "Hello! How can I assist you with requirements management today?",
                "mode": "greeting"
            })

        try:
            # Fetch requirements and stats
            requirements = Requirement.query.order_by(Requirement.ddate.desc()).all()
            stats = get_system_stats()
            
            # Format requirements concisely
            requirements_context = format_requirements_concise(requirements)

        except Exception as db_error:
            logging.error(f"Database error: {str(db_error)}", exc_info=True)
            return jsonify({"error": "Failed to load requirements data"}), 500

        try:
            # Detect response mode and user type
            response_mode = detect_response_mode(user_message)
            is_technical = detect_technical_user(user_message)
            tech_terms = explain_technical_terms(user_message)

            # Generate suggestions if requested
            suggestions = generate_requirement_suggestions(user_message, requirements)
            if suggestions:
                requirements_context += "\n\nSuggestions:\n" + suggestions

            # Build concise prompt
            prompt = build_concise_prompt(
                user_message,
                requirements_context,
                stats,
                response_mode,
                is_technical,
                tech_terms
            )

            # Configure API payload
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}],
                    "role": "USER"
                }],
                "generationConfig": {
                    "temperature": 0.7 if response_mode == "chatgpt" else (0.4 if is_technical else 0.6),
                    "topP": 0.95,
                    "maxOutputTokens": 1200,  # Adjusted token limit
                    "stopSequences": ["##END##"]
                }
            }

            # Execute API call
            try:
                headers = {"Content-Type": "application/json"}
                response = requests.post(
                    GEMINI_API_URL,
                    json=payload,
                    headers=headers,
                    params={"key": GEMINI_API_KEY},
                    timeout=10  # Reduced timeout
                )
                response.raise_for_status()
            except requests.exceptions.Timeout:
                logging.error("API request timed out")
                return jsonify({"error": "AI service timeout"}), 504
            except requests.exceptions.RequestException as re:
                logging.error(f"API connection error: {str(re)}")
                return jsonify({"error": "AI service unavailable"}), 502

            # Parse response
            try:
                gemini_response = response.json()
                bot_reply = gemini_response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "I couldn't process that.")
                
                # Check if the response is incomplete
                if not is_response_complete(bot_reply):
                    # If incomplete, request continuation
                    continuation_prompt = f"{prompt}\n\nContinue from: {bot_reply}"
                    continuation_payload = {
                        "contents": [{
                            "parts": [{"text": continuation_prompt}],
                            "role": "USER"
                        }],
                        "generationConfig": {
                            "temperature": 0.7 if response_mode == "chatgpt" else (0.4 if is_technical else 0.6),
                            "topP": 0.95,
                            "maxOutputTokens": 500,  # Additional tokens for continuation
                            "stopSequences": ["##END##"]
                        }
                    }
                    continuation_response = requests.post(
                        GEMINI_API_URL,
                        json=continuation_payload,
                        headers=headers,
                        params={"key": GEMINI_API_KEY},
                        timeout=10
                    )
                    continuation_response.raise_for_status()
                    continuation_data = continuation_response.json()
                    continuation_reply = continuation_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    bot_reply += " " + continuation_reply

            except (KeyError, JSONDecodeError, IndexError) as parse_error:
                logging.error(f"Response parsing failed: {str(parse_error)}")
                bot_reply = "There was an error processing your request"

            # Prepare concise response data
            response_data = {
                "response": bot_reply,
                "stats": stats,
                "requirements_analyzed": len(requirements),
                "mode": response_mode,
                "suggestions": bool(suggestions)
            }

            return jsonify(response_data)

        except Exception as api_error:
            logging.error(f"Processing error: {str(api_error)}", exc_info=True)
            return jsonify({"error": "Response generation failed"}), 500

    except Exception as e:
        logging.error(f"System error: {str(e)}", exc_info=True)
        return jsonify({"error": "Request processing failed"}), 500

def is_casual_greeting(message: str) -> bool:
    greetings = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "hi there"}
    return message.lower().strip() in greetings

def format_requirements_concise(requirements):
    if not requirements:
        return "No requirements"
    
    status_groups = {}
    for req in requirements:
        status = req.status.value.replace('_', ' ').title()
        status_groups.setdefault(status, []).append(req)
    
    result = []
    for status, reqs in status_groups.items():
        req_list = "\n".join(f"RQ-{req.id}: {simplify_text(req.requirement[:50])}..." for req in reqs)
        result.append(f"{status}:\n{req_list}")
    
    return "\n\n".join(result)

def detect_response_mode(query):
    query_lower = query.lower()
    if any(trigger in query_lower for trigger in ['explain', 'what is', 'how to']):
        return "chatgpt"
    if any(trigger in query_lower for trigger in ['suggest', 'improvement']):
        return "suggestion"
    return "technical" if detect_technical_user(query) else "business"

def generate_requirement_suggestions(query, existing_reqs):
    if not any(word in query.lower() for word in ['suggest', 'improvement']):
        return ""
    
    existing_ids = [req.id for req in existing_reqs] or [0]
    return "\n".join(f"[RQ-{max(existing_ids)+i+1}] {suggestion}" for i, suggestion in enumerate([
        "Automated compliance monitoring", "Real-time collaboration features", "AI-powered validation"
    ]))

def build_concise_prompt(query, context, stats, mode, is_technical, tech_terms):
    base = f"""**Project**: {stats.get('total', 0)} reqs | {stats.get('approved', 0)} approved
**Requirements**: {context}
**Query**: {query}"""
    
    if mode == "chatgpt":
        return f"{base}\n**Mode**: Explain with examples and steps"
    elif mode == "suggestion":
        return f"{base}\n**Mode**: Suggest improvements with rationale"
    return f"{base}\n**Mode**: Analyze for {'technical' if is_technical else 'business'} impact"

def detect_technical_user(query: str) -> bool:
    tech_terms = {'api', 'validation', 'traceability', 'sdlc'}
    business_terms = {'roi', 'cost', 'benefit', 'timeline'}
    text = query.lower()
    return sum(1 for term in tech_terms if term in text) > sum(1 for term in business_terms if term in text)

def explain_technical_terms(text: str) -> str:
    glossary = {'api': "Application Programming Interface", 'sdlc': "Software Development Life Cycle", 'validation': "Quality assurance process"}
    return "\n".join(f"- {term}: {desc}" for term, desc in glossary.items() if term in text.lower())

def simplify_text(text: str) -> str:
    replacements = {"shall": "must", "stakeholder": "user", "validation": "QA", "artifact": "doc"}
    return ' '.join(replacements.get(word.lower(), word) for word in text.split())

# New helper function to check response completeness
def is_response_complete(response: str) -> bool:
    """Check if the response is complete by looking for proper punctuation or stop sequence."""
    if not response:
        return False
    # Check if the response ends with proper punctuation or stop sequence
    if response.strip().endswith(('.', '!', '?', '##END##')):
        return True
    # Check if the response is logically complete
    if len(response.split()) >= 50:  # Arbitrary threshold for completeness
        return True
    return False

@app.route('/api/requirements/delete-all', methods=['DELETE'])
def delete_all_requirements():
    try:
        # Delete all requirements from the database
        num_deleted = db.session.query(Requirement).delete()
        db.session.commit()

        # Reset the requirement counter
        counter = RequirementCounter.query.first()
        if counter:
            counter.count = 0
            db.session.commit()

        return jsonify({
            "success": True,
            "message": f"Successfully deleted {num_deleted} requirements.",
            "deleted_count": num_deleted
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({
            "success": False,
            "message": f"An error occurred: {str(e)}"
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "database": "connected" if db.session.connection() else "disconnected",
        "ai_models": "loaded" if classifier else "unavailable",
        "gemini": "ready" if GEMINI_API_KEY else "missing_api_key"
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)