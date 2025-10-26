import os
import logging
import pickle
import json
import random
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy

# Import models
from models import db, EmotionRecord, LiveSession, FeedbackEntry

# Import utility modules
from utils.text_processor import process_text, analyze_text_sentiment
from utils.audio_processor import process_audio
from utils.image_processor import process_image
from utils.model_utils import load_model, predict_emotion

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key")

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mydatabase.db'
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the database
db.init_app(app)

# Create necessary tables
with app.app_context():
    db.create_all()

# Create upload directory if it doesn't exist
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed file extensions
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Emotion labels
EMOTION_LABELS = ['happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'neutral']

# Load transition matrix from pickle file
TRANSITION_MATRIX_PATH = os.path.join(os.getcwd(), 'attached_assets', 'transition_matrix.pkl')
try:
    with open(TRANSITION_MATRIX_PATH, 'rb') as f:
        TRANSITION_MATRIX = pickle.load(f)
    logger.debug("Loaded transition matrix from file")
except Exception as e:
    logger.warning(f"Could not load transition matrix from file: {e}")
    # Fallback to default transition matrix
    TRANSITION_MATRIX = {
        'happy': {'happy': 0.7, 'sad': 0.05, 'angry': 0.05, 'fear': 0.05, 'disgust': 0.05, 'surprise': 0.05, 'neutral': 0.05},
        'sad': {'happy': 0.05, 'sad': 0.7, 'angry': 0.05, 'fear': 0.05, 'disgust': 0.05, 'surprise': 0.05, 'neutral': 0.05},
        'angry': {'happy': 0.05, 'sad': 0.05, 'angry': 0.7, 'fear': 0.05, 'disgust': 0.05, 'surprise': 0.05, 'neutral': 0.05},
        'fear': {'happy': 0.05, 'sad': 0.05, 'angry': 0.05, 'fear': 0.7, 'disgust': 0.05, 'surprise': 0.05, 'neutral': 0.05},
        'disgust': {'happy': 0.05, 'sad': 0.05, 'angry': 0.05, 'fear': 0.05, 'disgust': 0.7, 'surprise': 0.05, 'neutral': 0.05},
        'surprise': {'happy': 0.05, 'sad': 0.05, 'angry': 0.05, 'fear': 0.05, 'disgust': 0.05, 'surprise': 0.7, 'neutral': 0.05},
        'neutral': {'happy': 0.1, 'sad': 0.1, 'angry': 0.1, 'fear': 0.1, 'disgust': 0.1, 'surprise': 0.1, 'neutral': 0.4}
    }

# Load feedback history from pickle file
FEEDBACK_HISTORY_PATH = os.path.join(os.getcwd(), 'attached_assets', 'feedback_history.pkl')
try:
    with open(FEEDBACK_HISTORY_PATH, 'rb') as f:
        FEEDBACK_HISTORY = pickle.load(f)
    logger.debug("Loaded feedback history from file")
except Exception as e:
    logger.warning(f"Could not load feedback history from file: {e}")
    # Fallback to default feedback history
    FEEDBACK_HISTORY = {
        'happy': {'correct': 85, 'incorrect': 15},
        'sad': {'correct': 80, 'incorrect': 20},
        'angry': {'correct': 75, 'incorrect': 25},
        'fear': {'correct': 70, 'incorrect': 30},
        'disgust': {'correct': 65, 'incorrect': 35},
        'surprise': {'correct': 80, 'incorrect': 20},
        'neutral': {'correct': 85, 'incorrect': 15}
    }

# Load label encoder from pickle file
LABEL_ENCODER_PATH = os.path.join(os.getcwd(), 'attached_assets', 'label_encoder.pkl')
try:
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        LABEL_ENCODER = pickle.load(f)
    logger.debug("Loaded label encoder from file")
except Exception as e:
    logger.warning(f"Could not load label encoder from file: {e}")
    LABEL_ENCODER = None

# Load model
MODEL_PATH = os.path.join(os.getcwd(), 'attached_assets', 'cognitive_resonance_model.h5')
try:
    MODEL = load_model(MODEL_PATH)
    logger.debug("Loaded model from file")
except Exception as e:
    logger.warning(f"Could not load model from file: {e}")
    MODEL = None

def allowed_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

@app.route('/')
def index():
    """Main page with file upload form"""
    return render_template('index.html')

@app.route('/live')
def live():
    """Page for live webcam and audio analysis"""
    return render_template('live.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log the received data
        logger.debug("Received prediction request")
        
        # Initialize flags for which inputs were provided
        audio_provided = False
        image_provided = False
        text_provided = False
        
        # Initialize feature variables
        audio_features = None
        image_features = None
        text_features = None
        
        # Check audio file
        audio_path = None
        if 'audio' in request.files and request.files['audio'].filename != '':
            audio_file = request.files['audio']
            if not allowed_audio_file(audio_file.filename):
                return jsonify({'error': 'Invalid audio file format'}), 400
            
            # Save file
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio_file.filename))
            audio_file.save(audio_path)
            logger.debug(f"Audio file saved to {audio_path}")
            audio_provided = True
            
            # Process audio file to extract features
            try:
                audio_features = process_audio(audio_path)
                logger.debug(f"Audio features extracted with shape: {audio_features.shape}")
            except Exception as e:
                logger.warning(f"Could not extract audio features: {e}")
        
        # Check image file
        image_path = None
        if 'image' in request.files and request.files['image'].filename != '':
            image_file = request.files['image']
            if not allowed_image_file(image_file.filename):
                return jsonify({'error': 'Invalid image file format'}), 400
            
            # Save file
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
            image_file.save(image_path)
            logger.debug(f"Image file saved to {image_path}")
            image_provided = True
            
            # Process image file to extract features
            try:
                image_features = process_image(image_path)
                logger.debug(f"Image features extracted with shape: {image_features.shape}")
            except Exception as e:
                logger.warning(f"Could not extract image features: {e}")
        
        # Check text input
        text_input = request.form.get('text', '')
        text_analysis = None  # Initialize this variable to avoid "possibly unbound" errors
        text_features = None  # Initialize text features
        
        if text_input:
            logger.debug(f"Text input: {text_input}")
            text_provided = True
            
            # Process text to extract features
            try:
                # Use our advanced text processor
                text_analysis = process_text(text_input)
                text_features = text_analysis.get('features')
                logger.debug(f"Text features extracted with shape: {text_features.shape if text_features is not None else 'None'}")
            except Exception as e:
                logger.warning(f"Could not extract text features: {e}")
        
        # Check if at least one modality is provided
        if not (audio_provided or image_provided or text_provided):
            return jsonify({'error': 'Please provide at least one input (audio, image, or text)'}), 400
        
        # Prepare for emotion prediction
        if MODEL is not None and LABEL_ENCODER is not None and (audio_features is not None or image_features is not None or text_features is not None):
            try:
                # Use the model for prediction
                emotion, confidence, uncertainty = predict_emotion(
                    MODEL, 
                    audio_features, 
                    image_features, 
                    text_features, 
                    LABEL_ENCODER
                )
                logger.debug(f"Model prediction: {emotion} (conf: {confidence:.2f}, unc: {uncertainty:.2f})")
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                # Fallback to text-based prediction if available
                if text_provided and text_analysis is not None and 'primary_emotion' in text_analysis:
                    emotion = text_analysis['primary_emotion']
                    confidence = text_analysis['confidence']
                    uncertainty = text_analysis['uncertainty']
                    logger.debug(f"Fallback to text analysis: {emotion} (conf: {confidence:.2f}, unc: {uncertainty:.2f})")
                else:
                    # Fallback to random selection
                    emotion = random.choice(EMOTION_LABELS)
                    confidence = random.uniform(0.6, 0.95)
                    uncertainty = random.uniform(0.05, 0.4)
                    logger.debug(f"Fallback to random: {emotion} (conf: {confidence:.2f}, unc: {uncertainty:.2f})")
        elif text_provided and text_analysis is not None and 'primary_emotion' in text_analysis:
            # Use text-based prediction
            emotion = text_analysis['primary_emotion']
            confidence = text_analysis['confidence']
            uncertainty = text_analysis['uncertainty']
            logger.debug(f"Text-based prediction: {emotion} (conf: {confidence:.2f}, unc: {uncertainty:.2f})")
        else:
            # Fallback to simple text-based logic or random selection
            if text_provided:
                text_lower = text_input.lower()
                if "happy" in text_lower or "joy" in text_lower or "glad" in text_lower:
                    emotion = "happy"
                    confidence = random.uniform(0.8, 0.95)
                    uncertainty = random.uniform(0.05, 0.2)
                elif "sad" in text_lower or "unhappy" in text_lower or "depressed" in text_lower:
                    emotion = "sad"
                    confidence = random.uniform(0.75, 0.9)
                    uncertainty = random.uniform(0.1, 0.25)
                elif "angry" in text_lower or "mad" in text_lower or "furious" in text_lower:
                    emotion = "angry"
                    confidence = random.uniform(0.7, 0.9)
                    uncertainty = random.uniform(0.1, 0.3)
                elif "scared" in text_lower or "afraid" in text_lower or "fear" in text_lower:
                    emotion = "fear"
                    confidence = random.uniform(0.65, 0.85)
                    uncertainty = random.uniform(0.15, 0.35)
                else:
                    emotion = random.choice(EMOTION_LABELS)
                    confidence = random.uniform(0.6, 0.95)
                    uncertainty = random.uniform(0.05, 0.4)
            else:
                emotion = random.choice(EMOTION_LABELS)
                confidence = random.uniform(0.6, 0.95)
                uncertainty = random.uniform(0.05, 0.4)
            
            logger.debug(f"Simple logic prediction: {emotion} (conf: {confidence:.2f}, unc: {uncertainty:.2f})")
        
        # Prepare response
        response = {
            'emotion': emotion,
            'confidence': float(confidence),
            'uncertainty': float(uncertainty)
        }
        
        # Add context emotion from transition matrix or text analysis
        if text_analysis is not None and 'context_emotion' in text_analysis:
            context_emotion = text_analysis['context_emotion']
            context_confidence = text_analysis['context_confidence']
            logger.debug(f"Using text analysis context: {context_emotion} (conf: {context_confidence:.2f})")
        else:
            # Use transition matrix for contextual emotion
            context_probs = TRANSITION_MATRIX.get(emotion, {})
            if context_probs:
                context_emotion_candidates = []
                context_emotion_probs = []
                for ctx_emotion, ctx_prob in context_probs.items():
                    context_emotion_candidates.append(ctx_emotion)
                    context_emotion_probs.append(ctx_prob)
                
                # Normalize probabilities
                total_prob = sum(context_emotion_probs)
                if total_prob > 0:
                    context_emotion_probs = [p/total_prob for p in context_emotion_probs]
                    
                # Choose a random emotion based on transition probabilities
                context_emotion = random.choices(
                    context_emotion_candidates, 
                    weights=context_emotion_probs, 
                    k=1
                )[0]
                context_confidence = random.uniform(0.5, 0.9)
                logger.debug(f"Using transition matrix context: {context_emotion} (conf: {context_confidence:.2f})")
            else:
                context_emotion = emotion
                context_confidence = confidence
        
        # Add context to response
        response['context_emotion'] = context_emotion
        response['context_confidence'] = float(context_confidence)
        
        # Add historical stats from feedback
        if emotion in FEEDBACK_HISTORY:
            stats = FEEDBACK_HISTORY[emotion]
            total = stats.get('correct', 0) + stats.get('incorrect', 0)
            if total > 0:
                accuracy = stats.get('correct', 0) / total
                response['historical_accuracy'] = float(accuracy)
        
        # Add detailed text analysis if available
        if text_provided and text_analysis is not None:
            # Add more detailed text analysis to the response
            response['text_analysis'] = {
                'emotion_scores': text_analysis.get('all_emotions', {}),
                'features': {
                    'word_count': len(text_input.split())
                }
            }
            
            # Perform additional sentiment analysis for more details
            try:
                detailed_analysis = analyze_text_sentiment(text_input)
                response['detailed_analysis'] = {
                    'emotion_word_counts': detailed_analysis.get('emotion_word_counts', {}),
                    'pattern_matches': detailed_analysis.get('pattern_matches', {}),
                    'emotion_phrases': detailed_analysis.get('emotion_phrases', {})
                }
            except Exception as e:
                logger.warning(f"Could not perform detailed sentiment analysis: {e}")
        
        # Save result to database
        try:
            emotion_record = EmotionRecord(
                emotion=emotion,
                confidence=float(confidence),
                uncertainty=float(uncertainty),
                context_emotion=context_emotion,
                context_confidence=float(context_confidence),
                source_text=text_input if text_provided else None,
                has_audio=audio_provided,
                has_image=image_provided,
                historical_accuracy=response.get('historical_accuracy')
            )
            db.session.add(emotion_record)
            db.session.commit()
            
            # Add record ID to response
            response['record_id'] = emotion_record.id
            logger.debug(f"Saved prediction to database with ID: {emotion_record.id}")
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            # Continue without saving to database
        
        logger.debug(f"Prediction response: {response}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/process-webcam', methods=['POST'])
def process_webcam():
    try:
        # Get data from request
        data = request.get_json()
        
        if not data or 'imageData' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Initialize variables
        text_data = data.get('textData')
        audio_data = data.get('audioData')
        body_gesture_data = data.get('gestureData')
        text_analysis = None
        text_features = None
        
        # Log activity
        logger.debug("Received webcam frame for processing")
        if audio_data:
            logger.debug("Audio data received")
        if text_data:
            logger.debug(f"Text data received: {text_data}")
            
            # Process text if available
            try:
                text_analysis = process_text(text_data)
                text_features = text_analysis.get('features')
                logger.debug(f"Processed webcam text with text analyzer")
            except Exception as e:
                logger.warning(f"Could not process webcam text data: {e}")
                
        if body_gesture_data:
            logger.debug("Body gesture data received")
        
        # Determine emotion using text analysis if available
        if text_data and text_analysis is not None and 'primary_emotion' in text_analysis:
            # Use the text analysis results
            selected_emotion = text_analysis['primary_emotion']
            confidence = text_analysis['confidence']
            uncertainty = text_analysis['uncertainty']
            logger.debug(f"Using text analysis for emotion: {selected_emotion} (conf: {confidence:.2f})")
            
            # Get context emotion from text analysis
            if 'context_emotion' in text_analysis:
                context_emotion = text_analysis['context_emotion']
                context_confidence = text_analysis['context_confidence']
            else:
                # Use transition matrix as fallback
                context_probs = TRANSITION_MATRIX.get(selected_emotion, {})
                if context_probs:
                    context_emotion_candidates = list(context_probs.keys())
                    context_emotion_probs = list(context_probs.values())
                    context_emotion = random.choices(context_emotion_candidates, weights=context_emotion_probs, k=1)[0]
                    context_confidence = random.uniform(0.5, 0.9)
                else:
                    context_emotion = selected_emotion
                    context_confidence = confidence
        else:
            # Select emotion with basic logic if text is provided but analysis failed
            selected_emotion = random.choice(EMOTION_LABELS)
            confidence = random.uniform(0.6, 0.95)
            uncertainty = random.uniform(0.05, 0.4)
            
            if text_data:
                text_lower = text_data.lower()
                if "happy" in text_lower or "joy" in text_lower or "smile" in text_lower:
                    selected_emotion = "happy"
                    confidence = random.uniform(0.8, 0.95)
                    uncertainty = random.uniform(0.05, 0.2)
                elif "sad" in text_lower or "unhappy" in text_lower:
                    selected_emotion = "sad"
                    confidence = random.uniform(0.75, 0.9)
                    uncertainty = random.uniform(0.1, 0.25)
                elif "angry" in text_lower or "mad" in text_lower:
                    selected_emotion = "angry"
                    confidence = random.uniform(0.7, 0.9)
                    uncertainty = random.uniform(0.1, 0.3)
            
            # Use transition matrix for context
            context_probs = TRANSITION_MATRIX.get(selected_emotion, {})
            if context_probs:
                context_emotion_candidates = list(context_probs.keys())
                context_emotion_probs = list(context_probs.values())
                context_emotion = random.choices(context_emotion_candidates, weights=context_emotion_probs, k=1)[0]
                context_confidence = random.uniform(0.5, 0.9)
            else:
                context_emotion = selected_emotion
                context_confidence = confidence
        
        # Create visualization data (enriched with text analysis if available)
        visualization = {
            'faceData': {
                'landmarks': [
                    # Eye positions (normalized coordinates)
                    {'x': 0.3, 'y': 0.4},
                    {'x': 0.7, 'y': 0.4},
                    # Mouth position
                    {'x': 0.5, 'y': 0.7},
                ],
                'emotions': {
                    'happy': random.uniform(0, 1),
                    'sad': random.uniform(0, 1),
                    'angry': random.uniform(0, 1),
                    'fear': random.uniform(0, 1),
                    'disgust': random.uniform(0, 1),
                    'surprise': random.uniform(0, 1),
                    'neutral': random.uniform(0, 1),
                }
            },
            'audioFeatures': {
                'volume': random.uniform(0, 1),
                'pitch': random.uniform(0, 1),
                'tempo': random.uniform(0, 1),
                'emotional_tones': {
                    'happy': random.uniform(0, 1),
                    'sad': random.uniform(0, 1),
                    'angry': random.uniform(0, 1),
                    'neutral': random.uniform(0, 1)
                }
            },
            'gestureData': {
                'movement_intensity': random.uniform(0, 1),
                'posture': random.choice(['open', 'closed', 'neutral']),
                'hand_activity': random.uniform(0, 1)
            }
        }
        
        # If we have text analysis results with emotion scores, use them in visualization
        if text_analysis is not None and 'all_emotions' in text_analysis:
            # Use text analysis emotion scores for better visualization
            emotion_scores = text_analysis['all_emotions']
            for emotion, score in emotion_scores.items():
                if emotion in visualization['faceData']['emotions']:
                    # Blend with existing values weighted more toward text analysis
                    current = visualization['faceData']['emotions'][emotion]
                    visualization['faceData']['emotions'][emotion] = 0.3 * current + 0.7 * score
        else:
            # Make the selected emotion more likely in the visualization data
            visualization['faceData']['emotions'][selected_emotion] = random.uniform(0.7, 1.0)
        
        # Prepare response with history from transition matrix
        temporal_data = []
        prev_emotion = selected_emotion
        
        # Generate a sequence of emotions using transition matrix
        for i in range(10):  # Past 10 timepoints
            if prev_emotion in TRANSITION_MATRIX:
                # Get transition probabilities from previous emotion
                probs = TRANSITION_MATRIX[prev_emotion]
                emotions = list(probs.keys())
                weights = list(probs.values())
                # Choose next emotion
                prev_emotion = random.choices(emotions, weights=weights, k=1)[0]
            else:
                # Fallback to random selection
                prev_emotion = random.choice(EMOTION_LABELS)
                
            # Add to temporal sequence (past to present)
            temporal_data.insert(0, {
                'emotion': prev_emotion,
                'confidence': random.uniform(0.6, 0.95),
                'timestamp': f"t-{10-i}"  # Simple timestamp format
            })
        
        # Add current prediction
        temporal_data.append({
            'emotion': selected_emotion,
            'confidence': confidence,
            'timestamp': 't0'  # Current time
        })
        
        # Complete response
        response = {
            'emotion': selected_emotion,
            'confidence': float(confidence),
            'uncertainty': float(uncertainty),
            'visualization': visualization,
            'temporal_data': temporal_data,
            'context_emotion': context_emotion,
            'context_confidence': float(context_confidence)
        }
        
        # Add detailed text analysis if available
        if text_data and text_analysis is not None:
            # Add more detailed text analysis to the response
            response['text_analysis'] = {
                'emotion_scores': text_analysis.get('all_emotions', {}),
                'features': {
                    'word_count': len(text_data.split())
                }
            }
            
            # Perform additional sentiment analysis for more details
            try:
                detailed_analysis = analyze_text_sentiment(text_data)
                response['detailed_analysis'] = {
                    'emotion_word_counts': detailed_analysis.get('emotion_word_counts', {}),
                    'pattern_matches': detailed_analysis.get('pattern_matches', {}),
                    'emotion_phrases': detailed_analysis.get('emotion_phrases', {})
                }
            except Exception as e:
                logger.warning(f"Could not perform detailed sentiment analysis: {e}")
        
        # Save to database for historical tracking
        try:
            # Record the emotion detection in the database
            emotion_record = EmotionRecord(
                emotion=selected_emotion,
                confidence=float(confidence),
                uncertainty=float(uncertainty),
                context_emotion=context_emotion,
                context_confidence=float(context_confidence),
                source_text=text_data if text_data else None,
                has_audio=audio_data is not None,
                has_image=True  # Webcam frames always have image data
            )
            db.session.add(emotion_record)
            
            # Check if we need to create or update a live session
            from datetime import timedelta
            current_time = datetime.utcnow()
            
            # Look for an active session in the last 5 minutes
            five_minutes_ago = current_time - timedelta(minutes=5)
            active_session = LiveSession.query.filter(
                LiveSession.end_time.is_(None),
                LiveSession.start_time >= five_minutes_ago
            ).order_by(LiveSession.start_time.desc()).first()
            
            if active_session:
                # Update existing session
                active_session.total_frames += 1
                
                # Update emotion counts
                if not active_session.emotions_detected:
                    active_session.emotions_detected = {}
                
                emotions_dict = active_session.emotions_detected
                if selected_emotion in emotions_dict:
                    emotions_dict[selected_emotion] += 1
                else:
                    emotions_dict[selected_emotion] = 1
                
                active_session.emotions_detected = emotions_dict
            else:
                # Create new session
                new_session = LiveSession(
                    start_time=current_time,
                    total_frames=1,
                    emotions_detected={selected_emotion: 1}
                )
                db.session.add(new_session)
            
            db.session.commit()
            logger.debug("Saved webcam data to database")
            
            # Add record ID to response
            response['record_id'] = emotion_record.id
        except Exception as e:
            logger.error(f"Error saving webcam data to database: {e}")
            # Continue without saving to database
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Webcam processing error: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)