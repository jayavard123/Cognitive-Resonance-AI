from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class EmotionRecord(db.Model):
    """Records of emotion analysis results"""
    id = db.Column(db.Integer, primary_key=True)
    emotion = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    uncertainty = db.Column(db.Float, nullable=False)
    context_emotion = db.Column(db.String(20), nullable=True)
    context_confidence = db.Column(db.Float, nullable=True)
    source_text = db.Column(db.Text, nullable=True)
    has_audio = db.Column(db.Boolean, default=False)
    has_image = db.Column(db.Boolean, default=False)
    historical_accuracy = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<EmotionRecord {self.emotion} ({self.confidence:.2f})>'
    
    def to_dict(self):
        """Convert record to dictionary for JSON response"""
        return {
            'id': self.id,
            'emotion': self.emotion,
            'confidence': self.confidence,
            'uncertainty': self.uncertainty,
            'context_emotion': self.context_emotion,
            'context_confidence': self.context_confidence,
            'has_audio': self.has_audio,
            'has_image': self.has_image,
            'historical_accuracy': self.historical_accuracy,
            'timestamp': self.timestamp.isoformat()
        }

class LiveSession(db.Model):
    """Records of live analysis sessions"""
    id = db.Column(db.Integer, primary_key=True)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    duration_seconds = db.Column(db.Integer, nullable=True)
    
    # Session statistics
    total_frames = db.Column(db.Integer, default=0)
    emotions_detected = db.Column(db.JSON, default=dict)  # Counts for each emotion
    
    def __repr__(self):
        return f'<LiveSession {self.id} ({self.start_time})>'
    
    def to_dict(self):
        """Convert session to dictionary for JSON response"""
        return {
            'id': self.id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'total_frames': self.total_frames,
            'emotions_detected': self.emotions_detected
        }

class FeedbackEntry(db.Model):
    """User feedback on emotion recognition results"""
    id = db.Column(db.Integer, primary_key=True)
    emotion_record_id = db.Column(db.Integer, db.ForeignKey('emotion_record.id'), nullable=True)
    predicted_emotion = db.Column(db.String(20), nullable=False)
    actual_emotion = db.Column(db.String(20), nullable=False)
    is_correct = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    emotion_record = db.relationship('EmotionRecord', backref=db.backref('feedback', lazy=True))
    
    def __repr__(self):
        return f'<FeedbackEntry {self.predicted_emotion} -> {self.actual_emotion}>'
    
    def to_dict(self):
        """Convert feedback to dictionary for JSON response"""
        return {
            'id': self.id,
            'predicted_emotion': self.predicted_emotion,
            'actual_emotion': self.actual_emotion,
            'is_correct': self.is_correct,
            'timestamp': self.timestamp.isoformat()
        }