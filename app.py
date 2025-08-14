from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
import uuid
from pathlib import Path
import logging
from video_analyzer import VideoAnalyzer
from docx_exporter import export_to_docx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Configuration
UPLOAD_FOLDER = 'uploads'
EXPORT_FOLDER = 'exports'
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPORT_FOLDER, exist_ok=True)

# Configure upload limits
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_file('index.html')

@app.route('/api/process-video', methods=['POST'])
def process_video():
    """Process uploaded video file and return analysis"""
    try:
        # Check if video file is provided
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        
        if video_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        file_ext = Path(video_file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Unsupported file type: {file_ext}'}), 400
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}{file_ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save uploaded file
        logger.info(f"Saving uploaded file: {filename}")
        video_file.save(filepath)
        
        # Initialize video analyzer
        try:
            analyzer = VideoAnalyzer()
            logger.info("Video analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize video analyzer: {e}")
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            return jsonify({'error': f'Configuration error: {str(e)}'}), 500
        
        # Process video
        logger.info(f"Starting video analysis for: {filename}")
        analysis_result = analyzer.analyze_video(
            video_path=filepath,
            video_name=f"video_{unique_id}",
            video_description="User uploaded video for analysis"
        )
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
            logger.info(f"Cleaned up temporary file: {filename}")
        except Exception as e:
            logger.warning(f"Failed to clean up file {filename}: {e}")
        
        return jsonify({
            'success': True,
            'analysis': analysis_result,
            'video_id': unique_id
        })
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        # Try to clean up any remaining files
        if 'filepath' in locals():
            try:
                os.remove(filepath)
            except:
                pass
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/export-docx', methods=['POST'])
def export_docx():
    """Export analysis results to DOCX format"""
    try:
        data = request.get_json()
        
        if not data or 'content' not in data:
            return jsonify({'error': 'No content provided'}), 400
        
        markdown_content = data['content']
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        filename = f"video_analysis_{unique_id}.docx"
        filepath = os.path.join(EXPORT_FOLDER, filename)
        
        # Export to DOCX
        logger.info(f"Exporting analysis to DOCX: {filename}")
        export_to_docx(markdown_content, filepath)
        
        # Send file and clean up after sending
        def remove_file(response):
            try:
                os.remove(filepath)
                logger.info(f"Cleaned up export file: {filename}")
            except Exception as e:
                logger.warning(f"Failed to clean up export file {filename}: {e}")
            return response
        
        return send_file(
            filepath,
            as_attachment=True,
            download_name='video-analysis-report.docx',
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
        
    except Exception as e:
        logger.error(f"Error exporting to DOCX: {str(e)}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Video Analysis API'
    })

@app.route('/api/test-config', methods=['GET'])
def test_config():
    """Test Azure services configuration"""
    try:
        from video_analyzer_original import VideoAnalyzer
        
        # Test analyzer initialization
        analyzer = VideoAnalyzer()
        
        # Test OpenAI connection
        openai_status = analyzer.test_openai_connection()
        
        return jsonify({
            'status': 'success',
            'video_indexer': 'configured',
            'openai': 'connected' if openai_status else 'connection_failed',
            'message': 'Configuration test completed'
        })
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Configuration test failed: {str(e)}'
        }), 500

@app.errorhandler(413)
def file_too_large(e):
    """Handle file size exceeded error"""
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting Video Analysis Server...")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Export folder: {EXPORT_FOLDER}")
    logger.info(f"Max file size: {MAX_FILE_SIZE / (1024*1024):.0f}MB")
    
    app.run(debug=True, host='0.0.0.0', port=8000)