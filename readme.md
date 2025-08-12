# Video Analysis Dashboard

A comprehensive web application for analyzing videos using Azure Video Indexer and Azure OpenAI, with an aesthetic frontend for uploading videos, viewing analysis results, and exporting reports to DOCX format.

## Features

- 🎥 **Video Upload & Preview**: Drag-and-drop video upload with instant preview
- 🔍 **AI-Powered Analysis**: Comprehensive video analysis using Azure Video Indexer
- 📊 **Smart Summarization**: GPT-4o generates human-readable analysis reports
- 📄 **Export to DOCX**: Convert analysis results to formatted Word documents
- 🎨 **Modern UI**: Clean, responsive interface with gradient backgrounds
- ⚡ **Real-time Processing**: Live updates during video processing

## Architecture

```
Frontend (HTML/CSS/JS) → Flask Backend → Azure Video Indexer → Azure OpenAI → DOCX Export
```

## File Structure

```
video-analysis-dashboard/
├── index.html              # Frontend interface
├── app.py                  # Flask backend server
├── video_analyzer.py       # Video analysis module
├── docx_exporter.py        # DOCX export utility
├── config.json             # Configuration file
├── requirements.txt        # Python dependencies
├── uploads/                # Temporary video uploads
├── exports/                # Generated DOCX files
└── README.md              # This file
```

## Prerequisites

### Azure Services Required:
1. **Azure Video Indexer Account**
   - Get account ID and subscription key
   - Ensure you have processing quota available

2. **Azure OpenAI Service**
   - Deploy GPT-4o model
   - Get endpoint URL and API key

### System Requirements:
- Python 3.8+
- Modern web browser
- Internet connection for Azure API calls

## Installation

### 1. Clone or Download Files

Save all the provided files in a single directory:
- `index.html`
- `app.py`
- `video_analyzer.py`
- `docx_exporter.py`
- `config.json`
- `requirements.txt`

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Azure Credentials

Edit `config.json` with your actual Azure credentials:

```json
{
  "video_indexer": {
    "location": "trial",
    "account_id": "YOUR_ACTUAL_ACCOUNT_ID",
    "subscription_key": "YOUR_ACTUAL_VIDEO_INDEXER_KEY"
  },
  "openai": {
    "endpoint": "https://YOUR_ACTUAL_ENDPOINT.cognitiveservices.azure.com/",
    "model_name": "gpt-4o",
    "deployment": "gpt-4o",
    "subscription_key": "YOUR_ACTUAL_OPENAI_KEY",
    "api_version": "2024-12-01-preview"
  }
}
```

### 4. Create Required Directories

```bash
mkdir uploads exports
```

## Usage

### 1. Start the Server

```bash
python app.py
```

The server will start on `http://localhost:8000`

### 2. Access the Dashboard

Open your web browser and navigate to:
```
http://localhost:8000
```

### 3. Analyze a Video

1. **Upload**: Click "Select Video File" and choose your video
2. **Preview**: The video will appear in the player
3. **Process**: Click "🚀 Process Video" to start analysis
4. **Wait**: Processing typically takes 5-15 minutes depending on video length
5. **Review**: View the summary and detailed analysis
6. **Export**: Click "📥 Export to DOCX" to download the report

## Supported Video Formats

- MP4 (recommended)
- AVI
- MOV
- MKV
- WMV
- FLV
- WebM

**Maximum file size**: 500MB

## API Endpoints

### Process Video
```
POST /api/process-video
Content-Type: multipart/form-data

Parameters:
- video: Video file upload

Returns:
- JSON with analysis results
```

### Export DOCX
```
POST /api/export-docx
Content-Type: application/json

Body:
{
  "content": "markdown_content_here"
}

Returns:
- DOCX file download
```

### Health Check
```
GET /api/health

Returns:
- Service status
```

## Analysis Features

The system provides comprehensive video analysis including:

### Content Analysis
- **Transcript**: Full speech-to-text with timestamps
- **Speaker Identification**: Multiple speaker detection
- **Scene Detection**: Automatic scene boundaries
- **Object Recognition**: Visual elements identification

### Insights Extraction
- **Named Entities**: People, places, organizations
- **Brand Detection**: Commercial brand identification
- **Topic Modeling**: Key themes and subjects
- **Sentiment Analysis**: Emotional tone tracking

### Visual Analysis
- **Face Recognition**: Face detection and identification
- **OCR Text**: On-screen text extraction
- **Keyframes**: Important visual moments
- **Content Moderation**: Safety and appropriateness checks

## Troubleshooting

### Common Issues

**1. "Processing failed" Error**
- Check your Azure credentials in `config.json`
- Ensure you have sufficient quota in Azure Video Indexer
- Verify video file format is supported

**2. "File too large" Error**
- Maximum file size is 500MB
- Compress your video or use a shorter clip

**3. "Export failed" Error**
- Check that the analysis completed successfully
- Ensure write permissions for the `exports/` directory

**4. Slow Processing**
- Video Indexer processing time depends on video length
- Typical processing time: 2-3x the video duration
- Large files and complex content take longer

### Debug Mode

To enable detailed logging, modify `app.py`:

```python
logging.basicConfig(level=logging.DEBUG)
```

### Testing with Sample Data

The frontend includes sample analysis data for testing the interface without processing actual videos.

## Security Considerations

- Store Azure credentials securely
- Don't commit `config.json` with real credentials to version control
- Consider using environment variables for production deployments
- Implement authentication for production use

## Production Deployment

For production deployment:

1. **Use Environment Variables** for sensitive configuration
2. **Add Authentication** to protect the application
3. **Configure HTTPS** for secure communications
4. **Set up Proper Logging** and monitoring
5. **Implement Rate Limiting** to prevent abuse
6. **Use a Production WSGI Server** like Gunicorn

Example production startup:
```bash
gunicorn -w 4 -b 0.0.0.0:8080 app:app
```

## License

This project is provided as-is for educational and development purposes.

## Support

For issues related to:
- **Azure Video Indexer**: Check Azure documentation
- **Azure OpenAI**: Refer to OpenAI API documentation
- **Application Issues**: Check the troubleshooting section above

---

**Note**: Replace placeholder credentials in `config.json` with your actual Azure service credentials before running the application.