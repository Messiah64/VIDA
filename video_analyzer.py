import time
import requests
import json
from pathlib import Path
from openai import AzureOpenAI
import logging

logger = logging.getLogger(__name__)

class VideoAnalyzer:
    """
    Video Analysis using Azure Video Indexer and Azure OpenAI
    """
    
    def __init__(self, config_file='config.json'):
        """Initialize with configuration"""
        self.load_config(config_file)
        
        # Initialize Azure OpenAI client - following exact working pattern
        try:
            self.openai_client = AzureOpenAI(
                api_version=self.config['openai']['api_version'],
                azure_endpoint=self.config['openai']['endpoint'],
                api_key=self.config['openai']['subscription_key'],
            )
            logger.info("Azure OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise
    
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_file} not found. Using default values.")
            self.config = self.get_default_config()
    
    def get_default_config(self):
        """Return default configuration (replace with your actual values)"""
        return {
            "video_indexer": {
                "location": "trial",
                "account_id": "ac09490f-3445-4638-91da-3f3e659fd268",
                "subscription_key": "f99caefac00047e0979f7a7b35144cdd"
            },
            "openai": {
                "endpoint": "https://khamb-me7vxhbb-eastus2.cognitiveservices.azure.com/",
                "model_name": "gpt-4o",
                "deployment": "gpt-4o",
                "subscription_key": "9fFeYtGLfGZxtysxr81gcn3yiBPginpV7uobhhWv8Uf2kpWDpPByJQQJ99BHACHYHv6XJ3w3AAAAACOGvfk3",
                "api_version": "2024-12-01-preview"
            },
            "processing": {
                "timeout_seconds": 900,
                "privacy": "Private"
            }
        }
    
    def get_access_token(self):
        """Get access token from Video Indexer"""
        vi_config = self.config['video_indexer']
        
        token_url = f"https://api.videoindexer.ai/Auth/{vi_config['location']}/Accounts/{vi_config['account_id']}/AccessTokenWithPermission"
        headers = {
            "Ocp-Apim-Subscription-Key": vi_config['subscription_key'],
            "Cache-Control": "no-cache"
        }
        params = {"permission": "Owner"}
        
        logger.info("Requesting access token from Video Indexer")
        resp = requests.get(token_url, headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        
        access_token = resp.text.strip().strip('"')
        logger.info("Access token retrieved successfully")
        return access_token
    
    def upload_video(self, access_token, video_path, video_name, video_description):
        """Upload video to Video Indexer"""
        vi_config = self.config['video_indexer']
        
        upload_url = f"https://api.videoindexer.ai/{vi_config['location']}/Accounts/{vi_config['account_id']}/Videos"
        upload_params = {
            "accessToken": access_token,
            "name": video_name,
            "description": video_description,
            "privacy": self.config['processing']['privacy'],
            "indexingPreset": "Advanced",
            "streamingPreset": "Default",
            "sendSuccessEmail": "false",
            "summarizedInsights": "true",
            "detectSourceLanguage": "true",
            "multiLanguage": "true"
        }
        
        logger.info(f"Uploading video: {video_path}")
        with open(video_path, "rb") as f:
            files = {"file": (Path(video_path).name, f, "video/mp4")}
            upload_resp = requests.post(upload_url, params=upload_params, files=files, timeout=300)
        
        upload_resp.raise_for_status()
        upload_data = upload_resp.json()
        video_id = upload_data.get("id")
        
        if not video_id:
            raise RuntimeError(f"Upload succeeded but no video ID returned. Response: {upload_data}")
        
        logger.info(f"Upload successful. Video ID: {video_id}")
        return video_id
    
    def wait_for_indexing(self, access_token, video_id):
        """Wait for video indexing to complete"""
        vi_config = self.config['video_indexer']
        timeout_seconds = self.config['processing']['timeout_seconds']
        
        index_url = f"https://api.videoindexer.ai/{vi_config['location']}/Accounts/{vi_config['account_id']}/Videos/{video_id}/Index"
        index_params = {
            "accessToken": access_token,
            "language": "English",
            "includeSummarizedInsights": "true"
        }
        
        logger.info("Waiting for video indexing to complete...")
        start_time = time.time()
        
        while True:
            idx_resp = requests.get(index_url, params=index_params, timeout=30)
            idx_resp.raise_for_status()
            idx_json = idx_resp.json()
            
            state = (idx_json.get("state") or "").lower()
            logger.info(f"Indexing state: {state}")
            
            if state == "processed":
                logger.info("Indexing complete!")
                return idx_json
            
            if state == "failed":
                raise RuntimeError(f"Indexing failed: {idx_json}")
            
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"Indexing timed out after {timeout_seconds} seconds.")
            
            time.sleep(10)
    
    def extract_insights(self, insights_json):
        """Extract structured insights from Video Indexer response"""
        video_insights = insights_json.get("videos", [])[0].get("insights", {})
        summarized = insights_json.get("summarizedInsights", {})
        
        def safe_get(key):
            return video_insights.get(key, [])
        
        structured_data = {
            "video_metadata": {
                "duration": insights_json.get("durationInSeconds", None),
                "summarized_insights": summarized
            },
            "transcript": safe_get("transcript"),
            "speakers": safe_get("speakers"),
            "labels": safe_get("labels"),
            "faces": safe_get("faces"),
            "named_people": safe_get("namedPeople"),
            "named_locations": safe_get("namedLocations"),
            "brands": safe_get("brands"),
            "topics": safe_get("topics"),
            "keywords": safe_get("keywords"),
            "sentiments": safe_get("sentiments"),
            "audio_effects": safe_get("audioEffects"),
            "audio_events": safe_get("audioEvents"),
            "visual_content_moderation": safe_get("visualContentModeration"),
            "ocr_text": safe_get("ocr"),
            "blocks": safe_get("blocks"),
            "shots": safe_get("shots"),
            "keyframes": safe_get("keyframes"),
            "annotations": safe_get("annotations"),
            "tags": safe_get("tags")
        }
        
        return structured_data
    
    def generate_summary(self, structured_data):
        """Generate human-readable summary using GPT-4o - following exact working pattern"""
        openai_config = self.config['openai']
        
        detailed_prompt = f"""
You are an expert multimedia content analyst.
You will receive the full structured metadata extracted from a video, including:
- Full transcript with timestamps and speakers
- Scene and shot breakdowns
- Visual labels, tags, and detected objects
- Named people, brands, and locations
- Detected emotions and sentiments
- Audio events and effects
- On-screen text from OCR
- Any moderation or safety flags

Your task:
1. Reconstruct a coherent, chronological narrative of the video, blending both visual and audio elements.
2. Identify the key characters, settings, and events.
3. Highlight any major changes in tone, emotion, or topic.
4. Create a scene-by-scene breakdown with timestamps.
5. Provide a final concise TL;DR summary (3–4 sentences).

The output should read like a human-written report, not raw data.

Here is the metadata JSON:
{json.dumps(structured_data, ensure_ascii=False)}
"""
        
        try:
            logger.info("Sending insights to GPT-4o for summarization...")
            
            # Following exact working pattern from original code
            response = self.openai_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a senior multimedia content summarizer."},
                    {"role": "user", "content": detailed_prompt}
                ],
                max_tokens=8192,
                temperature=0.6,
                top_p=1.0,
                model=openai_config['deployment']
            )
            
            final_summary = response.choices[0].message.content
            logger.info("Summary generation complete")
            return final_summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            # Return a fallback summary based on the structured data
            return self.generate_fallback_summary(structured_data)
    
    def generate_fallback_summary(self, structured_data):
        """Generate a basic summary when OpenAI is unavailable"""
        logger.info("Generating fallback summary from structured data")
        
        # Extract basic information
        transcript = structured_data.get('transcript', [])
        speakers = structured_data.get('speakers', [])
        duration = structured_data.get('video_metadata', {}).get('duration', 0)
        
        # Create basic summary
        summary = f"""# Video Analysis Report

## Summary
Video analysis completed successfully. Duration: {duration} seconds.

## Content Overview
- **Speakers Detected**: {len(speakers)} speaker(s) identified
- **Transcript Length**: {len(transcript)} transcript segments
- **Processing Status**: Analysis completed with basic extraction

## Transcript Preview
"""
        
        # Add first few transcript lines
        for i, segment in enumerate(transcript[:5]):
            if isinstance(segment, dict) and 'text' in segment:
                summary += f"- {segment.get('text', '')}\n"
        
        if len(transcript) > 5:
            summary += f"... and {len(transcript) - 5} more segments\n"
        
        summary += """
## Note
This is a basic summary generated due to OpenAI service limitations. 
For enhanced analysis, please check your Azure OpenAI configuration.
"""
        
        return summary
    
    def test_openai_connection(self):
        """Test OpenAI connection with a simple request - following exact working pattern"""
        try:
            response = self.openai_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, respond with 'Connection successful'"}
                ],
                max_tokens=10,
                temperature=0.7,
                top_p=1.0,
                model=self.config['openai']['deployment']
            )
            logger.info("OpenAI connection test successful")
            return True
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            return False
    
    def analyze_video(self, video_path, video_name, video_description):
        """
        Main method to analyze a video file
        
        Args:
            video_path (str): Path to the video file
            video_name (str): Name for the video
            video_description (str): Description for the video
            
        Returns:
            str: Human-readable analysis summary in markdown format
        """
        try:
            # Step 1: Get access token
            access_token = self.get_access_token()
            
            # Step 2: Upload video
            video_id = self.upload_video(access_token, video_path, video_name, video_description)
            
            # Step 3: Wait for indexing
            insights = self.wait_for_indexing(access_token, video_id)
            
            # Step 4: Extract insights
            structured_data = self.extract_insights(insights)
            
            # Step 5: Generate summary
            final_summary = self.generate_summary(structured_data)
            
            return final_summary
            
        except Exception as e:
            logger.error(f"Video analysis failed: {str(e)}")
            raise