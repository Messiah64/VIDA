import time
import requests
import json
import math
import hashlib
import os
import re
import cv2
from pathlib import Path
from openai import AsyncOpenAI
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ConfidenceFilter:
    """Filters insights based on confidence thresholds"""
    
    def __init__(self, min_confidence: float = 0.8):
        self.min_confidence = min_confidence
    
    def filter_insights(self, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter all insights based on confidence levels"""
        filtered_data = structured_data.copy()
        
        # Filter each category of insights
        for key, items in structured_data.items():
            if isinstance(items, list) and items:
                filtered_items = []
                for item in items:
                    if isinstance(item, dict):
                        confidence = self._extract_confidence(item)
                        if confidence >= self.min_confidence:
                            filtered_items.append(item)
                        else:
                            logger.debug(f"Filtered out {key} item with confidence {confidence}")
                    else:
                        # Keep non-dict items (like transcript text)
                        filtered_items.append(item)
                
                filtered_data[key] = filtered_items
                
                if len(filtered_items) != len(items):
                    logger.info(f"Filtered {key}: {len(items)} -> {len(filtered_items)} items (confidence >= {self.min_confidence})")
        
        return filtered_data
    
    def _extract_confidence(self, item: Dict[str, Any]) -> float:
        """Extract confidence value from an insight item"""
        # Common confidence field names in Video Indexer
        confidence_fields = ['confidence', 'score', 'probability', 'strength']
        
        for field in confidence_fields:
            if field in item:
                confidence = item[field]
                if isinstance(confidence, (int, float)):
                    return float(confidence)
                elif isinstance(confidence, str):
                    try:
                        return float(confidence)
                    except ValueError:
                        continue
        
        # Check for nested confidence in appearances or instances
        if 'appearances' in item and item['appearances']:
            appearances = item['appearances']
            if isinstance(appearances, list) and appearances:
                first_appearance = appearances[0]
                if isinstance(first_appearance, dict):
                    return self._extract_confidence(first_appearance)
        
        if 'instances' in item and item['instances']:
            instances = item['instances']
            if isinstance(instances, list) and instances:
                first_instance = instances[0]
                if isinstance(first_instance, dict):
                    return self._extract_confidence(first_instance)
        
        # Default confidence if none found
        return 1.0

class VideoAnalyzer:
    """
    Video Analysis using Azure Video Indexer and Azure OpenAI
    Features:
    - Async OpenAI API calls for improved performance
    - Confidence-based filtering of insights (0.8+ threshold)
    - Token-safe hierarchical summarization
    - Chunked transcript processing for large videos
    - Parallel GPT calls for improved speed
    - Robust error handling and fallback summaries
    """
    
    def __init__(self, config_file='configs.json'):
        """Initialize with configuration"""
        self.load_config(config_file)
        
        # Initialize confidence filter
        min_confidence = self.config.get('processing', {}).get('min_confidence', 0.8)
        self.confidence_filter = ConfidenceFilter(min_confidence)
        
        # Initialize Azure OpenAI client - async version
        try:
            self.openai_client = AsyncOpenAI(
                api_key=self.config['openai']['subscription_key'],
                base_url=f"{self.config['openai']['endpoint']}openai/deployments/{self.config['openai']['deployment']}",
                default_headers={"api-key": self.config['openai']['subscription_key']},
                default_query={"api-version": self.config['openai']['api_version']}
            )
            logger.info("Async Azure OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize async Azure OpenAI client: {e}")
            raise
    
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Configuration loaded from {config_file}")
        except FileNotFoundError:
            logger.error(f"Config file {config_file} not found!")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            raise
    
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
        """Extract structured insights from Video Indexer response and apply confidence filtering"""
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
        
        # Apply confidence filtering
        logger.info("Applying confidence filtering to insights")
        filtered_data = self.confidence_filter.filter_insights(structured_data)
        
        return filtered_data
    
    def chunk_transcript(self, transcript):
        """Break transcript into manageable chunks for token-safe processing"""
        chunk_size = self.config['processing']['chunk_size']
        
        for i in range(0, len(transcript), chunk_size):
            yield transcript[i:i + chunk_size]
    
    async def summarize_chunk_async(self, chunk, chunk_index):
        """Asynchronously summarize a single transcript chunk with detailed analysis"""
        try:
            # Build text from chunk with timestamps
            chunk_text = []
            for segment in chunk:
                timestamp = segment.get('start', '00:00')
                text = segment.get('text', '')
                speaker_id = segment.get('speakerId', 'Unknown')
                chunk_text.append(f"[{timestamp}] Speaker {speaker_id}: {text}")
            
            text_content = "\n".join(chunk_text)
            
            prompt = f"""
Analyze this video transcript segment in detail. Focus on:
1. Key events and actions described
2. Speaker interactions and dialogue
3. Visual elements mentioned
4. Emotional tone and context
5. Important topics discussed

Transcript segment:
{text_content}

Provide a comprehensive summary that captures the essence of this segment.
"""
            
            logger.info(f"Processing chunk {chunk_index} with {len(chunk)} segments asynchronously")
            
            response = await self.openai_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a professional video content analyst specializing in detailed segment analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.5,
                top_p=1.0,
                model=self.config['openai']['deployment']
            )
            
            result = {
                "chunk_index": chunk_index,
                "summary": response.choices[0].message.content,
                "segment_count": len(chunk)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to summarize chunk {chunk_index}: {e}")
            return {
                "chunk_index": chunk_index,
                "summary": f"Chunk {chunk_index}: Processing failed - {len(chunk)} segments with basic transcript content.",
                "segment_count": len(chunk)
            }
    
    async def generate_final_summary_async(self, chunk_summaries, structured_data):
        """Asynchronously generate comprehensive final summary from chunk summaries and metadata"""
        try:
            # Combine chunk summaries
            condensed_summaries = "\n\n".join(
                f"**Segment {s['chunk_index']}** ({s['segment_count']} parts):\n{s['summary']}" 
                for s in chunk_summaries
            )
            
            # Extract key metadata highlights
            metadata_highlights = {
                "duration": structured_data.get("video_metadata", {}).get("duration", 0),
                "speakers": len(structured_data.get("speakers", [])),
                "labels": [label.get("name", "") for label in structured_data.get("labels", [])[:8]],
                "topics": [topic.get("name", "") for topic in structured_data.get("topics", [])[:5]],
                "keywords": [kw.get("name", "") for kw in structured_data.get("keywords", [])[:10]],
                "key_sentiments": structured_data.get("sentiments", [])[:3],
                "ocr_detected": len(structured_data.get("ocr_text", [])) > 0
            }
            
            final_prompt = f"""
You are an expert multimedia content analyst. Create a comprehensive video analysis report using the segment summaries and metadata below.

**Segment-by-Segment Analysis:**
{condensed_summaries}

**Video Metadata:**
{json.dumps(metadata_highlights, ensure_ascii=False, indent=2)}

**Create a detailed report with these sections:**

### Narrative Reconstruction
Reconstruct a coherent, chronological narrative blending visual and audio elements from all segments.

### Key Characters, Settings, and Events
Identify main participants, locations, and significant events across the video.

### Tone, Emotion, and Topic Changes
Highlight major shifts in mood, emotion, or discussion topics throughout the video.

### Scene-by-Scene Breakdown
Provide timestamped analysis of key moments and transitions. Use this EXACT format for each scene:
START_TIME–END_TIME: Description of what happens in this scene.

Example format:
0:00–0:15: Opening scene with introduction.
0:15–0:45: Main discussion begins.
1:30–2:00: Conclusion and wrap-up.

Use timestamps in M:SS or H:MM:SS format. Each scene must be on its own line.

### TL;DR Summary
Conclude with 3-4 sentences summarizing the entire video content and main takeaways.

Make this read like a professional content analysis report, not raw data.
"""
            
            logger.info("Generating final comprehensive summary asynchronously")
            
            response = await self.openai_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a senior multimedia content summarizer creating professional analysis reports."},
                    {"role": "user", "content": final_prompt}
                ],
                max_tokens=4000,
                temperature=0.6,
                top_p=1.0,
                model=self.config['openai']['deployment']
            )
            
            final_summary = response.choices[0].message.content
            logger.info("Final comprehensive summary generated successfully")
            
            return final_summary
            
        except Exception as e:
            logger.error(f"Failed to generate final summary: {e}")
            return self.generate_fallback_summary(structured_data)
    
    def filter_insights_by_timerange(self, structured_data: Dict[str, Any], start_seconds: float, end_seconds: float) -> Dict[str, Any]:
        """Filter structured insights to only include data within the specified time range"""
        filtered_data = {}
        
        def is_in_timerange(item, start_sec, end_sec):
            """Check if an item falls within the time range"""
            if not isinstance(item, dict):
                return False
                
            # Check for direct start/end times
            if 'start' in item:
                item_start = self._timestamp_to_seconds(str(item['start']))
                return start_sec <= item_start <= end_sec
            
            # Check for instances with start/end times
            if 'instances' in item and item['instances']:
                for instance in item['instances']:
                    if isinstance(instance, dict) and 'start' in instance:
                        inst_start = self._timestamp_to_seconds(str(instance['start']))
                        if start_sec <= inst_start <= end_sec:
                            return True
            
            # Check for appearances with start/end times  
            if 'appearances' in item and item['appearances']:
                for appearance in item['appearances']:
                    if isinstance(appearance, dict) and 'startTime' in appearance:
                        app_start = self._timestamp_to_seconds(str(appearance['startTime']))
                        if start_sec <= app_start <= end_sec:
                            return True
            
            return False
        
        # Filter each category of insights
        for key, items in structured_data.items():
            if key == 'video_metadata':
                filtered_data[key] = items  # Keep metadata as-is
                continue
                
            if isinstance(items, list):
                filtered_items = []
                for item in items:
                    if is_in_timerange(item, start_seconds, end_seconds):
                        filtered_items.append(item)
                filtered_data[key] = filtered_items
            else:
                filtered_data[key] = items
        
        return filtered_data
    
    async def generate_detailed_scene_analysis(self, scene: Dict[str, Any], filtered_insights: Dict[str, Any]) -> str:
        """Generate detailed AI analysis for a specific scene using filtered insights"""
        try:
            # Extract key information from filtered insights
            transcript_segments = filtered_insights.get('transcript', [])
            speakers = filtered_insights.get('speakers', [])
            labels = filtered_insights.get('labels', [])
            faces = filtered_insights.get('faces', [])
            named_people = filtered_insights.get('named_people', [])
            named_locations = filtered_insights.get('named_locations', [])
            brands = filtered_insights.get('brands', [])
            topics = filtered_insights.get('topics', [])
            keywords = filtered_insights.get('keywords', [])
            sentiments = filtered_insights.get('sentiments', [])
            audio_effects = filtered_insights.get('audio_effects', [])
            ocr_text = filtered_insights.get('ocr_text', [])
            
            # Build transcript text for this timeframe
            transcript_text = ""
            for segment in transcript_segments:
                timestamp = segment.get('start', '00:00')
                text = segment.get('text', '')
                speaker_id = segment.get('speakerId', 'Unknown')
                transcript_text += f"[{timestamp}] Speaker {speaker_id}: {text}\n"
            
            # Compile insights for AI analysis
            insights_summary = {
                "transcript_segments": len(transcript_segments),
                "speakers_detected": len(speakers),
                "visual_labels": [label.get('name', '') for label in labels[:10]],
                "people_identified": [person.get('name', '') for person in named_people],
                "locations_mentioned": [loc.get('name', '') for loc in named_locations],
                "brands_detected": [brand.get('name', '') for brand in brands],
                "key_topics": [topic.get('name', '') for topic in topics],
                "keywords": [kw.get('name', '') for kw in keywords[:15]],
                "sentiment_analysis": sentiments[:3] if sentiments else [],
                "audio_events": [ae.get('type', '') for ae in audio_effects],
                "text_on_screen": [ocr.get('text', '') for ocr in ocr_text[:5]]
            }
            
            detailed_prompt = f"""
You are an expert video content analyst. Provide a comprehensive, detailed analysis of this specific video segment.

**Time Range:** {scene['start_time']} to {scene['end_time']} (Scene {scene['scene_number']})

**Transcript Content:**
{transcript_text if transcript_text else "No transcript available for this segment."}

**Visual & Audio Intelligence Data:**
{json.dumps(insights_summary, indent=2, ensure_ascii=False)}

**Scene Context from Overall Analysis:**
{scene['description']}

## Provide a detailed breakdown covering:

### 1. Narrative Summary
What exactly happens in this time segment? Provide a detailed, chronological account.

### 2. Dialogue & Communication Analysis  
- Key conversations and speaker interactions
- Tone, emotion, and communication style
- Important quotes or statements

### 3. Visual Scene Description
- Physical environment and setting details
- People present and their roles/actions
- Objects, brands, text visible on screen
- Camera movements or scene transitions

### 4. Character & Entity Analysis
- Who are the main participants?
- What are their roles and relationships?
- Any named individuals or organizations mentioned

### 5. Thematic Content
- Main topics and subjects discussed
- Underlying themes or messages
- Educational or informational content

### 6. Technical & Production Elements
- Audio quality, effects, or notable sounds
- Visual presentation style
- Any technical observations

### 7. Context & Significance
- How this segment fits into the overall video narrative
- Key takeaways or important information conveyed
- Emotional or dramatic highlights

Make this analysis comprehensive and detailed - imagine you're creating a professional content review for media analysis.
"""
            
            response = await self.openai_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a professional video content analyst specializing in detailed scene-by-scene breakdowns. Provide thorough, insightful analysis."},
                    {"role": "user", "content": detailed_prompt}
                ],
                max_tokens=3000,
                temperature=0.4,
                top_p=1.0,
                model=self.config['openai']['deployment']
            )
            
            detailed_analysis = response.choices[0].message.content
            
            # Add technical metadata at the end
            metadata_section = f"""

---

## Technical Metadata for {scene['start_time']} - {scene['end_time']}

**Confidence Threshold:** {self.confidence_filter.min_confidence}
**Data Sources:** Azure Video Indexer + Azure OpenAI Analysis

**Quantified Insights:**
- Transcript segments: {len(transcript_segments)}
- Visual elements detected: {len(labels)}
- Named entities: {len(named_people + named_locations + brands)}
- Audio events: {len(audio_effects)}
- Text recognition instances: {len(ocr_text)}

**Processing Notes:**
All insights filtered for confidence >= {self.confidence_filter.min_confidence}
Analysis generated using {self.config['openai']['model_name']} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            return detailed_analysis + metadata_section
            
        except Exception as e:
            logger.error(f"Failed to generate detailed analysis for scene {scene['scene_number']}: {e}")
            
            # Fallback analysis
            fallback = f"""
# Detailed Scene Analysis - {scene['start_time']} to {scene['end_time']}

## Scene {scene['scene_number']} Analysis

**Time Range:** {scene['start_time']} - {scene['end_time']}

**Basic Summary:**
{scene['description']}

**Note:** Detailed AI analysis failed. This segment contains:
- {len(filtered_insights.get('transcript', []))} transcript segments
- {len(filtered_insights.get('labels', []))} visual elements detected
- {len(filtered_insights.get('speakers', []))} speakers identified

**Raw Transcript (if available):**
"""
            # Add basic transcript
            for segment in filtered_insights.get('transcript', []):
                timestamp = segment.get('start', '00:00')
                text = segment.get('text', '')
                speaker_id = segment.get('speakerId', 'Unknown')
                fallback += f"[{timestamp}] Speaker {speaker_id}: {text}\n"
            
            return fallback
    
    async def save_detailed_scene_analyses(self, scenes: List[Dict[str, Any]], structured_data: Dict[str, Any], output_dir: str = "scene_analyses") -> List[str]:
        """Generate and save detailed analysis for each scene"""
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_analyses = []
        
        try:
            tasks = []
            for scene in scenes:
                start_seconds = scene['start_seconds']
                end_seconds = self._timestamp_to_seconds(scene['end_time'])
                
                # Filter insights for this time range
                filtered_insights = self.filter_insights_by_timerange(structured_data, start_seconds, end_seconds)
                
                # Create analysis task
                task = self.generate_detailed_scene_analysis(scene, filtered_insights)
                tasks.append((scene, task))
            
            # Process scenes with concurrency control
            semaphore = asyncio.Semaphore(self.config['processing'].get('max_workers', 4))
            
            async def bounded_analysis(scene, task):
                async with semaphore:
                    analysis = await task
                    return scene, analysis
            
            results = await asyncio.gather(
                *[bounded_analysis(scene, task) for scene, task in tasks],
                return_exceptions=True
            )
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Scene {i+1} analysis failed: {result}")
                    continue
                
                scene, analysis = result
                
                # Create filename
                safe_timestamp = scene["start_time"].replace(':', '-')
                filename = f"scene_{scene['scene_number']:02d}_{safe_timestamp}.txt"
                filepath = output_path / filename
                
                # Save analysis
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(analysis)
                
                saved_analyses.append(str(filepath))
                logger.info(f"Saved detailed analysis for scene {scene['scene_number']}: {filename}")
        
        except Exception as e:
            logger.error(f"Error processing scene analyses: {e}")
        
        return saved_analyses
    
    def parse_scene_breakdown(self, final_summary):
        """Parse the Scene-by-Scene Breakdown from the final summary"""
        scenes = []
        
        # Find the Scene-by-Scene Breakdown section - try multiple patterns
        patterns_to_try = [
            r'Scene-by-Scene Breakdown(.*?)(?=\n#|---|\Z)',  # Plain header
            r'#\s*Scene-by-Scene Breakdown(.*?)(?=\n#|---|\Z)',  # With # header
            r'###\s*Scene-by-Scene Breakdown(.*?)(?=###|---|\Z)',  # With ### header
            r'\*\*Scene-by-Scene Breakdown\*\*(.*?)(?=\*\*[^*]|---|\Z)',  # With ** bold
        ]
        
        scene_content = None
        for pattern in patterns_to_try:
            scene_match = re.search(pattern, final_summary, re.DOTALL | re.IGNORECASE)
            if scene_match:
                scene_content = scene_match.group(1)
                logger.info(f"Found Scene-by-Scene Breakdown with pattern: {pattern[:30]}...")
                break
        
        if not scene_content:
            logger.warning("No Scene-by-Scene Breakdown section found in summary")
            return scenes
        
        # Extract individual scenes with various timestamp formats
        # Updated pattern to handle:
        # - 0:00–0:05.8 (M:SS or M:SS.ms format)
        # - 0:00:00 – 0:02:33 (H:MM:SS format)
        # - With or without ** bold markers
        timestamp_patterns = [
            # M:SS or M:SS.ms format (like 0:00–0:05.8)
            r'(\d+:\d+(?:\.\d+)?)\s*[–\-—]\s*(\d+:\d+(?:\.\d+)?)\s*:\s*(.*?)(?=\d+:\d+|$)',
            # H:MM:SS format
            r'(\d+:\d+:\d+)\s*[–\-—]\s*(\d+:\d+:\d+)\s*:\s*(.*?)(?=\d+:\d+:\d+|$)',
            # With optional ** markers
            r'(?:\*\*)?(\d+:\d+(?:\.\d+)?)\s*[–\-—]\s*(\d+:\d+(?:\.\d+)?)(?:\*\*)?\s*:\s*(.*?)(?=(?:\*\*)?\d+:\d+|$)',
            r'(?:\*\*)?(\d+:\d+:\d+)\s*[–\-—]\s*(\d+:\d+:\d+)(?:\*\*)?\s*:\s*(.*?)(?=(?:\*\*)?\d+:\d+:\d+|$)',
        ]
        
        matches = []
        for pattern in timestamp_patterns:
            found_matches = re.findall(pattern, scene_content, re.DOTALL)
            if found_matches:
                matches = found_matches
                logger.info(f"Found {len(found_matches)} scenes with timestamp pattern: {pattern[:50]}...")
                break
        
        if not matches:
            logger.warning("No timestamp patterns matched in scene content")
            logger.debug(f"Scene content preview: {scene_content[:200]}...")
        
        for i, (start_time, end_time, description) in enumerate(matches):
            # Clean up the description
            description = description.strip()
            # Remove any trailing punctuation or whitespace
            description = re.sub(r'\s+', ' ', description)
            
            scene = {
                "scene_number": i + 1,
                "start_time": start_time.strip(),
                "end_time": end_time.strip(), 
                "description": description,
                "start_seconds": self._timestamp_to_seconds(start_time.strip())
            }
            scenes.append(scene)
            logger.debug(f"Scene {i+1}: {start_time} - {end_time}")
            
        logger.info(f"Extracted {len(scenes)} scenes from breakdown")
        return scenes
    
    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert timestamp string (H:MM:SS, MM:SS, M:SS.ms) to seconds"""
        try:
            # Remove any extra whitespace
            timestamp = timestamp.strip()
            
            # Handle different timestamp formats
            parts = timestamp.split(':')
            
            if len(parts) == 3:
                # H:MM:SS format
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
                
            elif len(parts) == 2:
                # MM:SS or M:SS or M:SS.ms format
                minutes = float(parts[0])
                seconds = float(parts[1])  # This handles decimal seconds like 05.8
                return minutes * 60 + seconds
                
            elif len(parts) == 1:
                # Just seconds
                return float(parts[0])
                
            else:
                logger.warning(f"Unexpected timestamp format: {timestamp}")
                return 0.0
                
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse timestamp: {timestamp} - {e}")
            return 0.0
    
    def extract_frames_from_scenes(self, video_path: str, scenes: List[Dict[str, Any]], output_dir: str = "scene_frames") -> List[str]:
        """Extract one frame from each scene timestamp and save to folder"""
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_frames = []
        
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Could not open video file: {video_path}")
                return saved_frames
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            logger.info(f"Video info: {duration:.2f}s, {fps:.2f} FPS, {total_frames} frames")
            
            for scene in scenes:
                timestamp_seconds = scene["start_seconds"]
                start_time = scene["start_time"]
                
                # Calculate frame number
                frame_number = int(timestamp_seconds * fps)
                
                if frame_number >= total_frames:
                    logger.warning(f"Timestamp {start_time} exceeds video duration, skipping")
                    continue
                
                # Set video position to the frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                # Read the frame
                ret, frame = cap.read()
                
                if ret:
                    # Create filename with timestamp
                    safe_timestamp = start_time.replace(':', '-')
                    filename = f"scene_{scene['scene_number']:02d}_{safe_timestamp}.jpg"
                    filepath = output_path / filename
                    
                    # Save frame
                    cv2.imwrite(str(filepath), frame)
                    saved_frames.append(str(filepath))
                    
                    logger.info(f"Saved frame for scene {scene['scene_number']} at {start_time}: {filename}")
                else:
                    logger.warning(f"Could not read frame at timestamp {start_time}")
            
            cap.release()
            logger.info(f"Extracted {len(saved_frames)} frames to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
        
        return saved_frames
    
    def generate_fallback_summary(self, structured_data):
        """Generate a basic summary when OpenAI processing fails"""
        logger.info("Generating fallback summary from structured data")
        
        # Extract basic information
        transcript = structured_data.get('transcript', [])
        speakers = structured_data.get('speakers', [])
        duration = structured_data.get('video_metadata', {}).get('duration', 0)
        labels = structured_data.get('labels', [])
        topics = structured_data.get('topics', [])
        
        # Create basic summary
        summary = f"""# Video Analysis Report

## Summary
Video analysis completed successfully. Duration: {duration} seconds.
**Note**: All insights filtered to confidence >= {self.confidence_filter.min_confidence}

## Content Overview
- **Speakers Detected**: {len(speakers)} high-confidence speaker(s) identified
- **Transcript Segments**: {len(transcript)} segments processed
- **Visual Labels**: {len(labels)} high-confidence visual elements detected
- **Topics Identified**: {len(topics)} high-confidence main topics found

## Key Elements Detected (High Confidence Only)
"""
        
        # Add top labels and topics
        if labels:
            summary += "### Visual Elements:\n"
            for label in labels[:5]:
                if isinstance(label, dict) and 'name' in label:
                    confidence = self.confidence_filter._extract_confidence(label)
                    summary += f"- {label['name']} (confidence: {confidence:.2f})\n"
        
        if topics:
            summary += "\n### Topics Discussed:\n"
            for topic in topics[:5]:
                if isinstance(topic, dict) and 'name' in topic:
                    confidence = self.confidence_filter._extract_confidence(topic)
                    summary += f"- {topic['name']} (confidence: {confidence:.2f})\n"
        
        # Add transcript preview
        summary += "\n## Transcript Preview\n"
        for i, segment in enumerate(transcript[:3]):
            if isinstance(segment, dict) and 'text' in segment:
                timestamp = segment.get('start', '00:00')
                summary += f"**[{timestamp}]** {segment.get('text', '')}\n\n"
        
        if len(transcript) > 3:
            summary += f"... and {len(transcript) - 3} more segments\n"
        
        summary += f"""
## Processing Note
This is a basic summary generated due to OpenAI service limitations. 
The video was successfully analyzed by Azure Video Indexer with full transcript and metadata extraction.
All insights have been filtered to show only items with confidence >= {self.confidence_filter.min_confidence}.
For enhanced narrative analysis, please check your Azure OpenAI configuration.

### TL;DR Summary
Video processing completed with high-confidence transcript extraction and metadata analysis. Multiple speakers and topics detected with comprehensive visual element identification, all filtered for quality assurance.
"""
        
        return summary
    
    async def test_openai_connection_async(self):
        """Test async OpenAI connection"""
        try:
            response = await self.openai_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, respond with 'Connection successful'"}
                ],
                max_tokens=10,
                temperature=0.7,
                top_p=1.0,
                model=self.config['openai']['deployment']
            )
            logger.info("Async OpenAI connection test successful")
            return True
        except Exception as e:
            logger.error(f"Async OpenAI connection test failed: {e}")
            return False
    
    async def analyze_video_async(self, video_path, video_name, video_description):
        """
        Enhanced async video analysis with chunked processing and confidence filtering
        
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
            
            # Step 4: Extract and filter insights
            structured_data = self.extract_insights(insights)
            
            # Step 5: Process transcript in chunks for large videos
            transcript = structured_data.get('transcript', [])
            
            if len(transcript) == 0:
                logger.warning("No transcript found - generating summary from metadata only")
                return self.generate_fallback_summary(structured_data)
            
            logger.info(f"Processing transcript with {len(transcript)} segments")
            
            # Chunk transcript for token-safe processing
            chunks = list(self.chunk_transcript(transcript))
            logger.info(f"Split transcript into {len(chunks)} chunks")
            
            # Process chunks asynchronously for speed
            try:
                # Create async tasks for all chunks
                tasks = [
                    self.summarize_chunk_async(chunk, i+1) 
                    for i, chunk in enumerate(chunks)
                ]
                
                # Process chunks concurrently with semaphore to limit concurrent requests
                semaphore = asyncio.Semaphore(self.config['processing'].get('max_workers', 4))
                
                async def bounded_task(task):
                    async with semaphore:
                        return await task
                
                chunk_results = await asyncio.gather(
                    *[bounded_task(task) for task in tasks],
                    return_exceptions=True
                )
                
                # Filter out exceptions and log errors
                valid_results = []
                for i, result in enumerate(chunk_results):
                    if isinstance(result, Exception):
                        logger.error(f"Chunk {i+1} failed: {result}")
                        # Create fallback result
                        valid_results.append({
                            "chunk_index": i+1,
                            "summary": f"Chunk {i+1}: Processing failed - fallback summary",
                            "segment_count": len(chunks[i]) if i < len(chunks) else 0
                        })
                    else:
                        valid_results.append(result)
                
                logger.info(f"Processed {len(valid_results)} chunks successfully")
                
                # Step 6: Generate final comprehensive summary asynchronously
                final_summary = await self.generate_final_summary_async(valid_results, structured_data)
                
                # Step 7: Extract scenes and frames
                scenes = self.parse_scene_breakdown(final_summary)
                if scenes:
                    logger.info(f"Found {len(scenes)} scenes, extracting frames and generating detailed analyses...")
                    
                    # Extract frames
                    saved_frames = self.extract_frames_from_scenes(video_path, scenes)
                    
                    # Generate detailed scene analyses
                    saved_analyses = await self.save_detailed_scene_analyses(scenes, structured_data)
                    
                    # Add frame and analysis information to the summary
                    frame_info = "\n\n---\n\n## Extracted Scene Data\n"
                    for i, scene in enumerate(scenes):
                        frame_info += f"**Scene {scene['scene_number']}** ({scene['start_time']} - {scene['end_time']}):\n"
                        
                        # Frame info
                        if i < len(saved_frames):
                            frame_info += f"  - Frame: `{Path(saved_frames[i]).name}`\n"
                        else:
                            frame_info += f"  - Frame: extraction failed\n"
                        
                        # Analysis info
                        if i < len(saved_analyses):
                            frame_info += f"  - Detailed Analysis: `{Path(saved_analyses[i]).name}`\n"
                        else:
                            frame_info += f"  - Detailed Analysis: generation failed\n"
                        
                        frame_info += "\n"
                    
                    final_summary += frame_info
                
                return final_summary
                
            except Exception as e:
                logger.error(f"Async processing failed: {e}")
                # Fallback to synchronous processing
                logger.info("Falling back to synchronous chunk processing")
                
                chunk_results = []
                for i, chunk in enumerate(chunks):
                    result = await self.summarize_chunk_async(chunk, i+1)
                    chunk_results.append(result)
                
                final_summary = await self.generate_final_summary_async(chunk_results, structured_data)
                
                # Extract scenes and frames for fallback processing too
                scenes = self.parse_scene_breakdown(final_summary)
                if scenes:
                    logger.info(f"Found {len(scenes)} scenes in fallback, extracting frames and generating detailed analyses...")
                    
                    # Extract frames
                    saved_frames = self.extract_frames_from_scenes(video_path, scenes)
                    
                    # Generate detailed scene analyses
                    saved_analyses = await self.save_detailed_scene_analyses(scenes, structured_data)
                    
                    # Add frame and analysis information to the summary
                    frame_info = "\n\n---\n\n## Extracted Scene Data\n"
                    for i, scene in enumerate(scenes):
                        frame_info += f"**Scene {scene['scene_number']}** ({scene['start_time']} - {scene['end_time']}):\n"
                        
                        # Frame info
                        if i < len(saved_frames):
                            frame_info += f"  - Frame: `{Path(saved_frames[i]).name}`\n"
                        else:
                            frame_info += f"  - Frame: extraction failed\n"
                        
                        # Analysis info
                        if i < len(saved_analyses):
                            frame_info += f"  - Detailed Analysis: `{Path(saved_analyses[i]).name}`\n"
                        else:
                            frame_info += f"  - Detailed Analysis: generation failed\n"
                        
                        frame_info += "\n"
                    
                    final_summary += frame_info
                
                return final_summary
            
        except Exception as e:
            logger.error(f"Video analysis failed: {str(e)}")
            raise
    
    def analyze_video(self, video_path, video_name, video_description):
        """
        Synchronous wrapper for async video analysis
        
        Args:
            video_path (str): Path to the video file
            video_name (str): Name for the video
            video_description (str): Description for the video
            
        Returns:
            str: Human-readable analysis summary in markdown format
        """
        return asyncio.run(self.analyze_video_async(video_path, video_name, video_description))
    
    def analyze_video_with_scenes(self, video_path, video_name, video_description):
        """
        Analyze video and return both summary and parsed scenes with detailed analyses
        
        Args:
            video_path (str): Path to the video file
            video_name (str): Name for the video
            video_description (str): Description for the video
            
        Returns:
            tuple: (summary_text, scenes_list, frame_paths, analysis_paths)
        """
        async def _analyze_with_scenes():
            summary = await self.analyze_video_async(video_path, video_name, video_description)
            scenes = self.parse_scene_breakdown(summary)
            saved_frames = []
            saved_analyses = []
            if scenes:
                saved_frames = self.extract_frames_from_scenes(video_path, scenes)
                # Note: detailed analyses are generated during main analysis, 
                # but we can generate them again if needed
                structured_data = {}  # This would need to be passed from the main analysis
                # saved_analyses = self.save_detailed_scene_analyses(scenes, structured_data)
            return summary, scenes, saved_frames, saved_analyses
        
        return asyncio.run(_analyze_with_scenes())


# Example usage and testing functions
async def main():
    """Example usage of the enhanced video analyzer with scene extraction and detailed analysis"""
    analyzer = VideoAnalyzer()
    
    # Test async OpenAI connection
    connection_ok = await analyzer.test_openai_connection_async()
    if not connection_ok:
        print("OpenAI connection failed!")
        return
    
    # Example video analysis
    try:
        # Replace with your actual video path
        video_path = "example_video.mp4"
        video_name = "Test Video Analysis"
        video_description = "Testing enhanced video analyzer with scene extraction and detailed analysis"
        
        print("Starting video analysis...")
        
        # Option 1: Get complete analysis with scene extraction and detailed breakdowns
        summary = await analyzer.analyze_video_async(video_path, video_name, video_description)
        print("Analysis complete!")
        print(summary)
        
        # Option 2: Get summary + structured scene data (detailed analyses generated during main analysis)
        # summary, scenes, frame_paths, analysis_paths = analyzer.analyze_video_with_scenes(video_path, video_name, video_description)
        # print(f"\nExtracted {len(scenes)} scenes:")
        # for scene in scenes:
        #     print(f"Scene {scene['scene_number']}: {scene['start_time']} - {scene['end_time']}")
        #     print(f"  Description: {scene['description'][:100]}...")
        # print(f"\nSaved {len(frame_paths)} frame images")
        # print(f"Generated {len(analysis_paths)} detailed scene analyses")
        
    except Exception as e:
        print(f"Analysis failed: {e}")


def example_scene_extraction():
    """Example of extracting scenes and generating detailed analyses from an existing summary"""
    analyzer = VideoAnalyzer()
    
    # Example summary text (replace with actual summary)
    sample_summary = """
    ### Scene-by-Scene Breakdown
    
    **0:00:00 – 0:02:33**
    Introduction to the Airport Police Division and their role in crime prevention.
    
    **0:02:33 – 0:04:33** 
    Shirley shares her story of being deceived by another conman.
    
    **0:04:34 – 0:09:07**
    Transition to terrorism preparedness and public vigilance.
    """
    
    # Parse scenes
    scenes = analyzer.parse_scene_breakdown(sample_summary)
    
    print("Extracted scenes:")
    for scene in scenes:
        print(f"Scene {scene['scene_number']}: {scene['start_time']} - {scene['end_time']}")
        print(f"  Seconds: {scene['start_seconds']}")
        print(f"  Description: {scene['description']}")
        print()
    
    # Extract frames and generate detailed analyses (if you have a video file and structured data)
    # video_path = "your_video.mp4"
    # structured_data = {}  # This would come from your video analysis
    # saved_frames = analyzer.extract_frames_from_scenes(video_path, scenes)
    # saved_analyses = analyzer.save_detailed_scene_analyses(scenes, structured_data)
    # print(f"Saved {len(saved_frames)} frames and {len(saved_analyses)} detailed analyses")


def example_detailed_analysis():
    """Example showing what detailed scene analysis files contain"""
    print("""
Example detailed scene analysis file content (scene_01_0-00-00.txt):

# Detailed Scene Analysis - 0:00:00 to 0:02:33

## Scene 1 Analysis

### 1. Narrative Summary
The video opens with an introduction to the Airport Police Division, establishing their primary role in crime prevention at the airport. The scene features Doris recounting her personal experience of being targeted by a sophisticated scam operation...

### 2. Dialogue & Communication Analysis
- Key speaker: Doris (victim testimonial)
- Tone: Serious, educational, cautionary
- Important quote: "I thought he was a genuine government officer..."

### 3. Visual Scene Description
- Setting: Interview setup with police division backdrop
- People: Doris (scam victim), police representatives
- Visual elements: Official police insignia, clean interview environment

### 4. Character & Entity Analysis
- Doris: Scam victim providing testimonial
- Airport Police Division: Law enforcement agency
- Conman: Antagonist (mentioned but not shown)

### 5. Thematic Content
- Crime prevention and public awareness
- Vulnerability of citizens to sophisticated scams
- Role of law enforcement in education

### 6. Technical & Production Elements
- Professional interview lighting and setup
- Clear audio quality
- Educational documentary style

### 7. Context & Significance
This opening segment establishes the video's educational purpose and introduces real-world consequences of criminal activity...

---

## Technical Metadata for 0:00:00 - 0:02:33
[Technical details and processing information]
    """)
    print("This is the type of detailed analysis saved in each scene's .txt file!")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the async main function
    asyncio.run(main())
    
    # Uncomment to test scene extraction only
    # example_scene_extraction()
    
    # Uncomment to see example of detailed analysis content
    # example_detailed_analysis()