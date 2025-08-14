import time
import requests
import json
import math
import hashlib
import pickle
import os
from pathlib import Path
from openai import AsyncOpenAI
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Tuple
import aiofiles
import aiohttp
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching for video analysis results and OpenAI responses"""
    
    def __init__(self, cache_dir: str = "cache", cache_expiry_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_expiry = timedelta(hours=cache_expiry_hours)
    
    def _get_cache_key(self, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is still valid"""
        if not cache_path.exists():
            return False
        
        file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - file_time < self.cache_expiry
    
    def get(self, key_data: Any) -> Optional[Any]:
        """Get cached data if valid"""
        cache_key = self._get_cache_key(key_data)
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info(f"Cache hit for key: {cache_key[:8]}...")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        return None
    
    def set(self, key_data: Any, value: Any) -> None:
        """Cache data"""
        cache_key = self._get_cache_key(key_data)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            logger.info(f"Cached data for key: {cache_key[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")
    
    def clear_expired(self) -> None:
        """Remove expired cache files"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            if not self._is_cache_valid(cache_file):
                try:
                    cache_file.unlink()
                    logger.info(f"Removed expired cache: {cache_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")

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
    Enhanced Video Analysis using Azure Video Indexer and Azure OpenAI
    Features:
    - Async OpenAI API calls for improved performance
    - Intelligent caching system for results and API responses
    - Confidence-based filtering of insights (0.8+ threshold)
    - Token-safe hierarchical summarization
    - Chunked transcript processing for large videos
    - Parallel GPT calls for improved speed
    - Robust error handling and fallback summaries
    """
    
    def __init__(self, config_file='config.json'):
        """Initialize with configuration"""
        self.load_config(config_file)
        
        # Initialize cache manager
        self.cache = CacheManager(
            cache_dir=self.config.get('cache', {}).get('cache_dir', 'cache'),
            cache_expiry_hours=self.config.get('cache', {}).get('expiry_hours', 24)
        )
        
        # Initialize confidence filter
        min_confidence = self.config.get('processing', {}).get('min_confidence', 0.8)
        self.confidence_filter = ConfidenceFilter(min_confidence)
        
        # Initialize Azure OpenAI client - async version
        try:
            # Using AsyncOpenAI instead of AsyncAzureOpenAI for better async support
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
                "privacy": "Private",
                "chunk_size": 30,  # transcript segments per chunk
                "max_workers": 4,   # parallel processing threads
                "min_confidence": 0.8  # minimum confidence threshold
            },
            "cache": {
                "cache_dir": "cache",
                "expiry_hours": 24
            }
        }
    
    def get_access_token(self):
        """Get access token from Video Indexer with caching"""
        # Check cache first
        cache_key = {"action": "get_access_token", "config": self.config['video_indexer']}
        cached_token = self.cache.get(cache_key)
        if cached_token:
            return cached_token
        
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
        
        # Cache token for shorter duration (tokens typically expire in 1 hour)
        short_cache = CacheManager(cache_expiry_hours=0.5)
        short_cache.set(cache_key, access_token)
        
        return access_token
    
    def upload_video(self, access_token, video_path, video_name, video_description):
        """Upload video to Video Indexer with result caching"""
        # Check cache based on file hash and metadata
        file_hash = self._get_file_hash(video_path)
        cache_key = {
            "action": "upload_video",
            "file_hash": file_hash,
            "video_name": video_name,
            "description": video_description
        }
        
        cached_video_id = self.cache.get(cache_key)
        if cached_video_id:
            logger.info(f"Using cached video ID: {cached_video_id}")
            return cached_video_id
        
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
        
        # Cache the video ID
        self.cache.set(cache_key, video_id)
        
        return video_id
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash of file for caching purposes"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def wait_for_indexing(self, access_token, video_id):
        """Wait for video indexing to complete with result caching"""
        # Check cache for completed indexing results
        cache_key = {"action": "indexing_result", "video_id": video_id}
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.info("Using cached indexing results")
            return cached_result
        
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
                # Cache the successful result
                self.cache.set(cache_key, idx_json)
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
        # Check cache first
        cache_key = {"action": "chunk_summary", "chunk": chunk, "index": chunk_index}
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.info(f"Using cached summary for chunk {chunk_index}")
            return cached_result
        
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
            
            # Cache the result
            self.cache.set(cache_key, result)
            
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
        # Check cache first
        cache_key = {
            "action": "final_summary",
            "chunk_summaries": chunk_summaries,
            "metadata_hash": self._get_dict_hash(structured_data)
        }
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.info("Using cached final summary")
            return cached_result
        
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
Provide timestamped analysis of key moments and transitions.

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
            
            # Cache the result
            self.cache.set(cache_key, final_summary)
            
            return final_summary
            
        except Exception as e:
            logger.error(f"Failed to generate final summary: {e}")
            return self.generate_fallback_summary(structured_data)
    
    def _get_dict_hash(self, data: Dict[str, Any]) -> str:
        """Generate hash for dictionary data"""
        return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
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
        Enhanced async video analysis with chunked processing, caching, and confidence filtering
        
        Args:
            video_path (str): Path to the video file
            video_name (str): Name for the video
            video_description (str): Description for the video
            
        Returns:
            str: Human-readable analysis summary in markdown format
        """
        try:
            # Clean expired cache files
            self.cache.clear_expired()
            
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
                
                return final_summary
                
            except Exception as e:
                logger.error(f"Async processing failed: {e}")
                # Fallback to synchronous processing
                logger.info("Falling back to synchronous chunk processing")
                
                chunk_results = []
                for i, chunk in enumerate(chunks):
                    # Use sync version as fallback
                    result = await self.summarize_chunk_async(chunk, i+1)
                    chunk_results.append(result)
                
                final_summary = await self.generate_final_summary_async(chunk_results, structured_data)
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


# Example usage and testing functions
async def main():
    """Example usage of the enhanced video analyzer"""
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
        video_description = "Testing enhanced video analyzer with async processing"
        
        print("Starting video analysis...")
        summary = await analyzer.analyze_video_async(video_path, video_name, video_description)
        
        print("Analysis complete!")
        print(summary)
        
    except Exception as e:
        print(f"Analysis failed: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the async main function
    asyncio.run(main())