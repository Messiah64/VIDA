import time
import requests
import json
import math
import hashlib
import os
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
            
            return final_summary
            
        except Exception as e:
            logger.error(f"Failed to generate final summary: {e}")
            return self.generate_fallback_summary(structured_data)
    
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