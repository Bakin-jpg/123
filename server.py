from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from playwright.async_api import async_playwright
import asyncio
from bs4 import BeautifulSoup
import re

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Scheduler for background scraping
scheduler = AsyncIOScheduler()

# =============================================================================
# MODELS
# =============================================================================

class Episode(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    episode_number: int
    title: Optional[str] = None
    url: Optional[str] = None
    iframe_url: Optional[str] = None
    thumbnail: Optional[str] = None
    duration: Optional[str] = None

class Anime(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    alternative_titles: Optional[List[str]] = []
    thumbnail: Optional[str] = None
    cover_image: Optional[str] = None
    synopsis: Optional[str] = None
    genres: List[str] = []
    rating: Optional[str] = None
    status: Optional[str] = None  # ongoing, completed
    total_episodes: Optional[int] = None
    release_year: Optional[int] = None
    studios: Optional[List[str]] = []
    type: Optional[str] = None  # TV, Movie, OVA, etc
    url: str
    episodes: List[Episode] = []
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AnimeListResponse(BaseModel):
    total: int
    data: List[Anime]

class ScraperStatus(BaseModel):
    is_running: bool
    last_run: Optional[datetime] = None
    anime_count: int
    message: str

# =============================================================================
# SCRAPER SERVICE
# =============================================================================

class AnimeScraperService:
    def __init__(self):
        self.base_url = "https://kickass-anime.ru"
        self.api_url = "https://kickass-anime.ru/api"
        self.is_scraping = False
        
    async def fetch_anime_list_api(self):
        """Fetch anime list from API"""
        import aiohttp
        try:
            logger.info(f"Fetching anime list from API...")
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/anime") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Fetched {len(data.get('result', []))} anime from API")
                        return data.get('result', [])
                    else:
                        logger.error(f"API returned status {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching from API: {e}")
            return []
    
    async def fetch_anime_detail_api(self, slug: str):
        """Fetch anime detail including episodes from website"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            page = await context.new_page()
            
            try:
                anime_url = f"{self.base_url}/{slug}"
                logger.info(f"Fetching episodes for: {anime_url}")
                
                await page.goto(anime_url, wait_until="networkidle", timeout=60000)
                await page.wait_for_timeout(5000)
                
                # Intercept API calls for episodes
                episodes = []
                
                async def handle_response(response):
                    nonlocal episodes
                    url = response.url
                    # Check for episode list API
                    if '/episodes' in url or 'episode' in url:
                        try:
                            if 'json' in response.headers.get('content-type', ''):
                                data = await response.json()
                                if isinstance(data, list):
                                    episodes = data
                                elif isinstance(data, dict) and 'episodes' in data:
                                    episodes = data['episodes']
                        except:
                            pass
                
                page.on("response", handle_response)
                
                # Try to load more episodes
                await page.wait_for_timeout(5000)
                
                # If no API intercepted, scrape from page
                if not episodes:
                    content = await page.content()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Find episode links
                    episode_links = soup.find_all('a', href=lambda x: x and '/ep-' in x)
                    
                    for idx, ep_link in enumerate(episode_links, 1):
                        ep_href = ep_link.get('href', '')
                        if ep_href:
                            # Extract episode number
                            ep_num_match = re.search(r'ep-(\d+)', ep_href)
                            ep_number = int(ep_num_match.group(1)) if ep_num_match else idx
                            
                            episodes.append({
                                'episode_number': ep_number,
                                'title': f"Episode {ep_number}",
                                'url': f"{self.base_url}{ep_href}" if not ep_href.startswith('http') else ep_href
                            })
                
                await browser.close()
                return episodes
                
            except Exception as e:
                logger.error(f"Error fetching anime detail: {e}")
                await browser.close()
                return []
    
    async def scrape_episode_iframe(self, episode_url: str):
        """Scrape iframe URL from episode page"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            page = await context.new_page()
            
            try:
                logger.info(f"Scraping episode iframe: {episode_url}")
                await page.goto(episode_url, wait_until="networkidle", timeout=60000)
                await page.wait_for_timeout(5000)
                
                # Try to find iframe
                iframe_urls = []
                
                # Method 1: Find iframe elements
                iframes = await page.query_selector_all('iframe')
                for iframe in iframes:
                    src = await iframe.get_attribute('src')
                    if src:
                        iframe_urls.append(src)
                
                # Method 2: Parse HTML for iframe
                content = await page.content()
                soup = BeautifulSoup(content, 'html.parser')
                iframe_elems = soup.find_all('iframe')
                for iframe in iframe_elems:
                    src = iframe.get('src', '')
                    if src and src not in iframe_urls:
                        iframe_urls.append(src)
                
                # Method 3: Check for video sources
                video_elems = soup.find_all('video')
                for video in video_elems:
                    src = video.get('src', '')
                    if src and src not in iframe_urls:
                        iframe_urls.append(src)
                    # Check source tags
                    sources = video.find_all('source')
                    for source in sources:
                        src = source.get('src', '')
                        if src and src not in iframe_urls:
                            iframe_urls.append(src)
                
                await browser.close()
                
                # Return first valid iframe URL
                return iframe_urls[0] if iframe_urls else None
                
            except Exception as e:
                logger.error(f"Error scraping episode iframe: {e}")
                await browser.close()
                return None
    
    async def run_full_scrape(self):
        """Run full scraping process using API"""
        if self.is_scraping:
            logger.info("Scraping already in progress, skipping...")
            return
        
        self.is_scraping = True
        logger.info("Starting full anime scraping via API...")
        
        try:
            # Fetch anime list from API
            anime_list = await self.fetch_anime_list_api()
            logger.info(f"Found {len(anime_list)} anime from API")
            
            # Process each anime
            for anime_data in anime_list[:50]:  # Process 50 anime per run
                try:
                    slug = anime_data.get('slug')
                    if not slug:
                        continue
                    
                    # Check if anime already exists
                    existing = await db.anime.find_one({"url": f"{self.base_url}/{slug}"}, {"_id": 0})
                    
                    if existing:
                        logger.info(f"Anime already cached: {anime_data.get('title')}")
                        continue
                    
                    # Get poster URL
                    poster = anime_data.get('poster', {})
                    poster_hq = poster.get('hq', '')
                    thumbnail_url = f"{self.base_url}/image/poster/{poster_hq}.webp" if poster_hq else None
                    
                    # Fetch episodes
                    episodes_data = await self.fetch_anime_detail_api(slug)
                    
                    # Create Anime object
                    anime_obj = Anime(
                        title=anime_data.get('title', anime_data.get('title_en', 'Unknown')),
                        alternative_titles=[anime_data.get('title_en')] if anime_data.get('title_en') else [],
                        url=f"{self.base_url}/{slug}",
                        thumbnail=thumbnail_url,
                        cover_image=thumbnail_url,
                        synopsis=anime_data.get('synopsis', ''),
                        genres=anime_data.get('genres', []),
                        status=anime_data.get('status', ''),
                        rating=None,
                        total_episodes=len(episodes_data),
                        release_year=anime_data.get('year'),
                        type=anime_data.get('type'),
                        episodes=[Episode(**ep) for ep in episodes_data]
                    )
                    
                    # Save to MongoDB
                    doc = anime_obj.model_dump()
                    doc['last_updated'] = doc['last_updated'].isoformat()
                    doc['created_at'] = doc['created_at'].isoformat()
                    
                    await db.anime.insert_one(doc)
                    logger.info(f"Cached anime: {anime_obj.title} ({len(episodes_data)} episodes)")
                    
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing anime {anime_data.get('title', 'unknown')}: {e}")
                    continue
            
            logger.info("Full scraping completed!")
            
        except Exception as e:
            logger.error(f"Error in full scrape: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_scraping = False

# Global scraper instance
scraper = AnimeScraperService()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@api_router.get("/")
async def root():
    return {
        "message": "Anime Scraper API",
        "endpoints": {
            "anime_list": "/api/anime",
            "anime_detail": "/api/anime/{anime_id}",
            "search": "/api/anime/search?q=query",
            "manual_scrape": "/api/scrape",
            "scraper_status": "/api/scrape/status"
        }
    }

@api_router.get("/anime", response_model=AnimeListResponse)
async def get_all_anime(skip: int = 0, limit: int = 50):
    """Get all cached anime from database"""
    try:
        # Get total count
        total = await db.anime.count_documents({})
        
        # Get anime with pagination
        anime_list = await db.anime.find({}, {"_id": 0}).skip(skip).limit(limit).to_list(limit)
        
        # Convert ISO strings back to datetime
        for anime in anime_list:
            if isinstance(anime.get('last_updated'), str):
                anime['last_updated'] = datetime.fromisoformat(anime['last_updated'])
            if isinstance(anime.get('created_at'), str):
                anime['created_at'] = datetime.fromisoformat(anime['created_at'])
        
        return {
            "total": total,
            "data": anime_list
        }
    except Exception as e:
        logger.error(f"Error fetching anime: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/anime/search")
async def search_anime(q: str, skip: int = 0, limit: int = 20):
    """Search anime by title or genre"""
    try:
        # Search by title or genre (case-insensitive)
        query = {
            "$or": [
                {"title": {"$regex": q, "$options": "i"}},
                {"genres": {"$regex": q, "$options": "i"}}
            ]
        }
        
        total = await db.anime.count_documents(query)
        results = await db.anime.find(query, {"_id": 0}).skip(skip).limit(limit).to_list(limit)
        
        # Convert ISO strings
        for anime in results:
            if isinstance(anime.get('last_updated'), str):
                anime['last_updated'] = datetime.fromisoformat(anime['last_updated'])
            if isinstance(anime.get('created_at'), str):
                anime['created_at'] = datetime.fromisoformat(anime['created_at'])
        
        return {
            "total": total,
            "query": q,
            "data": results
        }
    except Exception as e:
        logger.error(f"Error searching anime: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/anime/{anime_id}", response_model=Anime)
async def get_anime_detail(anime_id: str):
    """Get detailed anime info including episodes"""
    try:
        anime = await db.anime.find_one({"id": anime_id}, {"_id": 0})
        
        if not anime:
            raise HTTPException(status_code=404, detail="Anime not found")
        
        # Convert ISO strings
        if isinstance(anime.get('last_updated'), str):
            anime['last_updated'] = datetime.fromisoformat(anime['last_updated'])
        if isinstance(anime.get('created_at'), str):
            anime['created_at'] = datetime.fromisoformat(anime['created_at'])
        
        return anime
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching anime detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/anime/{anime_id}/episodes/{episode_number}/stream")
async def get_episode_stream(anime_id: str, episode_number: int):
    """Get streaming iframe URL for specific episode"""
    try:
        # Find anime
        anime = await db.anime.find_one({"id": anime_id}, {"_id": 0})
        
        if not anime:
            raise HTTPException(status_code=404, detail="Anime not found")
        
        # Find episode
        episode = next((ep for ep in anime.get('episodes', []) if ep['episode_number'] == episode_number), None)
        
        if not episode:
            raise HTTPException(status_code=404, detail="Episode not found")
        
        # If iframe_url not cached, scrape it
        if not episode.get('iframe_url') and episode.get('url'):
            logger.info(f"Scraping iframe for episode {episode_number}...")
            iframe_url = await scraper.scrape_episode_iframe(episode['url'])
            
            if iframe_url:
                # Update episode in database
                await db.anime.update_one(
                    {"id": anime_id, "episodes.episode_number": episode_number},
                    {"$set": {"episodes.$.iframe_url": iframe_url}}
                )
                episode['iframe_url'] = iframe_url
        
        return {
            "anime_id": anime_id,
            "episode_number": episode_number,
            "title": episode.get('title'),
            "iframe_url": episode.get('iframe_url'),
            "thumbnail": episode.get('thumbnail')
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching episode stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/scrape")
async def trigger_manual_scrape():
    """Manually trigger anime scraping"""
    if scraper.is_scraping:
        return {"message": "Scraping already in progress", "status": "running"}
    
    # Run scraping in background
    asyncio.create_task(scraper.run_full_scrape())
    
    return {"message": "Scraping started", "status": "started"}

@api_router.get("/scrape/status", response_model=ScraperStatus)
async def get_scraper_status():
    """Get current scraper status"""
    try:
        anime_count = await db.anime.count_documents({})
        
        # Get last scraped anime timestamp
        last_anime = await db.anime.find_one({}, {"_id": 0, "created_at": 1}, sort=[("created_at", -1)])
        last_run = None
        if last_anime and last_anime.get('created_at'):
            last_run = datetime.fromisoformat(last_anime['created_at']) if isinstance(last_anime['created_at'], str) else last_anime['created_at']
        
        return {
            "is_running": scraper.is_scraping,
            "last_run": last_run,
            "anime_count": anime_count,
            "message": "Scraper is running" if scraper.is_scraping else "Scraper is idle"
        }
    except Exception as e:
        logger.error(f"Error getting scraper status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/anime/cache")
async def clear_cache():
    """Clear all cached anime data"""
    try:
        result = await db.anime.delete_many({})
        return {
            "message": "Cache cleared successfully",
            "deleted_count": result.deleted_count
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# BACKGROUND SCHEDULER
# =============================================================================

async def scheduled_scrape():
    """Background job to scrape new anime"""
    logger.info("Running scheduled scrape...")
    await scraper.run_full_scrape()

@app.on_event("startup")
async def startup_event():
    """Run on app startup"""
    logger.info("Starting up Anime Scraper API...")
    
    # Create indexes
    await db.anime.create_index("id", unique=True)
    await db.anime.create_index("title")
    await db.anime.create_index("url", unique=True)
    
    # Schedule background scraping every 6 hours
    scheduler.add_job(scheduled_scrape, 'interval', hours=6, id='anime_scraper')
    scheduler.start()
    
    logger.info("Scheduler started - will scrape every 6 hours")
    
    # Run initial scrape if database is empty
    anime_count = await db.anime.count_documents({})
    if anime_count == 0:
        logger.info("Database empty, running initial scrape...")
        asyncio.create_task(scraper.run_full_scrape())

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
    scheduler.shutdown()
