"""
WHO Maternal Health Guidelines Scraper

Extracts maternal health guidance from WHO publications for training data.
Targets:
- WHO recommendations on antenatal care
- WHO recommendations on intrapartum care
- WHO recommendations on postnatal care
- IMCI (Integrated Management of Childhood Illness) guidelines
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import aiohttp
import requests
from bs4 import BeautifulSoup
import trafilatura
from PyPDF2 import PdfReader
import structlog
from tqdm import tqdm

logger = structlog.get_logger()


@dataclass
class MaternalHealthGuideline:
    """Represents a single maternal health guideline or recommendation."""
    source: str
    title: str
    category: str  # antenatal, intrapartum, postnatal, newborn
    content: str
    recommendation_strength: Optional[str] = None  # strong, conditional
    evidence_quality: Optional[str] = None  # high, moderate, low, very low
    url: Optional[str] = None
    extracted_date: str = None
    
    def __post_init__(self):
        if self.extracted_date is None:
            self.extracted_date = datetime.now().isoformat()


class WHOScraper:
    """Scrapes WHO maternal health guidelines."""
    
    # Key WHO publications for maternal health
    WHO_SOURCES = [
        {
            "name": "WHO ANC Recommendations",
            "url": "https://www.who.int/publications/i/item/9789241549912",
            "category": "antenatal",
            "type": "publication_page"
        },
        {
            "name": "WHO Intrapartum Care",
            "url": "https://www.who.int/publications/i/item/9789241550215", 
            "category": "intrapartum",
            "type": "publication_page"
        },
        {
            "name": "WHO Postnatal Care",
            "url": "https://www.who.int/publications/i/item/9789240045989",
            "category": "postnatal",
            "type": "publication_page"
        },
        {
            "name": "WHO Newborn Care",
            "url": "https://www.who.int/publications/i/item/9789241506649",
            "category": "newborn",
            "type": "publication_page"
        },
    ]
    
    # WHO Danger Signs (core content to extract)
    DANGER_SIGNS_MATERNAL = {
        "antenatal": [
            "Vaginal bleeding",
            "Convulsions or fits",
            "Severe headache with blurred vision",
            "Fever and too weak to get out of bed",
            "Severe abdominal pain",
            "Fast or difficult breathing",
            "Reduced or absent fetal movements",
            "Water breaks before labor starts",
        ],
        "intrapartum": [
            "Severe vaginal bleeding",
            "Convulsions",
            "Fever during labor",
            "Severe headache or visual disturbance",
            "Cord prolapse",
            "Prolonged labor (>12 hours)",
        ],
        "postnatal": [
            "Heavy vaginal bleeding",
            "Convulsions",
            "Fast or difficult breathing",
            "Fever and too weak to get out of bed",
            "Severe headache with blurred vision",
            "Calf pain, redness or swelling",
            "Severe abdominal pain",
        ]
    }
    
    DANGER_SIGNS_NEWBORN = [
        "Not feeding well or unable to breastfeed",
        "Convulsions or fits",
        "Fast breathing (60+ breaths/minute)",
        "Severe chest indrawing",
        "No spontaneous movement",
        "Fever (temperature 37.5°C or above)",
        "Low body temperature (below 35.5°C)",
        "Jaundice in first 24 hours",
        "Yellow palms and soles",
        "Bleeding from umbilical stump",
        "Red or draining umbilicus",
    ]
    
    def __init__(self, output_dir: str = "data/raw/who"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def extract_text_from_url(self, url: str) -> Optional[str]:
        """Extract main text content from a URL."""
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded, include_tables=True)
                return text
        except Exception as e:
            logger.error("Failed to extract from URL", url=url, error=str(e))
        return None
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        text_parts = []
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        except Exception as e:
            logger.error("Failed to extract from PDF", path=pdf_path, error=str(e))
        return "\n".join(text_parts)
    
    async def fetch_who_page(self, url: str) -> Optional[str]:
        """Fetch content from WHO publication page."""
        try:
            async with self.session.get(url, timeout=30) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")
                    
                    # Find main content area
                    main_content = soup.find("main") or soup.find("article") or soup.find("div", class_="content")
                    
                    if main_content:
                        return main_content.get_text(separator="\n", strip=True)
        except Exception as e:
            logger.error("Failed to fetch WHO page", url=url, error=str(e))
        return None
    
    def create_danger_signs_guidelines(self) -> list[MaternalHealthGuideline]:
        """Create structured guidelines from WHO danger signs."""
        guidelines = []
        
        # Maternal danger signs by stage
        for category, signs in self.DANGER_SIGNS_MATERNAL.items():
            for sign in signs:
                guideline = MaternalHealthGuideline(
                    source="WHO",
                    title=f"Danger Sign: {sign}",
                    category=category,
                    content=f"DANGER SIGN during {category} period: {sign}. "
                            f"This requires IMMEDIATE referral to a health facility. "
                            f"Do not delay - this is a medical emergency.",
                    recommendation_strength="strong",
                    evidence_quality="high",
                    url="https://www.who.int/publications/i/item/9789241550215"
                )
                guidelines.append(guideline)
        
        # Newborn danger signs
        for sign in self.DANGER_SIGNS_NEWBORN:
            guideline = MaternalHealthGuideline(
                source="WHO",
                title=f"Newborn Danger Sign: {sign}",
                category="newborn",
                content=f"DANGER SIGN in newborn: {sign}. "
                        f"This requires IMMEDIATE referral to a health facility. "
                        f"Keep the baby warm during transport. This is a medical emergency.",
                recommendation_strength="strong",
                evidence_quality="high",
                url="https://www.who.int/publications/i/item/9789241506649"
            )
            guidelines.append(guideline)
        
        return guidelines
    
    def create_anc_visit_guidelines(self) -> list[MaternalHealthGuideline]:
        """Create guidelines for antenatal care visits."""
        anc_contacts = [
            {
                "timing": "Up to 12 weeks",
                "contact": 1,
                "key_actions": [
                    "Confirm pregnancy and estimate due date",
                    "Screen for anemia, HIV, syphilis",
                    "Give iron and folic acid supplements",
                    "Provide tetanus vaccination if needed",
                    "Counsel on nutrition, birth preparedness, danger signs",
                ]
            },
            {
                "timing": "20 weeks",
                "contact": 2,
                "key_actions": [
                    "Monitor blood pressure",
                    "Check fetal growth and heartbeat",
                    "Screen for pre-eclampsia",
                    "Continue iron/folic acid supplements",
                    "Review birth plan",
                ]
            },
            {
                "timing": "26 weeks",
                "contact": 3,
                "key_actions": [
                    "Monitor blood pressure",
                    "Check fetal growth and movement",
                    "Screen for gestational diabetes if indicated",
                    "Counsel on birth preparedness",
                ]
            },
            {
                "timing": "30 weeks",
                "contact": 4,
                "key_actions": [
                    "Monitor blood pressure and check for edema",
                    "Assess fetal presentation",
                    "Review danger signs",
                    "Finalize birth plan",
                ]
            },
            {
                "timing": "34 weeks",
                "contact": 5,
                "key_actions": [
                    "Monitor blood pressure",
                    "Check fetal position",
                    "Discuss signs of labor",
                    "Plan for delivery location",
                ]
            },
            {
                "timing": "36 weeks",
                "contact": 6,
                "key_actions": [
                    "Monitor blood pressure",
                    "Confirm fetal presentation",
                    "Review labor signs and when to go to facility",
                ]
            },
            {
                "timing": "38 weeks",
                "contact": 7,
                "key_actions": [
                    "Monitor blood pressure and fetal wellbeing",
                    "Confirm delivery plans",
                    "Review newborn care",
                ]
            },
            {
                "timing": "40 weeks",
                "contact": 8,
                "key_actions": [
                    "Assess for signs of labor",
                    "Discuss post-term pregnancy management",
                    "Plan facility visit if labor hasn't started",
                ]
            },
        ]
        
        guidelines = []
        for visit in anc_contacts:
            actions_text = "\n".join([f"- {action}" for action in visit["key_actions"]])
            guideline = MaternalHealthGuideline(
                source="WHO ANC Recommendations 2016",
                title=f"ANC Contact {visit['contact']} ({visit['timing']})",
                category="antenatal",
                content=f"Antenatal Care Contact {visit['contact']} at {visit['timing']}:\n\n"
                        f"Key actions:\n{actions_text}",
                recommendation_strength="strong",
                url="https://www.who.int/publications/i/item/9789241549912"
            )
            guidelines.append(guideline)
        
        return guidelines
    
    async def scrape_all(self) -> list[MaternalHealthGuideline]:
        """Scrape all WHO sources and compile guidelines."""
        all_guidelines = []
        
        # Add structured danger signs
        logger.info("Creating danger signs guidelines...")
        danger_signs = self.create_danger_signs_guidelines()
        all_guidelines.extend(danger_signs)
        logger.info(f"Created {len(danger_signs)} danger sign guidelines")
        
        # Add ANC visit guidelines
        logger.info("Creating ANC visit guidelines...")
        anc_guidelines = self.create_anc_visit_guidelines()
        all_guidelines.extend(anc_guidelines)
        logger.info(f"Created {len(anc_guidelines)} ANC visit guidelines")
        
        # Scrape WHO publication pages for additional content
        logger.info("Scraping WHO publication pages...")
        for source in tqdm(self.WHO_SOURCES, desc="Scraping WHO sources"):
            content = await self.fetch_who_page(source["url"])
            if content:
                guideline = MaternalHealthGuideline(
                    source=source["name"],
                    title=source["name"],
                    category=source["category"],
                    content=content[:5000],  # Truncate long content
                    url=source["url"]
                )
                all_guidelines.append(guideline)
        
        return all_guidelines
    
    def save_guidelines(self, guidelines: list[MaternalHealthGuideline], filename: str = "who_guidelines.jsonl"):
        """Save guidelines to JSONL file."""
        output_path = self.output_dir / filename
        
        with open(output_path, "w") as f:
            for guideline in guidelines:
                f.write(json.dumps(asdict(guideline)) + "\n")
        
        logger.info(f"Saved {len(guidelines)} guidelines to {output_path}")
        return output_path


async def main():
    """Run the WHO scraper."""
    async with WHOScraper() as scraper:
        guidelines = await scraper.scrape_all()
        scraper.save_guidelines(guidelines)
        
        print(f"\n✅ Scraped {len(guidelines)} guidelines from WHO sources")
        print(f"📁 Output saved to: data/raw/who/who_guidelines.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
