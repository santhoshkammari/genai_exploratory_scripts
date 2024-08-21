# arxiv_fetcher.py

import asyncio
import time
from typing import List, Optional

import aiohttp
import pandas as pd
import yaml
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from arxiv_html_parser import ArxivHtmlProcessor
from langchain_custom.ollama import CustomChatOllama
from url_loading import UrlPaper as BaseUrlPaper

class UrlPaper(BaseModel):
    id: str
    title: str
    subjects: List[str]
    pdf_url: Optional[str] = None

class ResearchReport(BaseModel):
    papers: List[BaseUrlPaper] = Field(default_factory=list)

class ArxivFetcher:
    SELECTED_SUBJECTS = ["cs.MA", "cs.IR", "cs.AI", "cs.CL", "cs.NE", "cs.SE", "cs.MM", "cs.LG"]
    PRIORITY_ORDER = ["cs.MA", "cs.IR", "cs.AI", "cs.CL","cs.NE", "cs.SE", "cs.MM", "cs.LG"]

    def __init__(self, base_url: str, llm):
        self.base_url = base_url
        self.llm = llm
        self.arxiv_parser = ArxivHtmlProcessor()

    async def fetch_html(self, session: aiohttp.ClientSession, url: str) -> str:
        async with session.get(url) as response:
            return await response.text()

    def filter_papers(self, papers: List[UrlPaper]) -> List[UrlPaper]:
        return [paper for paper in papers if any(s in "".join(paper.subjects) for s in self.SELECTED_SUBJECTS)]

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        def get_highest_priority_subject(subjects):
            for priority in self.PRIORITY_ORDER:
                if priority in subjects:
                    return priority
            return None

        df['selected_subject'] = df['subjects'].apply(get_highest_priority_subject)
        return df.sort_values(
            by='selected_subject',
            key=lambda x: x.map({sub: idx for idx, sub in enumerate(self.PRIORITY_ORDER)})
        )

    async def fetch_papers(self, max_papers: Optional[int] = None) -> List[UrlPaper]:
        start_time = time.perf_counter()

        async with aiohttp.ClientSession() as session:
            html_content = await self.fetch_html(session, self.base_url)

        print(f"Fetched data: {time.perf_counter() - start_time:.2f}s, {len(html_content)} bytes")

        arxiv_result_list = self.arxiv_parser.process(html_content=html_content)
        papers = [UrlPaper(**result.model_dump()) for result in arxiv_result_list]
        print(f"Original papers: {len(papers)}")

        filtered_papers = self.filter_papers(papers)
        print(f"Filtered papers: {len(filtered_papers)}")

        df = pd.DataFrame([p.model_dump() for p in filtered_papers])
        processed_df = self.process_dataframe(df)
        processed_df.to_excel("fetched_papers.xlsx", index=False)

        return [UrlPaper(**row.to_dict()) for _, row in processed_df.iterrows()][:max_papers]

    def run(self, max_papers: Optional[int] = None) -> List[UrlPaper]:
        return asyncio.run(self.fetch_papers(max_papers))

def load_config(config_path: str = 'config.yml') -> dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def fetch_recent_cs_papers(max_papers=None, llm=None, URL=None):
    config = load_config()
    URL = config.get('url', "https://arxiv.org/list/cs/recent?skip=0&show=2000")
    max_papers = config.get('max_papers', 2000)

    # Set up LLM
    llm_base_url = config.get('llm', {}).get("base_url",None)
    if llm_base_url:
        llm = CustomChatOllama(model=config['llm']['model'], base_url=llm_base_url)
    else:
        llm = ChatOllama(model=config['llm']['model'])

    fetcher = ArxivFetcher(base_url=URL,llm=llm)
    papers = fetcher.run(max_papers=max_papers)
    return papers

if __name__ == "__main__":
    # URL = "https://arxiv.org/list/cs/recent?skip=0&show=2000"
    # llm = CustomChatOllama(model="gemma2:2b", base_url="http://192.168.162.147:8888")
    # max_papers = 2000
    fetch_recent_cs_papers()
