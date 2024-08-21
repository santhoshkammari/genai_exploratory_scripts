import asyncio
import time
import aiohttp
import pandas as pd
from arxiv import Search, Client
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from arxiv_html_parser import ArxivHtmlProcessor, ArxivResult


class UrlPaper(BaseModel):
    id: str
    title: str
    authors: List[str]
    comments: Optional[str]
    subjects: List[str]
    url: str = None
    keywords: Optional[List[str]] = []
    abstract: Optional[str] = None
    publication_date: Optional[str] = None
    additional_info: Optional[Dict[str, str]] = None
    code_links: Optional[List[str]] = None
    pdf_url: str = None
    simple_summary: Optional[str] = None
    key_points: Optional[List[str]] = None


class ResearchPaperFetcher:
    def __init__(self):
        self.arxiv_parser = ArxivHtmlProcessor()
        self.client = Client()

    async def fetch_html(self, session: aiohttp.ClientSession, url: str) -> str:
        async with session.get(url) as response:
            return await response.text()

    async def get_arxiv_result(self, query: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        search = Search(query=query, max_results=1)

        # Use asyncio.to_thread to run the synchronous arxiv search in a separate thread
        results = await asyncio.to_thread(list, self.client.results(search))

        ans = {"pdf_url":"","summary":""}
        if results:
            result = results[0]
            ans["pdf_url"] = result.pdf_url
            ans["summary"] = result.summary
        return ans

    async def update_paper_info(self, paper: UrlPaper, session: aiohttp.ClientSession) -> UrlPaper:
        res = await self.get_arxiv_result(paper.title, session)
        paper.abstract = res["summary"]
        paper.pdf_url = res["pdf_url"]
        return paper

    async def _update_papers_info(self, papers: List[UrlPaper], max_papers: Optional[int] = None) -> List[UrlPaper]:
        if max_papers:
            papers = papers[:max_papers]
        async with aiohttp.ClientSession() as session:
            tasks = [self.update_paper_info(paper, session) for paper in papers]
            updated_papers = await asyncio.gather(*tasks)
        return updated_papers

    async def run_async(self, url: str, max_papers: Optional[int] = None) -> List[UrlPaper]:
        start_time = time.perf_counter()

        async with aiohttp.ClientSession() as session:
            html_content = await self.fetch_html(session, url)

        print(f"Fetched data : {time.perf_counter() - start_time}s, {len(html_content)}")

        arxiv_result_list: List[ArxivResult] = self.arxiv_parser.process(html_content=html_content)
        papers = [UrlPaper(**result.model_dump()) for result in arxiv_result_list]
        print(f"Original papers: {len(papers)}")

        SELECTED_SUBJECTS = [
            "cs.SE", "cs.AI", "cs.LG", "cs.NE", "cs.MA", "cs.IR", "cs.CL", "cs.MM"
        ]
        papers = [paper for paper in papers if any(s in "".join(paper.subjects) for s in SELECTED_SUBJECTS)]
        print(f"Filtered papers: {len(papers)}")
        pd.DataFrame([p.model_dump() for p in papers]).to_excel("first_step.xlsx",index = False)
        updated_papers = await self._update_papers_info(papers, max_papers=max_papers)

        return updated_papers

    def run(self, url: str, max_papers: Optional[int] = None) -> List[UrlPaper]:
        return asyncio.run(self.run_async(url, max_papers))


# Example usage
if __name__ == "__main__":
    URL = "https://arxiv.org/list/cs/recent?skip=0&show=2000"
    fetcher = ResearchPaperFetcher()
    papers = fetcher.run(url=URL, max_papers=5)
    print(papers[0])
