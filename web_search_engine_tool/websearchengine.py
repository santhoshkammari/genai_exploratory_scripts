from crewai.utilities import Converter
from langchain_community.tools import DuckDuckGoSearchRun, ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from typing import List
from pydantic import BaseModel

import asyncio
import re
from typing import Any, Optional, Dict

import aiohttp
import trafilatura
from aiohttp import ClientTimeout
from crewai import Task, Crew, Agent
from googlesearch import search
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.tools import YouTubeSearchTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

##### Schemas #####
class BaseResult(BaseModel):
    query: str = ""
    google_search_result: str = ""
    arxiv_search_result: str = ""
    wikipedia_result: str = ""
    youtube_search_result: str = ""
    duckduckgo_search_result: str = ""


class Result(BaseModel):
    answers : List[BaseResult]

class QueryVariations(BaseModel):
    queries: List[str]

class ToolsResult(BaseModel):
    tool_results: List[Result]
##### Schemas #####



##### Tools #####
class GoogleSearchTool(BaseTool):
    """Tool that queries YouTube."""

    name: str = "google_search"
    description: str = (
        "search for latest information in internet"
    )
    html2text = Html2TextTransformer()
    google_search_config: dict
    google_search_llm: Any = None

    def _extract_main_content(self, html: str) -> str:
        """Extract the main content from HTML using trafilatura."""
        return trafilatura.extract(html) or ""

    def _google_search_terms(self,query, google_search_llm):
        if not google_search_llm:
            raise ValueError("GoogleSearchLLM is required")
        search_variation_generator = Agent(
            role='Search Variation Generator',
            goal='Generate five diverse search queries of the query with MAIN entities',
            backstory='You are an expert in information retrieval and query expansion, with a knack for generating varied and insightful search terms.',
            verbose=True,
            allow_delegation=False,
            llm=google_search_llm
        )
        search_variation_task = Task(
            description=f"Generate five diverse search terms for query to search in google: {query}\n",
            agent=search_variation_generator,
            expected_output="Respond only with five variation with numbers",
        )
        crew = Crew(
            agents=[search_variation_generator],
            tasks=[search_variation_task]
        )
        res = crew.kickoff({"query": query})
        ans = re.findall(r'(\d+)\.+\s*(.*?)(?=\n\d+\.*|\Z)', res.raw, re.DOTALL)
        try:
            ans = [query] + [y for x, y in ans]
        except:
            ans = [query]
        return ans

    async def _search(self, query: str) -> str:
        queries = self._google_search_terms(query,self.google_search_llm) if self.google_search_config.get("allow_search_term_variations",False) else [query]
        url_suffix_list = []
        for _q in queries:
            url_suffix_list.extend([_ for _ in search(_q,num_results=self.google_search_config.get("num_results",2))])

        async def fetch_url(session, url):
            try:
                async with session.get(url, timeout=ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        html = await response.text()
                        return self._extract_main_content(html)
            except Exception as e:
                return ""

        async with aiohttp.ClientSession() as session:
            tasks = [fetch_url(session, url) for url in url_suffix_list]
            processed_docs = await asyncio.gather(*tasks)

        return "\n".join(filter(None, processed_docs))

    async def _arun(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return await self._search(query)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool synchronously."""
        # We can't use asyncio.run() here because it tries to create a new event loop
        # Instead, we'll use asyncio.get_event_loop().run_until_complete()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._search(query))

class CustomWikipediaTool(BaseTool):
    name = "Wikipedia"
    description = "Useful for searching Wikipedia to get information on a wide range of topics."
    wikipedia_config: dict

    def _run(self, query: str) -> str:
        docs = WikipediaLoader(query = query, **self.wikipedia_config).load()
        return "\n".join([doc.page_content for doc in docs])

class CustomYoutubeSearchTool(YouTubeSearchTool):
    youtube_config: dict
    def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self._search(query, num_results=self.youtube_config.get("num_results",1))
##### Tools #####



##### Engine #####
class WebSearchEngine:
    def __init__(self, config: Dict[str, Any]=None, llm: Optional[Any] = None,tools: List[BaseTool] = None):
        self.llm = llm
        self.question_variation_llm = None
        self.google_search_llm = None
        self._init_llms()
        self.config = config if config else self._default_configs()
        self.tools: List[BaseTool] = self._initialize_tools() if tools is None else tools

    def _init_llms(self):
        if self.llm and self.google_search_llm is None:
            self.google_search_llm = self.llm
        if self.llm and self.question_variation_llm is None:
            self.question_variation_llm = self.llm

    def _initialize_tools(self):
        wikipedia_tool = CustomWikipediaTool(wikipedia_config=self.config.get("wikipedia_config",{}))
        arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(**self.config.get("arxiv_config",{})))
        youtube_search = CustomYoutubeSearchTool(youtube_config=self.config.get("youtube_config",{}))
        google_search = GoogleSearchTool(google_search_config=self.config.get("google_search_config",{}),
                                         google_search_llm=self.google_search_llm)
        duckduck_search_tool = DuckDuckGoSearchRun()

        tools = [google_search, duckduck_search_tool, youtube_search, arxiv, wikipedia_tool]
        return tools

    async def _process_query(self,search: BaseTool, query: str) -> BaseResult:
        if hasattr(search, 'arun'):
            search_result = await search.arun(query)
        else:
            search_result = search.run(query)

        if search.name == "google_search":
            return BaseResult(query=query, google_search_result=search_result)
        elif search.name == "arxiv":
            return BaseResult(query=query, arxiv_search_result=search_result)
        elif search.name == "Wikipedia":
            return BaseResult(query=query, wikipedia_result=search_result)
        elif search.name == "youtube_search":
            return BaseResult(query=query, youtube_search_result=search_result)
        elif search.name == "duckduckgo_search":
            return BaseResult(query=query, duckduckgo_search_result=search_result)


    def _get_query_variations(self,query,crew_config):
        print(f"State of Variations : {crew_config.get('allow_query_rewrite',False)}")

        if crew_config.get('allow_query_rewrite',False):
            if not self.question_variation_llm:
                raise ValueError("QuestionVariationLLM is required")
            class QVAgent(Agent):
                def get_output_converter(self, llm, text, model, instructions):
                    _queries = [p.strip() for p in text.split("#QSEP#") if p]
                    text = QueryVariations(queries=_queries).model_dump_json()
                    return Converter(text=text, llm=llm, model=model, instructions=instructions)

            question_variation_generator = QVAgent(
                role='Query Variation Generator',
                goal='Generate relevant variations of the input query ',
                backstory='You are an expert in information retrieval and natural language processing, with a knack for rephrasing queries to capture different aspects and intentions.',
                verbose=True,
                allow_delegation=False,
                llm=self.question_variation_llm
            )
            query_varition_task = Task(
                description=f"Generate a relevant variation of the input query: {query}\n",
                agent=question_variation_generator,
                expected_output=str(
                    "A list of {num_variations} unique and varied questions based on the original query, each of varation separated by #QSEP#. Each question should offer a different perspective or emphasis on the topic, without any numbering or extra formatting ."
                    " Start directly don't provide ouput like Here are five unique and varied questions based on the original query"
                ),
                output_pydantic=QueryVariations
            )

            crew = Crew(
                agents=[question_variation_generator],
                tasks=[query_varition_task],
                verbose=False
            )
            res = crew.kickoff(crew_config['kickoff'])
            query_variations = QueryVariations(queries=[query, *res.pydantic.queries])
        else:
            query_variations = QueryVariations(queries=[query])
        return query_variations


    async def __search_run(self,query: str, crew_config, tools) -> ToolsResult:
        query_variations = self._get_query_variations(query,crew_config)
        tool_results = []
        tasks = []
        for tool in tools:
            tasks.extend([self._process_query(tool, each_qvar) for each_qvar in query_variations.queries])
        _results = await asyncio.gather(*tasks)
        tool_results.append(Result(answers=_results))

        return ToolsResult(tool_results = tool_results)


    def search(self,query) -> BaseResult:
        final_results: ToolsResult = asyncio.run(self.__search_run(query,self.config.get("crew_config",{}),self.tools))
        final_res = {}
        for res in final_results.tool_results[0].answers:
            for q, v in res.model_dump().items():
                if v:
                    final_res[q] = final_res.get(q,"\n")+v if q!="query" else query
        return BaseResult(**final_res)

    def _default_configs(self):
        return dict(
            wikipedia_config=dict(
                load_max_docs=1,
                doc_content_chars_max=4000
            ),
            arxiv_config=dict(
                top_k_results=1,
                ARXIV_MAX_QUERY_LENGTH=300,
                load_max_docs=1,
                load_all_available_meta=False,
                doc_content_chars_max=40000
            ),
            youtube_config = dict(
                num_results=1
            ),
            google_search_config=dict(
                num_results=1,
                allow_search_term_variations = self.google_search_llm is not None
            ),
            crew_config=dict(
                kickoff=dict(
                    num_variations=3
                ),
                allow_query_rewrite = self.question_variation_llm is not None
            )
        )


if __name__ == "__main__":
    query = "what are some latest advancements of AI research field."
    from langchain_ollama.chat_models import ChatOllama
    engine = WebSearchEngine(llm = ChatOllama(model = "qwen2:0.5b"))
    print(engine.search(query=query))
