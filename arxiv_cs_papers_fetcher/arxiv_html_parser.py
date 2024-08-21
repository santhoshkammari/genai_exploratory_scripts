import re
from pathlib import Path
from typing import Tuple, List

from pydantic import BaseModel


class ArxivResult(BaseModel):
    title: str = "",
    authors: List[str] = []
    id: str = "",
    pdf_url: str = ""
    comments: str = "",
    subjects: List[str] = []

class ArxivHtmlProcessor:
    def __init__(self):
        self.pattern_dd = r"<dd>.*?</dd>"
        self.pattern_dt = r"<dt>.*?</dt>"

    def _get_divs(self,html_content):
        matches = re.findall(
            pattern=self.pattern_dd,
            string=html_content,
            flags=re.DOTALL  # To make '.' match newline characters as well

        )
        matches_dt = re.findall(
            pattern=self.pattern_dt,
            string=html_content,
            flags=re.DOTALL  # To make '.' match newline characters as well

        )
        matches = [(x, y) for x, y in zip(matches, matches_dt)]
        return matches

    def process(self, html_content: str) -> List[ArxivResult]:
        div_tuples = self._get_divs(html_content)
        processed_tuples = self._get_processed_tuples(div_tuples)
        return processed_tuples

    def _get_processed_tuples(self, div_tuples):
        results = []
        for example in div_tuples:
            _res = self._process_each_example(example)
            ar = ArxivResult(**_res)
            results.append(ar)
        return results

    def _process_each_example(self, example):
        div_processed_data = self._process_div_example(example[0])
        dt_processed_data = self._process_dt_example(example[1])
        return {**div_processed_data, **dt_processed_data}

    def _process_div_example(self, param):
        patterns = {
            "title": r"<div.*?Title.*?[^>]*>\s*(.*?)\s*</div>",
            "authors": r"<div class=['\"]list-authors['\"]>.*?>(.*?)</a>",
            "comments": r"<div class=['\"]list-comments mathjax['\"]>.*?<span class=['\"]descriptor['\"]>Comments:</span>\s*(.*?)\s*</div>",
            "subjects": r"<div class=['\"]list-subjects['\"]>.*?<span class=['\"]descriptor['\"]>Subjects:</span>\s*(.*?)\s*</div>",
        }

        # Extract information using regex patterns
        title = re.findall(patterns['title'], param, flags=re.DOTALL)
        authors = re.findall(patterns['authors'], param, flags=re.DOTALL)
        comments = re.findall(patterns['comments'], param, flags=re.DOTALL)
        subjects = re.findall(patterns['subjects'], param, flags=re.DOTALL)

        # Clean up and format results
        title = title[0].strip() if title else 'No title found'
        authors = [author.strip() for author in authors]
        comments = comments[0].strip() if comments else 'No comments found'
        subjects = subjects[0].strip() if subjects else 'No subjects found'
        data = {
            "title": title,
            "authors": authors,
            "comments": comments,
            "subjects": subjects
        }
        data = self._process_subjects(data)
        return data

    def _process_subjects(self, data):
        subjects_text = data["subjects"]
        # Extract tags like cs.AI, cs.CV
        subjects = re.findall(r'\((cs\.[^)]+)\)', subjects_text)
        data["subjects"] = subjects
        return data

    def _process_dt_example(self, param):
        patterns = {
        "id": re.compile(r'<a[^>]*Abstract[^>]*.*?arXiv:(.*?)[^\d]*</a>', re.DOTALL),

            "links": re.compile(r"<a href=\"(.*?)\" title=\"(.*?)\" id=\".*?\">(.*?)</a>", re.DOTALL),
        }

        # Extract information using regex patterns
        data = {}

        # Extract arXiv ID
        arxiv_match = patterns["id"].findall(param)
        data["id"] = next(iter(arxiv_match),"")

        # Extract links
        links = patterns["links"].findall(param)
        data["pdf_url"] = "https://arxiv.org" + links[0][0]
        return data


