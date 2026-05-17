import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .config import PageIndexConfig, GroqModel
from .tree_builder import build_tree
from .tree_search import TreeSearcher, SearchResult


class PageIndex:
    """
    High-level convenience class for vectorless reasoning-based RAG.

    Usage:
        pi = PageIndex(api_key="gsk_...")
        pi.load("report.pdf")
        
        answer = pi.ask("What are the key risks?")
        print(answer.text)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GroqModel.GPT_OSS_20B,
        verbose: bool = False,
        **config_kwargs: Any
    ):
        """
        Initialise PageIndex.

        Parameters
        ----------
        api_key : str or None
            Your Groq API key. If None, it will be read from the GROQ_API_KEY env var.
        model : str
            The Groq model to use. Defaults to the fast gpt-oss-20b.
        verbose : bool
            Enable verbose progress printing.
        **config_kwargs
            Any additional configuration options for PageIndexConfig.
        """
        self.config = PageIndexConfig(
            model=model,
            api_key=api_key,
            verbose=verbose,
            **config_kwargs
        )
        self.tree: Optional[Dict[str, Any]] = None
        self._searcher: Optional[TreeSearcher] = None

    def load(self, path: Union[str, Path]) -> "PageIndex":
        """
        Read a document (.pdf, .md, .txt) and build the reasoning tree index.
        """
        self.tree = build_tree(path, config=self.config)
        self._searcher = TreeSearcher(self.tree, config=self.config)
        return self

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the built tree index to a JSON file for later reuse.
        """
        if not self.tree:
            raise ValueError("No tree is loaded. Call load() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.tree, f, indent=2, ensure_ascii=False)

    def load_json(self, path: Union[str, Path]) -> "PageIndex":
        """
        Load a previously saved tree index from a JSON file.
        """
        with open(path, "r", encoding="utf-8") as f:
            self.tree = json.load(f)
        
        self._searcher = TreeSearcher(self.tree, config=self.config)
        return self

    def ask(self, question: str, history: Optional[list] = None) -> SearchResult:
        """
        Ask a question over the indexed document.

        Parameters
        ----------
        question : str
            The question you want to ask.
        history : list, optional
            A list of dicts representing conversation history,
            e.g. [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

        Returns
        -------
        SearchResult
            Contains `.text` (the answer), `.retrieved_nodes`, and reasoning metadata.
        """
        if not self._searcher:
            raise ValueError("No document loaded. Call load() or load_json() first.")
            
        return self._searcher.answer(question, history=history)
