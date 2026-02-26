"""CLI for looking up ICD-10-CM codes by natural-language condition description."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # makes bare imports work

import typer

from knowledge_base import KnowledgeBase
from retriever import TfidfRetriever

app = typer.Typer(help="Look up ICD-10-CM codes by natural-language condition.")


def _build_retriever() -> TfidfRetriever:
    typer.echo("Loading knowledge base…", err=True)
    kb = KnowledgeBase()
    return TfidfRetriever(kb)


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language condition description"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
    siblings: bool = typer.Option(
        False, "--siblings", "-s", help="Include sibling codes from the same category"
    ),
) -> None:
    retriever = _build_retriever()

    if siblings:
        results = retriever.search_with_siblings(query, top_k=top_k)
        for r in results:
            typer.echo(f"\n[{r.rank}] {r.code}  score={r.score}")
            typer.echo(f"    {r.description}")
            typer.echo(
                f"    Category: {r.category_code}  ({len(r.siblings)} sibling(s))"
            )
            for sib in r.siblings:
                marker = ">" if sib["code"] == r.code else " "
                typer.echo(f"      {marker} {sib['code']}  {sib['description']}")
    else:
        results = retriever.search(query, top_k=top_k)
        for r in results:
            typer.echo(f"\n[{r.rank}] {r.code}  score={r.score}")
            typer.echo(f"    {r.description}")

    if not results:
        typer.echo("No results found.", err=True)


if __name__ == "__main__":
    app()
