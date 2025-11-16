import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        rf"""
    # Graph RAG using Text2Cypher

    This is a demo app in marimo that allows you to query the Nobel laureate graph (that's managed in Kuzu) using natural language. A language model takes in the question you enter, translates it to Cypher via a custom Text2Cypher pipeline in Kuzu that's powered by DSPy. The response retrieved from the graph database is then used as context to formulate the answer to the question.

    > \- Powered by Kuzu, DSPy and marimo \-
    """
    )
    return


@app.cell
def _(mo):
    text_ui = mo.ui.text(value="Which scholars won prizes in Physics and were affiliated with University of Cambridge?", full_width=True)
    return (text_ui,)


@app.cell
def _(text_ui):
    text_ui
    return


@app.cell
def _(KuzuDatabaseManager, mo, run_graph_rag, text_ui):
    db_name = "nobel.kuzu"
    db_manager = KuzuDatabaseManager(db_name)

    question = text_ui.value

    with mo.status.spinner(title="Generating answer...") as _spinner:
        result = run_graph_rag([question], db_manager, True)[0]
        result_ori = run_graph_rag([question], db_manager, False)[0]
    # result may be empty or missing keys if the pipeline failed (e.g., LLM error,
    # DB query returned no rows, or an earlier step raised and returned an empty dict).
    # Guard against that to avoid KeyError when accessing result['query'].
    if not result or not isinstance(result, dict):
        mo.md("**No result produced by pipeline.** Check the server logs for errors (LLM auth, provider, or DB query failures).")
        # return "No answer available", "No query generated"

    query = result.get('query')
    answer_obj = result.get('answer')

    query_ori = result_ori.get('query')
    answer_obj_ori = result_ori.get('answer')

    # answer may be a DSPy output object with .response or a plain string/dict
    if hasattr(answer_obj, 'response'):
        answer_text = answer_obj.response
    elif isinstance(answer_obj, dict) and 'response' in answer_obj:
        answer_text = answer_obj['response']
    elif isinstance(answer_obj, str):
        answer_text = answer_obj
    else:
        answer_text = str(answer_obj)



    # answer may be a DSPy output object with .response or a plain string/dict
    if hasattr(answer_obj_ori, 'response'):
        answer_ori = answer_obj_ori.response
    elif isinstance(answer_obj_ori, dict) and 'response' in answer_obj_ori:
        answer_ori = answer_obj_ori['response']
    elif isinstance(answer_obj_ori, str):
        answer_ori = answer_obj_ori
    else:
        answer_ori = str(answer_obj_ori)

    return answer_ori, answer_text, query, query_ori


@app.cell
def _(answer_ori, answer_text, mo, query, query_ori, result):
    # Show cache status
    is_cached = result.get('metadata', {}).get('cached', False)
    cache_status = "✅ CACHED" if is_cached else "🔄 GENERATED"
    
    mo.vstack([
        mo.md(f"**Cache Status:** {cache_status}"),
        mo.hstack([
            mo.md(f"""### Query\n```{query}```"""),
            mo.md(f"""### Answer\n{answer_text}""")
        ]),
        
        mo.hstack([
            mo.md(f"""### Query Original\n```{query_ori}```"""),
            mo.md(f"""### Answer Original\n{answer_ori}""")
        ])
    ])    
    return

@app.cell
def _(GraphSchema, Query, dspy):
    class PruneSchema(dspy.Signature):
        """
        Understand the given labelled property graph schema and the given user question. Your task
        is to return ONLY the subset of the schema (node labels, edge labels and properties) that is
        relevant to the question.
            - The schema is a list of nodes and edges in a property graph.
            - The nodes are the entities in the graph.
            - The edges are the relationships between the nodes.
            - Properties of nodes and edges are their attributes, which helps answer the question.
        """

        question: str = dspy.InputField()
        input_schema: str = dspy.InputField()
        pruned_schema: GraphSchema = dspy.OutputField()


    class Text2Cypher(dspy.Signature):
        """
        Translate the question into a valid Cypher query that respects the graph schema.

        <SYNTAX>
        - When matching on Scholar names, ALWAYS match on the `knownName` property
        - For countries, cities, continents and institutions, you can match on the `name` property
        - Use short, concise alphanumeric strings as names of variable bindings (e.g., `a1`, `r1`, etc.)
        - Always strive to respect the relationship direction (FROM/TO) using the schema information.
        - When comparing string properties, ALWAYS do the following:
            - Lowercase the property values before comparison
            - Use the WHERE clause
            - Use the CONTAINS operator to check for presence of one substring in the other
        - DO NOT use APOC as the database does not support it.
        </SYNTAX>

        <RETURN_RESULTS>
        - If the result is an integer, return it as an integer (not a string).
        - When returning results, return property values rather than the entire node or relationship.
        - Do not attempt to coerce data types to number formats (e.g., integer, float) in your results.
        - NO Cypher keywords should be returned by your query.
        </RETURN_RESULTS>
        """

        question: str = dspy.InputField()
        input_schema: str = dspy.InputField()
        query: Query = dspy.OutputField()


    class AnswerQuestion(dspy.Signature):
        """
        - Use the provided question, the generated Cypher query and the context to answer the question.
        - If the context is empty, state that you don't have enough information to answer the question.
        - When dealing with dates, mention the month in full.
        """

        question: str = dspy.InputField()
        cypher_query: str = dspy.InputField()
        context: str = dspy.InputField()
        response: str = dspy.OutputField()
    return AnswerQuestion, PruneSchema, Text2Cypher


@app.cell
def _(GEMINI_API_KEY, dspy):
    # Using Google Gemini API directly (no OpenRouter)
    # Note: Not using BAMLAdapter to avoid "max depth exceeded" error
    # with deeply nested GraphSchema (Edge -> Node -> Property)
    # Increased max_tokens to avoid truncation with large exemplar store
    lm = dspy.LM(
        model="gemini/gemini-2.5-flash-lite",
        api_key=GEMINI_API_KEY,
        max_tokens=6000
    )
    dspy.configure(lm=lm)  # NO BAMLAdapter!
    return


@app.cell
def _(kuzu):
    class KuzuDatabaseManager:
        """Manages Kuzu database connection and schema retrieval."""

        def __init__(self, db_path: str = "ldbc_1.kuzu"):
            self.db_path = db_path
            self.db = kuzu.Database(db_path, read_only=True)
            self.conn = kuzu.Connection(self.db)

        @property
        def get_schema_dict(self) -> dict[str, list[dict]]:
            response = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'NODE' RETURN *;")
            nodes = [row[1] for row in response]  # type: ignore
            response = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'REL' RETURN *;")
            rel_tables = [row[1] for row in response]  # type: ignore
            relationships = []
            for tbl_name in rel_tables:
                response = self.conn.execute(f"CALL SHOW_CONNECTION('{tbl_name}') RETURN *;")
                for row in response:
                    relationships.append({"name": tbl_name, "from": row[0], "to": row[1]})  # type: ignore
            schema = {"nodes": [], "edges": []}

            for node in nodes:
                node_schema = {"label": node, "properties": []}
                node_properties = self.conn.execute(f"CALL TABLE_INFO('{node}') RETURN *;")
                for row in node_properties:  # type: ignore
                    node_schema["properties"].append({"name": row[1], "type": row[2]})  # type: ignore
                schema["nodes"].append(node_schema)

            for rel in relationships:
                edge = {
                    "label": rel["name"],
                    "from": rel["from"],
                    "to": rel["to"],
                    "properties": [],
                }
                rel_properties = self.conn.execute(f"""CALL TABLE_INFO('{rel["name"]}') RETURN *;""")
                for row in rel_properties:  # type: ignore
                    edge["properties"].append({"name": row[1], "type": row[2]})  # type: ignore
                schema["edges"].append(edge)
            return schema
    return (KuzuDatabaseManager,)


@app.cell
def _(BaseModel, Field):
    class Query(BaseModel):
        query: str = Field(description="Valid Cypher query with no newlines")


    class Property(BaseModel):
        name: str
        type: str = Field(description="Data type of the property")


    class Node(BaseModel):
        label: str
        properties: list[Property] | None


    # Simplified Edge to avoid deep nesting with Gemini API
    class SimpleEdge(BaseModel):
        label: str = Field(description="Relationship label")
        from_label: str = Field(alias="from", description="Source node label")
        to_label: str = Field(alias="to", description="Target node label")
        properties: list[Property] | None


    class GraphSchema(BaseModel):
        nodes: list[Node]
        edges: list[SimpleEdge]
    return GraphSchema, Query


@app.cell
def _(
    AnswerQuestion,
    Any,
    KuzuDatabaseManager,
    PruneSchema,
    Query,
    Text2Cypher,
    dspy,
    get_global_cache,
    text2cypher_enhanced,
):
    class GraphRAG(dspy.Module):
        """
        DSPy custom module that applies Text2Cypher to generate a query and run it
        on the Kuzu database, to generate a natural language response.

        Enhanced with:
        - Dynamic exemplar selection
        - Self-refinement loop with validation
        - Rule-based post-processing
        """

        def __init__(self, db_manager: KuzuDatabaseManager, use_enhanced: bool = True):
            self.prune = dspy.Predict(PruneSchema)
            self.generate_answer = dspy.ChainOfThought(AnswerQuestion)
            self.use_enhanced = use_enhanced

            if use_enhanced:
                # Use enhanced Text2Cypher with validation and refinement
                self.enhanced_text2cypher = text2cypher_enhanced.create_text2cypher_pipeline(
                    db_manager.conn,
                    exemplars_path="data/exemplars_comprehensive.json"
                )
            else:
                # Fallback to basic DSPy Text2Cypher
                self.text2cypher = dspy.ChainOfThought(Text2Cypher)

        def get_cypher_query(self, question: str, input_schema: str) -> tuple[Query, dict]:
            """
            Generate Cypher query with optional enhancement.

            Returns: (query, metadata) where metadata contains performance stats
            """
            prune_result = self.prune(question=question, input_schema=input_schema)
            schema = prune_result.pruned_schema

            if self.use_enhanced:
                # Use enhanced pipeline with self-refinement
                query_str, metadata = self.enhanced_text2cypher.generate(
                    question=question,
                    schema=schema.model_dump() if hasattr(schema, 'model_dump') else schema,
                    return_metadata=True
                )
                # Wrap in Query object for compatibility
                query_obj = Query(query=query_str)
                return query_obj, metadata
            else:
                # Use basic DSPy pipeline
                text2cypher_result = self.text2cypher(question=question, input_schema=schema)
                return text2cypher_result.query, {}

        def run_query(
            self, db_manager: KuzuDatabaseManager, question: str, input_schema: str
        ) -> tuple[str, list[Any] | None, dict]:
            """
            Run a query synchronously on the database.
            Returns: (query, results, metadata)
            """
            query_obj, metadata = self.get_cypher_query(question=question, input_schema=input_schema)
            query = query_obj.query if hasattr(query_obj, 'query') else str(query_obj)

            try:
                # Run the query on the database
                result = db_manager.conn.execute(query)
                results = [item for row in result for item in row]
            except RuntimeError as e:
                print(f"Error running query: {e}")
                results = None
            return query, results, metadata

        def forward(self, db_manager: KuzuDatabaseManager, question: str, input_schema: str):
            final_query, final_context, metadata = self.run_query(db_manager, question, input_schema)

            if final_context is None:
                print("Empty results obtained from the graph database. Please retry with a different question.")
                return {}
            else:
                answer = self.generate_answer(
                    question=question, cypher_query=final_query, context=str(final_context)
                )
                response = {
                    "question": question,
                    "query": final_query,
                    "answer": answer,
                    "metadata": metadata  # Include generation metadata
                }
                return response

        async def aforward(self, db_manager: KuzuDatabaseManager, question: str, input_schema: str):
            final_query, final_context, metadata = self.run_query(db_manager, question, input_schema)

            if final_context is None:
                print("Empty results obtained from the graph database. Please retry with a different question.")
                return {}
            else:
                answer = self.generate_answer(
                    question=question, cypher_query=final_query, context=str(final_context)
                )
                response = {
                    "question": question,
                    "query": final_query,
                    "answer": answer,
                    "metadata": metadata  # Include generation metadata
                }
                return response


    def run_graph_rag(questions: list[str], db_manager: KuzuDatabaseManager, use_enhanced: bool = True) -> list[Any]:
        """
        Run Graph RAG pipeline with optional enhanced Text2Cypher.

        Args:
            questions: List of questions to answer
            db_manager: Kuzu database manager
            use_enhanced: If True, use enhanced Text2Cypher with validation (default: True)
        """
        schema = str(db_manager.get_schema_dict)
        
        # Initialize cache (max_size=256 queries)
        cache = get_global_cache(max_size=256)
        
        rag = GraphRAG(db_manager, use_enhanced=use_enhanced)
        
        # Run pipeline with cache
        results = []
        for question in questions:
            # Check cache first
            cached_query = cache.get(question, schema)
            
            if cached_query:
                # Use cached Cypher query (fast path)
                try:
                    # Execute cached query directly
                    result = db_manager.conn.execute(cached_query)
                    context = [item for row in result for item in row]
                    
                    if context and len(context) > 0:
                        # Generate answer using cached query
                        answer = rag.generate_answer(
                            question=question,
                            cypher_query=cached_query,
                            context=str(context)
                        )
                        response = {
                            "question": question,
                            "query": cached_query,
                            "answer": answer,
                            "metadata": {"cached": True}
                        }
                        results.append(response)
                        continue
                except Exception as e:
                    # If cached query fails, fall through to regenerate
                    print(f"Cached query failed: {e}")
                    pass
            
            # Cache miss or cached query failed - generate new query
            response = rag(db_manager=db_manager, question=question, input_schema=schema)
            
            # Cache the generated query if successful
            if response and isinstance(response, dict) and 'query' in response:
                cache.put(question, response['query'], schema)
                if 'metadata' not in response:
                    response['metadata'] = {}
                response['metadata']['cached'] = False
            
            results.append(response)
        
        return results

    return (run_graph_rag,)


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    import os
    from textwrap import dedent
    from typing import Any

    import dspy
    import kuzu
    from dotenv import load_dotenv
    from pydantic import BaseModel, Field
    from cache import get_global_cache

    import text2cypher_enhanced

    load_dotenv('project.env')

    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return (
        Any,
        BaseModel,
        Field,
        GEMINI_API_KEY,
        dspy,
        get_global_cache,
        kuzu,
        mo,
        text2cypher_enhanced,
    )


if __name__ == "__main__":
    app.run()
