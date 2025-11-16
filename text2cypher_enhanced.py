"""
Enhanced Text2Cypher Module
Based on:
- Self-Refine (Madaan et al., NeurIPS 2023): Iterative refinement with self-feedback
- In-Context Learning (Brown et al., 2020): Few-shot exemplar retrieval
- Chain-of-Thought (Wei et al., 2022): Step-by-step reasoning for query generation

Key improvements:
1. Hybrid exemplar retrieval (BM25-style + semantic similarity)
2. Multi-stage validation (syntax → semantics → safety)
3. Adaptive self-refinement with error-type classification
4. Rule-based post-processing for robustness
"""

import json
import re
import time
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path


def strip_markdown_code_fence(query: str) -> str:
    """
    Remove markdown code fences from LLM-generated queries.
    Some LLMs wrap Cypher queries in ```cypher ... ``` blocks.
    """
    query = query.strip()
    # Remove ```cypher at start and ``` at end
    query = re.sub(r'^```(?:cypher)?\s*', '', query)
    query = re.sub(r'\s*```$', '', query)
    return query.strip()

import dspy
from pydantic import BaseModel, Field


class ExemplarStore:
    """Manages exemplar storage and retrieval with hybrid similarity scoring."""
    
    def __init__(self, exemplars_path: str = "data/exemplars.json"):
        self.exemplars = self._load_exemplars(exemplars_path)
    
    def _load_exemplars(self, path: str) -> List[Dict]:
        """Load exemplars from JSON file."""
        exemplar_file = Path(path)
        if not exemplar_file.exists():
            return []
        with open(exemplar_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def select_exemplars(self, question: str, k: int = 3) -> List[Dict]:
        """
        Select top-k most relevant exemplars using hybrid similarity.
        
        Uses lexical similarity (difflib) as baseline.
        Can be extended with semantic embeddings for better retrieval.
        """
        if not self.exemplars:
            return []
        
        scores = []
        for exemplar in self.exemplars:
            # Lexical similarity (character-level)
            lex_score = SequenceMatcher(
                None, 
                question.lower(), 
                exemplar['question'].lower()
            ).ratio()
            
            # Token-level similarity (simple word overlap)
            q_tokens = set(question.lower().split())
            e_tokens = set(exemplar['question'].lower().split())
            token_score = len(q_tokens & e_tokens) / max(len(q_tokens | e_tokens), 1)
            
            # Hybrid score (weighted combination)
            final_score = 0.6 * lex_score + 0.4 * token_score
            scores.append((final_score, exemplar))
        
        # Sort by score and return top-k
        scores.sort(reverse=True, key=lambda x: x[0])
        return [exemplar for _, exemplar in scores[:k]]


class QueryValidator:
    """Multi-stage query validation with detailed error reporting."""
    
    def __init__(self, conn):
        self.conn = conn
    
    def validate(self, query: str) -> Tuple[bool, str, str]:
        """
        Validate query through multiple stages.
        
        Returns: (is_valid, error_type, error_message)
        error_type: 'syntax' | 'semantic' | 'safety' | 'none'
        """
        # Stage 1: Safety check (prevent destructive operations)
        safety_valid, safety_msg = self._check_safety(query)
        if not safety_valid:
            return False, 'safety', safety_msg
        
        # Stage 2: Syntax validation via EXPLAIN
        syntax_valid, syntax_msg = self._check_syntax(query)
        if not syntax_valid:
            return False, 'syntax', syntax_msg
        
        # Stage 3: Semantic validation (dry run with LIMIT 0)
        semantic_valid, semantic_msg = self._check_semantics(query)
        if not semantic_valid:
            return False, 'semantic', semantic_msg
        
        return True, 'none', ''
    
    def _check_safety(self, query: str) -> Tuple[bool, str]:
        """Check for dangerous operations."""
        dangerous_keywords = ['DELETE', 'DROP', 'CREATE', 'SET', 'REMOVE', 'MERGE']
        query_upper = query.upper()
        
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return False, f"Query contains dangerous operation: {keyword}"
        
        return True, ''
    
    def _check_syntax(self, query: str) -> Tuple[bool, str]:
        """Validate syntax using EXPLAIN."""
        try:
            # Kuzu supports EXPLAIN for query plan inspection
            self.conn.execute(f"EXPLAIN {query}")
            return True, ''
        except Exception as e:
            error_msg = str(e)
            # Extract meaningful error information
            if 'Binder exception' in error_msg:
                return False, f"Schema binding error: {error_msg}"
            elif 'Parser exception' in error_msg:
                return False, f"Syntax error: {error_msg}"
            else:
                return False, error_msg
    
    def _check_semantics(self, query: str) -> Tuple[bool, str]:
        """Validate query executes successfully (dry run)."""
        try:
            # Simple dry run - just execute the query with LIMIT 1
            # Kuzu doesn't support wrapping queries in CALL, so we modify the query directly
            test_query = query
            
            # If query already has LIMIT, keep it; otherwise add LIMIT 1
            if 'LIMIT' not in test_query.upper():
                test_query = test_query.rstrip() + ' LIMIT 1'
            
            self.conn.execute(test_query)
            return True, ''
        except Exception as e:
            return False, f"Execution validation failed: {str(e)}"


class RuleBasedPostProcessor:
    """
    Apply deterministic rules to improve query robustness.
    
    Based on best practices from Neo4j and Kùzu documentation.
    """
    
    # Synonym mapping for common value variations
    VALUE_SYNONYMS = {
        'woman': 'female',
        'women': 'female',
        'man': 'male',
        'men': 'male',
        'us': 'USA',
        'united states': 'USA',
        'uk': 'United Kingdom',
        'britain': 'United Kingdom',
    }
    
    def process(self, query: str, schema: dict) -> str:
        """Apply all post-processing rules."""
        query = self._fix_property_names(query)  # Fix wrong property names FIRST
        query = self._fix_order_by_syntax(query)
        query = self._normalize_synonyms(query)
        query = self._enforce_lowercase_comparisons(query)
        query = self._ensure_property_projection(query, schema)
        query = self._add_safety_limits(query)
        query = self._normalize_whitespace(query)
        return query
    
    def _fix_property_names(self, query: str) -> str:
        """
        Fix common property name errors.
        
        Scholar nodes don't have 'name' property - they have 'fullName' and 'knownName'.
        Replace s.name with s.fullName (or s.knownName in some contexts).
        """
        # Pattern: variable.name where variable is likely a Scholar node
        # Common Scholar variable names: s, sch, scholar
        scholar_var_pattern = r'\b([s]|sch|scholar)\.name\b'
        
        # Replace with .fullName (more complete name)
        query = re.sub(scholar_var_pattern, r'\1.fullName', query, flags=re.IGNORECASE)
        
        return query
    
    def _normalize_synonyms(self, query: str) -> str:
        """
        Replace common synonym values with canonical database values.
        Example: gender = 'woman' → gender = 'female'
        """
        for synonym, canonical in self.VALUE_SYNONYMS.items():
            # Case-insensitive replacement in quoted strings
            # Pattern: 'word' or "word"
            pattern = rf"['\"]({synonym})['\"]"
            replacement = f"'{canonical}'"
            query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
        return query
    
    def _fix_order_by_syntax(self, query: str) -> str:
        """
        Fix common ORDER BY syntax errors.
        
        Patterns to fix:
        1. RETURN col1, col2, extra_col sort_col DESC -> RETURN col1, col2 ORDER BY sort_col DESC
        """
        # First check if ORDER BY already exists
        if 'ORDER BY' in query.upper():
            return query
        
        # Pattern: RETURN clause ending with comma, then extra_col sort_col DESC/ASC
        # Example: RETURN p.category, p.awardYear, p.motivation p.awardYear DESC
        # Group 1: RETURN p.category, p.awardYear (everything before last comma)
        # Group 2: p.motivation (extra column to remove)
        # Group 3: p.awardYear (actual sort column)
        # Group 4: DESC/ASC
        pattern = r'(RETURN\s+.+?),\s*(\w+(?:\.\w+)?)\s+(\w+(?:\.\w+)?)\s+(DESC|ASC)(?=\s+LIMIT|\s*$)'
        
        result = re.sub(pattern, r'\1 ORDER BY \3 \4', query, flags=re.IGNORECASE)
        
        return result
    
    def _enforce_lowercase_comparisons(self, query: str) -> str:
        """
        Ensure all string comparisons use case-insensitive matching.
        
        Transforms:
        1. prop = 'value' → toLower(prop) = 'value' (for string properties)
        2. prop CONTAINS 'value' → toLower(prop) CONTAINS 'value'
        
        Note: Lowercase the value in _normalize_synonyms, here we only add toLower() wrapper
        """
        # Pattern 1: Handle CONTAINS (already implemented)
        pattern_contains = r"(\w+\.\w+|\w+)\s+CONTAINS\s+'([^']*)'"
        
        def replace_contains(match):
            prop = match.group(1)
            value = match.group(2).lower()  # Lowercase the value
            if 'toLower' in prop:
                return match.group(0)
            return f"toLower({prop}) CONTAINS '{value}'"
        
        query = re.sub(pattern_contains, replace_contains, query, flags=re.IGNORECASE)
        
        # Pattern 2: Handle = comparisons for string properties (WHERE prop = 'string')
        # Only apply to properties that likely contain strings (name, category, gender, etc.)
        # Avoid applying to numeric properties (awardYear, prize_id, etc.)
        string_property_patterns = ['name', 'category', 'gender', 'motivation', 'fullName', 'knownName']
        
        for prop_pattern in string_property_patterns:
            # Match: property_name = 'value' where property contains one of the patterns
            pattern = rf"(\w*{prop_pattern}\w*)\s*=\s*'([^']*)'"
            
            def replace_equals(match):
                prop = match.group(1)
                value = match.group(2).lower()  # Lowercase the value
                if 'toLower' in prop:
                    return match.group(0)
                return f"toLower({prop}) = '{value}'"
            
            query = re.sub(pattern, replace_equals, query, flags=re.IGNORECASE)
        
        return query
    
    def _ensure_property_projection(self, query: str, schema: dict) -> str:
        """
        Ensure RETURN clause projects properties, not nodes.
        
        If RETURN contains bare node variable, append a default property.
        """
        # Find RETURN clause (match only the items, not ORDER BY/LIMIT)
        return_match = re.search(r'RETURN\s+(.+?)(?:\s+ORDER BY|\s+LIMIT|$)', query, re.IGNORECASE)
        if not return_match:
            return query
        
        return_clause = return_match.group(1).strip()
        
        # Check each return item
        items = [item.strip() for item in return_clause.split(',')]
        fixed_items = []
        
        for item in items:
            # Skip if already has property access (contains .) or is an aggregate/function
            if '.' in item or '(' in item or item.upper() in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']:
                fixed_items.append(item)
            else:
                # Bare variable - try to add a property
                # For Scholar nodes, prefer knownName; for others, use 'name'
                if any(scholar_pattern in query.upper() for scholar_pattern in ['SCHOLAR', ':Scholar']):
                    fixed_items.append(f"{item}.knownName")
                else:
                    fixed_items.append(f"{item}.name")
        
        # Replace only the RETURN items, preserve ORDER BY/LIMIT
        new_return_items = ', '.join(fixed_items)
        
        # Find where ORDER BY or LIMIT starts (if present)
        suffix_match = re.search(r'\s+(ORDER BY.+)$', query, re.IGNORECASE)
        if suffix_match:
            # Has ORDER BY and/or LIMIT
            suffix = suffix_match.group(1)
            prefix = query[:return_match.start()]
            query = f"{prefix}RETURN {new_return_items} {suffix}"
        else:
            # No ORDER BY/LIMIT, simple replacement
            prefix = query[:return_match.start()]
            query = f"{prefix}RETURN {new_return_items}"
        
        return query
    
    def _add_safety_limits(self, query: str) -> str:
        """Add LIMIT clause if not present to prevent massive result sets."""
        query = query.strip()
        
        # Don't add LIMIT if:
        # 1. Already present
        # 2. Contains aggregation functions (COUNT, SUM, AVG, MIN, MAX)
        # 3. Contains GROUP BY (aggregation needs all data)
        # 4. Contains DISTINCT with COUNT (needs all data to count unique)
        query_upper = query.upper()
        if ('LIMIT' in query_upper or 
            'COUNT(' in query_upper or 
            'SUM(' in query_upper or 
            'AVG(' in query_upper or 
            'MIN(' in query_upper or 
            'MAX(' in query_upper or
            'GROUP BY' in query_upper):
            return query
        
        # Check if query has RETURN clause - LIMIT should go AFTER RETURN
        if 'RETURN' not in query_upper:
            return query
        
        # Check if there's an ORDER BY clause - LIMIT should go after it
        if 'ORDER BY' in query_upper:
            # LIMIT goes after ORDER BY
            return query.rstrip() + ' LIMIT 100'
        
        # Otherwise, LIMIT goes right after the RETURN clause (at the end)
        return query.rstrip() + ' LIMIT 100'
    
    def _normalize_whitespace(self, query: str) -> str:
        """Normalize whitespace for consistency."""
        # Remove multiple spaces
        query = re.sub(r'\s+', ' ', query)
        # Remove newlines
        query = query.replace('\n', ' ')
        return query.strip()


class Text2CypherEnhanced:
    """
    Enhanced Text2Cypher with self-refinement and multi-stage validation.
    
    Architecture:
    1. Schema pruning (reduce context)
    2. Exemplar retrieval (few-shot learning)
    3. Query generation (LLM with CoT)
    4. Validation (syntax + semantics + safety)
    5. Self-refinement (if validation fails)
    6. Post-processing (rule-based cleanup)
    """
    
    def __init__(
        self,
        conn,
        exemplar_store: ExemplarStore,
        max_refinement_attempts: int = 3,
        exemplar_count: int = 3
    ):
        self.conn = conn
        self.exemplar_store = exemplar_store
        self.validator = QueryValidator(conn)
        self.post_processor = RuleBasedPostProcessor()
        self.max_refinement_attempts = max_refinement_attempts
        self.exemplar_count = exemplar_count
        
        # Define DSPy signatures
        self._setup_signatures()
    
    def _setup_signatures(self):
        """Define DSPy signatures for generation and repair."""
        
        class GenerateQuery(dspy.Signature):
            """
            Generate a Cypher query for the given question using the schema.
            
            Guidelines:
            - Use toLower() for all string comparisons
            - Always use CONTAINS for partial string matching
            - Respect relationship directions (FROM/TO)
            - Return only property values, not entire nodes
            - Use concise variable names (s, p, c, etc.)
            """
            question: str = dspy.InputField()
            schema: str = dspy.InputField()
            examples: str = dspy.InputField(desc="Similar example questions and their queries")
            reasoning: str = dspy.OutputField(desc="Step-by-step reasoning about query structure")
            query: str = dspy.OutputField(desc="Valid Cypher query")
        
        class RepairQuery(dspy.Signature):
            """
            Repair a Cypher query that failed validation.
            
            Analyze the error and generate a corrected query.
            """
            original_query: str = dspy.InputField()
            error_type: str = dspy.InputField(desc="Type of error: syntax, semantic, or safety")
            error_message: str = dspy.InputField()
            schema: str = dspy.InputField()
            repair_reasoning: str = dspy.OutputField(desc="Explanation of what was wrong and how to fix it")
            repaired_query: str = dspy.OutputField(desc="Corrected Cypher query")
        
        self.GenerateQuery = GenerateQuery
        self.RepairQuery = RepairQuery
    
    def generate(
        self,
        question: str,
        schema: dict,
        return_metadata: bool = False,
        enable_timing: bool = False
    ) -> str | Tuple[str, Dict[str, Any]]:
        """
        Generate Cypher query with self-refinement.
        
        Args:
            question: Natural language question
            schema: Pruned graph schema
            return_metadata: If True, return (query, metadata) with stats
            enable_timing: If True, record detailed timing for each stage
        
        Returns:
            Valid Cypher query string (or tuple if return_metadata=True)
        """
        metadata = {
            'refinement_attempts': 0,
            'initial_valid': False,
            'error_types': [],
            'exemplars_used': 0
        }
        
        # Initialize timing dict if enabled
        if enable_timing:
            timings = {}
        
        # Step 1: Retrieve relevant exemplars
        t_start = time.time() if enable_timing else None
        exemplars = self.exemplar_store.select_exemplars(question, k=self.exemplar_count)
        if enable_timing:
            timings['exemplar_retrieval'] = time.time() - t_start
        metadata['exemplars_used'] = len(exemplars)
        
        # Format exemplars for prompt
        examples_text = self._format_exemplars(exemplars)
        
        # Step 2: Generate initial query
        t_start = time.time() if enable_timing else None
        generator = dspy.ChainOfThought(self.GenerateQuery)
        schema_text = self._format_schema(schema)
        
        result = generator(
            question=question,
            schema=schema_text,
            examples=examples_text
        )
        if enable_timing:
            timings['query_generation'] = time.time() - t_start
        
        # Clean markdown code fences that some LLMs add
        candidate_query = strip_markdown_code_fence(result.query)
        
        # Step 3: Validate
        t_start = time.time() if enable_timing else None
        is_valid, error_type, error_msg = self.validator.validate(candidate_query)
        if enable_timing:
            timings['initial_validation'] = time.time() - t_start
        metadata['initial_valid'] = is_valid
        
        # Step 4: Self-refinement loop
        t_start = time.time() if enable_timing else None
        attempts = 0
        while not is_valid and attempts < self.max_refinement_attempts:
            metadata['error_types'].append(error_type)
            candidate_query = self._repair_query(
                candidate_query,
                error_type,
                error_msg,
                schema_text
            )
            is_valid, error_type, error_msg = self.validator.validate(candidate_query)
            attempts += 1
        if enable_timing:
            timings['refinement'] = time.time() - t_start
        
        metadata['refinement_attempts'] = attempts
        
        # Step 5: Post-processing (even if valid, apply cleanup rules)
        t_start = time.time() if enable_timing else None
        final_query = self.post_processor.process(candidate_query, schema)
        if enable_timing:
            timings['post_processing'] = time.time() - t_start
        
        # Final validation after post-processing
        t_start = time.time() if enable_timing else None
        is_valid, _, _ = self.validator.validate(final_query)
        if enable_timing:
            timings['final_validation'] = time.time() - t_start
        metadata['final_valid'] = is_valid
        
        # Add timing data to metadata if enabled
        if enable_timing:
            metadata['timings'] = timings
            metadata['total_time'] = sum(timings.values())
        
        if return_metadata:
            return final_query, metadata
        return final_query
    
    def _format_exemplars(self, exemplars: List[Dict]) -> str:
        """Format exemplars as few-shot examples."""
        if not exemplars:
            return "No similar examples available."
        
        examples = []
        for i, ex in enumerate(exemplars, 1):
            examples.append(
                f"Example {i}:\n"
                f"Question: {ex['question']}\n"
                f"Query: {ex['query']}\n"
                f"Note: {ex.get('explanation', '')}"
            )
        return "\n\n".join(examples)
    
    def _format_schema(self, schema: dict) -> str:
        """Format schema for prompt context."""
        lines = ["Graph Schema:"]
        
        # Format nodes
        if 'nodes' in schema:
            lines.append("\nNode Types:")
            for node in schema['nodes']:
                # Handle case where properties might be None
                properties = node.get('properties') or []
                props = ", ".join([p['name'] for p in properties if p and 'name' in p])
                lines.append(f"  - {node.get('label', 'Unknown')}: {props}")
        
        # Format relationships
        if 'edges' in schema:
            lines.append("\nRelationships:")
            for edge in schema['edges']:
                lines.append(f"  - ({edge.get('from', 'Node')})-[:{edge.get('label', 'REL')}]->({edge.get('to', 'Node')})")
        
        return "\n".join(lines)
    
    def _repair_query(
        self,
        original_query: str,
        error_type: str,
        error_message: str,
        schema: str
    ) -> str:
        """Attempt to repair a failed query using LLM."""
        repairer = dspy.ChainOfThought(self.RepairQuery)
        
        result = repairer(
            original_query=original_query,
            error_type=error_type,
            error_message=error_message,
            schema=schema
        )
        
        # Clean markdown code fences from repaired query too
        return strip_markdown_code_fence(result.repaired_query)


# Convenience function for backward compatibility
def create_text2cypher_pipeline(conn, exemplars_path: str = "data/exemplars.json"):
    """Factory function to create enhanced Text2Cypher pipeline."""
    exemplar_store = ExemplarStore(exemplars_path)
    # Use k=2 exemplars to avoid context truncation with large exemplar store (78 examples)
    return Text2CypherEnhanced(conn, exemplar_store, exemplar_count=2)
