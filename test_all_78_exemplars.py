"""
Final Comprehensive Test: Enhanced vs Baseline (78 exemplars)
Tests ALL exemplars and compares results with detailed analysis
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv
import kuzu
import dspy
from pydantic import BaseModel, Field

# Load environment
load_dotenv('project.env')
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

# Configure DSPy - use gemini-2.5-flash-lite (better rate limits)
lm = dspy.LM(model="gemini/gemini-2.5-flash-lite", api_key=GEMINI_API_KEY, max_tokens=6000, temperature=0.0)
dspy.configure(lm=lm)

import text2cypher_enhanced

# Schema classes for Baseline
class Property(BaseModel):
    name: str
    type: str

class Node(BaseModel):
    label: str
    properties: list[Property] | None

class SimpleEdge(BaseModel):
    label: str = Field(description="Relationship label")
    from_label: str = Field(alias="from")
    to_label: str = Field(alias="to")
    properties: list[Property] | None

class GraphSchema(BaseModel):
    nodes: list[Node]
    edges: list[SimpleEdge]

class PruneSchema(dspy.Signature):
    """Given a question and a graph schema, identify and return only the relevant subset of the schema needed to answer the question."""
    question: str = dspy.InputField()
    input_schema: str = dspy.InputField()
    pruned_schema: GraphSchema = dspy.OutputField()

class Query(BaseModel):
    query: str = Field(description="Valid Cypher query")

class Text2Cypher(dspy.Signature):
    """Generate a Cypher query for the given question using the provided schema."""
    question: str = dspy.InputField()
    input_schema: str = dspy.InputField()
    query: Query = dspy.OutputField()

# Database manager
class KuzuDatabaseManager:
    def __init__(self, db_path: str):
        self.db = kuzu.Database(db_path, read_only=True)
        self.conn = kuzu.Connection(self.db)
    
    @property
    def get_schema_dict(self):
        response = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'NODE' RETURN *;")
        nodes = [row[1] for row in response]
        response = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'REL' RETURN *;")
        rel_tables = [row[1] for row in response]
        relationships = []
        for tbl_name in rel_tables:
            response = self.conn.execute(f"CALL SHOW_CONNECTION('{tbl_name}') RETURN *;")
            for row in response:
                relationships.append({"name": tbl_name, "from": row[0], "to": row[1]})
        schema = {"nodes": [], "edges": []}
        for node in nodes:
            node_schema = {"label": node, "properties": []}
            node_properties = self.conn.execute(f"CALL TABLE_INFO('{node}') RETURN *;")
            for row in node_properties:
                node_schema["properties"].append({"name": row[1], "type": row[2]})
            schema["nodes"].append(node_schema)
        for rel in relationships:
            edge = {"label": rel["name"], "from": rel["from"], "to": rel["to"], "properties": []}
            rel_properties = self.conn.execute(f"""CALL TABLE_INFO('{rel["name"]}') RETURN *;""")
            for row in rel_properties:
                edge["properties"].append({"name": row[1], "type": row[2]})
            schema["edges"].append(edge)
        return schema

# Load exemplars
print("Loading exemplars...")
with open('data/exemplars.json', 'r', encoding='utf-8') as f:
    exemplars = json.load(f)

print(f"Found {len(exemplars)} exemplars\n")

# Initialize database
print("Initializing database...")
db_manager = KuzuDatabaseManager("nobel.kuzu")

# Initialize Enhanced pipeline
print("Loading Enhanced pipeline...")
enhanced = text2cypher_enhanced.create_text2cypher_pipeline(
    db_manager.conn,
    exemplars_path="data/exemplars.json"
)

# Initialize Baseline components
print("Loading Baseline pipeline...")
prune = dspy.Predict(PruneSchema)
baseline_text2cypher = dspy.ChainOfThought(Text2Cypher)

print("\n" + "=" * 100)
print(f"FINAL COMPREHENSIVE TEST: ALL {len(exemplars)} EXEMPLARS")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)

results = {
    'metadata': {
        'total_exemplars': len(exemplars),
        'test_date': datetime.now().isoformat(),
        'model': 'gemini/gemini-2.5-flash-lite',
        'max_tokens': 6000
    },
    'summary': {},
    'details': []
}

enhanced_total_results = 0
baseline_total_results = 0
enhanced_wins = 0
baseline_wins = 0
ties = 0
enhanced_errors = 0
baseline_errors = 0
schema_prune_failures = 0

for i, ex in enumerate(exemplars, 1):
    question = ex['question']
    expected_query = ex['query']
    
    print(f"\n[{i}/{len(exemplars)}] {question[:80]}...")
    
    # Get expected result count
    expected_count = 0
    try:
        expected_result = db_manager.conn.execute(expected_query)
        expected_count = sum(1 for _ in expected_result)
    except Exception as e:
        print(f"  ⚠️  Expected query failed: {str(e)[:50]}")
    
    # Prune schema
    schema = None
    try:
        time.sleep(3.0)  # Rate limiting - 3 seconds between requests (safer)
        prune_result = prune(question=question, input_schema=str(db_manager.get_schema_dict))
        schema = prune_result.pruned_schema
    except Exception as e:
        error_msg = str(e)
        print(f"  ✗ Schema pruning failed: {error_msg[:80]}")
        schema_prune_failures += 1
        results['details'].append({
            'index': i,
            'question': question,
            'expected': expected_count,
            'enhanced': {'count': 0, 'error': f'Schema pruning failed: {error_msg[:100]}', 'success': False, 'query': None},
            'baseline': {'count': 0, 'error': f'Schema pruning failed: {error_msg[:100]}', 'success': False, 'query': None},
            'winner': 'none'
        })
        continue
    
    # Test Enhanced
    enhanced_success = False
    enhanced_count = 0
    enhanced_error = None
    enhanced_query = None
    try:
        enhanced_query, metadata = enhanced.generate(
            question=question,
            schema=schema.model_dump(),
            return_metadata=True
        )
        result = db_manager.conn.execute(enhanced_query)
        enhanced_count = sum(1 for _ in result)
        enhanced_total_results += enhanced_count
        enhanced_success = True
    except Exception as e:
        enhanced_error = str(e)[:150]
        enhanced_errors += 1
    
    # Delay before Baseline
    time.sleep(3.0)
    
    # Test Baseline
    baseline_success = False
    baseline_count = 0
    baseline_error = None
    baseline_query = None
    try:
        baseline_result = baseline_text2cypher(
            question=question,
            input_schema=str(schema.model_dump())
        )
        baseline_query = baseline_result.query.query
        result = db_manager.conn.execute(baseline_query)
        baseline_count = sum(1 for _ in result)
        baseline_total_results += baseline_count
        baseline_success = True
    except Exception as e:
        baseline_error = str(e)[:150]
        baseline_errors += 1
    
    # Determine winner (based on proximity to expected)
    winner = 'none'
    if enhanced_success and baseline_success:
        enh_diff = abs(enhanced_count - expected_count)
        base_diff = abs(baseline_count - expected_count)
        if enh_diff < base_diff:
            winner = 'enhanced'
            enhanced_wins += 1
            print(f"  ✓ Enhanced: {enhanced_count} | Baseline: {baseline_count} → ENHANCED WINS")
        elif base_diff < enh_diff:
            winner = 'baseline'
            baseline_wins += 1
            print(f"  Enhanced: {enhanced_count} | ✓ Baseline: {baseline_count} → BASELINE WINS")
        else:
            winner = 'tie'
            ties += 1
            print(f"  = Enhanced: {enhanced_count} | Baseline: {baseline_count} → TIE")
    elif enhanced_success:
        winner = 'enhanced'
        enhanced_wins += 1
        print(f"  ✓ Enhanced: {enhanced_count} | ✗ Baseline: ERROR → ENHANCED WINS")
    elif baseline_success:
        winner = 'baseline'
        baseline_wins += 1
        print(f"  ✗ Enhanced: ERROR | ✓ Baseline: {baseline_count} → BASELINE WINS")
    else:
        print(f"  ✗ Both failed")
    
    # Record results
    results['details'].append({
        'index': i,
        'question': question,
        'expected': expected_count,
        'enhanced': {
            'count': enhanced_count,
            'error': enhanced_error,
            'success': enhanced_success,
            'query': enhanced_query
        },
        'baseline': {
            'count': baseline_count,
            'error': baseline_error,
            'success': baseline_success,
            'query': baseline_query
        },
        'winner': winner
    })

# Calculate summary statistics
total_valid = enhanced_wins + baseline_wins + ties
results['summary'] = {
    'total_exemplars': len(exemplars),
    'schema_prune_failures': schema_prune_failures,
    'valid_comparisons': total_valid,
    'enhanced': {
        'wins': enhanced_wins,
        'win_rate': f"{enhanced_wins/total_valid*100:.1f}%" if total_valid > 0 else "N/A",
        'total_results': enhanced_total_results,
        'errors': enhanced_errors,
        'error_rate': f"{enhanced_errors/len(exemplars)*100:.1f}%"
    },
    'baseline': {
        'wins': baseline_wins,
        'win_rate': f"{baseline_wins/total_valid*100:.1f}%" if total_valid > 0 else "N/A",
        'total_results': baseline_total_results,
        'errors': baseline_errors,
        'error_rate': f"{baseline_errors/len(exemplars)*100:.1f}%"
    },
    'ties': ties,
    'tie_rate': f"{ties/total_valid*100:.1f}%" if total_valid > 0 else "N/A"
}

# Print final summary
print("\n" + "=" * 100)
print("FINAL RESULTS SUMMARY")
print("=" * 100)
print(f"\nTotal Exemplars: {len(exemplars)}")
print(f"Schema Pruning Failures: {schema_prune_failures}")
print(f"Valid Comparisons: {total_valid}")

print(f"\n📊 PERFORMANCE COMPARISON:")
print(f"  Enhanced Wins:  {enhanced_wins}/{total_valid} ({results['summary']['enhanced']['win_rate']})")
print(f"  Baseline Wins:  {baseline_wins}/{total_valid} ({results['summary']['baseline']['win_rate']})")
print(f"  Ties:           {ties}/{total_valid} ({results['summary']['tie_rate']})")

print(f"\n📈 TOTAL RESULTS:")
print(f"  Enhanced:  {enhanced_total_results} results")
print(f"  Baseline:  {baseline_total_results} results")
print(f"  Difference: {enhanced_total_results - baseline_total_results:+d}")

print(f"\n⚠️  ERROR RATES:")
print(f"  Enhanced:  {enhanced_errors}/{len(exemplars)} ({results['summary']['enhanced']['error_rate']})")
print(f"  Baseline:  {baseline_errors}/{len(exemplars)} ({results['summary']['baseline']['error_rate']})")

if enhanced_wins > baseline_wins:
    margin = enhanced_wins - baseline_wins
    print(f"\n🎉 ENHANCED IS SUPERIOR by {margin} questions!")
elif baseline_wins > enhanced_wins:
    margin = baseline_wins - enhanced_wins
    print(f"\n⚠️  BASELINE IS BETTER by {margin} questions")
else:
    print(f"\n= TIED")

print("\n" + "=" * 100)

# Save detailed results to JSON
output_file = 'final_test_78_exemplars.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✓ Detailed results saved to: {output_file}")
print(f"✓ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)
