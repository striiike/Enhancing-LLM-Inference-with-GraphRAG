"""
Complete Relationship Fix Script: Add all missing relationships to the database
Can be used directly when rebuilding the database next time
"""
import kuzu
import polars as pl

print("=" * 80)
print("Completing Missing Relationships - Ensuring Database Integrity")
print("=" * 80)

# Read data
filepath = "./data/nobel.json"
df = pl.read_json(filepath).explode("prizes").unnest("prizes")
df = df.with_columns(
    pl.col("birthDate").str.replace("-00-00", "-01-01").str.to_date()
)

# Connect to database
db = kuzu.Database("nobel.kuzu")
conn = kuzu.Connection(db)

print("\n[Fix 1] City-IS_CITY_IN-Country (from birthPlace)")
print("-" * 80)
print("This relationship enables queries like 'born in China'")

result = conn.execute(
    """
    LOAD FROM $df
    WITH *
    WHERE birthPlaceCity IS NOT NULL AND birthPlaceCountryNow IS NOT NULL
    MATCH (ci:City {name: birthPlaceCity})
    MATCH (co:Country {name: birthPlaceCountryNow})
    MERGE (ci)-[r:IS_CITY_IN]->(co)
    RETURN count(DISTINCT r) AS num_new_rels
    """,
    parameters={"df": df}
)
count = result.get_as_pl()["num_new_rels"][0]
print(f"✓ Added {count} City-IS_CITY_IN-Country relationships (birth locations)")

print("\n[Fix 2] Country-IS_COUNTRY_IN-Continent (from birthPlace)")
print("-" * 80)
print("This relationship enables continent-level queries like 'born in Europe'")

result = conn.execute(
    """
    LOAD FROM $df
    WITH *
    WHERE birthPlaceCountryNow IS NOT NULL AND birthPlaceContinent IS NOT NULL
    MATCH (co:Country {name: birthPlaceCountryNow})
    MATCH (con:Continent {name: birthPlaceContinent})
    MERGE (co)-[rc:IS_COUNTRY_IN]->(con)
    RETURN count(DISTINCT rc) AS num_new_rels
    """,
    parameters={"df": df}
)
count = result.get_as_pl()["num_new_rels"][0]
print(f"✓ Added {count} Country-IS_COUNTRY_IN-Continent relationships (birth locations)")

print("\n\n[Verification] Test newly added relationships")
print("=" * 80)

# Test 1: Query by birth country
print("\nTest 1: Query Nobel laureates born in China")
result = conn.execute("""
    MATCH (s:Scholar)-[:BORN_IN]->(c:City)-[:IS_CITY_IN]->(co:Country {name: 'China'})
    RETURN count(s) as count
""")
count = result.get_as_pl()[0, 0]
print(f"Result: {count} scholars")
if count > 0:
    print("✓ PASS - Can query by birth country")
else:
    print("✗ FAIL - Relationships may not be correctly created")

# Test 2: Query by birth continent
print("\nTest 2: Count Nobel laureates by birth continent")
result = conn.execute("""
    MATCH (s:Scholar)-[:BORN_IN]->(:City)-[:IS_CITY_IN]->(:Country)-[:IS_COUNTRY_IN]->(con:Continent)
    RETURN con.name, count(DISTINCT s) as count
    ORDER BY count DESC
""")
rows = result.get_as_pl()
print(f"Result: Found {len(rows)} continents")
for row in rows.iter_rows():
    print(f"  {row[0]}: {row[1]} scholars")

if len(rows) > 0:
    print("✓ PASS - Can query by continent")
else:
    print("✗ FAIL - Continent relationships may not be correctly created")

# Test 3: Complex query - Scholars born in Europe but working at US institutions
print("\nTest 3: Scholars born in Europe but affiliated with institutions in USA")
result = conn.execute("""
    MATCH (s:Scholar)-[:BORN_IN]->(:City)-[:IS_CITY_IN]->(:Country)-[:IS_COUNTRY_IN]->(birth_cont:Continent {name: 'Europe'})
    MATCH (s)-[:AFFILIATED_WITH]->(i:Institution)-[:IS_LOCATED_IN]->(:City)-[:IS_CITY_IN]->(work_country:Country {name: 'USA'})
    RETURN count(DISTINCT s) as count
""")
count = result.get_as_pl()[0, 0]
print(f"Result: {count} scholars")
if count > 0:
    print("✓ PASS - Can perform complex cross-continent queries")

print("\n\n[Summary] Database Current Status")
print("=" * 80)

# Count all relationship types
result = conn.execute("CALL SHOW_TABLES() WHERE type = 'REL' RETURN *")
rel_tables = [row[1] for row in result]

print(f"\nTotal relationship tables: {len(rel_tables)}")
for rel in rel_tables:
    result = conn.execute(f"MATCH ()-[r:{rel}]->() RETURN count(r) as count")
    count = result.get_as_pl()[0, 0]
    print(f"  {rel}: {count} relationships")

print("\n✓ Database relationship completion finished!")
print("✓ Now supports more complex queries, improving Text2Cypher accuracy")
