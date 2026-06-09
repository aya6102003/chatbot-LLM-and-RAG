from neo4j import GraphDatabase
import json
import hashlib

# ───────── CONFIG ─────────
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

JSON_PATH = "structure_sciences.json"

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# ───────── HELPERS ─────────

def make_id(*parts):
    path = " > ".join([str(p) for p in parts if p])
    return hashlib.md5(path.encode("utf-8")).hexdigest()


# ───────── CONSTRAINTS ─────────

def create_constraints(tx):
    labels = ["Faculty", "General", "Department", "Level", "Program", "Category", "Specialization", "Year"]

    for label in labels:
        tx.run(f"""
        CREATE CONSTRAINT IF NOT EXISTS
        FOR (n:{label})
        REQUIRE n.id IS UNIQUE
        """)


# ───────── MAIN INSERT ─────────

def insert_structure(tx, data):

    faculty = data["faculty"]
    faculty_name = faculty["name"]
    faculty_id = make_id("FACULTY", faculty_name)

    tx.run("""
        MERGE (f:Faculty {id:$id})
        SET f.name = $name
    """, id=faculty_id, name=faculty_name)
    general_id = make_id("GENERAL", faculty_name)
    tx.run(""" MERGE (g:General {id:$id}) SET g.name='general' """, id=general_id)
    tx.run(""" MATCH (f:Faculty {id:$fac}) MATCH (g:General {id:$gen}) MERGE (f)-[:HAS_GENERAL]->(g) """, fac=faculty_id, gen=general_id)
    # ───────── Departments ─────────
    for dept in faculty.get("departments", []):

        dept_name = dept["name"]
        dept_id = make_id(faculty_name, dept_name)

        tx.run("""
            MATCH (f:Faculty {id:$fac})
            MERGE (d:Department {id:$id})
            SET d.name = $name
            MERGE (f)-[:HAS_DEPARTMENT]->(d)
        """, fac=faculty_id, id=dept_id, name=dept_name)

        # ───────── Levels ─────────
        for level in dept.get("levels", []):

            level_name = level["name"]
            level_id = make_id(faculty_name, dept_name, level_name)

            tx.run("""
                MATCH (d:Department {id:$dept})
                MERGE (l:Level {id:$id})
                SET l.name = $name
                MERGE (d)-[:HAS_LEVEL]->(l)
            """, dept=dept_id, id=level_id, name=level_name)

            # ───────── Programs ─────────
            for program in level.get("programs", []):
                program_name = program["name"]
                program_id = make_id(faculty_name, dept_name, level_name, program_name)

                tx.run("""
                    MATCH (l:Level {id:$level})
                    MERGE (p:Program {id:$id})
                    SET p.name = $name
                    MERGE (l)-[:HAS_PROGRAM]->(p)
                """, level=level_id, id=program_id, name=program_name)

                for year in program.get("years", []):
                    year_id = make_id(program_id, year)

                    tx.run("""
                        MATCH (p:Program {id:$program})
                        MERGE (y:Year {id:$id})
                        SET y.name = $name
                        MERGE (p)-[:HAS_YEAR]->(y)
                    """, program=program_id, id=year_id, name=year)

            # ───────── Categories (Master only) ─────────
            for cat in level.get("categories", []):

                cat_name = cat["type"]
                cat_id = make_id(faculty_name, dept_name, level_name, cat_name)

                tx.run("""
                    MATCH (l:Level {id:$level})
                    MERGE (c:Category {id:$id})
                    SET c.name = $name
                    MERGE (l)-[:HAS_CATEGORY]->(c)
                """, level=level_id, id=cat_id, name=cat_name)

                for spec in cat.get("specializations", []):
                    create_specialization(tx, cat_id,
                        [faculty_name, dept_name, level_name, cat_name],
                        spec
                    )

            # ───────── Direct specializations (Math / Physics case) ─────────
            for spec in level.get("specializations", []):
                create_specialization(tx, level_id,
                    [faculty_name, dept_name, level_name],
                    spec
                )


# ───────── SPECIALIZATION ─────────

def create_specialization(tx, parent_id, path, spec):

    spec_name = spec["name"]
    spec_id = make_id(*path, spec_name)

    tx.run("""
        MATCH (p {id:$parent})
        MERGE (s:Specialization {id:$id})
        SET s.name = $name
        MERGE (p)-[:HAS_SPECIALIZATION]->(s)
    """, parent=parent_id, id=spec_id, name=spec_name)

    for year in spec.get("years", []):
        year_id = make_id(spec_id, year)

        tx.run("""
            MATCH (s:Specialization {id:$spec})
            MERGE (y:Year {id:$id})
            SET y.name = $name
            MERGE (s)-[:HAS_YEAR]->(y)
        """, spec=spec_id, id=year_id, name=year)


# ───────── MAIN ─────────

def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    with driver.session() as session:

        session.execute_write(create_constraints)

        session.run("MATCH (n) DETACH DELETE n")

        session.execute_write(insert_structure, data)

    print("✅ Graph inserted successfully")


if __name__ == "__main__":
    main()

driver.close()