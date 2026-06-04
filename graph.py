from neo4j import GraphDatabase
import json
import hashlib

# ───────── CONFIG ─────────
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

JSON_PATH = "university_structure.json"

# ───────── CONNECT ─────────
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)


# ───────── HELPERS ─────────

def make_id(*parts):
    """
    Create deterministic unique ID from path.
    """
    path = " > ".join([str(p) for p in parts if p])
    return hashlib.md5(path.encode("utf-8")).hexdigest()


# ───────── CONSTRAINTS ─────────

def create_constraints(tx):

    labels = [
        "Faculte",
        "General",
        "Departement",
        "Filiere",
        "Niveau",
        "Parcours",
        "Specialite",
        "Annee"
    ]

    for label in labels:
        tx.run(f"""
        CREATE CONSTRAINT IF NOT EXISTS
        FOR (n:{label})
        REQUIRE n.id IS UNIQUE
        """)


# ───────── INSERT GRAPH ─────────

def insert_structure(tx, data):

    faculte_name = data.get("faculte", "Sciences")

    fac_id = make_id("FACULTE", faculte_name)

    tx.run("""
        MERGE (f:Faculte {id:$id})
        SET f.name = $name
    """, id=fac_id, name=faculte_name)

    # General Node
    general_id = make_id("GENERAL", faculte_name)

    tx.run("""
        MERGE (g:General {id:$id})
        SET g.name='general'
    """, id=general_id)

    tx.run("""
        MATCH (f:Faculte {id:$fac})
        MATCH (g:General {id:$gen})
        MERGE (f)-[:HAS_GENERAL]->(g)
    """, fac=fac_id, gen=general_id)

    # Departments
    for dept in data.get("departements", []):

        dept_name = dept["nom"]

        dept_id = make_id(
            faculte_name,
            dept_name
        )

        tx.run("""
            MATCH (f:Faculte {id:$fac})

            MERGE (d:Departement {id:$dept_id})
            SET d.name = $dept_name

            MERGE (f)-[:HAS_DEPARTEMENT]->(d)
        """,
        fac=fac_id,
        dept_id=dept_id,
        dept_name=dept_name)

        # Department with filieres
        if "filieres" in dept:

            for fil in dept["filieres"]:

                fil_name = fil["nom"]

                fil_id = make_id(
                    faculte_name,
                    dept_name,
                    fil_name
                )

                tx.run("""
                    MATCH (d:Departement {id:$dept})

                    MERGE (f:Filiere {id:$fid})
                    SET f.name = $fname

                    MERGE (d)-[:HAS_FILIERE]->(f)
                """,
                dept=dept_id,
                fid=fil_id,
                fname=fil_name)

                process_niveaux(
                    tx,
                    parent_id=fil_id,
                    parent_path=[
                        faculte_name,
                        dept_name,
                        fil_name
                    ],
                    node=fil
                )

        else:

            process_niveaux(
                tx,
                parent_id=dept_id,
                parent_path=[
                    faculte_name,
                    dept_name
                ],
                node=dept
            )


# ───────── PROCESS NIVEAUX ─────────

def process_niveaux(tx, parent_id, parent_path, node):

    for niv in node.get("niveaux", []):

        niv_name = niv["nom"]

        niv_id = make_id(
            *parent_path,
            niv_name
        )

        tx.run("""
            MATCH (p {id:$parent})

            MERGE (n:Niveau {id:$id})
            SET n.name = $name

            MERGE (p)-[:HAS_NIVEAU]->(n)
        """,
        parent=parent_id,
        id=niv_id,
        name=niv_name)

        # Doctorat etc.
        if (
            "specialites" not in niv
            and
            "parcours" not in niv
        ):
            continue

        # Master with parcours
        if "parcours" in niv:

            for parc in niv["parcours"]:

                parc_name = parc["nom"]

                parc_id = make_id(
                    *parent_path,
                    niv_name,
                    parc_name
                )

                tx.run("""
                    MATCH (n:Niveau {id:$niv})

                    MERGE (p:Parcours {id:$pid})
                    SET p.name = $pname

                    MERGE (n)-[:HAS_PARCOURS]->(p)
                """,
                niv=niv_id,
                pid=parc_id,
                pname=parc_name)

                for spec in parc.get("specialites", []):

                    create_specialite(
                        tx,
                        parent_id=parc_id,
                        parent_path=[
                            *parent_path,
                            niv_name,
                            parc_name
                        ],
                        spec=spec
                    )

        # Normal specialites
        if "specialites" in niv:

            for spec in niv["specialites"]:

                create_specialite(
                    tx,
                    parent_id=niv_id,
                    parent_path=[
                        *parent_path,
                        niv_name
                    ],
                    spec=spec
                )


# ───────── CREATE SPECIALITE ─────────

def create_specialite(
        tx,
        parent_id,
        parent_path,
        spec):

    spec_name = spec["nom"]

    spec_id = make_id(
        *parent_path,
        spec_name
    )

    tx.run("""
        MATCH (p {id:$parent})

        MERGE (s:Specialite {id:$id})
        SET s.name = $name

        MERGE (p)-[:HAS_SPECIALITE]->(s)
    """,
    parent=parent_id,
    id=spec_id,
    name=spec_name)

    for an in spec.get("annees", []):

        an_name = an["nom"]

        an_id = make_id(
            *parent_path,
            spec_name,
            an_name
        )

        tx.run("""
            MATCH (s:Specialite {id:$spec})

            MERGE (a:Annee {id:$id})
            SET a.name = $name

            MERGE (s)-[:HAS_ANNEE]->(a)
        """,
        spec=spec_id,
        id=an_id,
        name=an_name)


# ───────── MAIN ─────────

def main():

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    with driver.session() as session:

        session.execute_write(create_constraints)

        session.run("""
            MATCH (n)
            DETACH DELETE n
        """)

        session.execute_write(
            insert_structure,
            data
        )

    print("✅ Graph inserted successfully")


if __name__ == "__main__":
    main()

driver.close()