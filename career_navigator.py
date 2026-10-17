
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def get_driver():
    """Create and verify a Neo4j connection."""

    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
        raise ValueError(
            "Neo4j credentials are missing. Check your .env file."
        )

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    try:
        driver.verify_connectivity()
        return driver
    except Exception:
        driver.close()
        raise


def recommend_careers(resume_skills):
    """Compare resume skills against career requirements."""

    skills = [
        skill.strip().lower()
        for skill in resume_skills
        if isinstance(skill, str) and skill.strip()
    ]

    query = """
    MATCH (c:Career)-[:REQUIRES]->(s:Skill)

    WITH c,
         collect(s.name) AS requiredSkills,
         $resumeSkills AS candidateSkills

    WITH c, requiredSkills, candidateSkills,
         [skill IN requiredSkills
          WHERE toLower(skill) IN candidateSkills]
          AS matchedSkills

    RETURN
        c.name AS career,
        matchedSkills,
        [skill IN requiredSkills
         WHERE NOT toLower(skill) IN candidateSkills]
          AS missingSkills,
        size(matchedSkills) AS matchedCount,
        size(requiredSkills) AS requiredCount,
        round(
            100.0 * size(matchedSkills) /
            size(requiredSkills)
        ) AS coveragePercent

    ORDER BY coveragePercent DESC, career ASC
    """

    driver = get_driver()

    try:
        with driver.session() as session:
            result = session.run(
                query,
                resumeSkills=skills
            )
            return [record.data() for record in result]
    finally:
        driver.close()


if __name__ == "__main__":
    print("Testing Neo4j connection...")

    try:
        driver = get_driver()
        driver.close()

        print("SUCCESS: Connected to Neo4j Aura!")

        test_skills = ["Python", "SQL", "Git"]
        recommendations = recommend_careers(test_skills)

        print("\nCareer recommendations:")

        for career in recommendations:
            print(f"\nCareer: {career['career']}")
            print(f"Coverage: {career['coveragePercent']}%")
            print(f"Matched skills: {career['matchedSkills']}")
            print(f"Missing skills: {career['missingSkills']}")

    except Exception as error:
        print(f"ERROR: {error}")

def get_career_graph():
    """Fetch career nodes, skill nodes, and their relationships."""

    query = """
    MATCH (c:Career)-[:REQUIRES]->(s:Skill)
    RETURN c.name AS career, collect(s.name) AS skills
    ORDER BY career
    """

    driver = get_driver()

    try:
        with driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]
    finally:
        driver.close()
