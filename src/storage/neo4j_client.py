"""
Neo4j connection client for Graph RAG.
"""
import os
from typing import Optional, Any
from neo4j import GraphDatabase, Driver, Session, Result
from dotenv import load_dotenv

load_dotenv()


class Neo4jClient:
    """Client for interacting with Neo4j graph database."""
    
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j"
    ):
        """
        Initialize Neo4j client.
        
        Args:
            uri: Neo4j connection URI (default: from env)
            username: Neo4j username (default: from env)
            password: Neo4j password (default: from env)
            database: Database name to use
        """
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.environ.get("NEO4J_USERNAME", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "neo4j")
        self.database = database
        
        self._driver: Optional[Driver] = None
    
    def connect(self) -> None:
        """Establish connection to Neo4j."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
    
    def close(self) -> None:
        """Close the Neo4j connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
    
    def get_session(self) -> Session:
        """Get a new Neo4j session."""
        if self._driver is None:
            self.connect()
        return self._driver.session(database=self.database)
    
    def execute_query(
        self, 
        query: str, 
        parameters: Optional[dict] = None
    ) -> list[dict]:
        """
        Execute a Cypher query and return results as list of dictionaries.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        with self.get_session() as session:
            result: Result = session.run(query, parameters or {})
            return [dict(record) for record in result]
    
    def execute_write(
        self, 
        query: str, 
        parameters: Optional[dict] = None
    ) -> list[dict]:
        """
        Execute a write transaction.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records
        """
        def _write_fn(tx):
            result = tx.run(query, parameters or {})
            # Consume result within transaction
            return [dict(record) for record in result]
        
        with self.get_session() as session:
            return session.execute_write(_write_fn)
    
    def execute_read(
        self, 
        query: str, 
        parameters: Optional[dict] = None
    ) -> list[dict]:
        """
        Execute a read transaction.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records
        """
        def _read_fn(tx):
            result = tx.run(query, parameters or {})
            # Consume result within transaction
            return [dict(record) for record in result]
        
        with self.get_session() as session:
            return session.execute_read(_read_fn)
    
    def verify_connectivity(self) -> bool:
        """
        Verify that the connection to Neo4j is working.
        
        Returns:
            True if connected successfully
        """
        try:
            if self._driver is None:
                self.connect()
            return self._driver.verify_connectivity()
        except Exception as e:
            print(f"Neo4j connection verification failed: {e}")
            return False
    
    def create_constraints(self) -> None:
        """Create database constraints for the knowledge graph."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        ]
        
        for constraint in constraints:
            try:
                self.execute_query(constraint)
            except Exception as e:
                print(f"Constraint creation note: {e}")
    
    def clear_database(self) -> None:
        """Clear all nodes and relationships from the database."""
        self.execute_query("MATCH (n) DETACH DELETE n")
    
    def get_stats(self) -> dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with node and relationship counts
        """
        stats = {}
        
        # Count nodes by type
        node_counts = self.execute_read("""
            MATCH (n)
            RETURN labels(n)[0] as node_type, count(n) as count
        """)
        stats['nodes'] = {r['node_type']: r['count'] for r in node_counts}
        
        # Count relationships
        rel_counts = self.execute_read("""
            MATCH ()-[r]->()
            RETURN type(r) as rel_type, count(r) as count
        """)
        stats['relationships'] = {r['rel_type']: r['count'] for r in rel_counts}
        
        return stats
    
    def __enter__(self) -> "Neo4jClient":
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


# Singleton instance for convenience
_default_client: Optional[Neo4jClient] = None


def get_neo4j_client(
    uri: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    database: str = "neo4j"
) -> Neo4jClient:
    """
    Get or create a singleton Neo4j client.
    
    Args:
        uri: Neo4j connection URI
        username: Neo4j username
        password: Neo4j password
        database: Database name
        
    Returns:
        Neo4jClient instance
    """
    global _default_client
    if _default_client is None:
        _default_client = Neo4jClient(uri, username, password, database)
    return _default_client
