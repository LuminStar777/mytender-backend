import sqlite3


class ChunkSearcher:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def disconnect(self):
        if self.conn:
            self.conn.close()

    def find_chunk(self, search_term):
        query_find_chunk = '''
            SELECT id, c0
            FROM embedding_fulltext_search_content
            WHERE c0 LIKE ?
        '''
        self.cursor.execute(query_find_chunk, (f'%{search_term}%',))
        found_chunks = self.cursor.fetchall()
        return found_chunks

    def get_previous_chunk(self, chunk_id):
        query_previous_chunk = '''
            SELECT c0
            FROM embedding_fulltext_search_content
            WHERE id = ?
        '''
        self.cursor.execute(query_previous_chunk, (chunk_id - 1,))
        previous_chunk = self.cursor.fetchone()
        return previous_chunk

    def get_next_chunk(self, chunk_id):
        query_next_chunk = '''
            SELECT c0
            FROM embedding_fulltext_search_content
            WHERE id = ?
        '''
        self.cursor.execute(query_next_chunk, (chunk_id + 1,))
        next_chunk = self.cursor.fetchone()
        return next_chunk

    def search_and_fetch_neighbors(self, search_term):
        self.connect()
        found_chunks = self.find_chunk(search_term)

        if found_chunks:
            original_chunk = found_chunks[0]
            original_id = original_chunk[0]
            print("Found chunk:", original_chunk[1])

            # Get previous chunk
            previous_chunk = self.get_previous_chunk(original_id)
            if previous_chunk:
                print(f"Previous chunk (id={original_id - 1}):", previous_chunk[0])
            else:
                print(f"No valid previous chunk found for id={original_id - 1}")

            # Get next chunk
            next_chunk = self.get_next_chunk(original_id)
            if next_chunk:
                print(f"Next chunk (id={original_id + 1}):", next_chunk[0])
            else:
                print(f"No valid next chunk found for id={original_id + 1}")
        else:
            print("No chunks found with the given search term.")

        self.disconnect()
