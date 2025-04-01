from semantic_text_splitter import MarkdownSplitter  # pylint: disable=no-name-in-module

from config import CHUNK_SIZE


def split_text(text):
    splitter = MarkdownSplitter(CHUNK_SIZE)
    return splitter.chunks(text)
