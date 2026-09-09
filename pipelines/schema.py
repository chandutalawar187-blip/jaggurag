from typing import Any, List, Optional, Dict
try:
    from pydantic import BaseModel, Field
except ImportError:  # lightweight fallback
    class BaseModel:
        def __init__(self, **kw):
            self.__dict__.update(kw)
        def model_dump(self): return self.__dict__
    def Field(default=None, **_): return default

class Heading(BaseModel):
    text: str
    level: int = 1
class PageObject(BaseModel):
    label: str
    description: str = ""
class ImageLabel(BaseModel):
    label: str
    description: str = ""
class Table(BaseModel):
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
class ListItem(BaseModel):
    text: str
    level: int = 0
class Relationship(BaseModel):
    source: str
    relation: str
    target: str
class AlphabetItem(BaseModel):
    letter: str
    word: str
    page: Optional[int] = None
class PageAnalysis(BaseModel):
    page_number: int
    text: str = ""
    visual_required: bool = False
    headings: List[Heading] = Field(default_factory=list)
    objects: List[PageObject] = Field(default_factory=list)
    image_labels: List[ImageLabel] = Field(default_factory=list)
    tables: List[Table] = Field(default_factory=list)
    lists: List[ListItem] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    alphabet_items: List[AlphabetItem] = Field(default_factory=list)
class Document(BaseModel):
    source: str
    pages: List[PageAnalysis] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

def dump(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
