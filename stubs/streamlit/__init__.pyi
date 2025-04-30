from typing import Any, Optional, Callable, Dict, List, Union, TypeVar, Generic, overload

def set_page_config(
    page_title: Optional[str] = None,
    page_icon: Optional[str] = None,
    layout: str = "centered",
    initial_sidebar_state: str = "auto",
    menu_items: Optional[Dict[str, Any]] = None
) -> None: ...

def title(body: str) -> None: ...
def header(body: str) -> None: ...
def subheader(body: str) -> None: ...
def text(body: str) -> None: ...
def markdown(body: str) -> None: ...
def caption(body: str) -> None: ...
def code(body: str, language: Optional[str] = None) -> None: ...
def write(*args: Any, **kwargs: Any) -> None: ...

def info(body: str) -> None: ...
def success(body: str) -> None: ...
def warning(body: str) -> None: ...
def error(body: str) -> None: ...

def progress(value: float, text: Optional[str] = None) -> Any: ...
def spinner(text: Optional[str] = None) -> Any: ...
def status(text: str, expanded: bool = True) -> Any: ...

def button(
    label: str,
    key: Optional[str] = None,
    help: Optional[str] = None,
    on_click: Optional[Callable] = None,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None,
    type: str = "secondary",
    disabled: bool = False
) -> bool: ...

def text_input(
    label: str,
    value: str = "",
    max_chars: Optional[int] = None,
    key: Optional[str] = None,
    type: str = "default",
    help: Optional[str] = None,
    autocomplete: Optional[str] = None,
    on_change: Optional[Callable] = None,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None,
    placeholder: Optional[str] = None,
    disabled: bool = False,
    label_visibility: str = "visible"
) -> str: ...

def text_area(
    label: str,
    value: str = "",
    height: Optional[int] = None,
    max_chars: Optional[int] = None,
    key: Optional[str] = None,
    help: Optional[str] = None,
    on_change: Optional[Callable] = None,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None,
    placeholder: Optional[str] = None,
    disabled: bool = False,
    label_visibility: str = "visible"
) -> str: ...

def selectbox(
    label: str,
    options: List[Any],
    index: Optional[int] = None,
    format_func: Optional[Callable[[Any], str]] = None,
    key: Optional[str] = None,
    help: Optional[str] = None,
    on_change: Optional[Callable] = None,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None,
    disabled: bool = False,
    label_visibility: str = "visible"
) -> Any: ...

def radio(
    label: str,
    options: List[Any],
    index: Optional[int] = None,
    format_func: Optional[Callable[[Any], str]] = None,
    key: Optional[str] = None,
    help: Optional[str] = None,
    on_change: Optional[Callable] = None,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None,
    disabled: bool = False,
    horizontal: bool = False,
    label_visibility: str = "visible"
) -> Any: ...

def download_button(
    label: str,
    data: Union[str, bytes],
    file_name: Optional[str] = None,
    mime: Optional[str] = None,
    key: Optional[str] = None,
    help: Optional[str] = None,
    on_click: Optional[Callable] = None,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None,
    disabled: bool = False
) -> bool: ...

def columns(spec: List[int]) -> List[Any]: ...
def container() -> Any: ...
def empty() -> Any: ...
def expander(label: str, expanded: bool = False) -> Any: ...

def stop() -> None: ...
def rerun() -> None: ...

class SessionState:
    """Streamlit session state class."""
    processing_complete: bool
    current_space_id: Optional[str]
    download_progress: float
    total_fragments: int
    current_fragment: int
    regenerating_quotes: bool
    selected_media: Optional[str]
    url_history: Dict[str, str]
    loaded_space_id: Optional[str]
    
    def __init__(self) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    def __setattr__(self, name: str, value: Any) -> None: ...
    def __delattr__(self, name: str) -> None: ...
    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def __delitem__(self, key: str) -> None: ...
    def __contains__(self, key: str) -> bool: ...

session_state: SessionState

class PathWatcher:
    @staticmethod
    def watch_file(filepath: str, *args: Any, **kwargs: Any) -> Optional[bool]: ...

class watcher:
    path_watcher: PathWatcher

class sidebar:
    def __init__(self) -> None: ...
    def __enter__(self) -> Any: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...
    def title(self, body: str) -> None: ...
    def header(self, body: str) -> None: ...
    def subheader(self, body: str) -> None: ...
    def text(self, body: str) -> None: ...
    def markdown(self, body: str) -> None: ...
    def caption(self, body: str) -> None: ...
    def code(self, body: str, language: Optional[str] = None) -> None: ...
    def write(self, *args: Any, **kwargs: Any) -> None: ... 