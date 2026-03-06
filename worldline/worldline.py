
import worldline as wl

"""
Phase One:
"Worldlines"
These objects represent histories - sequences of recent events. One for 
objective events for the entire world, smaller ones for individual POVs that
will include thoughts and stuff. 

These will integrate heavily with Libraries somehow, important events getting
moved from Worldline to Library, and maybe Library recall methods might
directly return data from Worldlines. 

Oh, and what the Fate Engine plans to happen needs a Worldline too, independent
of what has already happened.

Wordlines in their entirety will NOT fit inside context windows, so only recent
events + what the Library recall system brings up will be available to the LLM.

Worldline objects may also be used for system logging purposes, for debugging
and such.
"""

class Worldline:
    def __init__(
        self,
        name: str,
        page_counter: wl.util.PageCounter | None = None,
        data: list[wl.util.Note] = [],
    ) -> None:
        self.name = name
        self.page_counter = wl.util.PageCounter() if page_counter is None else page_counter
        self.events = list(data)

    """
    Add an event to the worldline.
    If an event is provided, it will be added directly.
    If no event is provided, a new event will be created with the given title, content, and page, or the page counter's current page.
    """
    def add_event(
        self,
        event: wl.util.Note | None = None,
        title: str | None = None,
        content: str | None = None,
        page: int | None = None,
    ) -> None:
        if event is None:
            if page is None:
                page = self.page_counter.page
            event = wl.util.Note(title, content, page)
        self.events.append(event)

    def get_events(self) -> list[wl.util.Note]:
        return self.events

    def step_page(self) -> int:
        return self.page_counter.step()

    def __str__(self) -> str:
        return f"Worldline: {self.name}"