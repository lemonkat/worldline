"""

Phase One:

"Libraries"
Library objects represent banks of knowledge. 
One central Library is created for each story reprsenting the absolute truth, 
and there are additional Libraries for each Persona or perhaps societal perspective. 
Theoretically, these rules could be broken but those are going to be very complex edge cases 
and I don't want to deal with them for now.

Under the hood, Libraries are going to use some sort of vector embedding recall system like I used in a past project. 
Not sure exactly how it'll work and it will probably need some tuning.

Libraries will have methods for recalling pertinent info, as well as adding new info. 
New info may conflict older info and each Library will have some tuning params to determine how this happens - 
different Personas will do this differently.

There will be a base Library class, but actual Libraries will either be 
collections of Notes, data drawn from Worldlines, or unions of other Libraries.
"""

import numpy as np

import worldline as wl

class Library:
    def __init__(
        self, 
        name: str, 
        notes: list[wl.util.Note] | None = None,
        weight_relevance: float = 2.0,
        weight_importance: float = 1.0,
        summary_size: int = 10,
    ) -> None:
        self.name = name
        self.notes = notes or []
        self.weight_relevance = weight_relevance
        self.weight_importance = weight_importance
        self.summary_size = summary_size
    
    # query = None means just gather things that are generally important (or recent for Worldlines)
    def search(
        self, 
        query: str | wl.util.Note | None, 
        k: int = 1, 
    ) -> list[wl.util.Note]:
        return sorted(
            self.notes, 
            key=lambda note: self.score(query, note), 
            reverse=True
        )[:min(k, len(self.notes))]

    def score(
        self, 
        query: str | wl.util.Note | None, 
        item: wl.util.Note,
    ) -> float:
        score = item.importance * self.weight_importance
        if query is None:
            return score
        return score + np.dot(wl.util.get_emb(str(query)), item.emb) * self.weight_relevance

    def add(
        self, 
        note: wl.util.Note,
    ) -> None:
        self.notes.append(note)
        
    def __str__(self) -> str:
        return f"Library: {self.name}"

    @property
    def summary(self) -> wl.util.Note:
        return wl.util.Note(
            f"Pertinent information relating to {self.name}:",
            "\n".join([str(note) for note in self.search(None, self.summary_size)]),
            -1,
        )

"""
Phase Two:
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

class Worldline(Library):
    def __init__(
        self, 
        name: str, 
        notes: list[wl.util.Note] | None = None,
        weight_relevance: float = 2.0,
        weight_importance: float = 1.0,
        weight_recency: float = 1.0,
        t_decay: float = 0.95,
        t_global_base: float = 0.5,
        summary_size: int = 10,
    ) -> None:
        super().__init__(name, notes, weight_relevance, weight_importance, summary_size)
        self.weight_recency = weight_recency
        self.t_decay = t_decay
        self.t_global_base = t_global_base

    def score(
        self, 
        query: str | wl.util.Note | None, 
        item: wl.util.Note,
    ) -> float:
        base_score = super().score(query, item)
        if not isinstance(query, wl.util.Note):
            return base_score
        t_value = self.t_global_base if -1 in [query.page, item.page] else self.t_decay ** abs(query.page - item.page)
        return base_score + t_value * self.weight_recency

if __name__ == "__main__":
    wl.util.init()
    lib = Worldline("MAIN", notes=[
        wl.util.Note("breakfast", "jim ate breakfast at 3am. this is not when he usually eats breakfast.", 0),
        wl.util.Note("lunch", "jim ate lunch at 12pm.", 1),
        wl.util.Note("dinner", "jim ate dinner at 6pm.", 2),
    ])
    print(*lib.search("food", 2), sep="\n")