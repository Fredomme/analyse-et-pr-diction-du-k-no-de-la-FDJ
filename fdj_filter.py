import numpy as np

class FDJFilter:
    """
    Filtre FDJ paramétrable.
    On peut ajuster le comportement via des attributs ou
    des méthodes plus élaborées (score, weighting, etc.).
    """
    def __init__(self, hot_numbers, cold_numbers, suspect_nums,
                 top_pairs, suspicious_groups, memory_buffer=None,
                 max_boules=20):
        self.hot_numbers = hot_numbers
        self.cold_numbers = cold_numbers
        self.suspect_nums = suspect_nums
        self.top_pairs = top_pairs
        self.suspicious_groups = suspicious_groups
        self.memory_buffer = memory_buffer  # ex. les 10 derniers tirages
        self.max_boules = max_boules

    def filter_draw(self, draw):
        """
        Filtre "classique", qu'il sera possible d'affiner pour un scoring plus fin.
        """
        corrected = [x for x in draw if x not in self.cold_numbers]

        # Forcer l'ajout de suspects
        for sn in self.suspect_nums:
            if sn not in corrected and len(corrected) < self.max_boules:
                corrected.append(sn)

        # Ajouter les hot
        for hn in self.hot_numbers:
            if hn not in corrected and len(corrected) < self.max_boules:
                corrected.append(hn)

        # Ajouter paires
        for (a, b) in self.top_pairs:
            if a in corrected and b not in corrected and len(corrected) < self.max_boules:
                corrected.append(b)
            elif b in corrected and a not in corrected and len(corrected) < self.max_boules:
                corrected.append(a)

        # Mémoire
        if self.memory_buffer is not None:
            recent = set(x for row in self.memory_buffer[-10:] for x in row)
            for r in recent:
                if r not in corrected and len(corrected) < self.max_boules:
                    corrected.append(r)

        # Groupes suspects
        for grp in self.suspicious_groups:
            if len(set(grp).intersection(corrected)) >= 2:
                for g in grp:
                    if g not in corrected and len(corrected) < self.max_boules:
                        corrected.append(g)

        final_draw = sorted(set(corrected))[:self.max_boules]
        return final_draw
