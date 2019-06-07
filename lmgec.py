import argparse
from typing import Tuple, List
import os
import re
import spacy
from hunspell import Hunspell

from language_model import LanguageModel



class UnsupervisedGrammarCorrector:
    def __init__(self, threshold=0.96):
        basename = os.path.dirname(os.path.realpath(__file__))
        self.lm = LanguageModel()
        # Load spaCy
        self.nlp = spacy.load("en")
        # Hunspell spellchecker: https://pypi.python.org/pypi/CyHunspell
        # CyHunspell seems to be more accurate than Aspell in PyEnchant, but a bit slower.
        self.gb = Hunspell("en_GB-large", hunspell_data_dir=basename + '/resources/spelling/')
        # Inflection forms: http://wordlist.aspell.net/other/
        self.gb_infl = loadWordFormDict(basename + "/resources/agid-2016.01.19/infl.txt")
        # List of common determiners
        self.determiners = {"", "the", "a", "an"}
        # List of common prepositions
        self.prepositions = {"", "about", "at", "by", "for", "from", "in", "of", "on", "to", "with"}
        self.threshold = threshold

    def correct(self, sentence):
        # If the line is empty, preserve the newline in output and continue
        if not sentence:
            return ""
        best = sentence
        score = self.lm.score(best)

        while True:
            new_best, new_score = self.process(best)
            if new_best and new_score > score:
                best = new_best
                score = new_score
            else:
                break

        return best

    def process(self, sentence: str) -> Tuple[str, bool]:
        # Process sent with spacy
        proc_sent = self.nlp.tokenizer(sentence)
        self.nlp.tagger(proc_sent)
        # Calculate avg token prob of the sent so far.
        orig_prob = self.lm.score(proc_sent.text)
        # Store all the candidate corrected sentences here
        candidates = []
        # Process each token.
        for tok in proc_sent:
            # SPELLCHECKING
            # Spell check: tok must be alphabetical and not a real word.

            candidate_tokens = set()

            lower_cased_token = tok.lower_

            if lower_cased_token.isalpha() and not self.gb.spell(lower_cased_token):
                candidate_tokens |= set(self.gb.suggest(lower_cased_token))
            # MORPHOLOGY
            if tok.lemma_ in self.gb_infl:
                candidate_tokens |= self.gb_infl[tok.lemma_]
            # DETERMINERS
            if lower_cased_token in self.determiners:
                candidate_tokens |= self.determiners
            # PREPOSITIONS
            if lower_cased_token in self.prepositions:
                candidate_tokens |= self.prepositions

            candidate_tokens = [c for c in candidate_tokens if self.gb.spell(c)]

            if candidate_tokens:
                if tok.is_title:
                    candidate_tokens = [c.title() for c in candidate_tokens]
                elif tok.is_upper:
                    candidate_tokens = [c.upper() for c in candidate_tokens]

                candidates.extend(self._generate_candidates(tok.i, candidate_tokens, proc_sent))

        best_prob = orig_prob
        best = sentence

        for candidate in candidates:
            # Score the candidate sentence
            cand_prob = self.lm.score(candidate.text)
            print(candidate.text, self.lm.score(candidate.text), cand_prob)

            # Compare cand_prob against weighted orig_prob and best_prob
            if cand_prob > best_prob:
                best_prob = cand_prob
                best = candidate.text
        # Return the best sentence and a boolean whether to search for more errors
        return best, best_prob


    def _generate_candidates(self, tok_id, candidate_tokens, tokenized_sentence) -> List[str]:
        # Save candidates here.
        candidates = []

        prefix = tokenized_sentence[:tok_id]
        suffix = tokenized_sentence[tok_id+1:]
        # Loop through the input alternative candidates
        for token in candidate_tokens:
            candidate = prefix.text_with_ws
            if token:
                candidate += token + " "
            candidate += suffix.text_with_ws
            candidate = self.nlp.tokenizer(candidate)
            candidates.append(candidate)
        return candidates

def main(args):
    corrector = UnsupervisedGrammarCorrector()
    out_sents = open(args.out, "w")

    # Process each tokenized sentence
    with open(args.input_sents) as sents:
        for sent in sents:
            correction = corrector.correct(sent)
            out_sents.write(correction+"\n")


# Input: Path to Automatically Generated Inflection Database (AGID)
# Output: A dictionary; key is lemma, value is a set of word forms for that lemma
def loadWordFormDict(path):
    entries = open(path).read().strip().split("\n")
    form_dict = {}
    for entry in entries:
        entry = entry.split(": ")
        key = entry[0].split()
        forms = entry[1]
        # The word lemma
        word = key[0]
        # Ignore some of the extra markup in the forms
        forms = re.sub("[012~<,_!\?\.\|]+", "", forms)
        forms = re.sub("\{.*?\}", "", forms).split()
        # Save the lemma and unique forms in the form dict
        form_dict[word] = set([word]+forms)
    return form_dict

# Input 1: A token index indicating the target of the correction.
# Input 2: A list of candidate corrections for that token.
# Input 3: The current sentence as a list of token strings.
# Input 4: An error type weight
# Output: A dictionary. Key is a tuple: (tok_id, cand, weight),
# value is a list of strings forming a candidate corrected sentence.


if __name__ == "__main__":
    # Define and parse program input
    # parser = argparse.ArgumentParser()
    # parser.add_argument("input_sents", help="A text file containing 1 tokenized sentence per line.")
    # parser.add_argument("-o", "--out", help="The output correct text file, 1 tokenized sentence per line.", required=True)
    # parser.add_argument("-th", "--threshold", help="LM percent improvement threshold. Default: 0.96 requires scores to be at least 4% higher than the original.", type=float, default=0.96)
    # args = parser.parse_args()
    # # Run the program.
    # main(args)

    corrector = UnsupervisedGrammarCorrector()
    print(corrector.lm.score(("A apple tree has apple which is red.")))
    print(corrector.lm.score(("An apple tree has apple which is red.")))

    print(corrector.correct("A apple tree has apple which is red."))
