import argparse
import os
import re
import spacy
from hunspell import Hunspell

from language_model import LanguageModel



'''
Grammatical Error Correction (GEC) with a language model (LM) and minimal resources.
Corrects: non-word errors, morphological errors, some determiners and prepositions.
Does not correct: missing word errors, everything else.
'''

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
		self.det = {"", "the", "a", "an"}
		# List of common prepositions
		self.prep = {"", "about", "at", "by", "for", "from", "in", "of", "on", "to", "with"}
		self.threshold = threshold

	def correct(self, sentence):
		tokens = [tok.text for tok in self.nlp(sentence.strip())]
		# If the line is empty, preserve the newline in output and continue
		if not tokens:
			return ""
		# Search for and make corrections while has_errors is true
		has_errors = True
		# Iteratively correct errors one at a time until there are no more.
		while has_errors:
			tokens, has_errors = self.process(tokens)
		# Join all the tokens back together and upper case first char
		correction = " ".join(tokens)
		correction = correction[0].upper() + correction[1:]
		return correction

	def process(self, sent):
		# Process sent with spacy
		proc_sent =  self.nlp.tokenizer.tokens_from_list(sent)
		self.nlp.tagger(proc_sent)
		# Calculate avg token prob of the sent so far.
		orig_prob = self.lm.score(proc_sent.text) / len(proc_sent)
		# Store all the candidate corrected sentences here
		cand_dict = {}
		# Process each token.
		for tok in proc_sent:
			# SPELLCHECKING
			# Spell check: tok must be alphabetical and not a real word.
			if tok.text.isalpha() and not self.gb.spell(tok.text):
				cands = self.gb.suggest(tok.text)
				# Generate correction candidates
				if cands:
					cand_dict.update(generateCands(tok.i, cands, sent, self.threshold))
			# MORPHOLOGY
			if tok.lemma_ in self.gb_infl:
				cands = self.gb_infl[tok.lemma_]
				cand_dict.update(generateCands(tok.i, cands, sent, self.threshold))
			# DETERMINERS
			if tok.text in self.det:
				cand_dict.update(generateCands(tok.i, self.det, sent, self.threshold))
			# PREPOSITIONS
			if tok.text in self.prep:
				cand_dict.update(generateCands(tok.i, self.prep, sent, self.threshold))

		# Keep track of the best sent if any
		best_prob = float("-inf")
		best_sent = []
		# Loop through the candidate edits; edit[-1] is the error type weight
		for edit, cand_sent in cand_dict.items():
			# Score the candidate sentence
			cand_prob = self.lm.score(" ".join(cand_sent)) / len(cand_sent)
			# Compare cand_prob against weighted orig_prob and best_prob
			if cand_prob > edit[-1] * orig_prob and cand_prob > best_prob:
				best_prob = cand_prob
				best_sent = cand_sent
		# Return the best sentence and a boolean whether to search for more errors
		print(best_sent)
		if best_sent:
			return best_sent, True
		else:
			return sent, False











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
def generateCands(tok_id, cands, sent, weight):
	# Save candidates here.
	edit_dict = {}
	# Loop through the input alternative candidates
	for cand in cands:
		# Copy the input sentence
		new_sent = sent[:]
		# Change the target token with the current cand
		new_sent[tok_id] = cand
		# Remove empty strings from the list (for deletions)
		new_sent = list(filter(None, new_sent))
		# Give the edit a unique identifier
		edit_id = (tok_id, cand, weight)
		# Save non-empty sentences
		if new_sent: edit_dict[edit_id] = new_sent
	return edit_dict

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

	print(corrector.correct("Roses in the garden is red ."))
