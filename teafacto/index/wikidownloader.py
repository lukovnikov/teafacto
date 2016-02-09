import wikipedia as wp
import pickle
import argparse

class WikiDownloader():
	def __init__(self):
		self.found = 0
		self.total = 0
		self.disad = 0
		self.count = 0
		self.error = 0
		self.errorgrams = []


	def download(self, ngram, followdisambi=True):
		file = None
		try:
			self.total += 1
			page = wp.page(ngram, auto_suggest=False)
			print str(self.count) + ": \"" + ngram + "\" :found: " + page.title
			file = open("data/" + page.title + ".txt", "w")
			file.write(page.content.encode("utf-8"))
			file.close()
			self.found += 1
		except wp.exceptions.DisambiguationError as e:
			if followdisambi:
				options = e.options[:min(5, len(e.options))]
				print str(self.count) + ": " + ngram + " disambiguated ???"
				self.disad += 1
				for option in options:
					self.download(option, followdisambi=False)
		except wp.exceptions.PageError as e:
			print str(self.count) + ": " + ngram + " - not found !!!"
		except Exception as e:
			print e
			self.error += 1
			self.errorgrams.append(ngram)
		finally:
			if file is not None:
				file.close()

	def run(self, file="ngrams.pkl"):
		ngrams = pickle.load(open(file))
		inter = 100
		for ngram in ngrams:
			if self.count % inter == 0:
				print "\n\n%d/%d found, %d disambiguated, %d total %d errors\n\n" % (self.found, self.count, self.disad, self.total, self.error)
			self.download(ngram)
			self.count += 1
		pickle.dump(self.errorgrams, open(file+".errorgrams.pkl", "w"))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('file')
	args = parser.parse_args()
	wd = WikiDownloader()
	wd.run(args.file)
