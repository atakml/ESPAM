PACKAGE = smiles
MAINFILE = smiles

ifndef ANTLR4
abort:
	$(error variable ANTLR4 is not set)
endif

all: $(PACKAGE).g4
	antlr4 $^ -Dlanguage=Python3 -visitor

run: $(MAINFILE).py
	python3 -m $^

ex: $(MAINFILE).py
	python3 smiles.py test.smiles

clean:
	rm -rf *~ $(PACKAGE)*.py *.pyc *.tokens  __pycache* *.interp
