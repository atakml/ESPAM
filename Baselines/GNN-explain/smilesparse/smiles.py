import argparse

import antlr4
import matplotlib.pyplot as plt
import networkx as nx
from antlr4.error.ErrorListener import ErrorListener
from smilesparse.bond_inference import bond_inference, TooMuchBonds
from smilesparse.visitsmiles import VisitSmiles, Unsupported

from smilesparse.smilesLexer import smilesLexer
from smilesparse.smilesParser import smilesParser
import traceback


class CountErrorListener(ErrorListener):
    """Count number of errors.

    Parser provides getNumberOfSyntaxErrors(), but the Lexer
    apparently doesn't provide an easy way to know if an error occurred
    after the fact. Do the counting ourselves with a listener.
    """

    def __init__(self):
        super(CountErrorListener, self).__init__()
        self.count = 0

    def syntaxError(self, recognizer, offendingsymbol, line, column, msg, e):
        self.count += 1


def main():
    # command line
    parser = argparse.ArgumentParser(description='Exec/Type mu files.')
    parser.add_argument('path', type=str, help='file to exec and type')
    args = parser.parse_args()

    # lex and parse
    input_s = antlr4.FileStream(args.path, encoding='utf8')
    lexer = smilesLexer(input_s)
    counter = CountErrorListener()
    lexer.listeners.append(counter)
    stream = antlr4.CommonTokenStream(lexer)
    parser = smilesParser(stream)
    parser._listeners.append(counter)
    tree = parser.smiles()
    if counter.count > 0:
        exit(3)  # Syntax or lexicography errors occured

    # interpret Visitor
    parse_visitor = VisitSmiles()
    try:
        output = parse_visitor.visit(tree)
        plt.subplot(121)
        labs = {n: d["symbol"] for n, d in output.nodes(data=True)}
        pos = nx.spring_layout(output)
        nx.draw(output, pos)
        nx.draw_networkx_labels(output, pos, labels=labs)
        edge_labels = nx.get_edge_attributes(output, 'type')
        nx.draw_networkx_edge_labels(output, pos, labels=edge_labels)
        plt.show()
    except Unsupported:
        print(parse_visitor.log)


def smiles_to_graph(formula: str, **kwargs) -> nx.Graph:
    input_s = antlr4.InputStream(formula)
    lexer = smilesLexer(input_s)
    stream = antlr4.CommonTokenStream(lexer)
    parser = smilesParser(stream)
    tree = parser.smiles()
    parse_visitor = VisitSmiles()

    graph = parse_visitor.visit(tree)
    graph = bond_inference(graph)
    return graph
    '''except Unsupported as e:
        if kwargs.get("verbose"):
            print(parse_visitor.log)
    except TooMuchBonds as e:
        if kwargs.get("verbose"):
            print(e)'''


if __name__ == '__main__':
    main()
