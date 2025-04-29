# Generated from smiles.g4 by ANTLR 4.8
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .smilesParser import smilesParser
else:
    from smilesParser import smilesParser

# This class defines a complete generic visitor for a parse tree produced by smilesParser.

class smilesVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by smilesParser#smiles.
    def visitSmiles(self, ctx:smilesParser.SmilesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by smilesParser#baseAtom.
    def visitBaseAtom(self, ctx:smilesParser.BaseAtomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by smilesParser#wildcardAtom.
    def visitWildcardAtom(self, ctx:smilesParser.WildcardAtomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by smilesParser#aliphatic_organic.
    def visitAliphatic_organic(self, ctx:smilesParser.Aliphatic_organicContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by smilesParser#aromatic_organic.
    def visitAromatic_organic(self, ctx:smilesParser.Aromatic_organicContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by smilesParser#bracket_atom.
    def visitBracket_atom(self, ctx:smilesParser.Bracket_atomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by smilesParser#baseSymbol.
    def visitBaseSymbol(self, ctx:smilesParser.BaseSymbolContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by smilesParser#wildcardSymbol.
    def visitWildcardSymbol(self, ctx:smilesParser.WildcardSymbolContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by smilesParser#isotope.
    def visitIsotope(self, ctx:smilesParser.IsotopeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by smilesParser#element_symbols.
    def visitElement_symbols(self, ctx:smilesParser.Element_symbolsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by smilesParser#aromatic_symbols.
    def visitAromatic_symbols(self, ctx:smilesParser.Aromatic_symbolsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by smilesParser#chiral.
    def visitChiral(self, ctx:smilesParser.ChiralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by smilesParser#hcount.
    def visitHcount(self, ctx:smilesParser.HcountContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by smilesParser#charge.
    def visitCharge(self, ctx:smilesParser.ChargeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by smilesParser#class_.
    def visitClass_(self, ctx:smilesParser.Class_Context):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by smilesParser#bond.
    def visitBond(self, ctx:smilesParser.BondContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by smilesParser#ringbond.
    def visitRingbond(self, ctx:smilesParser.RingbondContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by smilesParser#branchedAtom.
    def visitBranchedAtom(self, ctx:smilesParser.BranchedAtomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by smilesParser#branch.
    def visitBranch(self, ctx:smilesParser.BranchContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by smilesParser#bondedChain.
    def visitBondedChain(self, ctx:smilesParser.BondedChainContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by smilesParser#singleAtom.
    def visitSingleAtom(self, ctx:smilesParser.SingleAtomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by smilesParser#terminator.
    def visitTerminator(self, ctx:smilesParser.TerminatorContext):
        return self.visitChildren(ctx)



del smilesParser