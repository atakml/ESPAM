# Generated from smiles.g4 by ANTLR 4.8
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .smilesParser import smilesParser
else:
    from smilesParser import smilesParser

# This class defines a complete listener for a parse tree produced by smilesParser.
class smilesListener(ParseTreeListener):

    # Enter a parse tree produced by smilesParser#smiles.
    def enterSmiles(self, ctx:smilesParser.SmilesContext):
        pass

    # Exit a parse tree produced by smilesParser#smiles.
    def exitSmiles(self, ctx:smilesParser.SmilesContext):
        pass


    # Enter a parse tree produced by smilesParser#baseAtom.
    def enterBaseAtom(self, ctx:smilesParser.BaseAtomContext):
        pass

    # Exit a parse tree produced by smilesParser#baseAtom.
    def exitBaseAtom(self, ctx:smilesParser.BaseAtomContext):
        pass


    # Enter a parse tree produced by smilesParser#wildcardAtom.
    def enterWildcardAtom(self, ctx:smilesParser.WildcardAtomContext):
        pass

    # Exit a parse tree produced by smilesParser#wildcardAtom.
    def exitWildcardAtom(self, ctx:smilesParser.WildcardAtomContext):
        pass


    # Enter a parse tree produced by smilesParser#aliphatic_organic.
    def enterAliphatic_organic(self, ctx:smilesParser.Aliphatic_organicContext):
        pass

    # Exit a parse tree produced by smilesParser#aliphatic_organic.
    def exitAliphatic_organic(self, ctx:smilesParser.Aliphatic_organicContext):
        pass


    # Enter a parse tree produced by smilesParser#aromatic_organic.
    def enterAromatic_organic(self, ctx:smilesParser.Aromatic_organicContext):
        pass

    # Exit a parse tree produced by smilesParser#aromatic_organic.
    def exitAromatic_organic(self, ctx:smilesParser.Aromatic_organicContext):
        pass


    # Enter a parse tree produced by smilesParser#bracket_atom.
    def enterBracket_atom(self, ctx:smilesParser.Bracket_atomContext):
        pass

    # Exit a parse tree produced by smilesParser#bracket_atom.
    def exitBracket_atom(self, ctx:smilesParser.Bracket_atomContext):
        pass


    # Enter a parse tree produced by smilesParser#baseSymbol.
    def enterBaseSymbol(self, ctx:smilesParser.BaseSymbolContext):
        pass

    # Exit a parse tree produced by smilesParser#baseSymbol.
    def exitBaseSymbol(self, ctx:smilesParser.BaseSymbolContext):
        pass


    # Enter a parse tree produced by smilesParser#wildcardSymbol.
    def enterWildcardSymbol(self, ctx:smilesParser.WildcardSymbolContext):
        pass

    # Exit a parse tree produced by smilesParser#wildcardSymbol.
    def exitWildcardSymbol(self, ctx:smilesParser.WildcardSymbolContext):
        pass


    # Enter a parse tree produced by smilesParser#isotope.
    def enterIsotope(self, ctx:smilesParser.IsotopeContext):
        pass

    # Exit a parse tree produced by smilesParser#isotope.
    def exitIsotope(self, ctx:smilesParser.IsotopeContext):
        pass


    # Enter a parse tree produced by smilesParser#element_symbols.
    def enterElement_symbols(self, ctx:smilesParser.Element_symbolsContext):
        pass

    # Exit a parse tree produced by smilesParser#element_symbols.
    def exitElement_symbols(self, ctx:smilesParser.Element_symbolsContext):
        pass


    # Enter a parse tree produced by smilesParser#aromatic_symbols.
    def enterAromatic_symbols(self, ctx:smilesParser.Aromatic_symbolsContext):
        pass

    # Exit a parse tree produced by smilesParser#aromatic_symbols.
    def exitAromatic_symbols(self, ctx:smilesParser.Aromatic_symbolsContext):
        pass


    # Enter a parse tree produced by smilesParser#chiral.
    def enterChiral(self, ctx:smilesParser.ChiralContext):
        pass

    # Exit a parse tree produced by smilesParser#chiral.
    def exitChiral(self, ctx:smilesParser.ChiralContext):
        pass


    # Enter a parse tree produced by smilesParser#hcount.
    def enterHcount(self, ctx:smilesParser.HcountContext):
        pass

    # Exit a parse tree produced by smilesParser#hcount.
    def exitHcount(self, ctx:smilesParser.HcountContext):
        pass


    # Enter a parse tree produced by smilesParser#charge.
    def enterCharge(self, ctx:smilesParser.ChargeContext):
        pass

    # Exit a parse tree produced by smilesParser#charge.
    def exitCharge(self, ctx:smilesParser.ChargeContext):
        pass


    # Enter a parse tree produced by smilesParser#class_.
    def enterClass_(self, ctx:smilesParser.Class_Context):
        pass

    # Exit a parse tree produced by smilesParser#class_.
    def exitClass_(self, ctx:smilesParser.Class_Context):
        pass


    # Enter a parse tree produced by smilesParser#bond.
    def enterBond(self, ctx:smilesParser.BondContext):
        pass

    # Exit a parse tree produced by smilesParser#bond.
    def exitBond(self, ctx:smilesParser.BondContext):
        pass


    # Enter a parse tree produced by smilesParser#ringbond.
    def enterRingbond(self, ctx:smilesParser.RingbondContext):
        pass

    # Exit a parse tree produced by smilesParser#ringbond.
    def exitRingbond(self, ctx:smilesParser.RingbondContext):
        pass


    # Enter a parse tree produced by smilesParser#branchedAtom.
    def enterBranchedAtom(self, ctx:smilesParser.BranchedAtomContext):
        pass

    # Exit a parse tree produced by smilesParser#branchedAtom.
    def exitBranchedAtom(self, ctx:smilesParser.BranchedAtomContext):
        pass


    # Enter a parse tree produced by smilesParser#branch.
    def enterBranch(self, ctx:smilesParser.BranchContext):
        pass

    # Exit a parse tree produced by smilesParser#branch.
    def exitBranch(self, ctx:smilesParser.BranchContext):
        pass


    # Enter a parse tree produced by smilesParser#bondedChain.
    def enterBondedChain(self, ctx:smilesParser.BondedChainContext):
        pass

    # Exit a parse tree produced by smilesParser#bondedChain.
    def exitBondedChain(self, ctx:smilesParser.BondedChainContext):
        pass


    # Enter a parse tree produced by smilesParser#singleAtom.
    def enterSingleAtom(self, ctx:smilesParser.SingleAtomContext):
        pass

    # Exit a parse tree produced by smilesParser#singleAtom.
    def exitSingleAtom(self, ctx:smilesParser.SingleAtomContext):
        pass


    # Enter a parse tree produced by smilesParser#terminator.
    def enterTerminator(self, ctx:smilesParser.TerminatorContext):
        pass

    # Exit a parse tree produced by smilesParser#terminator.
    def exitTerminator(self, ctx:smilesParser.TerminatorContext):
        pass



del smilesParser