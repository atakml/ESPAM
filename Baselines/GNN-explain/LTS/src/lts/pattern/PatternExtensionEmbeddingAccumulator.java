package lts.pattern;

import java.util.Collection;
import java.util.HashMap;

/** Maps EmbeddingEdges that create the same pattern together into the same PatternFactory*/
public class PatternExtensionEmbeddingAccumulator {

	private final PatternEdge parentPattern;
	private final HashMap<PatternExtension, PatternFactory> accumulator = new HashMap<>();
	
	public PatternExtensionEmbeddingAccumulator(PatternEdge parentPattern) {
		this.parentPattern = parentPattern;
	}

	public void addProposal(PatternExtension extension, int graphId, EmbeddingEdge embedding) {
		PatternFactory patternFactory = accumulator.get(extension);
		if (patternFactory == null) {
			accumulator.put(extension, patternFactory = new PatternFactory(makeExtendedPatternEdgeFromExtension(extension)));
		}
		patternFactory.addEmbedding(graphId, embedding);
	}

	private PatternEdge makeExtendedPatternEdgeFromExtension(PatternExtension extension) {
		PatternEdge extendedPattern = null;
		switch(extension.getType()) {
		case BASE:
			extendedPattern = new PatternEdgeBase();
			break;		
		case EXISTING_NODES:
			extendedPattern = new PatternEdgeBetweenExistingNodes(
					parentPattern,
					((PatternExtensionBetweenExistingNodes) extension).getPatternIndex1(),
					((PatternExtensionBetweenExistingNodes) extension).getPatternIndex2());
			break;
		case NEW_NODE:
			extendedPattern = new PatternEdgeToNewNode(
					parentPattern,
					((PatternExtensionToNewNode) extension).getPatternIndex());
			break;
		}
		return extendedPattern;
	}

	public Collection<PatternFactory> getPatternFactories() {
		return accumulator.values();
	}

}
