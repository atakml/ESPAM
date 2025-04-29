package lts.algorithm.search_history;

/**Pairs an edge count with a score discretized to a int bin. 
 * Used to the history table, which maps instances of this to upper bound score predictions.*/
public class EdgeAndScoreBin implements Comparable<EdgeAndScoreBin> {

	private int edges, scoreBin;

	public int getEdges() {
		return edges;
	}

	public void setEdges(int edges) {
		this.edges = edges;
	}

	public int getScoreBin() {
		return scoreBin;
	}

	public void setScoreBin(int score) {
		this.scoreBin = score;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + edges;
		result = prime * result + scoreBin;
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		EdgeAndScoreBin other = (EdgeAndScoreBin) obj;
		if (edges != other.edges)
			return false;
		if (scoreBin != other.scoreBin)
			return false;
		return true;
	}

	@Override
	public String toString() {
		return "EdgeAndScore [edges=" + edges + ", scoreBin=" + scoreBin + "]";
	}

	@Override
	public int compareTo(EdgeAndScoreBin o) {
		int c = this.edges - o.edges;
		if (c == 0) c = this.scoreBin - o.scoreBin;
		return c;
	}
	
}
