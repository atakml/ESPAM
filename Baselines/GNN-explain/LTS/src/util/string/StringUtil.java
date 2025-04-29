package util.string;

import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;

public class StringUtil {

	/**Returns a concatenation of the string representation of the given elements, separated by the given separator. */
	@SafeVarargs
	public static <T> String toSeparatedString(String separator, T...ts) {
		return toSeparatedString(separator, Arrays.asList(ts).iterator());
	}
	
	/**Returns a concatenation of the string representation of the given elements, separated by the given separator. */
	public static <T> String toSeparatedStringA(String separator, T[] ts) {
		return toSeparatedString(separator, Arrays.asList(ts).iterator());
	}
	
	/**Returns a concatenation of the string representation of the given elements, separated by the given separator. */
	public static <T> String toSeparatedString(String separator, Collection<T> ts) {
		return toSeparatedString(separator, ts.iterator());
	}
	
	/**Returns a concatenation of the string representation of the given elements, separated by the given separator. */
	public static <T> String toSeparatedString(String separator, Iterator<T> tIter) {
		StringBuilder sb = new StringBuilder();
		if (tIter.hasNext()) {
			sb.append(tIter.next().toString());
			while (tIter.hasNext()) {
				sb.append(separator);
				sb.append(tIter.next().toString());
			}
		}
		return sb.toString();
	}
	
}
