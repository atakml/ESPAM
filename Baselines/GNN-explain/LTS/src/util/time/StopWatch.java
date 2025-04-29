package util.time;


/**Records elapsed time. Behaves like you'd expect a stopwatch to.*/
public class StopWatch {

	private long prevElapsedTime = 0L;
	private long startTime;
	private boolean isRunning = false;
	
	/**Creates a new stopwatch. The stopwatch starts the moment it is created.*/
	public StopWatch() {
		start();
	}
	
	public void start() {
		if (!isRunning) {
			startTime = System.nanoTime();
			isRunning = true;
		}
	}
	
	public void pause() {
		if (isRunning) {
			long stopTime = System.nanoTime();
			prevElapsedTime += stopTime-startTime;
			isRunning = false;
		}
	}
	
	/**Pause and reset time*/
	public void stop() {
		if (isRunning) {
			prevElapsedTime = 0L;
			isRunning = false;
		}
	}
	
	public void reset() {
		prevElapsedTime = 0L;
		if (isRunning) {
			startTime = System.nanoTime();
		}
	}
	
	public long getTimeNanos() {
		long val = prevElapsedTime;
		if (isRunning) {
			long currTime = System.nanoTime();
			val += currTime-startTime;
		}
		return val;
	}
	
	public long getTimeMillis() {
		return getTimeNanos() / 1_000_000;
	}
	
}
