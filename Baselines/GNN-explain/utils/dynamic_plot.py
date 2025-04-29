import matplotlib.pyplot as plt

train, = plt.plot([], [])
valid, = plt.plot([], [])
xdata = []
ydata = []
axes = plt.gca()
axes.set_xlim(0, 200)
axes.set_ylim(0, 1)
line, = axes.plot(xdata, ydata, 'r-')

def update_line(t,v):
    xdata.append(len(xdata))
    ydata.append(t)
    line.set_xdata(xdata)
    line.set_ydata(ydata)
    plt.draw()
    plt.pause(1e-17)
