import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

PATH = 'results/mlp/test'
OPATH = 'results/mlp/plots/final'

E1 = np.load(PATH+'/result_E4.npy')
E2 = np.load(PATH+'/result_E5.npy')


M1 = np.load(PATH+'/result_M4.npy')*100./1902.
M2 = np.load(PATH+'/result_M5.npy')*100./1991.

### BOXPLOT 1
data = [E1,E2]

fig, ax1 = plt.subplots(figsize=(10,6))
fig.canvas.set_window_title('Test Sets Results for both MLP/SMO')
plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

bp = plt.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')

# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
              alpha=0.5)
# Hide these grid behind plot objects
ax1.set_axisbelow(True)
ax1.set_title('Comparison of Test Set Results of both MLP and SMO')
ax1.set_xlabel('Classification Problem')
ax1.set_ylabel('Normalised Logistic Error')

# Now fill the boxes with desired colors
boxColors = ['darkkhaki','royalblue']
numBoxes = len(data)
medians = range(numBoxes)
for i in range(numBoxes):
  box = bp['boxes'][i]
  boxX = []
  boxY = []
  for j in range(5):
      boxX.append(box.get_xdata()[j])
      boxY.append(box.get_ydata()[j])
  boxCoords = zip(boxX,boxY)
  # Alternate between Dark Khaki and Royal Blue
  k = i % 2
  boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
  ax1.add_patch(boxPolygon)
  # Now draw the median lines back over what we just filled in
  med = bp['medians'][i]
  medianX = []
  medianY = []
  for j in range(2):
      medianX.append(med.get_xdata()[j])
      medianY.append(med.get_ydata()[j])
      plt.plot(medianX, medianY, 'k')
      medians[i] = medianY[0]
  # Finally, overplot the sample averages, with horizontal alignment
  # in the center of each box
  plt.plot([np.average(med.get_xdata())], [np.average(data[i])],
           color='w', marker='*', markeredgecolor='k')


# Set the axes ranges and axes labels
ax1.set_xlim(0.5, numBoxes+0.5)
top = 0.1
bottom = 0.
ax1.set_ylim(bottom, top)
xtickNames = plt.setp(ax1, xticklabels=['MLP on MNIST 3vs5' , 'MLP on MNIST 4vs9'])
plt.setp(xtickNames, rotation=45, fontsize=8)

# Due to the Y-axis scale being different across samples, it can be
# hard to compare differences in medians across the samples. Add upper
# X-axis tick labels with the sample medians to aid in comparison
# (just use two decimal places of precision)
pos = np.arange(numBoxes)+1
upperLabels = [str(np.round(s, 4)) for s in medians]
weights = ['bold', 'semibold']
for tick,label in zip(range(numBoxes),ax1.get_xticklabels()):
   k = tick % 2
   ax1.text(pos[tick], top-(top*0.05), upperLabels[tick],
        horizontalalignment='center', size='x-small', weight=weights[k],
        color=boxColors[k])
        
        
plt.figtext(0.80, 0.015, '*', color='white', backgroundcolor='silver',
           weight='roman', size='medium')
plt.figtext(0.815, 0.013, ' Average Value', color='black', weight='roman',
           size='x-small')

#save/show           
filename = OPATH+'/comparaisonError'
#plt.show()
plt.savefig(filename+'.pdf')
plt.close()

#### PLT 2
data = [M1,M2, 0.9041]

print 'SD MLP 3-5 :',np.std(M1)
print 'SD MLP 4-9 :',np.std(M2)
fig, ax1 = plt.subplots(figsize=(10,6))
fig.canvas.set_window_title('Test Sets Results for both MLP/SMO')
plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

bp = plt.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')

# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
              alpha=0.5)
# Hide these grid behind plot objects
ax1.set_axisbelow(True)
ax1.set_title('Comparison of Test Set Results of both MLP and SMO')
ax1.set_xlabel('Classification Problem')
ax1.set_ylabel('Classification Mistakes [%]')

# Now fill the boxes with desired colors
boxColors = ['darkkhaki','royalblue']
numBoxes = len(data)
medians = range(numBoxes)
for i in range(numBoxes):
  box = bp['boxes'][i]
  boxX = []
  boxY = []
  for j in range(5):
      boxX.append(box.get_xdata()[j])
      boxY.append(box.get_ydata()[j])
  boxCoords = zip(boxX,boxY)
  # Alternate between Dark Khaki and Royal Blue
  k = i % 2
  boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
  ax1.add_patch(boxPolygon)
  # Now draw the median lines back over what we just filled in
  med = bp['medians'][i]
  medianX = []
  medianY = []
  for j in range(2):
      medianX.append(med.get_xdata()[j])
      medianY.append(med.get_ydata()[j])
      plt.plot(medianX, medianY, 'k')
      medians[i] = medianY[0]
  # Finally, overplot the sample averages, with horizontal alignment
  # in the center of each box
  plt.plot([np.average(med.get_xdata())], [np.average(data[i])],
           color='w', marker='*', markeredgecolor='k')


# Set the axes ranges and axes labels
ax1.set_xlim(0.5, numBoxes+0.5)
top = 3.
bottom = 0.
ax1.set_ylim(bottom, top)
xtickNames = plt.setp(ax1, xticklabels=['MLP on MNIST 3vs5' , 'MLP on MNIST 4vs9', 'SMO on MNIST 4vs9'])
plt.setp(xtickNames, rotation=45, fontsize=8)

# Due to the Y-axis scale being different across samples, it can be
# hard to compare differences in medians across the samples. Add upper
# X-axis tick labels with the sample medians to aid in comparison
# (just use two decimal places of precision)
pos = np.arange(numBoxes)+1
upperLabels = [str(np.round(s, 4)) for s in medians]
weights = ['bold', 'semibold']
for tick,label in zip(range(numBoxes),ax1.get_xticklabels()):
   k = tick % 2
   ax1.text(pos[tick], top-(top*0.05), upperLabels[tick],
        horizontalalignment='center', size='x-small', weight=weights[k],
        color=boxColors[k])
        
plt.figtext(0.80, 0.015, '*', color='white', backgroundcolor='silver',
           weight='roman', size='medium')
plt.figtext(0.815, 0.013, ' Average Value', color='black', weight='roman',
           size='x-small')
           
#save/show           
filename = OPATH+'/comparaisonMistakes'
#plt.show()
plt.savefig(filename+'.pdf')
plt.close()

