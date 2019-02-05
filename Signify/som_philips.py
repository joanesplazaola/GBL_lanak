# 0-Importar librerias
from minisom import MiniSom as msom

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from matplotlib.lines import Line2D

# 1- cargar datos
data = pd.read_csv('../data/HugeCSVMulticlass/Complete.csv')[:10000]

colsImp = ['DOS: atm_vocht', 'MEET: Licht RX', 'DOS: atm_druk', 'DOS: atm_temp', 'DOS: EPI_TEMP', 'lineNumber',
		   'Status']

# data = data[colsImp]
data.drop('Time', axis=1, inplace=True)
X = data.loc[:, data.columns != 'Status']
y = data.Status
X = X.values

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

# 3- Inicialización y aprendizaje del SOM Initialization and training
# seleccionar dimensión del SOM
ydim = 50
xdim = 50
# inicializar SOM
som = msom(xdim, ydim, X.shape[1], random_seed=145, )
som.pca_weights_init(X)
# empezar aprendizaje
print("Training...")
som.train_batch(X, 4000)  # random training
print("\n...ready!")

# enseñar SOM
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
plt.colorbar()

plt.show()

# plotear las clases (ya que las tenemos....)
labels = ['', 'healthy', 'defect1', 'defect2', 'defect3']
markers = ['o', 's', 'D', 'v', '^']
colors = ['C0', 'C1', 'C2', 'C3', 'C4']

custom_lines = []
for cnt, x in enumerate(labels):
	custom_lines.append(Line2D([0], [0], marker=markers[cnt], markerfacecolor='None',
							   markeredgecolor=colors[cnt], markersize=12, markeredgewidth=2))

for cnt, xx in enumerate(data):
	w = som.winner(xx)  # getting the winner
	# palce a marker on the winning position for the sample xx
	plt.plot(w[0] + .5, w[1] + .5, markers[y[cnt]], markerfacecolor='None',
			 markeredgecolor=colors[y[cnt]], markersize=12, markeredgewidth=2)
ax.legend(custom_lines, labels, loc='center right', bbox_to_anchor=(1.4, 0.5))
plt.axis([0, xdim, 0, ydim])
plt.show()
