import matplotlib
import numpy as np          # ermöglicht np-array
import matplotlib.pyplot as plt


# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))     # exp(x) berechnet e^x, auch für np.array

# Tangens Hyperbolicus: 
np.tanh(0)                          # berechnet tanh(x), auch für np.array


# ReLU
def relu(x):
    if x > 0:
        out = x
    else:
        out = 0
    return out

# Relufunktion vektorisieren, damit np.array damit verarbeitet werden können
relu_vec = np.vectorize(relu)


# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------


# Intervalle erzeugen
x1 = np.arange(-6, 7, 0.2)
x2 = np.arange(-4, 3, 1)

# Funktionswerte (y-Werte) aller drei Aktivierungsfunktionen berechnen
y_sigmoid = sigmoid(x1)    
y_tanh = np.tanh(x1)
y_relu = relu_vec(x2)

# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------

# Figure initialisieren
fig_activFunc = plt.figure(figsize=(16, 5))
ax_1 = fig_activFunc.add_subplot(131)           # add_subplot(131) steht für: Eine Row, Drei Columns, Index 1. Liefert ein "Axis" Objekt zum plotten zurück
ax_2 = fig_activFunc.add_subplot(132)
ax_3 = fig_activFunc.add_subplot(133)
# Sigmoid plotten
ax_1.plot(x1, y_sigmoid)
ax_1.grid(True)
ax_1.set_title('Sigmoid', fontsize=14)
# Tanh plotten
ax_2.plot(x1, y_tanh)
ax_2.grid(True)
ax_2.set_title('Tanh', fontsize=14)
# ReLU plotten
ax_3.plot(x2, y_relu)
ax_3.grid(True)
ax_3.set_title('ReLU', fontsize=14)

# Alle Plots anzeigen
plt.show()