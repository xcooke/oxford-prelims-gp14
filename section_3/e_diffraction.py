import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.odr import Model, RealData, ODR
#import dataframe_image as dfi

# x is accel voltage V

V = np.array([3.7, 3.9, 4.1, 4.5, 4.9, 5.2, 5.4, 5.6])*1e3
V_err = 0.1e3*np.ones(len(V))


# making curve linear

x = V**(-0.5)

# error in new linear - see section_3_linear_error.jpg

x_err = abs(-0.5*V**(-1.5)* V_err)


# y is D_r

y = np.array([1.7, 1.65, 1.6, 1.5, 1.45, 1.4, 1.4, 1.4])*1e-2
y_err = 0.1*1e-2*np.ones(len(x))


df = pd.DataFrame({"Anode Voltage (V)":V, "Anode Voltage Error (V)":V_err, "Anode Voltage ^2 (V^2)":x, "Anode Voltage ^2 Error (V^2)":x_err, "D_r (m)":y, "D_r error (m)":y_err})

#dfi.export(df,"section_3/e_diffraction_table.png")

df.to_csv("section_3/e_diffraction_table.csv")


# getting line of best fit

def linear(p, x):
    m, c = p
    return m*x + c

linear_model = Model(linear)

data = RealData(x, y, sx=x_err, sy=y_err)

odr = ODR(data, linear_model, beta0=[2.5, 0.0])

out = odr.run()


m, c = out.beta

m_err, c_err = out.sd_beta

# y = m*x + c

print("m", m, m_err)
print("c", c, c_err)

with open("section_3/e_diffraction_line_of_best_fit.txt", "w") as f:
    f.write(f"y=m*x+c\nm = {m} +- {m_err}\nc = {c} +- {c_err}")


x_fit = np.linspace(x.min(), x.max(), 1000)

y_fit = linear(out.beta, x_fit)


plt.errorbar(x, y, xerr=x_err, yerr=y_err, linestyle='None', marker='x', label="Measurements")
plt.plot(x_fit, y_fit, label="Linear Fit")

plt.title("Electron Diffraction")
plt.xlabel("Anode Voltage ^(-0.5) (V^(-0.5))")
plt.ylabel("Ring Diameter (n=1) (m)")
plt.legend()

plt.savefig("section_3/e_diffraction_graph", dpi=200)
plt.show()