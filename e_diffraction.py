from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import dataframe_image as dfi

# x is accel voltage V

V = (np.array([3.7, 3.9, 4.1, 4.5, 4.9, 5.2, 5.4, 5.6])*1e3)
V_err = 0.1e3*np.ones(len(V))

x_err = np.multiply(np.multiply(V_err, 1/V), V**(-0.5))
x = V**(-0.5)


# y is D_r

y = np.array([1.7, 1.65, 1.6, 1.5, 1.45, 1.4, 1.4, 1.4])*1e-2
y_err = 0.1*1e-2*np.ones(len(x))

def linear(p, x):
    m, c = p
    return m*x + c

from scipy.odr import Model, RealData, ODR

linear_model = Model(linear)

data = RealData(x, y, sx=x_err, sy=y_err)

odr = ODR(data, linear_model, beta0=[2.5, 0.0])

out = odr.run()

#out.pprint()


#x = np.round(x,3)
#x_err = np.round(x_err,3)

#y = np.round(y,4)
#y_err = np.round(y_err,4)

df = pd.DataFrame({"Anode Voltage (V)":V, "Anode Voltage Error (V)":V_err, "Anode Voltage ^2 (V)":x, "Anode Voltage Error ^2 (V)":x_err, "D_r (m)":y, "D_r error (m)":y_err})

#print(df)

dfi.export(df,"e_diffraction_table.png")


m, c = out.beta

#m = np.round(m, 3)
#c = np.round(c, 3)

m = "{:.3e}".format(m)
c = "{:.3e}".format(c)

m_err, c_err = out.sd_beta

m_err = np.round(m_err, 3)
c_err = np.round(c_err, 3)

m_err = "{:.3e}".format(m_err)
c_err = "{:.3e}".format(c_err)

print("m", m, m_err)
print("c", c, c_err)


x_fit = np.linspace(x.min(), x.max(), 1000)

y_fit = linear(out.beta, x_fit)

#plt.rcParams['text.usetex'] = True


"""
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
"""


plt.errorbar(x, y, xerr=x_err, yerr=y_err, linestyle='None', marker='x', label="Measurements")
plt.plot(x_fit, y_fit, label="Linear Fit")

#plt.text(0.22, 1, f"\(F=mx+c\)\n\(m={m}\pm{m_err}\)\n\(c={c}\pm{c_err}\)", horizontalalignment='center', verticalalignment='center')

plt.title("Electron Diffraction")
plt.xlabel("Anode Voltage ^2 (V)")
plt.ylabel("Ring Diameter (n=1) (m)")
plt.legend()

plt.savefig("e_diffraction_graph", dpi=200)
plt.show()