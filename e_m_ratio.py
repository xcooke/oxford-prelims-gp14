from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import dataframe_image as dfi

# x is anode voltage V (measured using meter) but aiming for 300, 280, 260, 240, 220, 200

x = np.array([299.9, 290.2, 279.8, 270.5, 259.9, 250.4, 240.1, 230.4, 220.5, 210.2, 200.4, 0])[-7:]
x_err = 0.1*np.ones(len(x))

# I is coil current I (measured using meter)

I = np.array([2.370, 2.311, 2.227, 2.028, 1.751, 1.678, 1.563, 1.481, 1.395, 1.275, 1.152, 0])[-7:]
I_err = 0.001*np.ones(len(x))

# y is magnetic field B (found from I)

"""
y = (4e-7*np.pi*124*I/(0.147*(5/4)**(3/2)))**2
#y_err = (5.7529e-13*(I**(-2)) + 1.2319e-9*I**2)**0.5
y_err = np.multiply((2*(1.2319e-9 * I**2 )**0.5), y)
"""

B = 4e-7*np.pi*124*I/(0.147*(5/4)**(3/2))
B_err = np.multiply((5.7529e-13*(I**(-2)) + 1.2319e-9*I**2)**0.5, B)

y = B**2
y_err = 2*B_err*y

def linear(p, x):
    m, c = p
    return m*x+c

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

df = pd.DataFrame({"Anode Voltage (V)":x, "Anode Voltage Error (V)":x_err, "Coil Current (A)":I, "Coil Current Error (A)":I_err, "Magnetic Field Strength (H)":B, "Magnetic Field Strength Error (H)":B_err, "B^2 (H)":y, "B^2 Error (H)":y_err})

#print(df)

dfi.export(df,"e_m_ratio_table.png")


m, c = out.beta

#m = np.round(m, 3)
#c = np.round(c, 3)

#m = "{:.3e}".format(m)
#c = "{:.3e}".format(c)

m_err, c_err = out.sd_beta

#m_err = np.round(m_err, 3)
#c_err = np.round(c_err, 3)

#m_err = "{:.3e}".format(m_err)
#c_err = "{:.3e}".format(c_err)

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

plt.title("e/m")
plt.xlabel("Anode Voltage (V)")
plt.ylabel("Magnetic Field Strength ^2 (H)")
plt.legend()

plt.savefig("e_m_ratio_graph", dpi=200)
plt.show()