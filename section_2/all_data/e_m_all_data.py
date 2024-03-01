import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.odr import Model, RealData, ODR
#import dataframe_image as dfi

# x is anode voltage V (measured using meter)

x = np.array([299.9, 290.2, 279.8, 270.5, 259.9, 250.4, 240.1, 230.4, 220.5, 210.2, 200.4])
x_err = 0.1*np.ones(len(x))

# I is coil current I (measured using meter)

I = np.array([2.370, 2.311, 2.227, 2.028, 1.751, 1.678, 1.563, 1.481, 1.395, 1.275, 1.152])
I_err = 0.001*np.ones(len(x))

# B is magnetic field - calculated from I - equation 4

B_k = 4e-7*np.pi*124/(0.147*(5/4)**(3/2)) # constant in B calc

B = B_k*I

# error propogation from error in a, I -> error in B - see section_2_error_in_B.jpg

# for uncertanties --> https://www.geol.lsu.edu/jlorenzo/geophysics/uncertainties/Uncertaintiespart2.html

B_err = B_k*(I_err**2*0.147**(-2) + I**2 * 0.001**2 * 0.147**(-4))**0.5

# making curve linear

y = B**2

# error in new linear - see section_2_linear_error.jpg

y_err = 2*B_err*B


df = pd.DataFrame({"Anode Voltage (V)":x, "Anode Voltage Error (V)":x_err, "Coil Current (A)":I, "Coil Current Error (A)":I_err, "Magnetic Field Strength (H)":B, "Magnetic Field Strength Error (H)":B_err, "B^2 (H^2)":y, "B^2 Error (H^2)":y_err})

#dfi.export(df,"section_2/e_m_all_data_table.png")

df.to_csv("section_2/all_data/e_m_all_data_table.csv")


# getting line of best fit

def linear(p, x):
    m, c = p
    return m*x+c

linear_model = Model(linear)

data = RealData(x, y, sx=x_err, sy=y_err)

odr = ODR(data, linear_model, beta0=[2.5, 0.0])

out = odr.run()


m, c = out.beta

m_err, c_err = out.sd_beta

# y = m*x + c

print("m", m, m_err)
print("c", c, c_err)

with open("section_2/all_data/e_m_all_data_line_of_best_fit.txt", "w") as f:
    f.write(f"y=m*x+c\nm = {m} +- {m_err}\nc = {c} +- {c_err}")


x_fit = np.linspace(x.min(), x.max(), 1000)

y_fit = linear(out.beta, x_fit)


plt.errorbar(x, y, xerr=x_err, yerr=y_err, linestyle='None', marker='x', label="Measurements")
plt.plot(x_fit, y_fit, label="Linear Fit")

plt.title("Finding e/m")
plt.xlabel("Anode Voltage (V)")
plt.ylabel("Magnetic Field Strength ^2 (H^2)")
plt.legend()

plt.savefig("section_2/all_data/e_m_all_data_graph", dpi=200)
#plt.show()