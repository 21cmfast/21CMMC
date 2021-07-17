# coding: utf-8
"""tutorial."""

# In[ ]:

# to get plots displayed in notebook
import matplotlib.pyplot as plt
import numpy as np

# import classy module
from classy import Class
from math import pi

# In[ ]:

# create instance of the class "Class"
LambdaCDM = Class()
# pass input parameters
LambdaCDM.set(
    {
        "omega_b": 0.022032,
        "omega_cdm": 0.12038,
        "h": 0.67556,
        "A_s": 2.215e-9,
        "n_s": 0.9619,
        "tau_reio": 0.0925,
    }
)
LambdaCDM.set({"output": "tCl,pCl,lCl,mPk", "lensing": "yes", "P_k_max_1/Mpc": 3.0})
# run class
LambdaCDM.compute()


# In[ ]:

# get all C_l output
cls = LambdaCDM.lensed_cl(2500)
# To check the format of cls
# Comment out cls.viewkeys()


# In[ ]:

ll = cls["ell"][2:]
cltt = cls["tt"][2:]
clee = cls["ee"][2:]
clpp = cls["pp"][2:]


# In[ ]:

# In[ ]:

# plot C_l^TT
plt.figure(1)
plt.xscale("log")
plt.yscale("linear")
plt.xlim(2, 2500)
plt.xlabel(r"$\ell$")
plt.ylabel(r"$[\ell(\ell+1)/2\pi]  C_\ell^\mathrm{TT}$")
plt.plot(ll, cltt * ll * (ll + 1) / 2.0 / pi, "r-")


# In[ ]:

plt.savefig("warmup_cltt.pdf")


# In[ ]:

# get P(k) at redhsift z=0

kk = np.logspace(-4, np.log10(3), 1000)
Pk = []
for k in kk:
    Pk.append(LambdaCDM.pk(k, 0.0))  # function .pk(k,z)


# In[ ]:

# plot P(k)
plt.figure(2)
plt.xscale("log")
plt.yscale("log")
plt.xlim(kk[0], kk[-1])
plt.xlabel(r"$k \,\,\,\, [h/\mathrm{Mpc}]$")
plt.ylabel(r"$P(k) \,\,\,\, [\mathrm{Mpc}/h]^3$")
plt.plot(kk, Pk, "b-")


# In[ ]:

plt.savefig("warmup_pk.pdf")


# In[ ]:

# optional: clear content of LambdaCDM (if you want to reuse it for another parameter set)
LambdaCDM.struct_cleanup()


# In[ ]:
