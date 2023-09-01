import numpy as np
import CoolProp.CoolProp as CP
import fluids
import matplotlib.pyplot as plt

print("Step 1: Calculate duty and define material properties")
# Cold-Side (Propane)
prop = 'Propane'
t_sat = 273.15-40 #K

# Hot-Side (Ethane)
eth = 'Ethane'
T_sat = 273.15-35 #K

# Calculating enthalpy of vaporization of propane
h_2_prop = CP.PropsSI('H','T', t_sat, 'Q', 1, prop) #J/kg
print('Saturated vapor enthalpy of Propane at {:.2f} K: {:.3f} kJ/kg'.format(t_sat,h_2_prop/1000))
h_1_prop = CP.PropsSI('H','T', t_sat, 'Q', 0, prop) #J/kg
print('Saturated liquid enthalpy of Propane at {:.2f} K: {:.3f} kJ/kg'.format(t_sat,h_1_prop/1000))
h_vap_prop = h_2_prop-h_1_prop #J/kg
print('Enthalpy of vaporization of Propane at {:.2f} K: {:.3f} kJ/kg'.format(t_sat,h_vap_prop/1000))

# Calculating enthalpy of vaporization of propane
h_2_eth = CP.PropsSI('H','T', T_sat, 'Q', 1, eth) #J/kg
print('\nSaturated vapor enthalpy of Ethane at {:.2f} K: {:.3f} kJ/kg'.format(T_sat,h_2_eth/1000))
h_1_eth = CP.PropsSI('H','T', T_sat, 'Q', 0, eth) #J/kg
print('Saturated liquid enthalpy of Ethane at {:.2f} K: {:.3f} kJ/kg'.format(T_sat,h_1_eth/1000))
h_con_eth = h_2_eth-h_1_eth #J/kg
print('Enthalpy of condensation of Ethane at {:.2f} K: {:.3f} kJ/kg'.format(T_sat,h_con_eth/1000))

# Material: Aluminium is preferred due to its high thermal conductivity and favorable corrosion characteristics in seawater environments. Initial dimensions are chosen from the Alumenco group catalog after consulting Table 19.3 (S&T, pg.1060). These will be adjusted based on U_overall and del_P.
k_plate = 209 #W/mK

# Defining physical properties of the 2 fluids

# Propane Viscosity
p_sat_prop = CP.PropsSI('P','T',t_sat,'Q',0,prop)
mu_prop_in = CP.PropsSI('VISCOSITY','T|liquid',t_sat,'P',p_sat_prop,prop) #Pas
mu_prop_out = CP.PropsSI('VISCOSITY','T|gas',t_sat,'P',p_sat_prop,prop) #Pas
mu_prop_mean = (mu_prop_in+mu_prop_out)/2 #Pas
print("\nDynamic Viscosity, propane: \nin: {:.3f} mPas, out: {:.3f} mPas, mean: {:.3f} mPas".format(mu_prop_in*1e3, mu_prop_out*1e3, mu_prop_mean*1e3))
# Ethane Viscosity
p_sat_eth = CP.PropsSI('P','T',t_sat,'Q',0,prop)
p_cr_eth = CP.PropsSI(eth,'pcrit')
mu_eth_in = CP.PropsSI('VISCOSITY','T|gas',T_sat,'P',p_sat_eth,eth) #Pas
mu_eth_out = CP.PropsSI('VISCOSITY','T|liquid',T_sat,'P',p_sat_eth,eth) #Pas
mu_eth_mean = (mu_eth_in+mu_eth_out)/2 #Pas
print("Dynamic Viscosity, ethane: \nin: {:.3f} mPas, out: {:.3f} mPas, mean: {:.3f} mPas".format(mu_eth_in*1e3, mu_eth_out*1e3, mu_eth_mean*1e3))

# Propane Density
rho_prop_in = CP.PropsSI('D', 'T|liquid', t_sat, 'P', p_sat_prop, prop) #kg/m^3
rho_prop_out = CP.PropsSI('D', 'T|gas', t_sat, 'P', p_sat_prop, prop) #kg/m^3
rho_prop_mean = (rho_prop_in+rho_prop_out)/2
nu_prop_mean = (1/rho_prop_in+1/rho_prop_out)/2
print("\nDensity, propane: \nin: {:.3f} kg/m^3, out: {:.3f} kg/m^3, mean: {:.3f} kg/m^3".format(rho_prop_in, rho_prop_out, rho_prop_mean))
# Ethane Density
rho_eth_in = CP.PropsSI('D', 'T|gas', T_sat, 'P', p_sat_eth, eth) #kg/m^3
rho_eth_out = CP.PropsSI('D', 'T|liquid', T_sat, 'P', p_sat_eth, eth) #kg/m^3
rho_eth_mean = (rho_eth_in+rho_eth_out)/2
nu_eth_mean = (1/rho_eth_in+1/rho_eth_out)/2
print("Density, ethane: \nin: {:.3f} kg/m^3, out: {:.3f} kg/m^3, mean: {:.3f} kg/m^3".format(rho_eth_in, rho_eth_out, rho_eth_mean))

# Propane Specific heat
c_prop_in = CP.PropsSI('CVMASS','T|liquid',t_sat,'Q',0,prop) #J/kgK
c_prop_out = CP.PropsSI('CVMASS','T|gas',t_sat,'Q',1,prop) #J/kgK
c_prop_mean = (c_prop_in+c_prop_out)/2
print("\nSpecific Heat, propane: \nin: {:.3f} J/kgK, out: {:.3f} J/kgK, mean: {:.3f} J/kgK".format(c_prop_in,c_prop_out,c_prop_mean))
# Ethane Specific heat
c_eth_in = CP.PropsSI('CVMASS','T|gas',T_sat,'Q',1,eth) #J/kgK
c_eth_out = CP.PropsSI('CVMASS','T|liquid',T_sat,'Q',0,eth) #J/kgK
c_eth_mean = (c_eth_in+c_eth_out)/2
print("Specific Heat, ethane: \nin: {:.3f} J/kgK, out: {:.3f} J/kgK, mean: {:.3f} J/kgK".format(c_eth_in,c_eth_out,c_eth_mean))

# Propane Thermal Conductivity
k_prop_in = CP.PropsSI('CONDUCTIVITY','T|liquid',t_sat,'Q',0,prop)
k_prop_out = CP.PropsSI('CONDUCTIVITY','T|gas',t_sat,'Q',1,prop)
k_prop_mean = (k_prop_in+k_prop_out)/2
print("\nThermal Conductivity, propane: \nin: {:.3f} W/mK, out: {:.3f} W/mK, mean: {:.3f} W/mK".format(k_prop_in,k_prop_out,k_prop_mean))
# Ethane Thermal Conductivity
k_eth_in = CP.PropsSI('CONDUCTIVITY','T|gas',T_sat,'Q',1,eth)
k_eth_out = CP.PropsSI('CONDUCTIVITY','T|liquid',T_sat,'Q',0,eth)
k_eth_mean = (k_eth_in+k_eth_out)/2
print("Thermal Conductivity, ethane: \nin: {:.3f} W/mK, out: {:.3f} W/mK, mean: {:.3f} W/mK".format(k_eth_in,k_eth_out,k_eth_mean))

# Ethane Surface Tension
sigma_eth_out = CP.PropsSI('SURFACE_TENSION','T',T_sat,'Q',0,eth) #N/m
print("\nSurface Tension, ethane: {:.3f} N/m".format(sigma_eth_out))

# Propane mass-flow rate
m_prop = 0.05 #kg/s

Q = m_prop*h_vap_prop #W
print("\nTotal duty: {:.3f} kW".format(Q/1000))

print("\nStep 2: Determine the unknown fluid flow rate from a heat balance")
# Ethane mass-flow rate
m_eth = m_prop*h_vap_prop/h_con_eth #K
print("Mass-flow rate for Ethane: {:.4f} kg/s".format(m_eth))

# Plotting Hot and Cold Temperature vs Inlet-Outlet
plt.clf()
plt.figure(1)
plt.plot([0,1],[T_sat,T_sat],'-r',label=r"Ethane $(C_2H_6)$")
plt.plot([0,1],[t_sat,t_sat],'-b',label=r"Propane $(C_3H_8)$")
plt.ylabel("Temperature [K]")
plt.legend()
ax = plt.gca()
ax.set_xlim([0, 1])
plt.savefig("stream-temp.png")
plt.savefig("stream-temp.pdf")

# Plotting Hot and Cold Temperature vs Inlet-Outlet
plt.clf()
plt.figure(1)
plt.plot([0,1],[1,0],'-r',label=r"Ethane $(C_2H_6)$")
plt.plot([0,1],[0,1],'-b',label=r"Propane $(C_3H_8)$")
plt.ylabel("Quality")
plt.legend()
ax = plt.gca()
ax.set_xlim([0, 1])
plt.savefig("stream-quali.png")
plt.savefig("stream-quali.pdf")

print("\nStep 3-5: Calculate the temperature difference")
T_m = T_sat-t_sat
print("ΔT_m = {:.3f} K".format(T_m))

print("\nStep 6: Estimate the overall heat transfer coefficient")
U_o = 2500 #W/m^2K # From Table 19.1, value for light organic fluid on bo
U_o_calc = 200 #W/m^2K

# Where we should start our loop, to iterate our Overall HT Coefficient value

i=0
while abs(U_o - U_o_calc)/U_o >= 0.0001:
    i+=1
    print("\nIteration-{:.0f}:".format(i))
    print("\nStep 7: Calculate the surface area required")
    A_tot = Q/(U_o*T_m)
    print("Total heat-exchanging area: {:.3f} m^2".format(A_tot))

    q = Q/A_tot
    print("Heat-flux: {:.3f} W/m^2".format(q))

    print("\nStep 8: Determine the number of plates required")
    L_p = 0.6 #m
    B_p = 0.3 #m

    t_p = 0.0005 #m, Plate thickness = 0.4 to 3.0 mm (Cite lecture slides, pg.16 | Actually from VDI)
    d_spacing = 0.003 #m, Plate Gap = 1.5 to 5.0 mm (Cite lecture slides, pg.16 | Actually from VDI)

    Y_p = 0.003 #m, Amplitude, assuming a small value for now
    lambda_p = Y_p*2*np.pi #m, Wavelength, assuming for X to be 1, can be altered later if required
    phi = np.pi/3 #chevron angle
    X_p = 2*np.pi*Y_p/lambda_p

    A_ref = L_p*B_p #m^2
    Phi = (1 + np.sqrt(1+X_p**2) + 4*np.sqrt(1+X_p**2/2))/6 # ratio wavy plate surface to its plane projection, defined as surface enhancements | 1.22 is typical for technical surfaces, which we get when 2*pi*Y_p = lambda_p (Cite lecture slides: pg.19 | Actually from VDI,pg.1516)

    D_h = 4*Y_p/Phi # Hydraulic Diameter

    A_p = Phi*A_ref
    N_p = np.ceil(A_tot/A_p)
    if N_p%2==0:
        N_p+=1

    A_tot = N_p*A_p
    print("Area of single plate: {:.3f} m\nNumber of plates: {:.0f}".format(A_p,N_p))

    N_HeX = 1 # Defining for completeness and to change it, if necessary in the future

    print("\nStep 9: Decide the flow arrangement and number of passes")
    # We keep the exchanger single pass, as this aids simplicity. A parallel flow is most appropriate for an evaporator-condenser. In a series flow, the evaporation/condensation happens only in the plates in the center of the heat-exchanger, which renders those in the ends of the exchanger ineffective. In a parallel flow, each plate is supplied with fresh fluid to be evaporated/condensed, thus maximizing the effectiveness of each and ensuring uniform heat transfer.

    print("\nStep 10: Calculate the film heat-transfer coefficients for each stream")

    # Yan & Lin Method
    G_prop = m_prop*2/(N_p-1)/B_p/d_spacing
    G_eth = m_eth*2/(N_p-1)/B_p/d_spacing
    G_eq_prop = G_prop*0.5*(1+np.sqrt(rho_prop_in/rho_prop_out))
    G_eq_eth = G_eth*0.5*(1+np.sqrt(rho_eth_out/rho_eth_in))
    print("Gap mass-fluxes:\nMass-flux, propane: {:.3f} kg/m^2s\nMass-flux, ethane: {:.3f} kg/m^2s\nEquivalent mass-flux, propane: {:.3f} kg/m^2s\nEquivalent mass-flux, ethane: {:.3f} kg/m^2s".format(G_prop,G_eth,G_eq_prop,G_eq_eth))

    # Prandtl number(s)
    Pr_prop = c_prop_mean*mu_prop_mean/k_prop_mean
    Pr_eth = c_eth_mean*mu_eth_mean/k_eth_mean
    print("\nLiquid prandtl number, propane: {:.3f}\nLiquid prandtl number, ethane: {:.3f}".format(Pr_prop,Pr_eth))

    # Reynolds Number(s)
    Re_prop =  G_prop*D_h/mu_prop_mean
    Re_eth = G_eth*D_h/mu_eth_mean
    Re_eq_prop = G_eq_prop*D_h/mu_prop_mean
    Re_eq_eth = G_eq_eth*D_h/mu_eth_mean
    print("\nReynolds Numbers:\nReynolds Number, propane: {:.3f}\nReynolds Number, ethane: {:.3f}\nEquivalent Reynolds Number, propane: {:.3f}\nEquivalent Reynolds Number, ethane: {:.3f}".format(Re_prop,Re_eth,Re_eq_prop,Re_eq_eth))
    # optimize for Re_eq < 10,000 or relations won't hold!

    # Boiling Number(s)
    Bo_eq_prop = q/h_vap_prop/G_eq_prop
    Bo_eq_eth = q/h_con_eth/G_eq_eth
    print("\nEquivalent boiling number, propane: {:.6f}\nEquivalent boiling number, ethane: {:.6f}".format(Bo_eq_prop,Bo_eq_eth))

    # Nusselt Number
    Nu_prop = 19.26*(Pr_prop**(1/3))*(Bo_eq_prop**0.3)*(Re_prop**0.5)*(0.5+0.5*np.sqrt(rho_prop_in/rho_prop_out))
    Nu_eth = 0.943*phi*(9.81*rho_prop_out*(rho_eth_out-rho_eth_in)*L_p**3*h_con_eth/mu_eth_out/k_eth_out/T_m)**0.25
    print("\nNusselt number, propane: {:.6f}".format(Nu_prop))
    print("Nusselt number, ethane: {:.6f}".format(Nu_eth))

    alpha_prop = Nu_prop*k_prop_mean/D_h
    alpha_eth = Nu_eth*k_eth_mean/D_h
    print("\nEvaporator heat transfer coefficient: {:.3f} W/m^2K".format(alpha_prop))
    print("Condenser heat transfer coefficient: {:.3f} W/m^2K".format(alpha_eth))

    # Overall Heat Transfer Coefficient
    U_o_calc_new = 1/(1/alpha_prop + 1/alpha_eth + t_p/k_plate*A_p)
    print("\nOverall Heat Transfer Coefficient: {:.6f} W/m^2K".format(U_o_calc_new))

    U_o = U_o_calc
    U_o_calc = U_o_calc_new

print("\nStep 13: Check the pressure drop for each stream")
f_prop = 31.21*Re_eq_prop**0.04557/Re_prop**0.5
Bd = (rho_eth_out-rho_eth_in)*9.81*D_h**2/sigma_eth_out
f_eth = (4.207-2.673*phi**(-0.46)) * (4200-5.41*Bd**1.2) * Re_eq_eth**(-0.95) * (p_sat_eth/p_cr_eth)**0.3

delta_P_f_prop = 2*nu_prop_mean*f_prop*G_prop**2*L_p/D_h
delta_P_a_prop = (1/rho_prop_out-1/rho_prop_in)*G_prop**2
delta_P_ele_prop = 9.81*L_p/nu_prop_mean
delta_P_man_prop = 0.75*G_prop**2*nu_prop_mean
delta_P_prop = delta_P_f_prop+delta_P_a_prop+delta_P_ele_prop+delta_P_man_prop

delta_P_eth = 2*nu_eth_mean*f_eth*G_eth**2*L_p/D_h
delta_P_tot = (delta_P_prop+delta_P_eth)*(N_p-1)/2
print("Single-channel pressure drops:\nPropane:\nFriction: {:.6f} Bar, Acceleration: {:.6f} Bar, Elevation: {:.6f} Bar, Manifold: {:.6f} Bar, Total: {:.3f} Bar\nEthane: {:.3f} Bar\n\nTotal pressure drop: {:.3f} Bar".format(delta_P_f_prop/1e5,delta_P_a_prop/1e5,delta_P_ele_prop/1e5,delta_P_man_prop/1e5,delta_P_prop/1e5,delta_P_eth/1e5,delta_P_tot/1e5))

print("\nStep 14: Calculate total cost")
cost = 643*A_tot**0.9
print("Total cost: €{:.3f}".format(cost))

# Cell Method

# Expressions for calculating heat transfer rate on each side and overall
Q_prop = m_prop * h_vap_prop
Q_eth = m_eth * h_con_eth
Q = U_o_calc_new * A_tot * T_m

