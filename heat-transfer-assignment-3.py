import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt

print("Step 1: Estimate the heat exchanger duty")
# Cold-Side (Propane)
air = 'Air'
t_in = 20 + 273.15 #K

# Hot-Side (Pentane)
pent = 'Pentane'
T_sat = 40 + 273.15 #K

# Propane mass-flow rate
m_pent = 5 #kg/s
m_air = 100 #kg/s

# Calculating enthalpy of condensation of pentane
h_2_pent = CP.PropsSI('H', 'T', T_sat, 'Q', 1, pent) #J/kg
print('\nSaturated vapor enthalpy of Pentane at {:.2f} K: {:.3f} kJ/kg'.format(T_sat, h_2_pent / 1000))
h_1_pent = CP.PropsSI('H', 'T', T_sat, 'Q', 0, pent) #J/kg
print('Saturated liquid enthalpy of Pentane at {:.2f} K: {:.3f} kJ/kg'.format(T_sat, h_1_pent / 1000))
h_con_pent = h_2_pent - h_1_pent #J/kg
print('Enthalpy of condensation of Pentane at {:.2f} K: {:.3f} kJ/kg'.format(T_sat,h_con_pent/1000))

Q = m_pent*h_con_pent #W
print("\nTotal duty: {:.3f} kW".format(Q/1000))

p_atm = 101325 #Pa
p_sat_pent = CP.PropsSI('P','T',T_sat,'Q',0,pent)
p_cr_pent = CP.PropsSI(pent, 'pcrit')

# Calculating heat capacity of air
c_air_in = CP.PropsSI('CPMASS','P',p_atm,'T',t_in,'air') #J/kgK
print('\nSpecific heat capacity of Air at {:.2f} K: {:.3f} kJ/kgK'.format(t_in,c_air_in/1000))
c_air_out = c_air_in
c_temp = 0
t_out = t_in

while(abs(c_air_out-c_temp)/c_air_out > 0.0001):
    c_temp = CP.PropsSI('CPMASS','P',p_atm,'T',t_out,'air')  # J/kgK

    c_air_mean = (c_air_in+c_temp)/2

    # Air exit temperature
    t_out = t_in + Q/(m_air*c_air_mean) #K
    #print("Exit temperature for Air: {:.4f} kg/s".format(t_out))

    c_air_out = CP.PropsSI('CPMASS','P',p_atm,'T',t_out,'air')  # J/kgK

c_air_mean = (c_air_in+c_temp)/2
print('Specific heat capacity of Air at {:.2f} K: {:.3f} kJ/kgK'.format(t_out,c_air_out/1000))

# Plotting Hot and Cold Temperature vs Inlet-Outlet
plt.clf()
plt.figure(1)
plt.plot([0,1],[T_sat,T_sat],'-r',label=r"Pentane $(C_5H_12)$")
plt.plot([0,1], [t_in, t_out], '-b', label=r"Air")
plt.ylabel("Temperature [K]")
plt.legend()
ax = plt.gca()
ax.set_xlim([0, 1])
plt.savefig("stream-temp.png")
plt.savefig("stream-temp.pdf")

print("Step 2: Collect physical property data")

# Air Viscosity
mu_air_in = CP.PropsSI('VISCOSITY','T',t_in,'P',p_atm,air) #Pas
mu_air_out = CP.PropsSI('VISCOSITY','T',t_out,'P',p_atm,air) #Pas
mu_air_mean = (mu_air_in + mu_air_out) / 2 #Pas
print("\nDynamic Viscosity, Air: \nin: {:.3f} mPas, out: {:.3f} mPas, mean: {:.3f} mPas".format(mu_air_in*1e3, mu_air_out*1e3, mu_air_mean*1e3))
# Pentane Viscosity
mu_pent_in = CP.PropsSI('VISCOSITY','T|gas',T_sat,'P',p_sat_pent,pent) #Pas
mu_pent_out = CP.PropsSI('VISCOSITY','T|liquid',T_sat,'P',p_sat_pent,pent) #Pas
mu_pent_mean = (mu_pent_in+mu_pent_out)/2 #Pas
print("Dynamic Viscosity, Pentane: \nin: {:.3f} mPas, out: {:.3f} mPas, mean: {:.3f} mPas".format(mu_pent_in*1e3, mu_pent_out*1e3, mu_pent_mean*1e3))

# Air Density
rho_air_in = CP.PropsSI('D','T',t_in,'P',p_atm,air) #kg/m^3
rho_air_out = CP.PropsSI('D','T',t_out,'P',p_atm,air) #kg/m^3
rho_air_mean = (rho_air_in+rho_air_out)/2
nu_air_mean = (1/rho_air_in+1/rho_air_out)/2
print("\nDensity, Air: \nin: {:.3f} kg/m^3, out: {:.3f} kg/m^3, mean: {:.3f} kg/m^3".format(rho_air_in, rho_air_out, rho_air_mean))
# Pentane Density
rho_pent_in = CP.PropsSI('D','T|gas',T_sat,'P',p_sat_pent,pent) #kg/m^3
rho_pent_out = CP.PropsSI('D','T|liquid',T_sat,'P',p_sat_pent,pent) #kg/m^3
rho_pent_mean = (rho_pent_in+rho_pent_out)/2
nu_pent_mean = (1/rho_pent_in+1/rho_pent_out)/2
print("Density, Pentane: \nin: {:.3f} kg/m^3, out: {:.3f} kg/m^3, mean: {:.3f} kg/m^3".format(rho_pent_in, rho_pent_out, rho_pent_mean))

# Pentane Specific heat
c_pent_in = CP.PropsSI('CVMASS','T|gas',T_sat,'Q',1,pent) #J/kgK
c_pent_out = CP.PropsSI('CVMASS','T|liquid',T_sat,'Q',0,pent) #J/kgK
c_pent_mean = (c_pent_in+c_pent_out)/2
print("\nSpecific Heat, Pentane: \nin: {:.3f} J/kgK, out: {:.3f} J/kgK, mean: {:.3f} J/kgK".format(c_pent_in, c_pent_out, c_pent_mean))

# Propane Thermal Conductivity
k_air_in = CP.PropsSI('CONDUCTIVITY','T',t_in,'P',p_atm,air)
k_air_out = CP.PropsSI('CONDUCTIVITY','T',t_out,'P',p_atm,air)
k_air_mean = (k_air_in+k_air_out)/2
print("\nThermal Conductivity, Air: \nin: {:.3f} W/mK, out: {:.3f} W/mK, mean: {:.3f} W/mK".format(k_air_in, k_air_out, k_air_mean))
# Pentane Thermal Conductivity
k_pent_in = CP.PropsSI('CONDUCTIVITY','T|gas',T_sat,'Q',1,pent)
k_pent_out = CP.PropsSI('CONDUCTIVITY','T|liquid',T_sat,'Q',0,pent)
k_pent_mean = (k_pent_in+k_pent_out)/2
print("Thermal Conductivity, Pentane: \nin: {:.3f} W/mK, out: {:.3f} W/mK, mean: {:.3f} W/mK".format(k_pent_in, k_pent_out, k_pent_mean))

# Pentane Surface Tension
sigma_pent_out = CP.PropsSI('SURFACE_TENSION','T',T_sat,'Q',0,pent) #N/m
print("\nSurface Tension, Pentane: {:.3f} N/m".format(sigma_pent_out))

#print("\nStep 3: Estimate tube side heat transfer coefficient")
# Overall HT Coefficient estimate: Condensing hydrocarbons: 300–600 | S&T, pg.1051

#print("\nStep 4: Consider internal fouling if applicable")
# Condensing organics F_pent = 5000 | S&T, pg.1053
F_pent = 5000

#print("\nStep 5: Estimate air side heat transfer coefficient")
# alpha_air = 80

#print("\nStep 6: Consider external fouling if applicable")
# F_air = 10000 | S&T, pg.1053
F_air = 10000

print("\nStep 7: Calculate the overall heat transfer coefficient")
# Taking worst-case estimate
U_o = 20 #W/m2K
U_o_calc = 21 #W/m^2K (Just to enter the loop)

print("\nStep 8: If not given, determine the ambient air temperature")
# Taking inlet air temp as ambient air temp
t_amb = t_in

print("\nStep 9: Calculate the temperature difference")
T_lm = (t_in-t_out)/np.log((T_sat-t_out)/(T_sat-t_in))
F_t = 0.9 # As per the slides and S&T, pg.1173 | Check once again for method of Taborek
T_m = F_t*T_lm
print("ΔT_m = {:.3f} K".format(T_m))

i = 1
while abs(U_o - U_o_calc)/U_o >= 0.0001:
    print("\nIteration #{:.0f}".format(i))

    print("\nStep 10: Estimate the required external area")
    A_tot = Q/(U_o*T_m)
    print("Total heat-exchanging area: {:.3f} m^2".format(A_tot))

    print("\nStep 11: Choose a tube diameter and length and determine number of tubes required")
    # We have chosen tube no. 11-40-20 from Schmole Laserfin for it's high surface area enhancement.
    d_1 = 0.020 #m (Outside dia.)
    d_o = d_1 #m (Outside dia., defining for convenience)
    s = 0.0015 #m (Wall thickness)
    d_i = d_1-2*s # (Inner dia.)

    A_a = 0.9 #m^2/m (Enhanced outside surface area)

    l_tube = 12 #m (Tube length)

    n_tube = A_tot/(l_tube*A_a) # (Number of tubes)
    print("Number of tubes: {:.0f}".format(n_tube))

    print("\nStep 12: Decide the tube layout; determine the number of tubes per row and the number of tube rows")
    # A staggered tube arrangement is chosen due to its higher performance compared to in-line arrangement. A staggered tube arrangement does however increase the pressure drop on the air side. This is considered in the design phase to comply with the limit of 0.3 bar pressure losses for the air side.

    # Choosing the pitch of the tubes
    s1 = 0.05
    s2 = 0.025 # Changed as the HT Coefficient improved

    n_tube_rows = 8 # set as per practical constraints
    n_tube_per_row = np.ceil(n_tube/n_tube_rows)

    n_tube = n_tube_rows*n_tube_per_row
    print("Number of tubes per row: {:.0f}".format(n_tube_per_row))

    print("\nStep 13: Determine the bundle area")
    b_bundle = s1*n_tube_per_row
    print("Bundle breadth: {:.3f} m".format(b_bundle))

    A_bundle = b_bundle*l_tube
    print("Bundle area: {:.3f} m^2".format(A_bundle))

    print("\nStep 14: Estimate air side flow rate")
    afr = m_air/rho_air_in
    print("Air-side flow rate: {:.3f} m^3/s".format(afr))

    u_approach = afr/A_bundle
    print("Approach velocity: {:.3f} m/s".format(u_approach))

    print("\nStep 15: Estimate pressure drop and fan power consumption")
    delta_P_air = 150 #Pa (from S&T pg. 1173)
    W_dot_fan = u_approach*s1*n_tube_per_row*l_tube*delta_P_air/(0.5*0.95)
    print("Fan power consumption: {:.3f} kW".format(W_dot_fan/1000))

    print("\nStep 16: Detailed Simulation and Design Iterations")
    u_exit = m_air/rho_air_out/A_bundle
    u_mean = (u_approach + u_exit)/2
    w_o = u_mean # Just refactoring to be in line with VDI variables

    # Air-side HT Coefficient
    d_fin = 0.040 # m, fin outer dia
    s_fin = 0.00231 # m, fin pitch
    h_fin = 0.010 # m, fin height
    t_fin = 0.0004 # m, fin thickness

    A0_by_As = (s1*s_fin)/(2*((s_fin*(np.sqrt((s2**2)+(s1/2)**2)-d_o)) + (np.sqrt((s2**2)+(s1/2)**2)-d_fin)*t_fin)) # Strained cross-section of flow | Lecture 9, slide #37
    print("A0_by_As: {:.3f}".format(A0_by_As))

    w_s = w_o*A0_by_As # flow velocity in the smallest cross-section
    print("w_s: {:.6f} m/s".format(w_s))

    Re_air = rho_air_mean*w_s*d_o/mu_air_mean
    print("Re_air: {:.3f}".format(Re_air))

    Pr_air = c_air_mean*mu_air_mean/k_air_mean
    print("Pr_air: {:.3f}".format(Pr_air))

    A_by_A_t0 = 1 + 2*((h_fin*(h_fin+d_o+t_fin))/(s_fin*d_o))
    print("A_by_A_t0: {:.3f}".format(A_by_A_t0))

    Nu_air = 0.38*(Re_air**0.6)*(A_by_A_t0**-0.15)*(Pr_air**(1/3))
    alpha_m = Nu_air*k_air_mean/d_o

    k_fin = 209 #W/mK | Aluminium, change if necessary

    phi = ((d_fin/d_o)-1)*(1 + 0.35*np.log(d_fin/d_o))
    X = phi*(d_o/2)*np.sqrt((2*alpha_m)/(k_fin*t_fin))
    eta_fin = np.tanh(X)/X

    A_fin_by_A = (2*0.25*np.pi*(d_fin**2 - d_o**2))/(np.pi*d_o*(s-t_fin) + 2*0.25*np.pi*(d_fin**2 - d_o**2))

    alpha_v = alpha_m*(1-(1-eta_fin)*A_fin_by_A) # virtual heat transfer coefficient
    print("Virtual Air-side HT Coefficient: {:.3f}".format(alpha_v))

    A_by_A_i = (A_a*l_tube)/(np.pi*d_i*l_tube)

    # Tube-side HT Coefficient (S&T, pg.1118)
    # Coefficient of condensation

    tau_h = m_pent/(np.pi*d_i*n_tube) # the horizontal tube loading, the condensate flow per unit length of tube, kg/ms
    print("tau_h: {:.3f}".format(tau_h))

    Re_pent = 4*tau_h/mu_pent_in # S&T, pg.1114
    Pr_pent = c_pent_mean*mu_pent_in/k_pent_in # S&T, pg.1114

    h_c_s = 0.76*k_pent_in*((rho_pent_out*(rho_pent_out - rho_pent_in)*9.81)/(mu_pent_in*tau_h))**(1/3)
    h_bk = 0.021*(k_pent_out/d_i)*(Re_pent**0.8)*(Pr_pent**0.43)*((1+(rho_pent_out/rho_pent_in)**0.5)/2)

    alpha_pent = np.maximum(h_c_s,h_bk)
    print("Tube-side HT Coefficient: {:.3f}".format(alpha_pent))

    # Overall HT Coefficient (VDI pg.1273)
    U_calc_new = 1/(1/alpha_v + A_by_A_i*(1/alpha_pent) + (d_o*np.log(d_o/d_i))/(2*k_fin) + 1/F_air + 1/F_pent)
    print("Overall HT Coefficient: {:.3f}".format(U_calc_new))

    U_o = U_o_calc
    U_o_calc = U_calc_new
    i += 1

print("\nStep 17: Check pressure drop in both streams")

# Tube-side Pressure Drop
# Since Re_pent ~ 100,000, use turbulent flow relations | VDI, pg.1057
zeta = 0.00540 + 0.3964/(Re_pent**0.3)
u_tube = m_pent/(rho_pent_mean*(0.25*np.pi*(d_i**2)))

delta_P_tube = (zeta*l_tube*rho_pent_mean*(u_tube**2))/2*d_i
print("Pressure Drop in Tube-side: {:.6f} bar".format(delta_P_tube/100000)) # <0.03 bar, no need to worry

# Air-side Pressure Drop
a = s1/d_o
b = s2/d_o
b_limit = 0.5*np.sqrt(2*a + 1)

# b > b_limit | so we can skip defining c
#c = (s_d/d_o)*np.sqrt((a/2)**2 + b**2)

w_e = (a/(a-1))*w_o # the mean flow velocity in the narrowest cross-section
Re = w_e*d_o*rho_air_mean/mu_air_mean

f_alv = (280*np.pi*((b**0.5 - 0.6)**2 + 0.75))/((4*a*b - np.pi)*a**1.6)
Xi_lam = f_alv/Re

f_atv = 2.5 + (1.2/(a-0.85)**1.08) + 0.4*(b/a - 1)**3 - 0.01*(a/b - 1)**3
Xi_turb = f_atv/(Re**0.25)
F_v = 1 - np.exp(-(Re+200)/1000)

Xi = Xi_lam + Xi_turb*F_v
n_MR = n_tube_rows

delta_P_air = Xi*n_MR*rho_air_mean*(w_e**2)/2
print("Pressure Drop in Air-side: {:.3f} Pa".format(delta_P_air))

eta_fan = 0.5
eta_motor = 0.95

u_approach = afr/A_bundle
print("\nApproach velocity: {:.3f} m/s".format(u_approach))
W_dot_fan = u_approach*s1*n_tube_per_row*l_tube*delta_P_air/(eta_fan*eta_motor) # fan power consumption
print("True Fan Power Consumption: {:.3f} kW".format(W_dot_fan/1000))

# Cost estimation
# Using known values to estimate the parameter a in the cost equation
cost_known = 193962 # $
A_tot_known = 9150.3 # m2

a_cost = cost_known/(A_tot_known**0.9)

cost_CAPEX = 3.5*a_cost*(A_tot**0.9)
print("\nCost of the Heat Exchanger (with installation | CAPEX): ${:.2f} ".format(cost_CAPEX))

cost_elec = 0.20 # maximum cost/kW/hr as per discussion during lecture
t_OP = 1*365*24 # hrs in 1 year/s
cost_OPEX = W_dot_fan*cost_elec*t_OP/1000
print("Annual cost of running the Heat Exchanger (OPEX): ${:.2f}".format(cost_OPEX))

print("Cost over 15 years: ${:.2f}".format(cost_CAPEX+15*cost_OPEX))

print("\nCell Method for Validation")
n_cells = 100
A_cell = A_tot/n_cells
m_cell = m_air/n_cells

x_in,x_out = 0,1
delta_x = ((x_in + x_out)/2)/n_cells
x_values = np.arange(x_in,x_out+delta_x, delta_x)

T_pent = T_sat # Temperature of pentane

delta_T_air_prop = (t_out-t_in)/n_cells # An approximation to get better property values of air
T_air_prop = np.arange(t_in,t_out+delta_T_air_prop,delta_T_air_prop)

T_air = np.zeros(n_cells)
T_air[0] = t_in
T_air[n_cells-1] = t_out

U_cell = np.zeros(n_cells)

i = 1
while i < n_cells:

    # Calculate enthalpy difference for a particular quality change
    h_in = CP.PropsSI('H', 'T', T_pent, 'Q', x_values[i], pent)  # J/kg
    h_out = CP.PropsSI('H', 'T', T_sat, 'Q', x_values[i-1], pent)  # J/kg
    delta_h = h_in - h_out  # J/kg

    # Calculate the specific heat of air
    c_avg = CP.PropsSI('CPMASS', 'P', p_atm, 'T', (T_air_prop[i]+T_air_prop[i-1])/2, air)  # J/kgK

    T_air[i] = T_air[i-1] + (m_pent*delta_h)/(c_avg*m_air)

    T_m = 0.9 * ((T_air[i-1]-T_air[i])/np.log((T_sat-T_air[i])/(T_sat-T_air[i-1])))

    U_cell[i-1] = (m_pent*delta_h)/(A_cell*T_m)

    i += 1

print("Average HT coefficient calculated from Cell Method: {:.2f} kW/m2K".format(np.mean(U_cell[:-1])))