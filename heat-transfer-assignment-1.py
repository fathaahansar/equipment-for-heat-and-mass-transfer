import numpy as np
import CoolProp.CoolProp as CP
import fluids
import matplotlib.pyplot as plt

print("Step 1: Define duty")
# Tube side (Water)
h2o = 'Water'
T_1 = 273.15+23.7 #K

# Shell side (Ammonia)
nh3 = 'Ammonia'
t_sat = 273.15 + 19.2 #K

# Calculating enthalpy of vaporization of ammonia
h_2_nh3 = CP.PropsSI('H','T', t_sat, 'Q', 1, nh3) #J/kg
print('Saturated vapor enthalpy of Ammonia at {:.2f} K: {:.3f} kJ/kg'.format(t_sat, h_2_nh3 / 1000))
h_1_nh3 = CP.PropsSI('H','T', t_sat, 'Q', 0, nh3) #J/kg
print('Saturated liquid enthalpy of Ammonia at {:.2f} K: {:.3f} kJ/kg'.format(t_sat, h_1_nh3 / 1000))
h_vap = h_2_nh3-h_1_nh3 #J/kg
print('Enthalpy of vaporization at {:.2f} K: {:.3f} kJ/kg'.format(t_sat, h_vap / 1000))

# Defining specific heat capacity of water at inlet conditions
c_h2o = CP.PropsSI('CVMASS','T',T_1,'Q',0,h2o) #J/kgK

#Total mass flow rates
m_nh3 = 4523 #kg/s
m_h2o = 396531 #kg/s

# Outlet temperature and duty
T_2 = T_1 - m_nh3*h_vap/m_h2o/c_h2o #K
Q = m_nh3*h_vap #W
print("Water exit temperature: {:.3f} K\nTotal duty: {:.3f} MW".format(T_2,Q/10**6))

plt.clf()
plt.figure(1)
plt.plot([0,1],[T_1,T_2],'-r',label=r"Seawater $(H_2O)$")
plt.plot([0,1],[t_sat,t_sat],'-b',label=r"Ammonia $(NH_3)$")
plt.ylabel("Temperature [K]")
plt.legend()
ax = plt.gca()
ax.set_xlim([0, 1])
plt.savefig("stream.png")
plt.savefig("stream.pdf")

print("\nStep 2: Collection of physical properties")
# Specific heat
c_h2o_in = CP.PropsSI('CVMASS','T|liquid',T_1,'Q',0,h2o) #J/kgK
c_h2o_out = CP.PropsSI('CVMASS','T|liquid',T_2,'Q',0,h2o) #J/kgK
c_h2o_mean = (c_h2o_in+c_h2o_out)/2
print("Specific Heat, water: \nin: {:.3f} J/kgK, out: {:.3f} J/kgK, mean: {:.3f} J/kgK".format(c_h2o_in,c_h2o_out,c_h2o_mean))
c_nh3_in = CP.PropsSI('CVMASS','T|liquid',t_sat,'Q',0,nh3) #J/kgK
c_nh3_out = CP.PropsSI('CVMASS','T|gas',t_sat,'Q',1,nh3) #J/kgK
c_nh3_mean = (c_nh3_in+c_nh3_out)/2
print("Specific Heat, ammonia: \nin: {:.3f} J/kgK, out: {:.3f} J/kgK, mean: {:.3f}J/kgK".format(c_nh3_in,c_nh3_out,c_nh3_mean))

# Thermal Conductivity
k_h2o_in = CP.PropsSI('CONDUCTIVITY','T|liquid',T_1,'Q',0,h2o)
k_h2o_out = CP.PropsSI('CONDUCTIVITY','T|liquid',T_2,'Q',0,h2o)
k_h2o_mean = (k_h2o_in+k_h2o_out)/2
print("Thermal Conductivity, water: \nin: {:.3f} W/mK, out: {:.3f} W/mK, mean: {:.3f} W/mK".format(k_h2o_in,k_h2o_out,k_h2o_mean))
k_nh3_in = CP.PropsSI('CONDUCTIVITY','T|liquid',t_sat,'Q',0,nh3)
k_nh3_out = CP.PropsSI('CONDUCTIVITY','T|gas',t_sat,'Q',1,nh3)
k_nh3_mean = (k_nh3_in+k_nh3_out)/2
print("Thermal Conductivity, ammonia: \nin: {:.3f} W/mK, out: {:.3f} W/mK, mean: {:.3f} W/mK".format(k_nh3_in,k_nh3_out,k_nh3_mean))

# Density
rho_h2o_in = CP.PropsSI('D', 'T|liquid', T_1, 'P', 1e5, h2o) #kg/m^3
rho_h2o_out = CP.PropsSI('D', 'T|liquid', T_2, 'P', 1e5, h2o) #kg/m^3
rho_h2o_mean = (rho_h2o_in+rho_h2o_out)/2
print("Density, water: \nin: {:.3f} kg/m^3, out: {:.3f} kg/m^3, mean: {:.3f} kg/m^3".format(rho_h2o_in, rho_h2o_out, rho_h2o_mean))
p_sat = CP.PropsSI('P','T',t_sat,'Q',0,nh3)
rho_nh3_in = CP.PropsSI('D', 'T|liquid', t_sat, 'P', p_sat, nh3) #kg/m^3
rho_nh3_out = CP.PropsSI('D', 'T|gas', t_sat, 'P', p_sat, nh3) #kg/m^3
rho_nh3_mean = (rho_nh3_in+rho_nh3_out)/2
print("Density, ammonia: \nin: {:.3f} kg/m^3, out: {:.3f} kg/m^3, mean: {:.3f} kg/m^3".format(rho_nh3_in, rho_nh3_out, rho_nh3_mean))

# Dynamic Viscosity
mu_h2o_in = CP.PropsSI('VISCOSITY','T|liquid',T_1,'P',1e5,h2o) #Pas
mu_h2o_out = CP.PropsSI('VISCOSITY','T|liquid',T_2,'P',1e5,h2o) #Pas
mu_h2o_mean = (mu_h2o_in+mu_h2o_out)/2
print("Dynamic Viscosity, water: \nin: {:.3f} mPas, out: {:.3f} mPas, mean: {:.3f} mPas".format(mu_h2o_in*1e3, mu_h2o_out*1e3, mu_h2o_mean*1e3))
mu_nh3_in = CP.PropsSI('VISCOSITY','T|liquid',t_sat,'P',p_sat,nh3) #Pas
mu_nh3_out = CP.PropsSI('VISCOSITY','T|gas',T_2,'P',p_sat,nh3) #Pas
mu_nh3_mean = (mu_nh3_in+mu_nh3_out)/2
print("Dynamic Viscosity, ammonia: \nin: {:.3f} mPas, out: {:.3f} mPas, mean: {:.3f} mPas".format(mu_nh3_in*1e3, mu_nh3_out*1e3, mu_nh3_mean*1e3))

# Surface Tension
sigma_nh3 = CP.PropsSI('SURFACE_TENSION','T',t_sat,'Q',0,nh3) #N/m
print("Surface Tension, ammonia: \nin: {:.3f} N/m".format(sigma_nh3))

print("\nStep 3: Assumption of overall heat transfer coefficient")
U_o = 1000 #W/m^2K
U_o_calc = 100 #W/m^2K (From running one iteration manually, we get this value)

# Where we should start our loop, to iterate our Overall HT Coefficient value

while abs(U_o - U_o_calc)/U_o >= 0.0001:

    print("\nStep 4: Decide number of tube passes and calculate the true temperature difference")
    T_lm = (T_1-T_2)/np.log((t_sat-T_1)/(t_sat-T_2))
    F_t = 1 #np.sqrt(R**2+1)*np.log((1-S)/(1-R*S))/(R-1)*np.log((2-S*(R+1-np.sqrt(R**2+1)))/(2-S*(R+1+np.sqrt(R**2+1))))
    T_m = F_t*T_lm
    print("ΔT_m = {:.3f} K".format(T_lm, F_t, T_m))

    print("\nStep 5: Determine the required heat transfer area")
    A_tot = Q / (U_o*T_m)
    print("Total heat-exchanging area: {:.3f} m^2".format(A_tot))

    q = Q/A_tot
    print("Heat Flux: {:.3f} W/m^2".format(q))

    print("\nStep 6: Decide type, tube size, material, layout. Assign fluid to shell and tube")

    # Type: U-tube, type BEU (S&T, pg.1055)

    # Tube Size: Outer dia, d_o = 25(mm) | Wall thickness, s = 2.5(mm) | For heavy fouling fluid, a dia of >= 25mm is preferred. We start with this. The wall thickness is chosen nominally even though the pressures in the system are high (~8 bar, shell side), because aluminium has sufficient tensile strength. The longest available length of a tube commercially is l_tube = 6(m). We will check the tube length/shell dia value to change this later (has to be between 5-10).

    N_HeX = 2700

    d_o = 0.025 #m
    l_tube = 6 #m
    s = 0.0025 #m
    A_tube = np.pi * d_o * l_tube
    d_i = d_o-s

    N_tubes = np.ceil(A_tot/(N_HeX*A_tube)/2)
    print('Number of tubes: {:.0f}'.format(N_tubes))

    # Material: Aluminium is preferred due to its high thermal conductivity and favorable corrosion characteristics in seawater environments. Initial dimensions are chosen from the Alumenco group catalog after consulting Table 19.3 (S&T, pg.1060). These will be adjusted based on U_overall and del_P.
    k_wall = 209 #W/mK

    print("\nStep 7: Tube Arrangement")
    # As we have a non-fouling fluid on the shell side, we can go for a triangular arrangement, to maximize the heat transfer rates. Since we also have a high pressure in the shell side, we can start with this and calculate pressure drop and ensure that it is below 0.3 bar. Tube pitch is taken as 1.5*d_o, which is the minimum because heat flux is too low for intolerable levels of vapor blanketing to occur.

    pitch_tube = 1.5 * d_o # Pitch of the tube, given in S&T, pg.1061
    n_1 = 2.207
    K_1 = 0.249

    print("\nStep 8: Calculate Shell Diameter")
    # Begin with calculating bundle diameter: D_b
    D_b = d_o * (N_tubes / K_1) ** (1 / n_1)
    print('Bundle Dia: {:.3f} m'.format(D_b))

    d_shell_inner = np.ceil(D_b*1.2*1000)/1000  # m
    print('Shell Inner Dia: {:.3f} mm'.format(d_shell_inner*1000))

    shell_thickness = 0.02 # Units: m
    d_shell_outer = d_shell_inner + 2*shell_thickness
    print('Shell Outer Dia: {:.3f} mm'.format(d_shell_outer*1000))

    print("\nStep 9: Estimating tube-side heat transfer coefficient")

    # Tube velocity
    u_tube = m_h2o / (N_HeX * N_tubes * rho_h2o_mean * (0.25 * np.pi * d_i ** 2))
    print("Velocity in tube-side: {:.3f} m/s".format(u_tube))

    # Reynold's number
    Re = (rho_h2o_mean*u_tube*d_i)/mu_h2o_mean
    print("Reynold's Number: {:.3f}".format(Re))

    # Prandtl number
    Pr = c_h2o_mean*mu_h2o_mean/k_h2o_mean
    print("Prandtl Number: {:.3f}".format(Pr))

    # Using Eagle and Ferguson equation (19.17) from S&T, pg.1080
    h_tube = (4200*(1.35+0.02*((T_1+T_2)/2))*(u_tube**0.8))/(d_i ** 0.2)
    print("Tube-Side Heat Transfer Coefficient: {:.3f} W/m^2K".format(h_tube))

    # Wall temperature approximation
    T_wall = T_1 - (U_o*(T_1-T_2)/h_tube)
    print("Tube-Side Wall Temperature: {:.3f} K".format(T_wall))
    mu_h2o_wall = CP.PropsSI('VISCOSITY','T|liquid',T_wall,'P',1e5,h2o) #Pas

    # Considering the fact that equation 19.17 is more specific for water in tubes, we will use the value obtained from this equation as it would account for the characteristics of water. And taking a lower heat transfer coefficient would ensure that we do not under-design with respect to heat transfer.

    print("\nStep 10: Calculate shell-side heat transfer coefficient")

    # On the shell-side, we will be using a kettle vaporizer, as we are not dealing with a high-viscosity or heavy-fouling fluid. We will be using the correlation by Forster and Zuber, 1955 (S&T eqn 19.41, pg 1134) to estimate the heat transfer coefficient, and the modified-Zuber correlation (S&T eqn 19.52, pg 1151) to estimate the critical heat flux.

    # Shell-side heat transfer coefficient
    p_wall = CP.PropsSI('P','T',T_wall,'Q',0,nh3)
    h_shell = 0.00122 * (k_nh3_in**0.79*c_nh3_in**0.45*rho_nh3_in**0.49/sigma_nh3**0.5/mu_nh3_in**0.29/h_vap**0.24/rho_nh3_out**0.24) * (T_wall-t_sat)**0.24 * (p_wall-p_sat)**0.75 #W/m^2K
    print("Shell-side heat transfer coefficient: {:.3f} W/m^2K".format(h_shell))

    # Critical heat flux
    q_cb = 0.41 * (pitch_tube / d_o) * (h_vap / np.sqrt(2 * N_tubes)) * (sigma_nh3 * 9.81 * (rho_nh3_in - rho_nh3_out) * rho_nh3_out ** 2) ** 0.25 #W/m^2
    print("Critical heat flux: {:.3f} W/m^2".format(q_cb))

    print("\nStep 11: Estimating overall heat-transfer coefficient")
    # Fouling factor for sea-water is taken as 2000 W/m^2K (S&T, table 19.2, pg 1053). No fouling is considered for ammonia as it is a clean fluid loop.
    U_o_calc_new = (1/h_shell + 1/2000 + d_o*np.log(d_o/d_i)/2/k_wall + d_o/h_tube/d_i)**(-1)
    print("U_o_calc: {:.3f} W/m^2K".format(U_o_calc_new))

    U_o = U_o_calc
    U_o_calc = U_o_calc_new

    #print(U_o)
    #print(U_o_calc)

print("\nStep 12: Estimating tube-side pressure drop")

#j_f = fluids.friction.friction_factor(Re=Re)
j_f = 4.25 * 10**-3 # From Figure 19.24
print("Friction Factor for Re={:.3f} (From Figure 19.24): {:.4f}".format(Re,j_f))

N_p = 2 # Number of passes

del_P_tube = N_p * ((8 * j_f * (l_tube/d_i) * (mu_h2o_mean/mu_h2o_wall)**-0.14) + 2.5) * rho_h2o_mean * (0.5 * u_tube**2)
print("Pressure Drop in tube: {:.3f} bar".format(del_P_tube/100000))

print("\nStep 13: Estimating shell-side pressure drop")
d_inlet = 0.2 #m
u_shell_in = m_nh3/rho_nh3_in/(0.25*np.pi*d_inlet**2)/N_HeX
print("Shell inlet velocity: {:.3f} m/s".format(u_shell_in))

u_max = 0.2*np.sqrt((rho_nh3_in-rho_nh3_out)/rho_nh3_out)
print("Maximum velocity: {:.3f} m/s".format(u_max))

u_surface = u_shell_in*rho_nh3_in*(0.25*np.pi*d_inlet**2)/rho_nh3_mean/l_tube/d_shell_inner
print("Surface velocity: {:.3f} m/s".format(u_surface))

u_shell_out = u_shell_in*rho_nh3_in/rho_nh3_out
print("Shell outlet velocity: {:.3f} m/s".format(u_shell_out))

del_P_shell = -rho_nh3_in*u_shell_in**2/2 + rho_nh3_out*u_shell_out**2/2 + rho_nh3_mean*9.81*d_shell_inner
print("Pressure Drop in shell: {:.3f} bar".format(del_P_shell/100000))

print("\nStep 14: Estimating the cost of heat exchanger")
# Formula for cost calculation from S&T Table 7.2, U-tube Kettle reboiler area, m2 s_lower = 10 s_upper = 500 a = 29,000 b = 400 n = 0.9

a = 29000
b = 400
S = A_tot/N_HeX
n = 0.9

cost = a + b*S**n

print("Single Heat Exchanger Cost for S={:.3f}: €{:.2f}".format(S,cost))
print("Cost of all Heat Exchangers: €{:.2f}".format(cost*N_HeX))