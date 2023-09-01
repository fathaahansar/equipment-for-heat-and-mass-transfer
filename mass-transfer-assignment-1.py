import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import CoolProp.CoolProp as CP

x_benzene = np.linspace(0,1,201)
y_benzene = x_benzene # Declaring this variable just to create a grid, will give clarity while stepping
y_equi = 0.7731*x_benzene**3 - 1.9825*x_benzene**2 + 2.2087*x_benzene + 0.0078 # Equilibrium curve from given equilibrium data

x_D = 0.97 # Distillate mole fraction (Benzene)
x_B = 1 - 0.98 # Bottom product mole fraction (Benzene)
z_F = 0.4
R = 3.5 # Reflux ratio kmol reflux/kmol product

# 1(a) The mass flow of top and bottom products
F = 2.5 # kg/s

# Hand calculated
D = 1 # kg/s
B = 1.5 # kg/s

print("Question A")
print("Distillate Mass Flow Rate: {:.3f} kg/s".format(D))
print("Bottoms Mass Flow Rate: {:.3f} kg/s".format(B))

# Computing intersection points with the equilibrium curve
eq_curve = LineString(np.column_stack((x_benzene,y_equi))) # equilibrium curve
ff_deg_line = LineString(np.column_stack((x_benzene,x_benzene))) # 45-deg line
dist_curve = LineString(np.column_stack(([x_D,x_D],[0,1]))) # distillate line
bot_curve = LineString(np.column_stack(([x_B,x_B],[0,1]))) # bottom product line
feed_curve = LineString(np.column_stack(([z_F,z_F],[0,1]))) # feed line
x_axis = LineString(np.column_stack((x_benzene,np.zeros(x_benzene.size))))

# At the 45-deg line
y_B = x_B
y_D = x_D
y_F = z_F

# q-line
T_feed = 295 # units: K
T_b_mean = 0.4*353.3 + 0.6*383.8 # Average boiling point, units: K
c_mean = 1.84*1000 # Specific heat of mixture, units: J/kgK
h_vap = 30*10**6 # Molar Latent Heat, units: W/kmol
M_mean = (1-z_F)*92.14 + z_F*78

q = (h_vap + (M_mean*c_mean)*(T_b_mean-T_feed))/h_vap  #1 - D/F # quality = 1 - liquid/total feed
m_q = q/(q-1) # Slope
c_q = y_F - m_q*z_F # x-intercept
y_q = m_q*x_benzene + c_q # data set for future manipulation

# Rectifying-Section Operating Line
y_rect = (R/(R+1))*x_benzene + (1/(R+1))*x_D # data set for future manipulation

# Finding the intersection points to trim the lines
q_line = LineString(np.column_stack((x_benzene,y_q))) # q-line
rect_line = LineString(np.column_stack((x_benzene,y_rect))) # Rectifying-Section Operating Line

pt1_rect = [rect_line.intersection(q_line).xy[0][0],rect_line.intersection(q_line).xy[1][0]]
pt2_rect = [rect_line.intersection(ff_deg_line).xy[0][0],rect_line.intersection(ff_deg_line).xy[1][0]]

# Rectifying-Section partial line for executing conditional statements
x_rect_line_partial = np.linspace(pt1_rect[0],pt2_rect[0],201)
y_rect_line_partial = np.linspace(pt1_rect[1],pt2_rect[1],201)
rect_line_partial = LineString(np.column_stack((x_rect_line_partial,y_rect_line_partial)))

pt1_q = pt1_rect # same point
pt2_q = [q_line.intersection(ff_deg_line).xy[0][0],q_line.intersection(ff_deg_line).xy[1][0]]
pt3_q = [q_line.intersection(eq_curve).xy[0][0],q_line.intersection(eq_curve).xy[1][0]] # for extending the stripping line upto the equilibrium curve

# q line partial for executing conditional statements
x_q_line_partial = np.linspace(pt2_q[0],pt3_q[0],201)
y_q_line_partial = np.linspace(pt2_q[1],pt3_q[1],201)
q_line_partial = LineString(np.column_stack((x_q_line_partial,y_q_line_partial)))

# Stripping-Section Operating Line | Since we don't know boilup ratio, we can graphically plot the line
pt1_strip = pt1_rect # same point
pt2_strip = [x_B,y_B]
m_strip = (pt2_strip[1] - pt1_strip[1])/(pt2_strip[0] - pt1_strip[0]) # slope

c_strip = pt2_strip[1] - m_strip*pt2_strip[0] # x-intercept
y_strip = m_strip*x_benzene + c_strip # data set for future manipulation
strip_line = LineString(np.column_stack((x_benzene,y_strip))) # Stripping-Section Operating Line
pt3_strip = [strip_line.intersection(eq_curve).xy[0][0],strip_line.intersection(eq_curve).xy[1][0]] # for extending the stripping line upto the equilibrium curve

x_strip_line_partial = np.linspace(pt1_rect[0],pt3_strip[0],201)
y_strip_line_partial = np.linspace(pt1_rect[1],pt3_strip[1],201)
strip_line_partial = LineString(np.column_stack((x_strip_line_partial,y_strip_line_partial))) # Partial line for giving conditional statements

print("\nQuestion B")
# ------------------------------------ For OPTIMAL feed ------------------------------------
plt.figure(2)
plt.clf()

# Plotting background curves and lines
plt.plot(x_benzene, y_equi, color='blue', label='Equilibrium Curve')
plt.plot(x_benzene, x_benzene, color='red', label='45-degree Line')
plt.plot([x_D, x_D], [0, y_D], ':b', color='gray')
plt.plot([x_B, x_B], [0, y_B], ':b', color='gray')
plt.plot([z_F, z_F], [0, y_F], ':b', color='gray')
plt.plot([pt1_q[0], pt2_q[0]], [pt1_q[1], pt2_q[1]], color='gray')
plt.plot([pt1_rect[0], pt2_rect[0]], [pt1_rect[1], pt2_rect[1]], color='gray')
plt.plot([pt2_strip[0], pt1_strip[0]], [pt2_strip[1], pt1_strip[1]], color='gray')

# Stepping operation to find the number of plates and position of the feed
i = 0  # step counter
flag = 0  # indicator variable to know when to stop the loop
feed_step = 0  # indicator variable for feed step identification

# Iterating for optimal feed conditions
while flag != 1:

    if i == 0:
        # Step-1 | Point 1 is for start and Point 2 is for finish every single time it appears
        step1_h_line = LineString(np.column_stack((x_benzene, y_D * np.ones(x_benzene.size))))  # Step-1 Horizontal Line
        step1_h_pt1 = [x_D, y_D]  # the start point of h1 (horizontal,first step)
        step1_h_pt2 = [step1_h_line.intersection(eq_curve).xy[0][0],step1_h_line.intersection(eq_curve).xy[1][0]]  # the end point of h1 (horizontal,first step)

        step1_v_line = LineString(np.column_stack((step1_h_pt2[0] * np.ones(y_benzene.size), y_benzene)))  # Step-1 Vertical Line
        step1_v_pt1 = step1_h_pt2  # the start point of v1 (vertical,first step)
        step1_v_pt2 = [step1_v_line.intersection(rect_line).xy[0][0],step1_v_line.intersection(rect_line).xy[1][0]]  # the end point of v1 (vertical,first step)

        # Plotting the stepping operation
        plt.plot([step1_h_pt1[0], step1_h_pt2[0]], [step1_h_pt1[1], step1_h_pt2[1]], color='green')
        plt.plot([step1_v_pt1[0], step1_v_pt2[0]], [step1_v_pt1[1], step1_v_pt2[1]], color='green')

        stepn_minus_1_h_pt2 = step1_h_pt2  # assigning value to a variable, to store it for the next step
        stepn_minus_1_v_pt2 = step1_v_pt2  # assigning value to a variable, to store it for the next step
        stepn_minus_1_h_line = step1_h_line  # assigning value to a variable, to store it for the next step
        stepn_minus_1_v_line = step1_v_line  # assigning value to a variable, to store it for the next step

        print("\nStep #{:.0f}".format(i))
        print("H | Point 1: {:.3f},{:.3f} | Point 2: {:.3f},{:.3f}".format(step1_h_pt1[0], step1_h_pt1[1], step1_h_pt2[0], step1_h_pt2[1]))
        print("V | Point 1: {:.3f},{:.3f} | Point 2: {:.3f},{:.3f}".format(step1_v_pt1[0], step1_v_pt1[1], step1_v_pt2[0], step1_v_pt2[1]))
    else:

        # Step > 1
        # just putting in a condition for the stepping to stop
        if (stepn_minus_1_v_line.intersection(x_axis).xy[0][0] - x_B) / stepn_minus_1_v_line.intersection(x_axis).xy[0][0] > 0.01:
            stepn_h_line = LineString(np.column_stack((x_benzene, stepn_minus_1_v_pt2[1] * np.ones(x_benzene.size))))  # Horizontal Line
            stepn_h_pt1 = stepn_minus_1_v_pt2
            stepn_h_pt2 = [stepn_h_line.intersection(eq_curve).xy[0][0], stepn_h_line.intersection(eq_curve).xy[1][0]]

            stepn_v_line = LineString(np.column_stack((stepn_h_pt2[0] * np.ones(y_benzene.size), y_benzene)))  # Vertical Line
            stepn_v_pt1 = stepn_h_pt2

            # just to check if the stepping line intersects with the stripping line | checking it before so that we select the stripping line in the vertical line
            if stepn_h_line.intersection(q_line_partial).is_empty == False & feed_step == 0:
                feed_step = i
                print("\nFeed Step: {:.0f}".format(feed_step))

            if feed_step == 0:
                stepn_v_pt2 = [stepn_v_line.intersection(rect_line).xy[0][0],stepn_v_line.intersection(rect_line).xy[1][0]]
            else:
                stepn_v_pt2 = [stepn_v_line.intersection(strip_line).xy[0][0],stepn_v_line.intersection(strip_line).xy[1][0]]

            # Plotting the stepping operation
            plt.plot([stepn_h_pt1[0], stepn_h_pt2[0]], [stepn_h_pt1[1], stepn_h_pt2[1]], color='green')
            plt.plot([stepn_v_pt1[0], stepn_v_pt2[0]], [stepn_v_pt1[1], stepn_v_pt2[1]], color='green')

            print("\nStep #{:.0f}".format(i))
            print("H | Point 1: {:.3f},{:.3f} | Point 2: {:.3f},{:.3f}".format(stepn_h_pt1[0], stepn_h_pt1[1], stepn_h_pt2[0], stepn_h_pt2[1]))
            print("V | Point 1: {:.3f},{:.3f} | Point 2: {:.3f},{:.3f}".format(stepn_v_pt1[0], stepn_v_pt1[1], stepn_v_pt2[0], stepn_v_pt2[1]))

            stepn_minus_1_h_pt2 = stepn_h_pt2  # assigning value to a variable, to store it for the next step
            stepn_minus_1_v_pt2 = stepn_v_pt2  # assigning value to a variable, to store it for the next step
            stepn_minus_1_h_line = stepn_h_line  # assigning value to a variable, to store it for the next step
            stepn_minus_1_v_line = stepn_v_line  # assigning value to a variable, to store it for the next step
        else:
            flag = 1  # iteration stops

    i += 1

plt.xlabel("x, Mole fraction of benzene in liquid")
plt.ylabel("y, Mole fraction of benzene in the vapor")

plt.xlim([0, 1])
plt.ylim([0, 1])

plt.legend()
plt.savefig("optimal.png")

print("\nQuestion C")
# Calculating boil-up ratio
V_B = 1/(m_strip-1)
print("Boil up ratio V_B: {:.3f} kmol/kmol of feed".format(V_B))

M_avg = (1-z_F)*92.14 + z_F*78
print("Average Molar Mass: {:.3f} g".format(M_avg))

F_kmol = F/M_avg
print("Feed rate: {:.3f} kmol/s".format(F_kmol))

rate_vapor = V_B*F_kmol # kmol/s
print("Vapor flow rate: {:.3f} kmol/s".format(rate_vapor))

h_latent = 30*10**6 # Units: W/kmol
Q = rate_vapor*h_latent # W
print("Heat Load: {:.3f} MW".format(Q/10**6))

H_steam = CP.PropsSI('H','P',240000,'Q',1,'Water') # J/kg
print("Latent Heat of Steam at 240 kN/m2: {:.3f} kJ/kg".format(H_steam/1000))

mass_steam = Q/H_steam
print("Mass flow rate of steam: {:.3f} kg/s".format(mass_steam))

print("\nQuestion D")
density_molar_top = CP.PropsSI("Dmolar", "T", 353.3, "P", 101325, "Benzene")/1000 # kmol/m3
vol_flow_rate_top = rate_vapor/density_molar_top
print("Volume flow rate of benzene at top: {:.3f} m3/s".format(vol_flow_rate_top))

velocity_vapor = 1 # m/s
area = vol_flow_rate_top/velocity_vapor
print("Cross-sectional area of column (top): {:.3f} m2".format(area))
dia = np.sqrt(4*area/np.pi)
print("Diameter of the column (top): {:.3f} m".format(dia))

density_molar_bottom = CP.PropsSI("Dmolar", "T", 383.8, "P", 101325, "Toluene")/1000 # kmol/m3
vol_flow_rate_bottom = rate_vapor/density_molar_bottom
print("\nVolume flow rate of benzene at bottom: {:.3f} m3/s".format(vol_flow_rate_bottom))

area = vol_flow_rate_bottom/velocity_vapor
print("Cross-sectional area of column (bottom): {:.3f} m2".format(area))
dia = np.sqrt(4*area/np.pi)
print("Diameter of the column (bottom): {:.3f} m".format(dia))

print("\nQuestion E")
velocity_vapor = 0.75 # m/s

area_free_bottom = 0.88 * vol_flow_rate_bottom/velocity_vapor
print("Cross-sectional area of column (bottom): {:.3f} m2".format(area_free_bottom))
dia = np.sqrt(4*area_free_bottom/np.pi)
print("Diameter of the column (bottom): {:.3f} m".format(dia))

area_free_top = 0.88 * vol_flow_rate_top/velocity_vapor
print("\nCross-sectional area of column (top): {:.3f} m2".format(area_free_top))
dia = np.sqrt(4*area_free_top/np.pi)
print("Diameter of the column (top): {:.3f} m".format(dia))

print("\nQuestion F")

# Minimum reflux ratio = m_rect_limit/(1-m_rect_limit)
m_rect_limit = (pt3_q[1] - y_D)/(pt3_q[0] - x_D)
min_reflux_ratio = m_rect_limit/(1-m_rect_limit)
print("Minimum Reflux Ratio: {:.3f}".format(min_reflux_ratio))

# ------------------------------------ For MINIMUM PLATES ------------------------------------

plt.figure(2)
plt.clf()

# Plotting background curves and lines
plt.plot(x_benzene, y_equi, color='blue', label='Equilibrium Curve')
plt.plot(x_benzene, x_benzene, color='red', label='45-degree Line')
plt.plot([x_D, x_D], [0, y_D], ':b', color='gray')
plt.plot([x_B, x_B], [0, y_B], ':b', color='gray')
plt.plot([z_F, z_F], [0, y_F], ':b', color='gray')

# Stepping operation to find the number of plates and position of the feed
i = 0  # step counter
flag = 0  # indicator variable to know when to stop the loop
feed_step = 0  # indicator variable for feed step identification

# Iterating for optimal feed conditions
while flag != 1:

    if i == 0:
        # Step-1 | Point 1 is for start and Point 2 is for finish every single time it appears
        step1_h_line = LineString(np.column_stack((x_benzene, y_D * np.ones(x_benzene.size))))  # Step-1 Horizontal Line
        step1_h_pt1 = [x_D, y_D]  # the start point of h1 (horizontal,first step)
        step1_h_pt2 = [step1_h_line.intersection(eq_curve).xy[0][0],step1_h_line.intersection(eq_curve).xy[1][0]]  # the end point of h1 (horizontal,first step)

        step1_v_line = LineString(np.column_stack((step1_h_pt2[0] * np.ones(y_benzene.size), y_benzene)))  # Step-1 Vertical Line
        step1_v_pt1 = step1_h_pt2  # the start point of v1 (vertical,first step)
        step1_v_pt2 = [step1_v_line.intersection(ff_deg_line).xy[0][0],step1_v_line.intersection(ff_deg_line).xy[1][0]]  # the end point of v1 (vertical,first step)

        # Plotting the stepping operation
        plt.plot([step1_h_pt1[0], step1_h_pt2[0]], [step1_h_pt1[1], step1_h_pt2[1]], color='green')
        plt.plot([step1_v_pt1[0], step1_v_pt2[0]], [step1_v_pt1[1], step1_v_pt2[1]], color='green')

        stepn_minus_1_h_pt2 = step1_h_pt2  # assigning value to a variable, to store it for the next step
        stepn_minus_1_v_pt2 = step1_v_pt2  # assigning value to a variable, to store it for the next step
        stepn_minus_1_h_line = step1_h_line  # assigning value to a variable, to store it for the next step
        stepn_minus_1_v_line = step1_v_line  # assigning value to a variable, to store it for the next step

        print("\nStep #{:.0f}".format(i))
        print("H | Point 1: {:.3f},{:.3f} | Point 2: {:.3f},{:.3f}".format(step1_h_pt1[0], step1_h_pt1[1], step1_h_pt2[0], step1_h_pt2[1]))
        print("V | Point 1: {:.3f},{:.3f} | Point 2: {:.3f},{:.3f}".format(step1_v_pt1[0], step1_v_pt1[1], step1_v_pt2[0], step1_v_pt2[1]))
    else:

        # Step > 1
        # just putting in a condition for the stepping to stop
        if (stepn_minus_1_v_line.intersection(x_axis).xy[0][0] - x_B) / stepn_minus_1_v_line.intersection(x_axis).xy[0][0] > 0.01:
            stepn_h_line = LineString(np.column_stack((x_benzene, stepn_minus_1_v_pt2[1] * np.ones(x_benzene.size))))  # Horizontal Line
            stepn_h_pt1 = stepn_minus_1_v_pt2
            stepn_h_pt2 = [stepn_h_line.intersection(eq_curve).xy[0][0], stepn_h_line.intersection(eq_curve).xy[1][0]]

            stepn_v_line = LineString(np.column_stack((stepn_h_pt2[0] * np.ones(y_benzene.size), y_benzene)))  # Vertical Line
            stepn_v_pt1 = stepn_h_pt2
            stepn_v_pt2 = [stepn_v_line.intersection(ff_deg_line).xy[0][0],stepn_v_line.intersection(ff_deg_line).xy[1][0]]

            # Plotting the stepping operation
            plt.plot([stepn_h_pt1[0], stepn_h_pt2[0]], [stepn_h_pt1[1], stepn_h_pt2[1]], color='green')
            plt.plot([stepn_v_pt1[0], stepn_v_pt2[0]], [stepn_v_pt1[1], stepn_v_pt2[1]], color='green')

            print("\nStep #{:.0f}".format(i))
            print("H | Point 1: {:.3f},{:.3f} | Point 2: {:.3f},{:.3f}".format(stepn_h_pt1[0], stepn_h_pt1[1], stepn_h_pt2[0], stepn_h_pt2[1]))
            print("V | Point 1: {:.3f},{:.3f} | Point 2: {:.3f},{:.3f}".format(stepn_v_pt1[0], stepn_v_pt1[1], stepn_v_pt2[0], stepn_v_pt2[1]))

            stepn_minus_1_h_pt2 = stepn_h_pt2  # assigning value to a variable, to store it for the next step
            stepn_minus_1_v_pt2 = stepn_v_pt2  # assigning value to a variable, to store it for the next step
            stepn_minus_1_h_line = stepn_h_line  # assigning value to a variable, to store it for the next step
            stepn_minus_1_v_line = stepn_v_line  # assigning value to a variable, to store it for the next step
        else:
            flag = 1  # iteration stops

    i += 1

plt.xlabel("x, Mole fraction of benzene in liquid")
plt.ylabel("y, Mole fraction of benzene in the vapor")

plt.xlim([0, 1])
plt.ylim([0, 1])

plt.legend()
plt.savefig("minimum_plates.png")