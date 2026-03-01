'''

Convertible Bond Pricer
Using Longstaff-Schwartz LSMC

'''



import numpy as np
from numpy.polynomial import laguerre 
import matplotlib.pyplot as plt

SIMS = 10_000
face_value = 100
T = 2
steps = 252 * T
m = 4 # degree of the Laguerre Polynomials
N = 1
P = face_value / N
S0 = 100.0
sigma = 0.1
q = 0
r = 0.04

def stock_price(S0, sigma, r, q, T, steps, sims):
    X = np.full(shape=(steps+1, sims), fill_value=S0)
    Z = np.random.normal(0, 1, size=(steps, sims))
    dt = T/steps
    for i in range(1, steps+1):
        X[i] = X[i-1] + (r - q) * X[i-1] * dt + sigma * X[i-1] * np.sqrt(dt) * Z[i-1, :]
        
    return X.T

def convertible_bond_pricer(steps: int, N: int = 50, face_value: float = 1000, sims: int = SIMS, m:int=10,
                            coupon_annual_rate: float = 0.0, freq_coupon: int = 12):
    # n intervals in [0,T]
    dt = T/ steps
    coupon = np.zeros(steps+1)
    if coupon_annual_rate > 0.0:
        coupon_amount = face_value * coupon_annual_rate / freq_coupon
        pay_times = np.arange(1, int(T * freq_coupon) + 1) / freq_coupon
        pay_idx = np.round(pay_times / dt).astype(int)
        pay_idx = pay_idx[(pay_idx >= 1) & (pay_idx <= steps)]
        coupon[pay_idx] = coupon_amount
    
    
    paths = stock_price(S0, sigma, r, q, T, steps, sims)
    ST = paths[:, -1]
    
    discount = np.exp(-r*dt)
    
    reedem_T = face_value + coupon[-1]
    ConvB = np.array(np.maximum(N * ST, reedem_T))
    P = face_value / N
    
    # diagnostics
    exercise_time = np.full(sims, steps, dtype=int)
    diag_store = None
    boundary = np.full(steps, np.nan)
    diag_t = steps // 2
    
    for t in range(steps-2, -1, -1):
        
        curr_price = paths[:, t]
        conv_values = curr_price * N
        
        Y_hold = discount * (ConvB + coupon[t+1])
        
        if np.any(curr_price > P):
            itm_curr_price = curr_price[curr_price > P]
            
            alphas = laguerre.lagfit(itm_curr_price / P, Y_hold[curr_price > P], m) 
            Y = laguerre.lagval(curr_price / P, alphas) # continuation values
            
            
            exercise_idx = (curr_price > P) & (conv_values > Y)
            
            # diagnostics
            grid = np.linspace(curr_price.min(), curr_price.max(), 200)
            Cg = laguerre.lagval(grid / P, alphas)
            diff = (N * grid) - Cg
            cross = np.where(diff >= 0)[0]
            if len(cross) > 0:
                boundary[t] = grid[cross[0]]
            
            if t == diag_t:
                diag_store = {"t": t, "S": curr_price.copy(),
                              "Y_hold": Y_hold.copy(), 
                              "Y": Y.copy(), "conv": conv_values.copy(),
                              "exercise": exercise_idx.copy(), "P": P}
            
        else:
            exercise_idx = np.zeros_like(curr_price, dtype=bool)
            
        ConvB = Y_hold # continue
        ConvB[exercise_idx] = conv_values[exercise_idx] # exercise
        exercise_time[exercise_idx] = t
        
    return ConvB, paths, exercise_time, coupon, boundary, diag_store

def r_style_axes(ax):
    ax.set_facecolor("white")
    ax.grid(False)
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(direction="out", length=4, width=1, colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")

plt.rcParams.update({
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.size": 11,
})


vals, paths, ex_time, coupon, boundary, diag = convertible_bond_pricer(steps=steps, N=N, face_value=face_value,
                                                                       sims=SIMS, m=m, coupon_annual_rate= 0.00,
                                                                       freq_coupon=2)

price0 = np.mean(vals)
stderr = np.std(vals, ddof=1) / np.sqrt(len(vals))
print(price0, stderr)


# Distribution of the time-0 LSMC estimator
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(vals, bins=50, edgecolor="black", facecolor="lightgray")
ax.set_title("LSMC convertible: distribution of pathwise PV at t=0")
ax.set_xlabel("Present value per path")
ax.set_ylabel("Frequency")
r_style_axes(ax)
plt.savefig("fig_pv_hist.pdf", format="pdf")
plt.show()

# Continuation vs conversion at one time slice
t = diag["t"]
S = diag["S"]
Y_hold = diag["Y_hold"]
Y = diag["Y"]
conv = diag["conv"]
exercise = diag["exercise"]
Pconv = diag["P"]

fig = plt.figure()
ax = fig.add_subplot(111)
mask = S > Pconv

S_sc = S[mask]
Y_sc = Y_hold[mask]

x_lo, x_hi = np.quantile(S_sc, [0.01, 0.99])
y_lo, y_hi = np.quantile(Y_sc, [0.01, 0.99])

x_pad = 0.05 * (x_hi - x_lo)
y_pad = 0.05 * (y_hi - y_lo)

ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

ax.plot(S[mask], Y_hold[mask], "o", markersize=2, markerfacecolor="none", markeredgecolor="black", linestyle="None")
idx = np.argsort(S)
ax.plot(S[idx], Y[idx], "-", color="black", linewidth=1.0) # continuation curve
ax.plot(S[idx], conv[idx], "--", color="black", linewidth=1.0) # conversion line
ax.set_title(f"Continuation vs conversion at step t={t}")
ax.set_xlabel(r"$S_{t}$")
ax.set_ylabel("Value")
r_style_axes(ax)
plt.savefig("fig_continuation_vs_conversion.pdf", format="pdf")
plt.show()

# Estimated conversion boundary over time
times = np.arange(steps) * (T / steps)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(times, boundary, "-", color="black", linewidth=1.0)
ax.set_title("Estimated conversion boundary over time")
ax.set_xlabel("Time (years)")
ax.set_ylabel(r"Boundary stock price $S^*(t)$")
r_style_axes(ax)
plt.savefig("fig_boundary.pdf", format="pdf")
plt.show()

# Histogram of conversion times
ex_years = ex_time * (T / steps)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(ex_years, bins=40, edgecolor="black", facecolor="lightgray")
ax.set_title("Distribution of estimated conversion times")
ax.set_xlabel("Conversion time (years)")
ax.set_ylabel("Frequency")
r_style_axes(ax)
plt.savefig("fig_exercise_hist.pdf", format="pdf")
plt.show()

### 

Mvis = 50
paths_vis = stock_price(S0, sigma, r, q, T, steps, Mvis)

Stn   = paths_vis[:, -1]
Stnm1 = paths_vis[:, -2]

deg = 6
dt = T / steps
df = np.exp(-r * dt)
cashflow_tn = np.maximum(N * Stn, face_value)
disc_cashflows = df * cashflow_tn

coeffs = laguerre.lagfit(Stnm1 / P, disc_cashflows, deg)

s_linspace = np.linspace(Stnm1.min(), Stnm1.max(), 300)
contvalues = laguerre.lagval(s_linspace / P, coeffs)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

ax.plot(Stnm1, disc_cashflows, "o",
        markersize=4, markerfacecolor="none", markeredgecolor="black",
        linestyle="None", label="Discounted Cashflows")

ax.plot(s_linspace, contvalues, "-",
        color="black", linewidth=1.2,
        label=f"Fitted Laguerre Series, degree={deg}")

ax.set_title("Discounted Cashflows vs Underlying Price")
ax.set_xlabel(r"Underlying $S(t_{n-1})$")
ax.set_ylabel(r"Discounted cashflow to $t_{n-1}$")
ax.legend(loc="upper left", frameon=False)

r_style_axes(ax)
plt.savefig("LaguerreFitConvertibleBond.pdf", format="pdf")
plt.show()