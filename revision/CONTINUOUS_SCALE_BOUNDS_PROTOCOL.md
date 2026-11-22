# Continuous monotone scale bounds

This extension tests whether the fixed forecast rankings inferred from the eight-segment scale analysis depend on its discretization. Keep the paired forecast ledgers and weights from the original analysis. Use one independent transform per site and slope budgets 0.10, 0.25 and 0.50. Score MSE differences at all four horizons. The primary setting remains the 0.25 slope budget. Compare the continuous bound with the existing feasible witnesses at both EcoIrrigate sites and all 35 USCRN stations. Also compare with 8-, 16- and 32-segment interpolation bounds at the two EcoIrrigate sites. This analysis uses the existing evaluated origins.

## Exact score grid and continuous bound

An admissible map $g$ is absolutely continuous on the observed scoring range $I$, fixes both endpoints and has derivative in $[1-\delta,1+\delta]$ almost everywhere. Each squared error is the double integral of $g'(u)g'(v)$ over the square formed by the interval between forecast and observation. Define $K(u,v)$ as the mean signed coverage of those squares, counting the model with weight +1 and persistence with weight −1. The paired MSE difference is $D(g)=\iint K(u,v)g'(u)g'(v)\,du\,dv$.

Write $g'=1+e$, so $|e|\le\delta$ and $\int_I e=0$. Let $\ell(u)=\int_I K(u,v)\,dv$ and let $c$ be a width-weighted median of $\ell$. Expanding and using symmetry of $K$ gives

$$
D(g)-D(\mathrm{id})=2\int_I [\ell(u)-c]e(u)\,du+\iint_{I^2}K(u,v)e(u)e(v)\,du\,dv.
$$

The absolute difference is bounded by $2\delta\int_I|\ell-c|+\delta^2\iint_{I^2}|K|$. The median minimizes the absolute-deviation integral. The first term gives the exact maximum magnitude of the linear perturbation since $e$ can take opposite extremes above and below the median and balance its integral on its level set. The quadratic term is bounded in absolute value, so the final interval can be wider than the attainable range.

For both signs, integrate the slope that optimizes the linear term and directly score the resulting map under the full squared loss. The balancing slope is constant on the median level set. Report these feasible scores alongside the earlier eight-segment witnesses. The candidate maps optimize the first-order term only.

The sorted union of forecasts, observations and range endpoints makes $K$ constant on each grid cell pair. Interpolation at all these values preserves every error exactly. Sweep the first coordinate and maintain signed difference counts across the second coordinate to evaluate the integrals without materializing the matrix. For $k$ intervals and $n$ observations, sorted event processing and row scans cost $O(n\log n+k^2)$ time and $O(k+n)$ working storage. The implementation includes a numerical allowance of $10^{-9}$ squared percentage points. The balanced-slope bound can also be evaluated on samples for which identical values are repeated.

## Comparison with the coarse-grid approach

The linear interpolant $G$ on an equally spaced grid also shares the endpoints and has feasible slopes because each secant slope averages $g'$. For $g(v)-g(u)-[G(v)-G(u)]$, full grid cells between $u$ and $v$ cancel. A partial interval of length $l$ in a cell of length $h$ contributes at most $2\delta l(1-l/h)$ in magnitude. This follows by subtracting $l/h$ times the integral of $g'$ over the cell from its integral over the partial interval. Sum the bounds over the partial cells.

Write the difference bound as $E(u,v)$ and $L=1+\delta$. For observation $y$, forecast $m$ and persistence $p$, the absolute change in the paired squared loss difference is bounded by both

- $2L[|m-y|E(m,y)+|p-y|E(p,y)]$, from the difference of squares for each forecast error, and
- $L[(|m-y|+|p-y|)E(m,p)+|m-p|(E(m,y)+E(p,y))]$, from factoring the paired difference as $[g(m)-g(p)][g(m)+g(p)-2g(y)]$.

Average the smaller of the two bounds for each pair to get a deterministic margin for the grid. Widen the lifted linear-programming interval for each grid by its margin and intersect those widened intervals with the continuous bound. Combine site intervals by row counts. The resulting interval encloses all full-family MSE differences on the fixed sample, subject to numerical allowances. Existing feasible piecewise linear witnesses are members of the full class. If an outer interval crosses zero, require feasible transforms demonstrating both signs to report a reversal.

## Verification

Compare the sweep with dense rectangle integration and the first-order term with linear programming on a small example. Check the bounds by direct scoring with finer-grid maps. Assert that each feasible witness and identity score lie inside its interval. The derivative budgets are sensitivity assumptions. The specified family excludes site drift and additional measurement error. Statistical sampling is addressed in the separate eight-segment block resampling analysis.

## Results at the primary slope budget

The continuous analysis preserves the EcoIrrigate 1-, 6- and 12-hour signs on the fixed sample. Grid refinement alone leaves these signs unresolved at 32 intervals once the interpolation margin is included. The eight-segment maps already demonstrate the 24-hour reversal.

| EcoIrrigate horizon | 32-segment bound including interpolation margin | Continuous bound from signed coverage |
|---|---:|---:|
| 1 h | [−0.0046, +0.0423] | [+0.0083, +0.0274] |
| 6 h | [−0.1150, +0.0335] | [−0.0664, −0.0118] |
| 12 h | [−0.2627, +0.0268] | [−0.1839, −0.0474] |
| 24 h | [−0.2487, +0.6257] | [−0.1347, +0.4925] |

The USCRN one-hour eight-segment outer interval stays above zero at this budget. The continuous method constructs station-specific maps that change the fixed-sample difference from +0.021858 to −0.000633 squared percentage points. The manuscript verifier reconstructs this reversal independently from one-dimensional error-length updates and a fractional allocation of the slope budget. The 1-hour Ridge versus persistence difference under the constructed maps is small.

## Slope budget required for a reversal

For each dataset and horizon, bound the smallest slope budget at which any admissible family of site maps changes the sign of the pooled MSE gap. Apply the same fixed empirical ledgers and row weights as above. For a lower bound, use the width-weighted median bound at each site and pool the first- and second-order coefficients by row counts. The positive solution of $A\delta+B\delta^2=|D(\mathrm{id})|-10^{-9}$ gives a certified threshold. Every budget strictly below it preserves the nominal sign.

For an upper bound, each site can choose the identity, either of the two first-order maps, or an eight-segment witness from the 0.10, 0.25 and 0.50 runs. Express each witness as $g_\delta(x)=x+\delta d(x)$ with endpoint-fixed displacement and $|d'|\le1$. Its paired MSE difference is exactly quadratic in $\delta$. Evaluate the resulting polynomials at budgets 0.001 through 0.999 in steps of 0.001. At each budget, choose the feasible site maps most adverse to the nominal sign and pool their actual gaps by row count. The first budget with an opposing sign gives an upper bound. An absent upper bound means this candidate set did not demonstrate a reversal within the searched range.

These brackets concern a site-specific scale map at a single fixed horizon and sample. The maps may differ across horizons. The thresholds quantify sensitivity to the assumed scale family.
