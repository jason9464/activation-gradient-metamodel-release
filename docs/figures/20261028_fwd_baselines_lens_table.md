# Forward-only baselines: logit lens

Plan: docs/notes/20261028_fwd_baselines_plan.md. Δ = identification AUROC of a bundle's own behaviour description on showing minus none bundles, within organism, pooled over quirk, backdoor and rare.

Reproduction of the published lens values: passed (max |diff| 0.0, any config 0.0).

## Contrasts (confirmatory categories use simultaneous 97.5% intervals across four rows)

| section | a | b | norm | family | a − b | individual 95% | simultaneous 97.5% | category |
|---|---|---|---|---|---|---|---|---|
| primary | grad_L38_center 0.349 | actresp_L38_center 0.225 | z | pooled | +0.125 | +0.085, +0.167 | +0.072, +0.177 | lead remains |
| primary | grad_L38_center 0.349 | actrespl_L38_center 0.267 | z | pooled | +0.082 | +0.044, +0.125 | +0.030, +0.133 | unresolved |
| controls | grad_L38_center 0.349 | actrespn_L38_center 0.225 | z | pooled | +0.124 | +0.085, +0.166 | +0.071, +0.177 | lead remains |
| controls | actrespl_L38_center 0.267 | actrespn_L38_center 0.225 | z | pooled | +0.043 | +0.032, +0.054 | +0.028, +0.057 | loss better |
| descriptive | grad_L38_center 0.290 | actresp_L38_center 0.202 | pct | pooled | +0.087 | +0.050, +0.127 |  |  |
| descriptive | grad_L38_raw 0.345 | actresp_L38_raw 0.226 | z | pooled | +0.120 | +0.083, +0.159 |  |  |
| descriptive | grad_L38_center 0.401 | actresp_L38_center 0.325 | z | quirk | +0.076 | +0.014, +0.136 |  |  |
| descriptive | grad_L38_center 0.346 | actresp_L38_center 0.258 | z | backdoor | +0.088 | +0.011, +0.168 |  |  |
| descriptive | grad_L38_center 0.301 | actresp_L38_center 0.091 | z | rare | +0.210 | +0.150, +0.272 |  |  |
| descriptive | grad_L38_center 0.290 | actrespl_L38_center 0.237 | pct | pooled | +0.053 | +0.018, +0.092 |  |  |
| descriptive | grad_L38_raw 0.345 | actrespl_L38_raw 0.264 | z | pooled | +0.081 | +0.042, +0.120 |  |  |
| descriptive | grad_L38_center 0.401 | actrespl_L38_center 0.336 | z | quirk | +0.065 | +0.003, +0.125 |  |  |
| descriptive | grad_L38_center 0.346 | actrespl_L38_center 0.307 | z | backdoor | +0.039 | -0.031, +0.114 |  |  |
| descriptive | grad_L38_center 0.301 | actrespl_L38_center 0.159 | z | rare | +0.142 | +0.084, +0.202 |  |  |
| descriptive | grad_L38_center 0.349 | act_L38_center 0.228 | z | pooled | +0.121 | +0.081, +0.163 |  |  |
| descriptive | grad_L38_center 0.349 | actw_L38_center 0.292 | z | pooled | +0.058 | +0.018, +0.100 |  |  |
| descriptive | grad_L38_center 0.349 | actrespg_L38_center 0.295 | z | pooled | +0.054 | +0.014, +0.097 |  |  |
| descriptive | grad_L38_center 0.349 | actrespm_L38_center 0.309 | z | pooled | +0.041 | +0.000, +0.085 |  |  |
| descriptive | grad_L38_center 0.349 | actrespn_L38_center 0.225 | z | pooled | +0.124 | +0.085, +0.166 |  |  |
| descriptive | actresp_L38_center 0.225 | act_L38_center 0.228 | z | pooled | -0.004 | -0.008, +0.000 |  |  |
| descriptive | actrespl_L38_center 0.267 | actresp_L38_center 0.225 | z | pooled | +0.043 | +0.032, +0.053 |  |  |
| descriptive | actrespg_L38_center 0.295 | actrespl_L38_center 0.267 | z | pooled | +0.028 | +0.020, +0.035 |  |  |
| descriptive | actrespl_L38_center 0.267 | actw_L38_center 0.292 | z | pooled | -0.024 | -0.034, -0.015 |  |  |
| descriptive | actrespl_L38_center 0.267 | actrespn_L38_center 0.225 | z | pooled | +0.043 | +0.032, +0.054 |  |  |
| descriptive | actrespg_L38_center 0.295 | actw_L38_center 0.292 | z | pooled | +0.004 | -0.003, +0.010 |  |  |
| descriptive | grad_L30_center 0.330 | oerr_L40_center 0.333 | z | pooled | -0.003 | -0.033, +0.029 |  |  |
| descriptive | grad_L38_center 0.349 | oerr_L40_center 0.333 | z | pooled | +0.016 | -0.007, +0.038 |  |  |
| descriptive | grad_L39_center 0.326 | oerr_L40_center 0.333 | z | pooled | -0.007 | -0.018, +0.005 |  |  |
| descriptive | oerr_L40_center 0.333 | actrespl_L38_center 0.267 | z | pooled | +0.066 | +0.025, +0.107 |  |  |
| descriptive | text_center 0.362 | actresp_L38_center 0.225 | z | pooled | +0.138 | +0.100, +0.177 |  |  |

## All configs (z)

| config | Δ pooled | quirk | backdoor | rare | top-20 hit (showing) | present-token AUROC |
|---|---|---|---|---|---|---|
| grad_L30_center | 0.330 [0.298, 0.363] | 0.416 | 0.311 | 0.264 | 0.379 | 0.558 |
| grad_L30_raw | 0.323 [0.293, 0.358] | 0.411 | 0.308 | 0.251 | 0.135 | 0.421 |
| grad_L31_center | 0.338 [0.306, 0.372] | 0.424 | 0.320 | 0.271 | 0.395 | 0.560 |
| grad_L31_raw | 0.333 [0.299, 0.367] | 0.421 | 0.312 | 0.265 | 0.161 | 0.428 |
| grad_L32_center | 0.345 [0.314, 0.378] | 0.419 | 0.327 | 0.288 | 0.400 | 0.563 |
| grad_L32_raw | 0.339 [0.307, 0.374] | 0.417 | 0.322 | 0.278 | 0.175 | 0.424 |
| grad_L33_center | 0.334 [0.302, 0.371] | 0.409 | 0.310 | 0.283 | 0.413 | 0.562 |
| grad_L33_raw | 0.327 [0.294, 0.364] | 0.398 | 0.313 | 0.269 | 0.178 | 0.419 |
| grad_L34_center | 0.350 [0.319, 0.386] | 0.409 | 0.333 | 0.307 | 0.427 | 0.565 |
| grad_L34_raw | 0.341 [0.310, 0.377] | 0.392 | 0.337 | 0.295 | 0.184 | 0.428 |
| grad_L35_center | 0.347 [0.316, 0.384] | 0.409 | 0.335 | 0.298 | 0.440 | 0.565 |
| grad_L35_raw | 0.336 [0.304, 0.371] | 0.393 | 0.329 | 0.285 | 0.237 | 0.442 |
| grad_L36_center | 0.351 [0.317, 0.387] | 0.405 | 0.335 | 0.313 | 0.444 | 0.568 |
| grad_L36_raw | 0.340 [0.306, 0.378] | 0.391 | 0.326 | 0.302 | 0.269 | 0.448 |
| grad_L37_center | 0.353 [0.319, 0.387] | 0.404 | 0.342 | 0.312 | 0.459 | 0.597 |
| grad_L37_raw | 0.341 [0.308, 0.377] | 0.394 | 0.334 | 0.294 | 0.364 | 0.445 |
| grad_L38_center | 0.349 [0.314, 0.385] | 0.401 | 0.346 | 0.301 | 0.440 | 0.575 |
| grad_L38_raw | 0.345 [0.311, 0.380] | 0.402 | 0.345 | 0.288 | 0.291 | 0.457 |
| grad_L39_center | 0.326 [0.293, 0.361] | 0.383 | 0.321 | 0.274 | 0.421 | 0.515 |
| grad_L39_raw | 0.314 [0.277, 0.350] | 0.391 | 0.331 | 0.219 | 0.202 | 0.326 |
| act_L30_center | 0.187 [0.152, 0.219] | 0.291 | 0.212 | 0.059 | 0.118 | 0.635 |
| act_L30_raw | 0.155 [0.123, 0.191] | 0.266 | 0.153 | 0.047 | 0.005 | 0.719 |
| act_L35_center | 0.213 [0.177, 0.246] | 0.305 | 0.237 | 0.097 | 0.178 | 0.663 |
| act_L35_raw | 0.189 [0.154, 0.225] | 0.291 | 0.192 | 0.084 | 0.015 | 0.812 |
| act_L38_center | 0.228 [0.195, 0.261] | 0.326 | 0.267 | 0.093 | 0.173 | 0.667 |
| act_L38_raw | 0.229 [0.196, 0.261] | 0.334 | 0.261 | 0.092 | 0.003 | 0.885 |
| actw_L30_center | 0.228 [0.193, 0.262] | 0.307 | 0.252 | 0.124 | 0.156 | 0.623 |
| actw_L30_raw | 0.193 [0.157, 0.230] | 0.282 | 0.192 | 0.104 | 0.006 | 0.714 |
| actw_L35_center | 0.266 [0.231, 0.300] | 0.324 | 0.285 | 0.189 | 0.227 | 0.652 |
| actw_L35_raw | 0.237 [0.201, 0.275] | 0.306 | 0.238 | 0.166 | 0.018 | 0.809 |
| actw_L38_center | 0.292 [0.259, 0.325] | 0.342 | 0.330 | 0.204 | 0.227 | 0.658 |
| actw_L38_raw | 0.283 [0.247, 0.320] | 0.343 | 0.308 | 0.198 | 0.003 | 0.883 |
| text_center | 0.362 [0.329, 0.398] | 0.384 | 0.401 | 0.301 | 0.402 |  |
| text_raw | 0.357 [0.323, 0.393] | 0.380 | 0.382 | 0.308 | 0.261 |  |
| actresp_L30_center | 0.186 [0.151, 0.219] | 0.294 | 0.205 | 0.059 | 0.116 | 0.640 |
| actresp_L30_raw | 0.154 [0.121, 0.189] | 0.265 | 0.147 | 0.048 | 0.007 | 0.713 |
| actresp_L35_center | 0.210 [0.173, 0.244] | 0.305 | 0.230 | 0.095 | 0.178 | 0.672 |
| actresp_L35_raw | 0.184 [0.149, 0.222] | 0.285 | 0.183 | 0.085 | 0.019 | 0.806 |
| actresp_L38_center | 0.225 [0.190, 0.257] | 0.325 | 0.258 | 0.091 | 0.177 | 0.678 |
| actresp_L38_raw | 0.226 [0.192, 0.258] | 0.332 | 0.251 | 0.094 | 0.005 | 0.885 |
| actrespl_L30_center | 0.217 [0.180, 0.251] | 0.307 | 0.241 | 0.104 | 0.157 | 0.636 |
| actrespl_L30_raw | 0.183 [0.147, 0.222] | 0.282 | 0.182 | 0.084 | 0.010 | 0.713 |
| actrespl_L35_center | 0.246 [0.210, 0.281] | 0.316 | 0.271 | 0.153 | 0.224 | 0.667 |
| actrespl_L35_raw | 0.217 [0.181, 0.255] | 0.298 | 0.217 | 0.136 | 0.026 | 0.805 |
| actrespl_L38_center | 0.267 [0.232, 0.302] | 0.336 | 0.307 | 0.159 | 0.224 | 0.672 |
| actrespl_L38_raw | 0.264 [0.229, 0.299] | 0.345 | 0.291 | 0.157 | 0.007 | 0.884 |
| actrespn_L30_center | 0.182 [0.146, 0.215] | 0.292 | 0.201 | 0.054 | 0.113 | 0.640 |
| actrespn_L30_raw | 0.150 [0.117, 0.184] | 0.263 | 0.143 | 0.044 | 0.007 | 0.712 |
| actrespn_L35_center | 0.208 [0.171, 0.242] | 0.306 | 0.231 | 0.087 | 0.173 | 0.671 |
| actrespn_L35_raw | 0.181 [0.145, 0.218] | 0.284 | 0.180 | 0.078 | 0.019 | 0.805 |
| actrespn_L38_center | 0.225 [0.191, 0.257] | 0.327 | 0.261 | 0.087 | 0.175 | 0.677 |
| actrespn_L38_raw | 0.225 [0.191, 0.258] | 0.332 | 0.250 | 0.092 | 0.004 | 0.885 |
| actrespm_L30_center | 0.275 [0.238, 0.316] | 0.316 | 0.279 | 0.229 | 0.233 | 0.597 |
| actrespm_L30_raw | 0.233 [0.196, 0.272] | 0.273 | 0.214 | 0.212 | 0.020 | 0.695 |
| actrespm_L35_center | 0.306 [0.269, 0.343] | 0.330 | 0.295 | 0.295 | 0.336 | 0.625 |
| actrespm_L35_raw | 0.266 [0.225, 0.307] | 0.279 | 0.240 | 0.279 | 0.037 | 0.789 |
| actrespm_L38_center | 0.309 [0.270, 0.343] | 0.331 | 0.306 | 0.290 | 0.313 | 0.638 |
| actrespm_L38_raw | 0.315 [0.276, 0.353] | 0.331 | 0.305 | 0.308 | 0.007 | 0.877 |
| actrespg_L30_center | 0.233 [0.196, 0.267] | 0.311 | 0.254 | 0.132 | 0.163 | 0.626 |
| actrespg_L30_raw | 0.196 [0.159, 0.234] | 0.281 | 0.194 | 0.112 | 0.009 | 0.706 |
| actrespg_L35_center | 0.268 [0.230, 0.303] | 0.321 | 0.281 | 0.201 | 0.237 | 0.658 |
| actrespg_L35_raw | 0.235 [0.198, 0.274] | 0.297 | 0.231 | 0.178 | 0.025 | 0.801 |
| actrespg_L38_center | 0.295 [0.259, 0.330] | 0.342 | 0.326 | 0.218 | 0.241 | 0.666 |
| actrespg_L38_raw | 0.289 [0.254, 0.326] | 0.343 | 0.310 | 0.214 | 0.007 | 0.882 |
| oerr_L40_center | 0.333 [0.299, 0.369] | 0.385 | 0.328 | 0.286 | 0.447 | 0.540 |
| oerr_L40_raw | 0.330 [0.294, 0.367] | 0.395 | 0.323 | 0.273 | 0.395 | 0.385 |
