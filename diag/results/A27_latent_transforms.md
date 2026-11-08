| Space | Readout | Transform | VAL ID pct (31) | VAL top-1 | VAL top-3 | VAL top-10 | TRAIN ID pct (278) | TRAIN top-1 | TRAIN top-10 |
|---|---|---|---|---|---|---|---|---|---|
| gene expression (no model) | expr | raw | 0.996 | 0.919 | 0.984 | 1.000 | 0.986 | 0.655 | 0.921 |
| public scGPT | cls | raw | 0.944 | 0.677 | 0.887 | 0.952 | 0.904 | 0.286 | 0.638 |
| public scGPT | cls | whiten | 0.984 | 0.774 | 0.935 | 1.000 | 0.970 | 0.432 | 0.770 |
| public scGPT | cls | remove_pcs10 | 0.972 | 0.710 | 0.887 | 0.984 | 0.942 | 0.333 | 0.691 |
| public scGPT | cls | lda50 | 0.992 | 0.871 | 0.968 | 1.000 | – | – | – |
| public scGPT | tok | raw | 0.950 | 0.726 | 0.871 | 0.952 | 0.921 | 0.322 | 0.655 |
| public scGPT | tok | whiten | 0.985 | 0.806 | 0.952 | 1.000 | 0.967 | 0.419 | 0.784 |
| public scGPT | tok | remove_pcs10 | 0.967 | 0.758 | 0.887 | 0.968 | 0.944 | 0.347 | 0.718 |
| public scGPT | tok | lda50 | 0.983 | 0.855 | 0.952 | 0.984 | – | – | – |
| blood-scGPT | cls | raw | 0.905 | 0.581 | 0.790 | 0.871 | 0.850 | 0.228 | 0.507 |
| blood-scGPT | cls | whiten | 0.978 | 0.758 | 0.935 | 1.000 | 0.962 | 0.417 | 0.755 |
| blood-scGPT | cls | remove_pcs10 | 0.962 | 0.726 | 0.903 | 0.968 | 0.940 | 0.340 | 0.665 |
| blood-scGPT | cls | lda50 | 0.988 | 0.806 | 0.952 | 1.000 | – | – | – |
| blood-scGPT | tok | raw | 0.901 | 0.629 | 0.758 | 0.871 | 0.834 | 0.227 | 0.495 |
| blood-scGPT | tok | whiten | 0.973 | 0.726 | 0.903 | 0.984 | 0.963 | 0.442 | 0.761 |
| blood-scGPT | tok | remove_pcs10 | 0.973 | 0.677 | 0.855 | 1.000 | 0.950 | 0.362 | 0.683 |
| blood-scGPT | tok | lda50 | 0.985 | 0.823 | 0.952 | 1.000 | – | – | – |
| blood-JEPA | cls | raw | 0.666 | 0.081 | 0.210 | 0.548 | 0.633 | 0.009 | 0.095 |
| blood-JEPA | cls | whiten | 0.782 | 0.226 | 0.516 | 0.790 | 0.727 | 0.056 | 0.221 |
| blood-JEPA | cls | remove_pcs10 | 0.768 | 0.242 | 0.403 | 0.694 | 0.676 | 0.038 | 0.158 |
| blood-JEPA | cls | lda50 | 0.756 | 0.194 | 0.419 | 0.758 | – | – | – |
| blood-JEPA | tok | raw | 0.644 | 0.065 | 0.242 | 0.516 | 0.629 | 0.009 | 0.099 |
| blood-JEPA | tok | whiten | 0.810 | 0.242 | 0.532 | 0.806 | 0.738 | 0.041 | 0.239 |
| blood-JEPA | tok | remove_pcs10 | 0.808 | 0.177 | 0.435 | 0.758 | 0.701 | 0.047 | 0.200 |
| blood-JEPA | tok | lda50 | 0.790 | 0.226 | 0.435 | 0.742 | – | – | – |
