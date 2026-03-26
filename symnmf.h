#ifndef SYMNMF_H
#define SYMNMF_H

/* Memory management functions exposed to the Python C API */
double **allocate_matrix(int rows, int cols);
void free_matrix(double **mat, int n);

/* Core mathematical functions for SymNMF goals */
double **create_sym_matrix(double **data_points, int n, int d);
double *create_diag_matrix(double **sym_mat, int n);
double **create_norm_matrix(double **sym_mat, double *diag_arr, int n);
double **optimize_H(double **W, double **H, int n, int k);

#endif