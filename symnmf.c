#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "symnmf.h"

#define MAX_ITER 300
#define EPSILON 1e-4
#define BETA 0.5

void err_msg_and_terminate()
{
    printf("An Error Has Occurred\n");
    exit(1);
}

/*
 * Reads input from a text file using a Two-Pass approach.
 * Modifies n (rows) and d (dimensions) in place, returns the 2D array.
 */
double **read_input(const char *file_name, int *n, int *d)
{
    FILE *f;
    double **data_points;
    int i, j, c;
    int prev_c = '\n';

    *n = 0;
    *d = 1;

    f = fopen(file_name, "r");
    if (f == NULL)
        err_msg_and_terminate();

    /* Counting rows (n) and dimensions (d) */
    while ((c = fgetc(f)) != EOF)
    {
        if (c == '\n')
            if (prev_c != '\n')
                (*n)++;
        if (c == ',' && *n == 0)
            (*d)++;
        prev_c = c;
    }
    /* **************************************NOT SURE ABOUT THIS */
    if (prev_c != '\n' && prev_c != EOF)
        (*n)++;

    data_points = (double **)malloc(*n * sizeof(double *));
    if (data_points == NULL)
        err_msg_and_terminate();
    rewind(f);

    for (i = 0; i < *n; i++)
    {
        data_points[i] = (double *)malloc(*d * sizeof(double));
        if (data_points[i] == NULL)
            err_msg_and_terminate();
        for (j = 0; j < *d; j++)
        {
            fscanf(f, "%lf%c", &data_points[i][j], &c);
        }
    }

    fclose(f);
    return data_points;
}

double squared_euclidean_distance(double *p, double *q, int d)
{
    double sum = 0.0;
    int i;
    for (i = 0; i < d; i++)
    {
        sum += (p[i] - q[i]) * (p[i] - q[i]);
    }
    return sum;
}

void free_matrix(double **mat, int n)
{
    int i;
    if (mat == NULL)
        return;

    for (i = 0; i < n; i++)
    {
        free(mat[i]);
    }
    free(mat);
}

double **allocate_matrix(int rows, int cols)
{
    int i;
    double **mat = (double **)malloc(rows * sizeof(double *));
    if (mat == NULL)
        err_msg_and_terminate();

    for (i = 0; i < rows; i++)
    {
        mat[i] = (double *)calloc(cols, sizeof(double));
        if (mat[i] == NULL)
            err_msg_and_terminate();
    }
    return mat;
}

double **create_sym_matrix(double **data_points, int n, int d)
{
    int i, j;
    double **sym_mat = allocate_matrix(n, n);

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (i != j)
            {
                sym_mat[i][j] = exp((squared_euclidean_distance(data_points[i], data_points[j], d)) * (-0.5));
            }
        }
    }
    return sym_mat;
}

double *create_diag_matrix(double **sym_mat, int n)
{
    int i, j;
    double *diag_arr = (double *)calloc(n, sizeof(double));
    if (diag_arr == NULL)
        err_msg_and_terminate();
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            diag_arr[i] += sym_mat[i][j];
        }
    }
    return diag_arr;
}

double **create_norm_matrix(double **sym_mat, double *diag_arr, int n)
{
    int i, j;
    double **norm_mat = allocate_matrix(n, n);
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (diag_arr[i] * diag_arr[j] > 0)
                norm_mat[i][j] = ((sym_mat[i][j]) / (sqrt(diag_arr[i] * diag_arr[j])));
            else
                norm_mat[i][j] = 0.0;
        }
    }
    return norm_mat;
}

void mult_matrices(double **mat1, double **mat2, double **result_mat, int rows1, int common_dim, int cols2)
{
    /*  number of cols in mat1 = number of rows in mat2 = common_dim */
    int i, j, k;

    for (i = 0; i < rows1; i++)
    {
        for (j = 0; j < cols2; j++)
        {
            result_mat[i][j] = 0.0;

            for (k = 0; k < common_dim; k++)
            {
                result_mat[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
}

void transpose_mat(double **mat, double **t_mat, int rows, int cols)
{
    int i, j;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            t_mat[j][i] = mat[i][j];
        }
    }
}

double squared_frobenius_norm(double **mat1, double **mat2, int rows, int cols)
{
    int i, j;
    double sum = 0.0;
    double diff;

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            diff = mat1[i][j] - mat2[i][j];
            sum += (diff * diff);
        }
    }

    return sum;
}

/* Computes the next iteration of H and stores it in next_H */
void compute_next_H(double **W, double **H, double **WH, double **Ht,
                    double **HtH, double **HHtH, double **next_H, int n, int k)
{
    int i, j;
    /* transposes H and stores it in Ht */
    transpose_mat(H, Ht, n, k);

    /* multiply matrices */
    mult_matrices(W, H, WH, n, n, k);
    mult_matrices(Ht, H, HtH, k, n, k);
    mult_matrices(H, HtH, HHtH, n, k, k);

    /* calculate next_H */
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < k; j++)
        {
            next_H[i][j] = H[i][j] * (1.0 - BETA + BETA * (WH[i][j] / HHtH[i][j]));
        }
    }
}

void update_H(double **H, double **next_H, int n, int k)
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < k; j++)
        {
            H[i][j] = next_H[i][j];
        }
    }
}

double **optimize_H(double **W, double **H, int n, int k)
{
    int iter_num = 0;
    double dist;

    /* Allocate working memory once to prevent performance overhead */
    double **WH = allocate_matrix(n, k);
    double **Ht = allocate_matrix(k, n);
    double **HtH = allocate_matrix(k, k);
    double **HHtH = allocate_matrix(n, k);
    double **next_H = allocate_matrix(n, k);

    do
    {
        compute_next_H(W, H, WH, Ht, HtH, HHtH, next_H, n, k);
        dist = squared_frobenius_norm(next_H, H, n, k);
        update_H(H, next_H, n, k);
        iter_num++;
    } while (iter_num < MAX_ITER && dist >= EPSILON);

    /* Free working memory */
    free_matrix(WH, n);
    free_matrix(Ht, k);
    free_matrix(HtH, k);
    free_matrix(HHtH, n);
    free_matrix(next_H, n);

    return H;
}

void print_matrix(double **mat, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf("%.4f", mat[i][j]);
            if (j < n - 1)
                printf(",");
        }
        printf("\n");
    }
}

void print_diagonal_matrix(double *diag_arr, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (i == j)
                printf("%.4f", diag_arr[i]);
            else
                printf("0.0000");

            if (j < n - 1)
                printf(",");
        }
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    char *goal;
    int n, d, is_sym, is_ddg, is_norm;
    double **data_points, **sym_mat, **norm_mat;
    double *diag_arr = NULL;

    if (argc != 3)
        err_msg_and_terminate();
    goal = argv[1];

    is_sym = (strcmp(goal, "sym") == 0);
    is_ddg = (strcmp(goal, "ddg") == 0);
    is_norm = (strcmp(goal, "norm") == 0);

    if (!is_sym && !is_ddg && !is_norm)
        err_msg_and_terminate();

    data_points = read_input(argv[2], &n, &d);
    sym_mat = create_sym_matrix(data_points, n, d);

    if (is_sym)
        print_matrix(sym_mat, n);

    if (is_ddg || is_norm)
        diag_arr = create_diag_matrix(sym_mat, n);

    if (is_ddg)
        print_diagonal_matrix(diag_arr, n);

    if (is_norm)
    {
        norm_mat = create_norm_matrix(sym_mat, diag_arr, n);
        print_matrix(norm_mat, n);
        free_matrix(norm_mat, n);
    }
    if (diag_arr != NULL)
        free(diag_arr);
    free_matrix(sym_mat, n);
    free_matrix(data_points, n);

    return 0;
}