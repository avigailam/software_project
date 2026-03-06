#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

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

double **allocate_matrix(int n)
{
    int i;
    double **mat = (double **)malloc(n * sizeof(double *));
    if (mat == NULL)
        err_msg_and_terminate();

    for (i = 0; i < n; i++)
    {
        mat[i] = (double *)calloc(n, sizeof(double));
        if (mat[i] == NULL)
            err_msg_and_terminate();
    }
    return mat;
}

double **create_sym_matrix(double **data_points, int n, int d)
{
    int i, j;
    double **sym_mat = allocate_matrix(n);

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
    double **norm_mat = allocate_matrix(n);
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